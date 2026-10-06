"""test_rolling_e2e.py — 方案 C 端到端實測（含 A/B 對照）

把音訊「按真實速度」餵給後端 WebSocket，量兩個關鍵數字：
  · 首次上屏延遲：說話開始 → 畫面上出現任何文字
  · 定稿延遲：    說話開始 → 定稿字幕

然後同一段音訊跑兩次：滾動重識別 關 / 開，直接對比。

用法：
  python test_rolling_e2e.py               # 跑內建案例（ja.wav）
  python test_rolling_e2e.py xxx.wav       # 指定音訊

注意：這支腳本會啟一個測試用後端在 8010 埠，不會干擾 8000 的主服務。
"""
import asyncio
import json
import os
import sys
import time
import wave

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

PORT = 8010
URL = f"ws://127.0.0.1:{PORT}/stream"
SAMPLE_RATE = 16000
CHUNK = 1600          # 100ms


def load_wav(path):
    with wave.open(path) as w:
        sr, n = w.getframerate(), w.getnframes()
        a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if sr != SAMPLE_RATE:
        a = a[(np.arange(int(len(a) * SAMPLE_RATE / sr)) * sr / SAMPLE_RATE).astype(int)]
    return a


def speech_offset(audio):
    """回傳語音起點秒數（用來把延遲算在「真的開始說話」上，而不是檔案開頭）。"""
    e = np.abs(audio)
    thr = max(e.max() * 0.02, 1e-4)
    nz = np.where(e > thr)[0]
    return (nz[0] / SAMPLE_RATE) if len(nz) else 0.0


async def run_case(url, audio, target_lang, rolling, label):
    import websockets

    ws = await websockets.connect(url, max_size=None)
    await ws.send(json.dumps({
        "event": "config",
        "ollama_url": "http://127.0.0.1:11434",
        "model_name": "qwen3:4b-instruct",
        "deepseek_key": "",
        "min_silence": 0.35,
        "max_speech": 10.0,
        "source_lang": "auto",
        "target_lang": target_lang,
        "asr_engine": "sensevoice",
        "stream_translate": True,
        "rolling_asr": rolling,
    }))
    await asyncio.sleep(0.6)

    events = []
    t0 = time.perf_counter()

    async def sender():
        for i in range(0, len(audio), CHUNK):
            pcm = (np.clip(audio[i:i + CHUNK], -1, 1) * 32767).astype(np.int16)
            await ws.send(pcm.tobytes())
            want = (i + CHUNK) / SAMPLE_RATE
            dt = want - (time.perf_counter() - t0)
            if dt > 0:
                await asyncio.sleep(dt)

    async def receiver():
        try:
            async for msg in ws:
                d = json.loads(msg)
                if isinstance(d, dict) and str(d.get("event", "")).startswith("subtitle"):
                    events.append((time.perf_counter() - t0, d))
        except Exception:
            pass

    st = asyncio.create_task(sender())
    rt = asyncio.create_task(receiver())
    await st
    await asyncio.sleep(5.0)          # 等定稿與翻譯收尾
    rt.cancel()
    await ws.close()

    # ── 統計 ──
    off = speech_offset(audio)
    first_show = None
    final = None
    partials = []
    for t, d in events:
        ev = d.get("event")
        if ev in ("subtitle_begin", "subtitle_partial") and first_show is None:
            first_show = (t - off, d)
        if ev == "subtitle_partial":
            key = ((d.get("text_raw") or "").strip(), (d.get("text_zh") or "").strip())
            if not partials or partials[-1][1] != key:
                partials.append((t - off, key))
        if ev == "subtitle":
            final = (t - off, d)

    print(f"\n{'='*72}")
    print(f"  {label}   (rolling_asr={rolling})")
    print(f"{'='*72}")
    if first_show:
        print(f"  首次上屏  {first_show[0]:5.2f}s  {str(first_show[1].get('text_zh') or first_show[1].get('text_raw'))[:52]}")
    else:
        print("  首次上屏  —— 沒有收到任何字幕事件")
    if final:
        print(f"  定稿      {final[0]:5.2f}s  {str(final[1].get('text_zh'))[:52]}")
    else:
        print("  定稿      —— 逾時未收到")

    if partials:
        print(f"\n  滾動過程共 {len(partials)} 次更新：")
        for t, (rw, zh) in partials:
            if zh:
                print(f"    {t:5.2f}s  [原文] {rw}")
                print(f"    {'':5s}  [譯文] {zh}")
            else:
                print(f"    {t:5.2f}s  [原文] {rw}")
    return {
        "first_show": first_show[0] if first_show else None,
        "final": final[0] if final else None,
        "n_partials": len(partials),
    }


async def main():
    import uvicorn
    import main as backend

    wavs = sys.argv[1:] or [os.path.join(
        BASE_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17", "test_wavs", "ja.wav")]

    cfg = uvicorn.Config(backend.app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(cfg)
    srv = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.2)
    print(f"測試後端已啟動於 :{PORT}")

    results = []
    for wav in wavs:
        audio = load_wav(wav)
        name = os.path.basename(wav)
        print(f"\n\n{'#'*72}\n# 音訊 {name}   長度 {len(audio)/SAMPLE_RATE:.2f}s\n{'#'*72}")
        for rolling in (False, True):
            label = "A 方案：等 VAD 定段（原行為）" if not rolling else "C 方案：滾動重識別"
            try:
                r = await run_case(URL, audio, "zh-CN", rolling, f"[{name}] {label}")
                r["name"] = name
                r["rolling"] = rolling
                results.append(r)
            except Exception as e:
                print(f"\n  以 {label} 執行時出錯: {type(e).__name__}: {e}")
            await asyncio.sleep(1.0)

    print(f"\n\n{'='*72}\n  對照總表（秒，從「真的開始說話」起算）\n{'='*72}")
    print(f"  {'音訊':<12}{'方案':<10}{'首次上屏':>10}{'定稿':>10}{'滾動版數':>10}")
    for r in results:
        tag = "滾動" if r["rolling"] else "原分段"
        fs = f"{r['first_show']:.2f}" if r["first_show"] is not None else "—"
        fn = f"{r['final']:.2f}" if r["final"] is not None else "—"
        print(f"  {r['name']:<12}{tag:<10}{fs:>10}{fn:>10}{r['n_partials']:>10}")

    server.should_exit = True
    await srv


if __name__ == "__main__":
    asyncio.run(main())
