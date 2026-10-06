import os
import sys
import re
import json
import time
import asyncio
from pathlib import Path


# ── 讓本機回環請求繞過系統代理 ────────────────────────────────────────
# 為什麼非做不可（實測過，不是理論）：
#   httpx 預設 trust_env=True，會從 Windows 登錄檔讀 IE/WinINET 代理設定。
#   urllib 會拿 ProxyOverride 裡的 127.* 判斷 bypass，**但 httpx 不這樣做** ——
#   它只認 getproxies()["no"] 這個鍵，而登錄檔那條路徑不會給出 "no"，
#   於是 httpx 對「所有 http:// 請求」都掛上代理，本機回環也不例外。
#
# 後果（三個都是「本機回環走代理」才會有的）：
#   1. 每次呼叫 Ollama 都多一趟「連代理 → 代理再連回本機」的往返。字幕管線
#      每秒要打數次，開「滾動字幕」後更頻繁，這個來回純屬白費。
#   2. 代理沒開或掛掉時，本機 Ollama 也跟著連不上 —— 服務明明就在本機，
#      卻因為繞了代理而失效。
#   3. Ollama 永遠在本機，本來就沒有任何理由走代理。
#
# 註：這不是「避免把代理打爆」的防禦措施。曾懷疑本機回環的高頻短連接會打掛
#   Clash 內核，後續以 8 份內核日誌交叉比對後已推翻該推論（命中 0 次照樣崩、
#   命中 200 次反而活了 8 分鐘；崩潰棧最內層是 HTTPS 的 crypto/tls，而 Ollama
#   走的是明文 HTTP）。真正的根因是舊內核以 go1.18beta1 建置。
def _ensure_loopback_no_proxy():
    cur = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [p.strip() for p in cur.split(",") if p.strip()]
    for host in ("127.0.0.1", "localhost", "::1"):
        if host not in parts:
            parts.append(host)
    merged = ",".join(parts)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


_ensure_loopback_no_proxy()


# Add NVIDIA pip packages' DLL paths to Windows DLL search directory
def add_nvidia_dll_paths():
    dll_dirs = []
    
    # 1. 偵測本機開發環境的 NVIDIA CUDA/cuDNN 依賴套件路徑
    site_packages_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if site_packages_nvidia.exists() and site_packages_nvidia.is_dir():
        print(f"偵測到本機 NVIDIA 依賴套件路徑: {site_packages_nvidia}")
        for path in site_packages_nvidia.glob("**/bin"):
            if path.is_dir():
                dll_dirs.append(str(path.resolve()))
                
    # 2. 針對 PyInstaller 打包後的路徑，尋找根目錄與 sherpa_onnx DLL 目錄
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        # 增加根目錄 (內含 CUDA/cuDNN DLLs)
        dll_dirs.append(base_dir)
        
        # 增加 _internal/sherpa_onnx/lib (內含 sherpa-onnx, onnxruntime DLLs)
        sherpa_lib_dir = os.path.join(base_dir, "_internal", "sherpa_onnx", "lib")
        if os.path.exists(sherpa_lib_dir):
            dll_dirs.append(sherpa_lib_dir)

    # 執行載入
    for dll_dir in dll_dirs:
        try:
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(dll_dir)
                print(f"已成功載入 DLL 目錄 (add_dll_directory): {dll_dir}")
        except Exception as e:
            print(f"載入 DLL 目錄 {dll_dir} 失敗 (add_dll_directory): {e}")
            
    if dll_dirs:
        # 同時將這些目錄加入 PATH 環境變數，確保 C++ 底層 DLL 可以透過傳統搜尋尋找到依賴項
        os.environ["PATH"] = ";".join(dll_dirs) + ";" + os.environ["PATH"]
        print("已將所有依賴 DLL 目錄加入 OS PATH 變數！")

add_nvidia_dll_paths()

import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import sherpa_onnx
from opencc import OpenCC

# 初始化簡繁體轉換器
cc_s2t = OpenCC('s2t')
cc_t2s = OpenCC('t2s')

app = FastAPI(title="Studio0808_LiveCaption Backend")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
import sys
if getattr(sys, 'frozen', False):
    # 打包後的執行檔路徑（LiveCaptionServer.exe 所在的目錄）
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # 開發模式下的 Python 檔案路徑
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VAD_MODEL_PATH = os.path.join(BASE_DIR, "silero_vad.onnx")

# Find SenseVoice directory
SENSE_VOICE_DIR = os.path.join(BASE_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
SENSE_VOICE_MODEL = os.path.join(SENSE_VOICE_DIR, "model.int8.onnx")
SENSE_VOICE_TOKENS = os.path.join(SENSE_VOICE_DIR, "tokens.txt")

# Global models initialized on startup / on demand
vad_detector = None
asr_recognizer = None
current_asr_engine = None  # 追蹤當前加載的 ASR 引擎名稱："sensevoice" 或 "whisper"
current_whisper_lang = None  # 追蹤當前 Whisper 加載的語言

def init_vad():
    global vad_detector
    if vad_detector is not None:
        return
    if not os.path.exists(VAD_MODEL_PATH):
        raise FileNotFoundError(f"找不到 VAD 模型: {VAD_MODEL_PATH}，請先執行 download_models.py")
    
    print("正在初始化 Silero VAD 模型...")
    vad_config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=VAD_MODEL_PATH,
            threshold=0.4,
            min_silence_duration=0.35,  # 實測 0.35s 最平衡：空檔比 0.5s 少 250ms，
                                        # 且 0.30s 以內的自然停頓仍不會被切分
            min_speech_duration=0.15,
            max_speech_duration=10.0,  # 最長單句 10 秒強迫切分
        ),
        sample_rate=16000,
    )
    vad_detector = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

def release_asr_model():
    global asr_recognizer, current_asr_engine, current_whisper_lang
    if asr_recognizer is not None:
        print(f"正在卸載當前 AI 模型資源 ({current_asr_engine})...")
        asr_recognizer = None
        current_asr_engine = None
        current_whisper_lang = None
        import gc
        gc.collect()
        print("資源釋放完成。")

def init_sensevoice():
    global asr_recognizer, current_asr_engine
    if asr_recognizer is not None and current_asr_engine == "sensevoice":
        return
        
    release_asr_model()
    
    if not os.path.exists(SENSE_VOICE_MODEL):
        raise FileNotFoundError(f"找不到 SenseVoice 模型: {SENSE_VOICE_MODEL}，請先執行 download_models.py")
        
    # 嘗試以 GPU (CUDA) 載入，若失敗則 Fallback 到 CPU 載入
    try:
        print("嘗試以 GPU (CUDA) 初始化 SenseVoice ASR 模型...")
        asr_recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=SENSE_VOICE_MODEL,
            tokens=SENSE_VOICE_TOKENS,
            num_threads=4,
            use_itn=True,
            provider="cuda",
        )
        print("SenseVoice GPU (CUDA) 模型載入成功！")
    except Exception as cuda_err:
        print(f"SenseVoice GPU 初始化失敗 ({cuda_err})，將自動降級至 CPU 推理模式...")
        asr_recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=SENSE_VOICE_MODEL,
            tokens=SENSE_VOICE_TOKENS,
            num_threads=4,
            use_itn=True,
            provider="cpu",
        )
        print("SenseVoice 離線 CPU 模型載入成功！")
        
    current_asr_engine = "sensevoice"
    print("SenseVoice 離線 AI 模型載入成功！")

def init_whisper(lang: str = "auto"):
    global asr_recognizer, current_asr_engine, current_whisper_lang
    
    # 決定 whisper 語言代碼
    whisper_lang = lang
    if whisper_lang == "auto":
        whisper_lang = ""
    elif whisper_lang in ["zh-TW", "zh-CN"]:
        whisper_lang = "zh"
        
    # 如果已加載 Whisper 且語言相同，就直接返回
    if asr_recognizer is not None and current_asr_engine == "whisper" and current_whisper_lang == whisper_lang:
        return
        
    release_asr_model()
    
    # 優先尋找 whisper-small，如果沒有則嘗試 whisper-base
    WHISPER_DIR = os.path.join(BASE_DIR, "sherpa-onnx-whisper-small")
    if not os.path.exists(WHISPER_DIR):
        WHISPER_DIR = os.path.join(BASE_DIR, "sherpa-onnx-whisper-base")
        
    if not os.path.exists(WHISPER_DIR):
        raise FileNotFoundError(f"找不到 Whisper 模型資料夾 (請執行 download_models.py): {WHISPER_DIR}")
        
    # 探測潛在檔名 (同時相容 small 和 base 規格)
    possible_encoders = [
        "whisper-small-encoder.int8.onnx", "whisper-small-encoder.onnx",
        "small-encoder.int8.onnx", "small-encoder.onnx",
        "whisper-base-encoder.int8.onnx", "whisper-base-encoder.onnx", 
        "base-encoder.int8.onnx", "base-encoder.onnx", 
        "encoder.int8.onnx", "encoder.onnx"
    ]
    possible_decoders = [
        "whisper-small-decoder.int8.onnx", "whisper-small-decoder.onnx",
        "small-decoder.int8.onnx", "small-decoder.onnx",
        "whisper-base-decoder.int8.onnx", "whisper-base-decoder.onnx", 
        "base-decoder.int8.onnx", "base-decoder.onnx", 
        "decoder.int8.onnx", "decoder.onnx"
    ]
    possible_tokens = [
        "whisper-small-tokens.txt", "small-tokens.txt",
        "whisper-base-tokens.txt", "base-tokens.txt", "tokens.txt"
    ]
    
    encoder_path = next((os.path.join(WHISPER_DIR, n) for n in possible_encoders if os.path.exists(os.path.join(WHISPER_DIR, n))), None)
    decoder_path = next((os.path.join(WHISPER_DIR, n) for n in possible_decoders if os.path.exists(os.path.join(WHISPER_DIR, n))), None)
    tokens_path = next((os.path.join(WHISPER_DIR, n) for n in possible_tokens if os.path.exists(os.path.join(WHISPER_DIR, n))), None)
    
    if not encoder_path or not decoder_path or not tokens_path:
        raise FileNotFoundError(f"找不到完整的 Whisper 模型檔案 (需要 encoder, decoder, tokens) 於: {WHISPER_DIR}")
        
    # 嘗試以 GPU (CUDA) 載入，若失敗則 Fallback 到 CPU 載入
    try:
        print(f"嘗試以 GPU (CUDA) 初始化 Whisper ASR 模型 (語言: '{whisper_lang}' | 來源: '{lang}')...")
        asr_recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder_path,
            decoder=decoder_path,
            tokens=tokens_path,
            num_threads=4,
            language=whisper_lang,
            task="transcribe",
            provider="cuda",  # 啟用 CUDA GPU 推理
        )
        print("Whisper GPU (CUDA) 模型載入成功！")
    except Exception as cuda_err:
        print(f"GPU 初始化失敗 ({cuda_err})，將自動降級至 CPU 推理模式...")
        asr_recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder_path,
            decoder=decoder_path,
            tokens=tokens_path,
            num_threads=4,
            language=whisper_lang,
            task="transcribe",
            provider="cpu",
        )
        print("Whisper 離線 CPU 模型載入成功！")
        
    current_asr_engine = "whisper"
    current_whisper_lang = whisper_lang
    print("Whisper 離線 AI 模型載入成功！")

def init_models():
    # 啟動時只初始化 VAD，ASR 延遲到 WebSocket 連線與配置時載入
    init_vad()

# 追蹤閒置連線與釋放記憶體/顯存相關變數
active_connections = 0
last_active_time = None

async def monitor_idle_timeout():
    global asr_recognizer, vad_detector, active_connections, last_active_time
    import gc
    import time
    # 閒置超時時間設定為 10 分鐘 (600 秒)
    IDLE_TIMEOUT = 600
    
    while True:
        await asyncio.sleep(30)  # 每 30 秒檢查一次
        if active_connections == 0 and last_active_time is not None:
            elapsed = time.time() - last_active_time
            if elapsed >= IDLE_TIMEOUT:
                if asr_recognizer is not None or vad_detector is not None:
                    print(f"後端閒置已達 {IDLE_TIMEOUT // 60} 分鐘，開始主動釋放 AI 模型資源...")
                    release_asr_model()
                    vad_detector = None
                    gc.collect()
                    print("後端 AI 模型資源釋放完成！")

@app.on_event("startup")
def startup_event():
    try:
        init_models()
        # 啟動閒置監控工作
        asyncio.create_task(monitor_idle_timeout())
    except Exception as e:
        print(f"啟動初始化失敗: {e}")

def clean_sense_voice_text(text: str) -> str:
    # 移除 SenseVoice 的特有標籤，例如 <|zh|>, <|NEUTRAL|>, <|speech|> 等
    cleaned = re.sub(r'<\|.*?\|>', '', text)
    # 清理多餘空白
    return cleaned.strip()

# 全局變數，用來快取 Ollama 是否在線，避免每次都等待 3 秒超時
ollama_online = True

LANG_MAP = {
    "zh-TW": "繁體中文 (Traditional Chinese)",
    "zh-CN": "簡體中文 (Simplified Chinese)",
    "en": "英文 (English)",
    "ja": "日文 (Japanese)",
    "ko": "韓文 (Korean)",
    "es": "西班牙文 (Spanish)",
    "fr": "法文 (French)",
    "de": "德文 (German)",
    "ru": "俄文 (Russian)",
}

def build_system_prompt(target_lang: str, model_name: str) -> str:
    """依目標語言組出 system prompt（串流與非串流共用，確保兩種路徑行為一致）。

    ⚠️ system prompt 本身的用字會「傳染」給模型：簡體目標若用繁體撰寫 prompt，
       小模型（qwen2.5:3b 等）會大量漏出繁體字（實測 5/11 句），並偶爾漏譯。
       因此簡體目標必須用簡體 prompt，繁體目標用繁體 prompt，其餘語言用繁體版。
    """
    target_name = LANG_MAP.get(target_lang, "繁體中文")
    if target_lang == "zh-TW":
        return (
            "你是一個專業的影片字幕即時翻譯官。請將使用者輸入的影片語音字幕，翻譯成簡短流暢的繁體中文。"
            "輸出必須是「繁體中文（台灣用語）」，絕對禁止輸出任何簡體字。"
            "只輸出翻譯後的譯文本身，並輸出為「單獨一行純文字」；"
            "嚴禁輸出原文、注音、拼音、解釋、引言、括號標註、清單符號、換行或任何額外標記。"
        )
    if target_lang in ("zh-CN", "zh", "zh-Hans"):
        return (
            "你是一个专业的影片字幕即时翻译官。请把用户输入的影片语音字幕，翻译成简短流畅的简体中文。"
            "输出必须全部使用简体中文汉字，禁止使用繁体字。"
            "只输出翻译后的译文本身，并输出为「单独一行纯文字」；"
            "严禁输出原文、注音、拼音、解释、引言、括号标注、清单符号、换行或任何额外标记。"
        )
    return (
        f"你是一個專業的影片字幕即時翻譯官。請將使用者輸入的影片語音字幕，翻譯成簡短流暢的{target_name}。"
        f"輸出必須是{target_name}。"
        "只輸出翻譯後的譯文本身，並輸出為「單獨一行純文字」；"
        "嚴禁輸出原文、注音、拼音、解釋、引言、括號標註、清單符號、換行或任何額外標記。"
    )


def add_no_think(system_prompt: str, model_name: str) -> str:
    """Qwen3 系列預設會先吐一長串思考鏈（thinking），字幕即時場景必須關閉，
    否則延遲暴增數倍。/no_think 由 Qwen3 官方 chat template 識別。"""
    if model_name and model_name.lower().startswith("qwen3"):
        return system_prompt + " /no_think"
    return system_prompt


def convert_script(text: str, target_lang: str) -> str:
    """繁體/簡體目標各過一道 OpenCC，消除殘留異體字（天气→天氣、酋長→酋长）。
    OpenCC 為逐字映射且冪等，因此串流時對「累積文本」反覆轉換也安全。"""
    if not text:
        return text
    if target_lang == "zh-TW":
        return cc_s2t.convert(text)
    if target_lang in ("zh-CN", "zh", "zh-Hans"):
        return cc_t2s.convert(text)
    return text


def flatten_text(text: str) -> str:
    """壓平殘留的換行與多餘空白，消除「分行/分段」型亂碼。"""
    return " ".join(text.split())


def deepseek_system_prompt(target_lang: str) -> str:
    target_name = LANG_MAP.get(target_lang, "繁體中文")
    return (
        f"你是一個專業的影片字幕即時翻譯官。請將輸入的影片語音字幕，翻譯成簡短流暢的{target_name}。"
        "請只輸出翻譯後的文字，不要包含任何解釋、引言或額外標記，保持字數與原句差不多。"
    )


async def translate_text(text: str, target_lang: str, ollama_url: str, model_name: str, deepseek_key: str = None) -> str:
    global ollama_online
    """
    三軌翻譯引擎：支援線上 DeepSeek API、本機 Ollama、以及免費的 Google Translate API (終極備用)。
    """
    target_name = LANG_MAP.get(target_lang, "繁體中文")

    # 1. 優先嘗試 DeepSeek API (若有提供 API Key)
    if deepseek_key and len(deepseek_key.strip()) > 10:
        try:
            _t0 = time.perf_counter()
            # 逾時由 3s 放寬至 8s：實測單句約 0.45~0.9s，但雲端 API 偶有抖動。
            # 逾時後會落到 Ollama(通常未啟動) 與 Google(部分網路不可達) 兩道無效備援，
            # 最終輸出「[未翻譯]」比多等 1~2 秒更糟，故給予合理餘裕。
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {deepseek_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": deepseek_system_prompt(target_lang)},
                            {"role": "user", "content": text}
                        ],
                        "temperature": 0.3
                    }
                )
                if response.status_code == 200:
                    res_json = response.json()
                    translated = res_json["choices"][0]["message"]["content"].strip()
                    print(f"  ➔ [DeepSeek 翻譯] (耗時 {(time.perf_counter()-_t0)*1000:.0f} ms)")
                    return translated
        except Exception as e:
            print(f"DeepSeek 翻譯失敗: {e}")

    # 2. 本機離線 Ollama 翻譯
    if ollama_url and ollama_online:
        try:
            # 依目標語言組 system prompt，並對 Qwen3 關閉思考鏈（細節見 build_system_prompt）
            system_prompt = add_no_think(build_system_prompt(target_lang, model_name), model_name)
            _t0 = time.perf_counter()
            # trust_env=False：Ollama 永遠在本機，絕不該繞系統代理（詳見檔頭 _ensure_loopback_no_proxy）
            async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
                # 改用 system + 乾淨 prompt（套用模型 chat 模板），比補全式 prompt 更穩定，避免亂碼輸出
                response = await client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model_name,
                        "system": system_prompt,
                        "prompt": text,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 80
                        }
                    }
                )
                if response.status_code == 200:
                    res_json = response.json()
                    # 先壓平殘留換行與多餘空白，再過一道 OpenCC 修正字形殘留：
                    #   繁體目標 s2t (天气→天氣)、簡體目標 t2s (酋長→酋长)。兩者皆冪等。
                    translated = convert_script(flatten_text(res_json.get("response", "")), target_lang)
                    if translated:
                        print(f"  ➔ [Ollama 翻譯 ({model_name})] (耗時 {(time.perf_counter()-_t0)*1000:.0f} ms)")
                        return translated
        except Exception as e:
            # 區分兩種失敗：服務沒開 vs 模型載入中(冷啟動)
            if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
                # 連不上 Ollama 服務 → 本工作階段跳過，避免每句都空等逾時
                print(f"本機 Ollama 連線失敗 ({type(e).__name__})，服務可能未啟動，本工作階段後續將跳過 Ollama。")
                ollama_online = False
            else:
                # 多為 ReadTimeout：服務有回應但模型正在載入(冷啟動)。
                # 本句先用後備引擎，但「不」永久關閉 Ollama，下一句模型暖機後會自動接手。
                print(f"本機 Ollama 本次回應逾時 ({type(e).__name__})，可能為模型載入中；本句改用後備引擎，下一句再試 Ollama。")

    # 3. 終極備用：免費 Google Translate Web API (免 Key、免配置、即開即用)
    try:
        _t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=3.0) as client:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": "auto",
                "tl": target_lang,
                "dt": "t",
                "q": text
            }
            response = await client.get(url, params=params)
            if response.status_code == 200:
                res_json = response.json()
                translated = "".join([part[0] for part in res_json[0] if part[0]])
                print(f"  ➔ [Google 翻譯] (耗時 {(time.perf_counter()-_t0)*1000:.0f} ms)")
                return translated.strip()
    except Exception as e:
        print(f"Google 翻譯失敗: {e}")
        
    # 如果都失敗，則返回原文字
    print(f"  ➔ [所有翻譯引擎皆失敗，保留原文]")
    return f"[未翻譯] {text}"


async def translate_text_stream(text: str, target_lang: str, ollama_url: str, model_name: str, deepseek_key: str = None):
    """串流版翻譯：逐塊 yield「截至目前為止的累積譯文」（不是增量片段）。

    為什麼 yield 累積文本而不是 delta？
      字幕前端只做「整行替換」，累積文本讓前端不必自己拼接，
      且天然免疫封包亂序/丟失造成的文字錯位。

    契約：只有在真的吐出過至少一個 chunk 時才算成功。
      若不支援串流或連線失敗而「一個 chunk 都沒吐」，呼叫方會退回非串流的
      translate_text()，確保字幕永遠不會空白。
    """
    global ollama_online

    # 1. 雲端 DeepSeek API（OpenAI 相容 SSE：每行 "data: {json}"，以 "data: [DONE]" 收尾）
    if deepseek_key and len(deepseek_key.strip()) > 10:
        try:
            _t0 = time.perf_counter()
            acc = ""
            async with httpx.AsyncClient(timeout=8.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {deepseek_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": deepseek_system_prompt(target_lang)},
                            {"role": "user", "content": text}
                        ],
                        "temperature": 0.3,
                        "stream": True
                    }
                ) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                break
                            try:
                                delta = json.loads(payload)["choices"][0]["delta"].get("content") or ""
                            except Exception:
                                continue  # 心跳/空行/格式異常 → 跳過該塊
                            if not delta:
                                continue
                            acc += delta
                            yield flatten_text(acc)
                        if acc.strip():
                            print(f"  ➔ [DeepSeek 串流翻譯] (耗時 {(time.perf_counter()-_t0)*1000:.0f} ms)")
                            return
        except Exception as e:
            print(f"DeepSeek 串流翻譯失敗 ({type(e).__name__}): {e}")

    # 2. 本機 Ollama（NDJSON：每行一個完整 JSON，直到 done=true）
    if ollama_url and ollama_online:
        try:
            system_prompt = add_no_think(build_system_prompt(target_lang, model_name), model_name)
            _t0 = time.perf_counter()
            acc = ""
            # trust_env=False：Ollama 永遠在本機，不該繞系統代理（詳見檔頭 _ensure_loopback_no_proxy）
            async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
                async with client.stream(
                    "POST",
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model_name,
                        "system": system_prompt,
                        "prompt": text,
                        "stream": True,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 80
                        }
                    }
                ) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line).get("response", "")
                            except Exception:
                                continue
                            if not chunk:
                                continue
                            acc += chunk
                            # 每塊都重新壓平 + 轉字形（兩者皆冪等，對累積文本安全）
                            yield convert_script(flatten_text(acc), target_lang)
                        if acc.strip():
                            print(f"  ➔ [Ollama 串流翻譯 ({model_name})] (耗時 {(time.perf_counter()-_t0)*1000:.0f} ms)")
                            return
        except Exception as e:
            # 區分兩種失敗：服務沒開 vs 模型載入中(冷啟動)
            if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
                print(f"本機 Ollama 連線失敗 ({type(e).__name__})，服務可能未啟動，本工作階段後續將跳過 Ollama。")
                ollama_online = False
            else:
                print(f"本機 Ollama 串流翻譯失敗 ({type(e).__name__})，本句將退回非串流路徑。")

    # 3. Google 免費通道不支援串流 → 這裡直接放棄，由呼叫方退回 translate_text() 處理


# ─────────────────────────────────────────────────────────────────────
# 滾動重識別（方案 C）
# ─────────────────────────────────────────────────────────────────────
# 背景：這個管線原本是「等 VAD 定段 → 才 ASR → 才翻譯」，說話人不停頓就一個
# 字都不出。實測逐字稿裡典型一段 5~9 秒、最長 20 秒 —— 首字延遲有 96% 花在
# 「等停頓」，ASR 與翻譯合計只佔 4%。
#
# 業界做法一致：Zoom / Google Meet / 騰訊會議 都是「邊說邊出臨時結果，定段
# 後原地覆蓋成定稿」。學界同樣有明確對應 —— Whisper-Streaming 的
# LocalAgreement-2、IWSLT 2022 同傳任務、EMNLP 2023 demo 的 "revision mode"
#（論文原文：「the last hypothesis except the common prefix is sent as
# unstable output」，且不穩定文本同樣送進翻譯模型）。
# Google Cloud STT 的 API 直接把這件事寫成欄位：isFinal=false 的 interim
# result「may change」，另附 stability 0.0~1.0。
# Zoom 的官方使用說明也直白：「if you pause slightly at the end of a sentence
# the captions will often go back and correct a word based on context」。
#
# 因此使用者看到的「文字邊打邊改」不是瑕疵，而是這類系統的正常型態。
#
# 實作要點：沿用 sherpa-onnx VAD 的 current_segment（說話中的即時音訊），
# 每 ROLL_INTERVAL 秒對「整段累積音訊」重跑一次 ASR。因為每次都是從頭重跑，
# 不穩定的只有尾端 —— 實測臨時文本與定稿的公共前綴單調增長（9%→40%→94%），
# 11 輪中 0 次回縮，所以不會出現「字消失」的閃爍。
ROLL_INTERVAL = 0.5      # 臨時 ASR 節奏（秒）
ROLL_MIN_AUDIO = 0.6     # 短於這個秒數不值得跑
ROLL_MAX_AUDIO = 15.0    # 長於這個秒數就跳過該輪（單次解碼會拖太久）


def plan_translation(raw_text, lang_tag, target_lang, source_lang, engine):
    """決定一句話該怎麼產出顯示文本 → (kind, payload, recognized_lang)。

    kind = 'raw'   直接沿用原文（沒開翻譯，或識別語言已等於目標語言）
           'local' 本地 OpenCC 簡繁轉換（中文對中文，零成本）
           'mt'    需送翻譯引擎

    定稿流程與滾動臨時字幕共用這一份判斷，避免兩處條件漂移
    （例如只有定稿會處理 zh→zh-TW，臨時字幕卻漏掉）。

    lang_tag 要傳 SenseVoice 的 result.lang（例如 '<|zh|>'）。
    ⚠ 這裡曾有一個長期潛伏的 bug：舊寫法是在 result.text 裡搜尋 '<|zh|>'，
      但 sherpa-onnx 根本不會把語言標籤寫進 text，於是 recognized_lang 永遠
      是 'auto'，中文→中文那段本地 OpenCC 分支是**死碼**，
      所有中文語音都被多送一次 LLM「翻譯」，不但白費 GPU，還會把內容改壞
      （實測：「开饭时间」被翻成「开放时间」，語意直接跑掉）。
    """
    recognized = "auto"
    if engine == "whisper":
        if source_lang != "auto":
            recognized = source_lang
    else:
        t = (lang_tag or "").strip().lower()
        # 粵語獨立成一類：它的書面語就是粵語，不能拿去跟繁體/簡體中文做
        # OpenCC 互轉（那只轉字形、不轉詞彙），必須交給翻譯模型。
        # 面板沒有粵語目標選項，因此這樣分類不會改變既有行為。
        if "<|yue|>" in t or t in ("yue", "cantonese"):
            recognized = "yue"
        elif "<|zh|>" in t or t in ("zh", "chinese", "mandarin"):
            recognized = "zh"
        elif "<|en|>" in t or t in ("en", "english"):
            recognized = "en"
        elif "<|ja|>" in t or t in ("ja", "japanese"):
            recognized = "ja"
        elif "<|ko|>" in t or t in ("ko", "korean"):
            recognized = "ko"

    if target_lang == "none":
        return "raw", raw_text, recognized
    if target_lang == recognized:
        return "raw", raw_text, recognized
    if recognized == "zh" and target_lang in ("zh-TW", "zh-CN"):
        return "local", convert_script(raw_text, target_lang), recognized
    return "mt", raw_text, recognized


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    global active_connections, asr_recognizer, vad_detector, last_active_time
    await websocket.accept()
    active_connections += 1
    print(f"WebSocket 客戶端已連線。當前連線數: {active_connections}")
    
    # 確保 VAD 已載入
    if vad_detector is None:
        init_vad()
        
    # 建立會議記錄存檔目錄與檔案
    import datetime
    transcripts_dir = os.path.join(os.path.dirname(BASE_DIR), "transcripts")
    try:
        os.makedirs(transcripts_dir, exist_ok=True)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file_path = os.path.join(transcripts_dir, f"transcript_{now_str}.md")
        with open(transcript_file_path, "w", encoding="utf-8") as f:
            f.write(f"# LiveCaption 語音辨識與翻譯會議紀錄\n\n")
            f.write(f"*   **開始時間**：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"*   **存檔路徑**：{transcript_file_path}\n")
            f.write(f"---\n\n")
        print(f"已建立本次會議/影片紀錄存檔：{transcript_file_path}")
    except Exception as e:
        print(f"建立紀錄存檔目錄或檔案失敗: {e}")
        transcript_file_path = None
    
    # 建立連線專用的獨立 VAD 實例以避免不同連線互相干擾
    vad_config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=VAD_MODEL_PATH,
            threshold=0.4,
            min_silence_duration=0.35,
            min_speech_duration=0.15,
            max_speech_duration=10.0,
        ),
        sample_rate=16000,
    )
    local_vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
    
    # 設定選項 (預設為 Ollama 本地)
    ollama_url = "http://127.0.0.1:11434"
    model_name = "qwen3:4b-instruct"
    deepseek_key = None
    source_lang = "auto"
    target_lang = "none"
    asr_engine = "sensevoice"
    # 串流翻譯：開啟後譯文逐字回推，首字約 100~300ms 就出現在畫面上
    stream_translate = False
    # 滾動重識別：說話中就持續出臨時字幕，不等 VAD 定段
    rolling_asr = True

    # ── 滾動重識別的連線內狀態 ──
    # 放在 dict 裡是為了讓巢狀協程能直接改，不必用 nonlocal 到處宣告
    rs = {"task": None, "stop": True, "raw": "", "zh": "", "start": None}
    # sherpa-onnx 的解碼是同步 C++ 呼叫，丟到執行緒池才不會把「收音訊」一起卡住；
    # 但同一時間只能有一個解碼在跑（多個 stream 共用同一份模型權重）。
    asr_lock = asyncio.Lock()

    async def asr_decode_async(samples):
        """在執行緒池裡跑一次 ASR，回傳 (文字, 語言標籤)。

        語言標籤取自 result.lang（SenseVoice 會給 '<|zh|>' 這種值）。
        不要試圖從 result.text 裡找標籤 —— 那裡不會有。
        """
        def _run():
            st = asr_recognizer.create_stream()
            st.accept_waveform(16000, samples)
            asr_recognizer.decode_stream(st)
            return st.result.text, (getattr(st.result, "lang", "") or "")

        async with asr_lock:
            if asr_recognizer is None:
                return "", ""
            return await asyncio.get_running_loop().run_in_executor(None, _run)

    async def settle_rolling():
        """定段時先收掉滾動迴圈。

        必須「等它結束」而不是直接 cancel：解碼跑在執行緒池裡，cancel 不會
        真的中止那個 C++ 呼叫，定稿若緊接著解碼就有並發存取的風險。等待上限
        取 2s，正常情況下迴圈會在一個回合內（≤400ms）自己看到 stop 退出。
        """
        rs["stop"] = True
        task = rs["task"]
        rs["task"] = None
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
        except Exception:
            pass

    async def rolling_loop():
        """方案 C 主迴圈：說話中每 ROLL_INTERVAL 秒出一版臨時字幕。"""
        rs["stop"] = False
        rs["raw"] = ""
        rs["zh"] = ""
        rs["start"] = None
        try:
            while not rs["stop"]:
                await asyncio.sleep(ROLL_INTERVAL)
                if rs["stop"]:
                    break

                seg = local_vad.current_segment
                if seg is None:
                    continue
                samples = seg.samples
                n_sec = len(samples) / 16000.0
                if n_sec < ROLL_MIN_AUDIO or n_sec > ROLL_MAX_AUDIO:
                    continue

                raw_full, lang_tag = await asr_decode_async(samples)
                if rs["stop"]:
                    break
                if current_asr_engine == "whisper":
                    raw = raw_full.strip()
                else:
                    raw = clean_sense_voice_text(raw_full)
                # 文本沒變就不用重送、更不用重翻
                if not raw or raw == rs["raw"]:
                    continue
                rs["raw"] = raw
                start_time = seg.start / 16000.0
                rs["start"] = start_time

                kind, payload, _ = plan_translation(
                    raw, lang_tag, target_lang, source_lang, current_asr_engine
                )

                # 原文先上屏：這一步不花任何模型成本。
                # 但需要翻譯時，若直接把「上一輪的譯文」墊在後面，會出現原文已更新、
                # 譯文還停在上一句的錯位（實測：原文「开放时间早上9点」配譯文
                # 「待放时间」），雙語模式下很難讀。所以只有這一行還沒出現過任何
                # 文字時才先送原文，之後一律等譯文一起送，確保同行原文與譯文對得上。
                if kind != "mt" or not rs["zh"]:
                    await websocket.send_json({
                        "event": "subtitle_partial",
                        "text_raw": raw,
                        "text_zh": rs["zh"] if kind == "mt" else payload,
                        "start": start_time,
                        "duration": n_sec,
                    })

                if kind != "mt":
                    rs["zh"] = payload
                    continue

                # 臨時譯文：臨時結果反正會被定稿覆蓋，不做串流。
                # 不需要額外的節流 —— 迴圈每次都會先 sleep(ROLL_INTERVAL)，
                # 節奏天然就由 ASR + MT 的耗時決定（實測約每秒一版）。
                zh = await translate_text(raw, target_lang, ollama_url, model_name, deepseek_key)
                if rs["stop"]:
                    break
                if not zh:
                    continue
                rs["zh"] = zh
                await websocket.send_json({
                    "event": "subtitle_partial",
                    "text_raw": raw,
                    "text_zh": zh,
                    "start": start_time,
                    "duration": n_sec,
                })
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"滾動重識別迴圈中止 ({type(e).__name__}): {e}")

    try:
        while True:
            # 接收前端發送的封包
            message = await websocket.receive()
            
            # 如果收到的是文字設定訊息
            if "text" in message:
                try:
                    config_data = json.loads(message["text"])
                    if config_data.get("event") == "config":
                        ollama_url = config_data.get("ollama_url", ollama_url).rstrip("/")
                        model_name = config_data.get("model_name", model_name)
                        deepseek_key = config_data.get("deepseek_key", deepseek_key)
                        source_lang = config_data.get("source_lang", source_lang)
                        target_lang = config_data.get("target_lang", target_lang)
                        asr_engine = config_data.get("asr_engine", asr_engine)
                        stream_translate = bool(config_data.get("stream_translate", stream_translate))
                        rolling_asr = bool(config_data.get("rolling_asr", rolling_asr))
                        global ollama_online
                        ollama_online = True

                        # 面板上把滾動關掉時，正在跑的那一輪要收手，否則會繼續往
                        # 同一行寫臨時字幕。
                        if not rolling_asr:
                            await settle_rolling()
                        
                        # 動態加載或切換 ASR 引擎
                        if asr_engine == "whisper":
                            init_whisper(source_lang)
                        else:
                            init_sensevoice()
                            
                        # 動態重新設定 VAD 參數以降低延遲
                        min_silence = config_data.get("min_silence", 0.35)
                        max_speech = config_data.get("max_speech", 10.0)
                        
                        vad_config = sherpa_onnx.VadModelConfig(
                            silero_vad=sherpa_onnx.SileroVadModelConfig(
                                model=VAD_MODEL_PATH,
                                threshold=0.4,
                                min_silence_duration=min_silence,
                                min_speech_duration=0.15,
                                max_speech_duration=max_speech,
                            ),
                            sample_rate=16000,
                        )
                        local_vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
                        print(f"已更新後端設定: ASR={asr_engine}, Ollama={ollama_url}, Model={model_name}, SourceLang={source_lang}, TargetLang={target_lang}, 串流翻譯={'開' if stream_translate else '關'}, 滾動重識別={'開' if rolling_asr else '關'}")
                        print(f"已動態更新 VAD 設定: min_silence={min_silence}s, max_speech={max_speech}s")
                except Exception as e:
                    print(f"解析設定訊息或更新 VAD 失敗: {e}")
                continue
                
            # 如果收到的是二進位音訊數據
            if "bytes" in message:
                audio_bytes = message["bytes"]
                if not audio_bytes:
                    continue
                
                # 轉成 16kHz float32 NumPy 陣列
                # 前端會以 16-bit signed PCM (Int16) 發送
                pcm_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # 餵給 VAD
                local_vad.accept_waveform(pcm_data)

                # ── 滾動重識別：講話中就開始出臨時字幕，不等 VAD 定段 ──
                # is_speech_detected() 在說話中為真、VAD 定段後轉為假，
                # 正好可以拿來啟動／收掉臨時字幕迴圈。
                if rolling_asr:
                    if local_vad.is_speech_detected():
                        if rs["task"] is None or rs["task"].done():
                            rs["task"] = asyncio.create_task(rolling_loop())
                    else:
                        rs["stop"] = True

                # 檢查是否有切分好的語音段落
                while not local_vad.empty():
                    # 定段了：先讓滾動迴圈收手，避免臨時字幕和定稿搶著寫同一行
                    await settle_rolling()

                    speech_segment = local_vad.front
                    samples = speech_segment.samples
                    # 注意：sherpa-onnx 的 speech_segment.start 單位是「取樣點 (samples)」不是秒。
                    # 16kHz 下必須除以 16000 才是秒數，否則日誌會出現 [316064.00s] 這種離譜數值，
                    # 且寫入逐字稿的相對時間也會算錯。
                    # 另外 current_segment.start 與 front.start 實測完全相同，
                    # 所以臨時字幕和定稿會落在同一個 start → 前端能原地改字。
                    start_time = speech_segment.start / 16000.0
                    
                    # 確保 ASR 模型已載入
                    if len(samples) > 0:
                        if asr_recognizer is None:
                            if asr_engine == "whisper":
                                init_whisper(source_lang)
                            else:
                                init_sensevoice()

                        # 走執行緒池，避免 1~3 秒音訊的解碼（40~200ms）把
                        # 事件迴圈卡住、連帶拖慢後續音訊的接收。
                        full_asr_text, lang_tag = await asr_decode_async(samples)
                        
                        if current_asr_engine == "whisper":
                            raw_text = full_asr_text.strip()
                        else:
                            raw_text = clean_sense_voice_text(full_asr_text)
                        
                        if raw_text:
                            # 判定這句話該怎麼產出顯示文本（與臨時字幕共用同一份判斷）
                            kind, payload, recognized_lang = plan_translation(
                                raw_text, lang_tag, target_lang, source_lang, current_asr_engine
                            )

                            if kind == "raw":
                                translated_text = payload
                                if target_lang == "none":
                                    print(f"ASR 識別 [{start_time:.2f}s] (僅顯示原文): {raw_text}")
                                else:
                                    print(f"ASR 識別 [{start_time:.2f}s] (識別與目標一致 '{target_lang}'，跳過翻譯): {raw_text}")
                            elif kind == "local":
                                # 都是中文，使用本地 OpenCC
                                translated_text = payload
                                raw_text = translated_text  # 同步字形，方便前端去重
                                print(f"ASR 識別 [{start_time:.2f}s] (中文本地 CC 轉換): {translated_text}")
                            else:
                                # 進行翻譯
                                print(f"ASR 識別 [{start_time:.2f}s] (目標語言 '{target_lang}'): {raw_text}")
                                seg_duration = len(samples) / 16000.0
                                # 滾動模式下先墊上臨時譯文，避免定稿接手時字幕行閃成空白
                                carry_zh = ""
                                if (rolling_asr and rs["zh"]
                                        and rs["start"] is not None
                                        and abs(rs["start"] - start_time) < 1e-6):
                                    carry_zh = rs["zh"]
                                if stream_translate:
                                    # ① 先送「開始」事件：前端立刻顯示原文 + 閃爍游標，
                                    #    使用者不必盯著空白等翻譯完成。
                                    await websocket.send_json({
                                        "event": "subtitle_begin",
                                        "text_raw": raw_text,
                                        "text_zh": carry_zh,
                                        "start": start_time,
                                        "duration": seg_duration
                                    })
                                    translated_text = ""
                                    # ② 邊產生邊推送累積譯文
                                    async for partial in translate_text_stream(
                                        raw_text, target_lang, ollama_url, model_name, deepseek_key
                                    ):
                                        translated_text = partial
                                        await websocket.send_json({
                                            "event": "subtitle_chunk",
                                            "text_raw": raw_text,
                                            "text_zh": partial,
                                            "start": start_time,
                                            "duration": seg_duration
                                        })
                                    if not translated_text:
                                        # ③ 串流引擎一個字都沒吐（服務未啟動 / 不支援 SSE）
                                        #    → 退回非串流，確保一定有字幕
                                        print("  ➔ 串流翻譯無輸出，本句改用非串流路徑")
                                        translated_text = await translate_text(
                                            raw_text, target_lang, ollama_url, model_name, deepseek_key
                                        )
                                else:
                                    translated_text = await translate_text(
                                        raw_text, target_lang, ollama_url, model_name, deepseek_key
                                    )
                                print(f"翻譯結果: {translated_text}")
                            
                            # 傳回給前端外掛
                            await websocket.send_json({
                                "event": "subtitle",
                                "text_raw": raw_text,
                                "text_zh": translated_text,
                                "start": start_time,
                                "duration": len(samples) / 16000.0
                            })

                            # 定稿已送出，清掉本段的滾動狀態，避免殘留到下一段
                            # （否則下一段的臨時字幕會先閃出上一段的譯文）
                            rs["raw"] = ""
                            rs["zh"] = ""
                            rs["start"] = None
                            
                            # 寫入本機逐字稿存檔
                            if transcript_file_path:
                                try:
                                    abs_time = datetime.datetime.now().strftime("%H:%M:%S")
                                    m, s = divmod(int(start_time), 60)
                                    h, m = divmod(m, 60)
                                    rel_time = f"{h:02d}:{m:02d}:{s:02d}"
                                    
                                    with open(transcript_file_path, "a", encoding="utf-8") as f:
                                        f.write(f"### 🕒 [{abs_time} | 影片 {rel_time}]\n")
                                        f.write(f"*   **原文**：{raw_text}\n")
                                        f.write(f"*   **中文**：{translated_text}\n\n")
                                except Exception as file_err:
                                    print(f"寫入紀錄檔失敗: {file_err}")
                            
                    local_vad.pop()

    except WebSocketDisconnect:
        print("WebSocket 客戶端已中斷連線。")
    except Exception as e:
        print(f"WebSocket 發生錯誤: {e}")
    finally:
        # 斷線時把滾動迴圈收乾淨，否則它會繼續對已關閉的 websocket 送事件
        try:
            await settle_rolling()
        except Exception:
            pass
        active_connections = max(0, active_connections - 1)
        # 注意：這裡原本寫著 `import time`，會讓 time 變成此函式的區域變數，
        # 連帶讓巢狀的 rolling_loop 讀不到它（NameError）。
        # 模組頂端已有 import time，直接用即可。
        last_active_time = time.time()
        print(f"WebSocket 客戶端已中斷連線。當前連線數: {active_connections}")

if __name__ == "__main__":
    import uvicorn
    # 預設執行在 8000 埠
    uvicorn.run(app, host="127.0.0.1", port=8000)
