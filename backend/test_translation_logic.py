"""test_translation_logic.py — 翻譯決策與文字後處理的純邏輯單元測試

與 backend/ 下其他測試不同，本檔**不需要任何模型檔、不需要 Ollama、不需要音訊、
不佔 GPU、不連網路**，只驗證純函式，約 2 秒跑完。適合每次改動翻譯流程後隨手跑。

用法（在 backend/ 目錄，或專案根目錄都可以）：
    .venv\\Scripts\\python.exe test_translation_logic.py
    python backend/test_translation_logic.py

覆蓋範圍（皆為本次改動新增或重構的邏輯）：
    plan_translation     語言判定 → raw / local(OpenCC) / mt 三路分流
    build_system_prompt  目標語言 → 對應字形的 system prompt
    add_no_think         Qwen3 關閉思考鏈
    convert_script       簡繁轉換（含滾動字幕所需的冪等性）
    flatten_text         壓平換行
    _ensure_loopback_no_proxy  回環不走代理（含冪等性）

未覆蓋：WebSocket 串流、ASR 解碼、Ollama 串流輸出、擴充功能的 DOM 行為。
這些需要真實模型與瀏覽器，分別由 test_rolling_e2e.py / test_stream_e2e.py 與手動驗證負責。
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import main  # noqa: E402


PASSED = 0
FAILED = []


def check(label, cond, detail=""):
    """記錄一條斷言。刻意不用 assert，讓一次執行能報出全部問題。"""
    global PASSED
    if cond:
        PASSED += 1
        print(f"  [ok]   {label}")
    else:
        FAILED.append(label)
        print(f"  [FAIL] {label}" + (f"  ->  {detail}" if detail else ""))


def section(title):
    print(f"\n{title}")


# 只用「簡體獨有」與「繁體獨有」的字，避免誤判同時存在的共用字（如 翻、文）。
SIMPLIFIED_ONLY = "简译语输单严标体个头业请将记换额"
TRADITIONAL_ONLY = "這個專業譯請將輸記換額體單嚴標"


# ---------------------------------------------------------------------------
# 1. plan_translation：語言判定與三路分流
# ---------------------------------------------------------------------------
def test_plan_translation():
    section("1. plan_translation — 語言判定與分流")
    p = main.plan_translation

    # --- 關閉翻譯 / 識別語言等於目標語言 ---
    kind, payload, _ = p("hello", "<|en|>", "none", "auto", "sensevoice")
    check("target=none 時直接沿用原文", kind == "raw" and payload == "hello", f"got {kind!r}/{payload!r}")

    kind, payload, rec = p("hello", "<|en|>", "en", "auto", "sensevoice")
    check("識別語言 == 目標語言時不送翻譯", kind == "raw" and rec == "en", f"got {kind!r}/{rec!r}")

    # --- 中文 → 中文：走本地 OpenCC，零模型成本 ---
    kind, payload, rec = p("天气很好", "<|zh|>", "zh-TW", "auto", "sensevoice")
    check(
        "中文語音 + 繁中目標 → local 且確實轉成繁體",
        kind == "local" and payload == "天氣很好" and rec == "zh",
        f"got {kind!r}/{payload!r}/{rec!r}",
    )

    kind, payload, _ = p("天氣很好", "<|zh|>", "zh-CN", "auto", "sensevoice")
    check("中文語音 + 簡中目標 → local 且確實轉成簡體", kind == "local" and payload == "天气很好", f"got {kind!r}/{payload!r}")

    # --- 粵語是獨立一類：書面語就是粵語，OpenCC 只轉字形不轉詞彙，必須交給翻譯模型 ---
    kind, _, rec = p("今日天氣", "<|yue|>", "zh-TW", "auto", "sensevoice")
    check("粵語 → 繁中必須走 mt（不可被當成中文做 OpenCC）", kind == "mt" and rec == "yue", f"got {kind!r}/{rec!r}")

    kind, _, rec = p("今日天氣", "<|yue|>", "zh-CN", "auto", "sensevoice")
    check("粵語 → 簡中同樣走 mt", kind == "mt" and rec == "yue", f"got {kind!r}/{rec!r}")

    # --- 外語 → 中文 ---
    kind, _, rec = p("hello", "<|en|>", "zh-TW", "auto", "sensevoice")
    check("英文 → 繁中走 mt", kind == "mt" and rec == "en", f"got {kind!r}/{rec!r}")

    kind, _, rec = p("こんにちは", "<|ja|>", "zh-TW", "auto", "sensevoice")
    check("日文 → 繁中走 mt", kind == "mt" and rec == "ja", f"got {kind!r}/{rec!r}")

    # --- 迴歸守衛 1：識別語言必須取自 lang_tag 參數 ---
    # 舊寫法是在 result.text 裡搜尋 '<|zh|>'，但 sherpa-onnx 從不把語言標籤寫進 text，
    # 於是 recognized_lang 恆為 'auto'，中文→中文的 OpenCC 分支變成死碼：
    # 所有中文語音都被多送一次 LLM「翻譯」，白費 GPU 還會改壞內容
    # （實測「开饭时间」被翻成「开放时间」）。此斷言確保該分支活著。
    kind, _, rec = p("天气", "<|zh|>", "zh-TW", "auto", "sensevoice")
    check(
        "迴歸守衛：中文→繁中走 local（OpenCC 分支不得再是死碼）",
        kind == "local" and rec == "zh",
        f"got {kind!r}/{rec!r}",
    )

    # --- 迴歸守衛 2：raw_text 內出現標籤字面量時，不得被當成語言來源 ---
    kind, payload, rec = p("<|zh|> 天气", "", "zh-TW", "auto", "sensevoice")
    check(
        "迴歸守衛：語言只認 lang_tag，不掃描 raw_text 內容",
        rec == "auto" and kind == "mt" and payload == "<|zh|> 天气",
        f"got {kind!r}/{rec!r}",
    )

    # --- lang_tag 的各種寫法 ---
    for tag, want in [("<|zh|>", "zh"), ("zh", "zh"), ("ZH", "zh"), (" mandarin ", "zh"),
                      ("<|en|>", "en"), ("english", "en"), ("<|ja|>", "ja"),
                      ("<|ko|>", "ko"), ("<|yue|>", "yue"), ("cantonese", "yue")]:
        _, _, rec = p("x", tag, "zh-TW", "auto", "sensevoice")
        check(f"lang_tag={tag!r} → {want}", rec == want, f"got {rec!r}")

    _, _, rec = p("x", "", "zh-TW", "auto", "sensevoice")
    check("空 lang_tag → auto", rec == "auto", f"got {rec!r}")

    _, _, rec = p("x", None, "zh-TW", "auto", "sensevoice")
    check("None lang_tag → auto（不得拋錯）", rec == "auto", f"got {rec!r}")

    _, _, rec = p("x", "<|th|>", "zh-TW", "auto", "sensevoice")
    check("未知語言 → auto 並走 mt", rec == "auto", f"got {rec!r}")

    # --- whisper 引擎：語言來自 source_lang，且不看 lang_tag ---
    _, _, rec = p("x", "<|zh|>", "zh-TW", "ja", "whisper")
    check("whisper 用 source_lang 判定語言", rec == "ja", f"got {rec!r}")

    kind, payload, _ = p("天氣", "<|en|>", "zh-TW", "zh", "whisper")
    check("whisper 判為中文時同樣走 local", kind == "local" and payload == "天氣", f"got {kind!r}/{payload!r}")

    _, _, rec = p("x", "<|zh|>", "zh-TW", "auto", "whisper")
    check("whisper source_lang=auto 時不猜語言（忽略 lang_tag）", rec == "auto", f"got {rec!r}")

    # --- 非 local 路徑必須原樣回傳，不得先動字 ---
    raw = "don't change me 天气"
    _, payload, _ = p(raw, "<|en|>", "zh-TW", "auto", "sensevoice")
    check("mt 路徑原樣回傳未經改動的原文", payload == raw, f"got {payload!r}")


# ---------------------------------------------------------------------------
# 2. build_system_prompt / add_no_think
# ---------------------------------------------------------------------------
def test_system_prompt():
    section("2. build_system_prompt / add_no_think — prompt 組裝")
    b = main.build_system_prompt

    tw = b("zh-TW", "qwen2.5:7b-instruct")
    check("繁中 prompt 為繁體字形", "繁體中文" in tw and "翻譯官" in tw)
    bad = [c for c in tw if c in SIMPLIFIED_ONLY]
    check("繁中 prompt 不得混入簡體字（會傳染給模型）", not bad, f"混入 {bad}")

    cn = b("zh-CN", "qwen2.5:7b-instruct")
    check("簡中 prompt 為簡體字形", "简体中文" in cn and "翻译官" in cn)
    bad = [c for c in cn if c in TRADITIONAL_ONLY]
    check("簡中 prompt 不得混入繁體字", not bad, f"混入 {bad}")

    for alias in ("zh", "zh-Hans"):
        p2 = b(alias, "qwen2.5:7b-instruct")
        check(f"別名 {alias!r} 等同 zh-CN", p2 == cn)

    ja = b("ja", "qwen2.5:7b-instruct")
    check("其他語言帶入 LANG_MAP 語言名", "日文" in ja, f"got {ja[:40]!r}")

    unknown = b("xx-YY", "qwen2.5:7b-instruct")
    check("未知語言碼回退繁體中文", "繁體中文" in unknown)

    check("prompt 不含換行（要求單行輸出）", "\n" not in tw and "\n" not in cn)

    # add_no_think
    a = main.add_no_think
    base = b("zh-TW", "qwen3:4b-instruct")
    check("qwen3 加上 /no_think", a(base, "qwen3:4b-instruct").endswith(" /no_think"))
    check("模型名大小寫不敏感", a(base, "Qwen3:8b").endswith(" /no_think"))
    check("/no_think 不破壞原 prompt", a(base, "qwen3:4b-instruct").startswith(base))

    check("qwen2.5 不應加 /no_think", a(base, "qwen2.5:7b-instruct") == base)
    check("空模型名不應加 /no_think", a(base, "") == base)
    check("None 模型名不應拋錯", a(base, None) == base)

    # 兩者串接後仍須維持字形不變（Rolling/定稿兩條路徑共用同一份組裝）
    q3 = a(b("zh-CN", "qwen3:4b-instruct"), "qwen3:4b-instruct")
    bad = [c for c in q3 if c in TRADITIONAL_ONLY]
    check("串接 /no_think 後簡中 prompt 仍無繁體字", not bad, f"混入 {bad}")


# ---------------------------------------------------------------------------
# 3. convert_script / flatten_text
# ---------------------------------------------------------------------------
def test_text_helpers():
    section("3. convert_script / flatten_text — 文字後處理")
    c = main.convert_script

    check("簡→繁", c("天气", "zh-TW") == "天氣", f"got {c('天气', 'zh-TW')!r}")
    check("簡→繁（詞彙級）", c("酋长", "zh-TW") == "酋長", f"got {c('酋长', 'zh-TW')!r}")
    check("繁→簡", c("天氣", "zh-CN") == "天气", f"got {c('天氣', 'zh-CN')!r}")
    check("繁→簡（詞彙級）", c("酋長", "zh-CN") == "酋长", f"got {c('酋長', 'zh-CN')!r}")
    check("zh 別名等同 zh-CN", c("天氣", "zh") == "天气")
    check("空字串原樣回傳", c("", "zh-TW") == "")
    check("None 原樣回傳（不得拋錯）", c(None, "zh-TW") is None)
    check("非中文目標不改字", c("天气", "ja") == "天气")
    check("混合中英不破壞英文", c("Hello 世界", "zh-TW") == "Hello 世界")

    # 滾動字幕每 0.5 秒會對「累積文本」重新轉換一次，因此冪等性是硬需求。
    for txt, lang in [("天气很好", "zh-TW"), ("天氣很好", "zh-CN"),
                      ("Hello 世界", "zh-TW"), ("", "zh-TW")]:
        once = c(txt, lang)
        twice = c(once, lang)
        check(f"convert_script 冪等 ({lang}, {txt!r})", once == twice, f"{once!r} vs {twice!r}")

    f = main.flatten_text
    got = f("a\nb")
    check("換行壓成空白", got == "a b", f"got {got!r}")
    got = f("  a \n\n  b  ")
    check("多餘空白壓平", got == "a b", f"got {got!r}")
    check("空字串回傳空字串", f("") == "")
    got = f("你好\n世界")
    check("中日文換行壓平", got == "你好 世界", f"got {got!r}")
    check("正常單一空白不受影響", f("hello world") == "hello world")
    check("flatten_text 冪等", f(f(" a \n b ")) == f(" a \n b "))


# ---------------------------------------------------------------------------
# 4. _ensure_loopback_no_proxy
# ---------------------------------------------------------------------------
def test_loopback_no_proxy():
    section("4. _ensure_loopback_no_proxy — 回環不走代理")
    fn = main._ensure_loopback_no_proxy
    saved = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}

    def restore():
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    try:
        # 情境 A：全新環境
        os.environ.pop("NO_PROXY", None)
        os.environ.pop("no_proxy", None)
        fn()
        val = os.environ.get("NO_PROXY", "")
        check("三個回環位址都寫入 NO_PROXY",
              all(h in val.split(",") for h in ("127.0.0.1", "localhost", "::1")), f"got {val!r}")
        check("同時寫入小寫 no_proxy（httpx/requests 讀法不同）",
              os.environ.get("no_proxy") == val, f"got {os.environ.get('no_proxy')!r}")

        # 情境 B：冪等 —— 本函式在模組載入時就被呼叫，重複呼叫不得累積重複項
        fn()
        fn()
        parts = [x for x in os.environ.get("NO_PROXY", "").split(",") if x.strip()]
        check("重複呼叫不產生重複項目", len(parts) == len(set(parts)), f"got {parts}")
        check("重複呼叫後仍恰好 3 項", len(parts) == 3, f"got {parts}")

        # 情境 C：保留使用者原有的 NO_PROXY
        os.environ["NO_PROXY"] = "example.com,10.0.0.0/8"
        fn()
        val = os.environ["NO_PROXY"]
        check("保留使用者既有條目", "example.com" in val and "10.0.0.0/8" in val, f"got {val!r}")
        check("既有條目之後仍補上回環", "127.0.0.1" in val, f"got {val!r}")

        # 情境 D：髒資料（多餘空白與空項）
        os.environ["NO_PROXY"] = "  , example.com ,, "
        fn()
        parts = [x.strip() for x in os.environ["NO_PROXY"].split(",")]
        check("清掉空白與空項", "" not in parts, f"got {parts}")
        check("髒資料中的有效項仍保留", "example.com" in parts, f"got {parts}")
    finally:
        restore()


def main_run():
    print("=" * 66)
    print(" LiveCaption 純邏輯單元測試（無需模型 / 網路 / GPU）")
    print("=" * 66)

    test_plan_translation()
    test_system_prompt()
    test_text_helpers()
    test_loopback_no_proxy()

    print("\n" + "=" * 66)
    if FAILED:
        print(f"=== [FAILED] {len(FAILED)} 項失敗 / 共 {PASSED + len(FAILED)} 項 ===")
        for name in FAILED:
            print(f"    - {name}")
        sys.exit(1)
    print(f"=== [ALL PASSED] 全部 {PASSED} 項通過 ===")
    print("=" * 66)


if __name__ == "__main__":
    main_run()
