import os
import sys
import re
import json
import time
import asyncio
from pathlib import Path

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
            min_silence_duration=0.5,  # 0.5 秒靜音判定為說話結束
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
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {deepseek_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": f"你是一個專業的影片字幕即時翻譯官。請將輸入的影片語音字幕，翻譯成簡短流暢的{target_name}。請只輸出翻譯後的文字，不要包含任何解釋、引言或額外標記，保持字數與原句差不多。"},
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
            # 針對目標語言強化指令；繁體中文額外強制不可輸出簡體字
            if target_lang == "zh-TW":
                lang_rule = "輸出必須是「繁體中文（台灣用語）」，絕對禁止輸出任何簡體字。"
            else:
                lang_rule = f"輸出必須是{target_name}。"
            system_prompt = (
                f"你是一個專業的影片字幕即時翻譯官。請將使用者輸入的影片語音字幕，翻譯成簡短流暢的{target_name}。"
                f"{lang_rule}"
                "只輸出翻譯後的譯文本身，並輸出為「單獨一行純文字」；"
                "嚴禁輸出原文、注音、拼音、解釋、引言、括號標註、清單符號、換行或任何額外標記。"
            )
            _t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=15.0) as client:
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
                    translated = res_json.get("response", "").strip()
                    # 壓平殘留的換行與多餘空白，消除「分行/分段」型亂碼
                    translated = " ".join(translated.split())
                    # 繁體中文目標：再過一道 OpenCC s2t，徹底消除簡體殘留 (天气→天氣、钱→錢)
                    if target_lang == "zh-TW" and translated:
                        translated = cc_s2t.convert(translated)
                    if translated:
                        print(f"  ➔ [Ollama 翻譯 ({model_name})] (耗時 {(time.perf_counter()-_t0)*1000:.0f} ms)")
                        return translated
        except Exception as e:
            import traceback
            print(f"本機 Ollama 翻譯調用失敗 ({type(e).__name__}: {e})。詳細錯誤資訊如下：")
            traceback.print_exc()
            print("本工作階段後續將自動跳過 Ollama，避免連線超時延遲。")
            ollama_online = False

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

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    global active_connections, asr_recognizer, vad_detector
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
            min_silence_duration=0.5,
            min_speech_duration=0.15,
            max_speech_duration=10.0,
        ),
        sample_rate=16000,
    )
    local_vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
    
    # 設定選項 (預設為 Ollama 本地)
    ollama_url = "http://127.0.0.1:11434"
    model_name = "qwen2.5:7b-instruct"
    deepseek_key = None
    source_lang = "auto"
    target_lang = "none"
    asr_engine = "sensevoice"

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
                        global ollama_online
                        ollama_online = True
                        
                        # 動態加載或切換 ASR 引擎
                        if asr_engine == "whisper":
                            init_whisper(source_lang)
                        else:
                            init_sensevoice()
                            
                        # 動態重新設定 VAD 參數以降低延遲
                        min_silence = config_data.get("min_silence", 0.5)
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
                        print(f"已更新後端設定: ASR={asr_engine}, Ollama={ollama_url}, Model={model_name}, SourceLang={source_lang}, TargetLang={target_lang}")
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
                
                # 檢查是否有切分好的語音段落
                while not local_vad.empty():
                    speech_segment = local_vad.front
                    samples = speech_segment.samples
                    start_time = speech_segment.start
                    
                    # 確保 ASR 模型已載入
                    if len(samples) > 0:
                        if asr_recognizer is None:
                            if asr_engine == "whisper":
                                init_whisper(source_lang)
                            else:
                                init_sensevoice()
                                
                        stream = asr_recognizer.create_stream()
                        stream.accept_waveform(16000, samples)
                        asr_recognizer.decode_stream(stream)
                        
                        full_asr_text = stream.result.text
                        
                        if current_asr_engine == "whisper":
                            raw_text = full_asr_text.strip()
                        else:
                            raw_text = clean_sense_voice_text(full_asr_text)
                        
                        if raw_text:
                            # 偵測 ASR 識別語音標籤
                            recognized_lang = "auto"
                            if current_asr_engine == "whisper":
                                if source_lang != "auto":
                                    recognized_lang = source_lang
                            else:
                                if "<|zh|>" in full_asr_text or "<|yue|>" in full_asr_text:
                                    recognized_lang = "zh"
                                elif "<|en|>" in full_asr_text:
                                    recognized_lang = "en"
                                elif "<|ja|>" in full_asr_text:
                                    recognized_lang = "ja"
                                elif "<|ko|>" in full_asr_text:
                                    recognized_lang = "ko"

                            # 判定是否需要翻譯
                            if target_lang == "none":
                                # 僅顯示原文
                                translated_text = raw_text
                                print(f"ASR 識別 [{start_time:.2f}s] (僅顯示原文): {raw_text}")
                            elif target_lang == recognized_lang:
                                # 識別語言與目標翻譯語言一致，跳過翻譯
                                translated_text = raw_text
                                print(f"ASR 識別 [{start_time:.2f}s] (識別與目標一致 '{target_lang}'，跳過翻譯): {raw_text}")
                            elif recognized_lang == "zh" and target_lang in ["zh-TW", "zh-CN"]:
                                # 都是中文，使用本地 OpenCC
                                if target_lang == "zh-TW":
                                    translated_text = cc_s2t.convert(raw_text)
                                    raw_text = translated_text  # 同步為繁體，方便前端去重
                                else:
                                    translated_text = cc_t2s.convert(raw_text)
                                    raw_text = translated_text  # 同步為簡體
                                print(f"ASR 識別 [{start_time:.2f}s] (中文本地 CC 轉換): {translated_text}")
                            else:
                                # 進行翻譯
                                print(f"ASR 識別 [{start_time:.2f}s] (目標語言 '{target_lang}'): {raw_text}")
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
        active_connections = max(0, active_connections - 1)
        import time
        last_active_time = time.time()
        print(f"WebSocket 客戶端已中斷連線。當前連線數: {active_connections}")

if __name__ == "__main__":
    import uvicorn
    # 預設執行在 8000 埠
    uvicorn.run(app, host="127.0.0.1", port=8000)
