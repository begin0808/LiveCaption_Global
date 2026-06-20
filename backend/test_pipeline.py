import os
import wave
import math
import struct
import numpy as np
import sherpa_onnx

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAD_MODEL_PATH = os.path.join(BASE_DIR, "silero_vad.onnx")
SENSE_VOICE_DIR = os.path.join(BASE_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
SENSE_VOICE_MODEL = os.path.join(SENSE_VOICE_DIR, "model.int8.onnx")
SENSE_VOICE_TOKENS = os.path.join(SENSE_VOICE_DIR, "tokens.txt")
DUMMY_WAV_PATH = os.path.join(BASE_DIR, "dummy_test.wav")

def create_dummy_wav(path, duration=2.0, freq=440, sample_rate=16000):
    print(f"正在建立測試用 Dummy WAV: {path} (頻率 {freq}Hz, 長度 {duration}s)...")
    num_samples = int(duration * sample_rate)
    with wave.open(path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(num_samples):
            val = int(16384.0 * math.sin(2.0 * math.pi * freq * i / sample_rate))
            data = struct.pack('<h', val)
            wav.writeframesraw(data)
    print("Dummy WAV 建立成功！")

def test_pipeline():
    # 1. 檢查模型檔案
    if not os.path.exists(VAD_MODEL_PATH) or not os.path.exists(SENSE_VOICE_MODEL):
        print("模型檔案遺失！請先執行 download_models.py 下載模型。")
        return False

    # 2. 建立 dummy wav 檔用於測試
    create_dummy_wav(DUMMY_WAV_PATH)

    try:
        # 3. 測試 VAD 初始化
        print("1. 正在測試 VAD 初始化...")
        vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=VAD_MODEL_PATH,
                threshold=0.5,
                min_silence_duration=0.5,
                min_speech_duration=0.25,
                max_speech_duration=10.0,
            ),
            sample_rate=16000,
        )
        vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
        print("VAD 初始化成功！")

        # 4. 測試 ASR 初始化
        print("2. 正在測試 SenseVoice ASR 初始化...")
        asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=SENSE_VOICE_MODEL,
            tokens=SENSE_VOICE_TOKENS,
            num_threads=4,
            use_itn=True,
        )
        print("ASR 初始化成功！")

        # 5. 讀取 dummy wav 並餵給 VAD & ASR
        print("3. 正在讀取測試音訊並進行解碼測試...")
        with wave.open(DUMMY_WAV_PATH, 'rb') as wav:
            frames = wav.readframes(wav.getnframes())
            # 轉換為 float32
            pcm_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 測試 VAD 接受波形
            vad.accept_waveform(pcm_data)
            print("VAD 音訊資料餵入成功！")
            
            # 由於 dummy wav 是純 sine wave，可能不會被判定為人類說話（Speech）
            # 所以我們也直接對 ASR 解碼進行測試，確保解碼器正常
            print("4. 測試 ASR 直接解碼...")
            stream = asr.create_stream()
            stream.accept_waveform(16000, pcm_data)
            asr.decode_stream(stream)
            print(f"ASR 解碼成功！解碼輸出: {stream.result.text}")

        # 6. 清理測試檔
        if os.path.exists(DUMMY_WAV_PATH):
            os.remove(DUMMY_WAV_PATH)
            print("清理測試用臨時音檔。")

        print("\n=== [SUCCESS] 本地 ASR & VAD 基礎管道測試全部通過！ ===")
        return True

    except Exception as e:
        print(f"\n=== [ERROR] 測試失敗: {e} ===")
        if os.path.exists(DUMMY_WAV_PATH):
            os.remove(DUMMY_WAV_PATH)
        return False

if __name__ == "__main__":
    test_pipeline()
