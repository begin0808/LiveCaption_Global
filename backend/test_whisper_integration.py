import os
import sys
import gc
import numpy as np

# 模擬 init_whisper 邏輯
import main

try:
    print("1. 測試初始化 VAD...")
    main.init_vad()
    print("VAD 成功！")
    
    print("\n2. 測試初始化 SenseVoice...")
    main.init_sensevoice()
    print("SenseVoice 成功！")
    
    print("\n3. 測試切換至 Whisper-base (語言: 'es' 西班牙文)...")
    main.init_whisper("es")
    print("Whisper-base 成功！")
    
    print("\n4. 測試切換回 SenseVoice...")
    main.init_sensevoice()
    print("SenseVoice 切回成功！")
    
    print("\n5. 測試切換至 Whisper-base (語言: 'auto')...")
    main.init_whisper("auto")
    print("Whisper-base auto 成功！")
    
    print("\n6. 測試 ASR 解碼 (Whisper)...")
    # 建立 2 秒的靜音資料並進行解碼
    samples = np.zeros(32000, dtype=np.float32)
    stream = main.asr_recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    main.asr_recognizer.decode_stream(stream)
    print(f"解碼測試成功！輸出 text: '{stream.result.text}'")
    
    print("\n7. 測試釋放資源...")
    main.release_asr_model()
    print("資源釋放成功！")
    
    print("\n=== [ALL PASSED] 本地雙引擎加載與切換集成測試通過！ ===")
    
except Exception as e:
    print(f"\n=== [FAILED] 測試失敗: {e} ===")
    sys.exit(1)
