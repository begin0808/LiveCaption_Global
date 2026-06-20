import os
import urllib.request
import tarfile
import shutil

# Target directory
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Model download URLs
SENSE_VOICE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

def download_file(url, dest_path):
    print(f"正在下載: {url} -> {dest_path}")
    
    def report_progress(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, (read_so_far * 100) // total_size)
            print(f"\r進度: {percent}% ({read_so_far // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="")
        else:
            print(f"\r已下載: {read_so_far // (1024*1024)}MB", end="")

    urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
    print("\n下載完成！")

def main():
    # 1. 下載並設定 Silero VAD
    vad_dest = os.path.join(MODEL_DIR, "silero_vad.onnx")
    if not os.path.exists(vad_dest):
        print("--- 下載 Silero VAD ---")
        download_file(VAD_URL, vad_dest)
    else:
        print("Silero VAD 已存在，跳過。")

    # 2. 下載並解壓縮 SenseVoice-Small ONNX
    sense_voice_archive = os.path.join(MODEL_DIR, "sense_voice.tar.bz2")
    sense_voice_dir = os.path.join(MODEL_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
    
    if not os.path.exists(sense_voice_dir):
        print("--- 下載 SenseVoice-Small ---")
        if not os.path.exists(sense_voice_archive):
            download_file(SENSE_VOICE_URL, sense_voice_archive)
        
        print("正在解壓縮 SenseVoice-Small (這可能需要一些時間)...")
        try:
            with tarfile.open(sense_voice_archive, "r:bz2") as tar:
                tar.extractall(path=MODEL_DIR)
            print("解壓縮完成！")
        except Exception as e:
            print(f"解壓縮失敗: {e}")
            return
        
        # 刪除壓縮檔
        if os.path.exists(sense_voice_archive):
            os.remove(sense_voice_archive)
            print("清除臨時壓縮檔。")
    else:
        print("SenseVoice-Small 資料夾已存在，跳過。")
        
    print("\n所有模型均已下載且設定完成！")

if __name__ == "__main__":
    main()
