import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, "dist", "LiveCaptionServer")

def run_command(cmd, cwd=None):
    print(f"執行指令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, shell=True)
    if result.returncode != 0:
        print(f"錯誤: 指令執行失敗 (代碼 {result.returncode})")
        sys.exit(result.returncode)

def main():
    print("=== Studio0808 LiveCaption 後端打包工具 ===")
    
    # 1. 切換工作目錄到 backend
    os.chdir(BASE_DIR)
    
    # 2. 確保虛擬環境的 pip 與 pyinstaller 存在
    exe_ext = ".exe" if sys.platform == "win32" else ""
    python_path = os.path.join(BASE_DIR, ".venv", "Scripts", f"python{exe_ext}")
    pyinstaller_path = os.path.join(BASE_DIR, ".venv", "Scripts", f"pyinstaller{exe_ext}")
    
    if not os.path.exists(python_path):
        print(f"未找到虛擬環境的 Python (路徑: {python_path})，請確認您已在 backend 目錄建立 .venv")
        sys.exit(1)
        
    print("正在檢查並安裝 PyInstaller...")
    run_command([python_path, "-m", "pip", "install", "pyinstaller"])
    
    # 3. 執行 PyInstaller 打包 (onedir 模式，利於載入外部大型模型)
    # --collect-data opencc 用於確保簡繁轉換字典能被正確包入 exe
    print("正在使用 PyInstaller 編譯 Python 程式為 Windows 執行檔...")
    pyinstaller_cmd = [
        pyinstaller_path,
        "--name", "LiveCaptionServer",
        "--onedir",
        "--clean",
        "-y",
        "--collect-data", "opencc",
        "main.py"
    ]
    run_command(pyinstaller_cmd)
    
    # 4. 複製 AI 模型檔案到打包後的資料夾中
    print("正在複製 AI 模型至發布資料夾...")
    
    # VAD 模型
    vad_src = os.path.join(BASE_DIR, "silero_vad.onnx")
    vad_dst = os.path.join(DIST_DIR, "silero_vad.onnx")
    if os.path.exists(vad_src):
        print("複製 VAD 模型...")
        shutil.copy2(vad_src, vad_dst)
    else:
        print("警告: 找不到 silero_vad.onnx！請先執行 download_models.py")
        
    # SenseVoice ASR 模型
    asr_dir_name = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
    asr_src = os.path.join(BASE_DIR, str(asr_dir_name))
    asr_dst = os.path.join(DIST_DIR, str(asr_dir_name))
    if os.path.exists(asr_src):
        print("複製 SenseVoice ASR 模型目錄 (檔案較大，請稍候)...")
        if os.path.exists(asr_dst):
            shutil.rmtree(asr_dst)
        shutil.copytree(asr_src, asr_dst)
    else:
        print("警告: 找不到 SenseVoice 模型目錄！請先執行 download_models.py")
        
    # Whisper ASR 模型 (優先小模型，若無則複製 base)
    whisper_dir_name = "sherpa-onnx-whisper-small"
    whisper_src = os.path.join(BASE_DIR, whisper_dir_name)
    if not os.path.exists(whisper_src):
        whisper_dir_name = "sherpa-onnx-whisper-base"
        whisper_src = os.path.join(BASE_DIR, whisper_dir_name)
        
    whisper_dst = os.path.join(DIST_DIR, whisper_dir_name)
    if os.path.exists(whisper_src):
        print(f"複製 Whisper ASR 模型目錄 ({whisper_dir_name}，檔案較大，請稍候)...")
        if os.path.exists(whisper_dst):
            shutil.rmtree(whisper_dst)
        shutil.copytree(whisper_src, whisper_dst)
        # 僅保留 int8 模型，剔除較大的 fp32 .onnx (程式預設載入 int8，可省約 925MB)
        for f in os.listdir(whisper_dst):
            if f.endswith(".onnx") and ".int8." not in f:
                os.remove(os.path.join(whisper_dst, f))
                print(f"已剔除 fp32 模型以節省空間: {f}")
    else:
        print("警告: 找不到 Whisper 模型目錄！請先執行 download_models.py")

    # 4.5 複製本機虛擬環境安裝的 NVIDIA CUDA / cuDNN DLLs (讓編譯後的外殼可以直接使用 GPU)
    site_packages_nvidia = os.path.join(BASE_DIR, ".venv", "Lib", "site-packages", "nvidia")
    if os.path.exists(site_packages_nvidia):
        print("\n偵測到虛擬環境中存在 NVIDIA CUDA 套件，正在複製 GPU 加速所需的 DLL 檔案...")
        print("(此過程會複製大約 1.3GB 檔案，請耐心等候)...")
        copied_count = 0
        for root, dirs, files in os.walk(site_packages_nvidia):
            for file in files:
                if file.lower().endswith(".dll"):
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(DIST_DIR, file)
                    try:
                        # 避免重複複製同名檔案
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                            copied_count += 1
                    except Exception as e:
                        print(f"複製 {file} 失敗: {e}")
        print(f"CUDA/cuDNN DLL 複製完成！共複製了 {copied_count} 個 DLL 檔案至發布目錄。")
        
    # 4.6 複製 sherpa_onnx 遺漏的 DLL 動態庫 (含 CUDA Provider 等)
    sherpa_lib_src = os.path.join(BASE_DIR, ".venv", "Lib", "site-packages", "sherpa_onnx", "lib")
    sherpa_lib_dst = os.path.join(DIST_DIR, "_internal", "sherpa_onnx", "lib")
    if os.path.exists(sherpa_lib_src):
        print("\n正在複製 sherpa_onnx 遺漏的 DLL 動態庫 (含 CUDA Provider 等)...")
        os.makedirs(sherpa_lib_dst, exist_ok=True)
        copied_sherpa_count = 0
        copied_root_count = 0
        for file in os.listdir(sherpa_lib_src):
            if file.lower().endswith(".dll"):
                src_file = os.path.join(sherpa_lib_src, file)
                
                # 複製到 _internal 下的 lib 目錄
                dst_file = os.path.join(sherpa_lib_dst, file)
                try:
                    shutil.copy2(src_file, dst_file)
                    copied_sherpa_count += 1
                except Exception as e:
                    print(f"複製 sherpa_onnx DLL {file} 失敗: {e}")
                
                # 同時複製到打包根目錄 (DIST_DIR)，雙重確保 Windows DLL 搜尋能載入 CUDA Execution Provider 及其依賴
                dst_root_file = os.path.join(DIST_DIR, file)
                try:
                    if not os.path.exists(dst_root_file):
                        shutil.copy2(src_file, dst_root_file)
                        copied_root_count += 1
                except Exception as e:
                    print(f"複製 sherpa_onnx DLL {file} 至根目錄失敗: {e}")
        print(f"sherpa_onnx DLL 複製完成！共複製了 {copied_sherpa_count} 個 DLL 至 lib 目錄，{copied_root_count} 個新 DLL 至根目錄。")
        
    # 5. 建立便利的「啟動服務.bat」批次檔
    print("\n正在建立一鍵啟動批次檔...")
    bat_path = os.path.join(DIST_DIR, "點我啟動【即時字幕】後端服務.bat")
    bat_content = """@echo off
chcp 65001 > nul
title Studio0808 LiveCaption Server
echo ===================================================
echo   正在啟動 Studio0808 LiveCaption 語音翻譯後端伺服器...
echo ===================================================
echo.
LiveCaptionServer.exe
echo.
echo 伺服器已結束運行。
pause
"""
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)
        
    print("\n=== [SUCCESS] 打包完成！ ===")
    print(f"打包成品路徑：{DIST_DIR}")
    print("您可以直接將該資料夾壓縮為 ZIP 檔分享給他人使用！")
    print("使用者只需解壓縮後，點選「點我啟動【即時字幕】後端服務.bat」即可啟動伺服器，無需安裝 Python 或執行指令！")

if __name__ == "__main__":
    main()
