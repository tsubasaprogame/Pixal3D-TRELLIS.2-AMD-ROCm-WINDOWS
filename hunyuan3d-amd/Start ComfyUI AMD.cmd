@echo off
chcp 65001 >nul
set "COMFY=%~dp0runtime\ComfyUI"
if not exist "%COMFY%\.venv\Scripts\python.exe" (
  echo ComfyUI AMD is not installed. Run "Install Hunyuan3D AMD.cmd" first.
  pause
  exit /b 1
)
pushd "%COMFY%"
start "" "http://127.0.0.1:8190"
".venv\Scripts\python.exe" main.py --listen 127.0.0.1 --port 8190
popd
pause
