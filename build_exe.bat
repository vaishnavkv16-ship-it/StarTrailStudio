@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python -m pip install pyinstaller
.venv\Scripts\pyinstaller --noconfirm --onefile --windowed --name StarTrailStudio --collect-all rawpy --collect-all pillow_heif --collect-all imageio_ffmpeg --exclude-module tkinter launcher.py
echo.
echo Done. Single file: dist\StarTrailStudio.exe
pause
