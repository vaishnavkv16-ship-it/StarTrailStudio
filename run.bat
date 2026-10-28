@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python launcher.py
if errorlevel 1 (
  echo.
  echo The app stopped with an error shown above.
  pause
)
