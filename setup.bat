@echo off
cd /d "%~dp0"
echo === Star Trail Studio setup ===
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version || goto nopython
if not exist .venv (
  echo Creating environment...
  %PY% -m venv .venv || goto fail
)
echo Installing packages (first time takes a few minutes)...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || goto fail
echo.
echo Setup finished. Now double-click run.bat
pause
exit /b 0
:nopython
echo.
echo Python was not found. Install 64-bit Python 3.10+ from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during install, then run setup.bat again.
pause
exit /b 1
:fail
echo.
echo Setup failed - see the error above.
pause
exit /b 1
