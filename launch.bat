@echo off
rem ===== Vintage Story Ore Finder =====
rem Double-click this file to open the app.
cd /d "%~dp0"
python app.py
if errorlevel 1 (
  echo.
  echo The app exited with an error. Press any key to close.
  pause >nul
)
