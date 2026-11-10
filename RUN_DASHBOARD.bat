@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Project environment not found. Please double-click START_PROJECT.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m streamlit run app\streamlit_app.py
if errorlevel 1 pause
endlocal
