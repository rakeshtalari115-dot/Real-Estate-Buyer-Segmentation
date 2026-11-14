@echo off
setlocal
cd /d "%~dp0"
title Real Estate Buyer Segmentation - Setup and Launch

echo ======================================================
echo  REAL ESTATE BUYER SEGMENTATION - FIRST TIME SETUP
echo ======================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    py -3.11 -m venv .venv >nul 2>nul
    if errorlevel 1 py -3 -m venv .venv
) else (
    python -m venv .venv
)
if errorlevel 1 goto :error

call .venv\Scripts\activate.bat
if errorlevel 1 goto :error

echo [1/4] Updating pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [2/4] Installing project dependencies (internet connection required)...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [3/4] Rebuilding cleaned data, model, metrics and charts...
python train_model.py
if errorlevel 1 goto :error

echo [4/4] Starting Streamlit dashboard...
python -m streamlit run app\streamlit_app.py
if errorlevel 1 goto :error

goto :end

:error
echo.
echo SETUP OR LAUNCH FAILED.
echo Please check the error shown above. See README.md - Troubleshooting.
echo Keep this window open and share a screenshot of the error if you need help.
pause
:end
endlocal
