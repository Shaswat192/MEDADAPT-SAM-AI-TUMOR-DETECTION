@echo off
setlocal
title MedAdapt-SAM Auto-Setup and Launcher

echo ============================================================
echo      🧠 MedAdapt-SAM: Automatic Setup ^& Launcher 🚀
echo ============================================================
echo.

:: 1. Check for Python
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+ from python.org
    pause
    exit /b
)
echo [OK] Python detected.

:: 2. Upgrade Pip
echo [2/4] Initializing environment (Upgrading Pip)...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] Pip is ready.

:: 3. Install Requirements
echo [3/4] Installing Medical AI dependencies...
echo This may take a few minutes (downloading Torch, SAM, and OpenCV)...
echo.
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [RETRYING] Individual install for core components...
    python -m pip install streamlit torch torchvision opencv-python matplotlib pandas numpy pillow streamlit-drawable-canvas segment-anything
)

:: 4. Launch Dashboard
echo.
echo [4/4] Launching Premium Clinical Dashboard...
echo ============================================================
echo.
python -m streamlit run streamlit_app/app.py

if %errorlevel% neq 0 (
    echo [ERROR] Dashboard failed to launch. Checking logs...
    pause
)

pause
