@echo off
title Multilingual Image Captioning - Installation
color 0A

echo.
echo ============================================================
echo    MULTILINGUAL IMAGE CAPTIONING - INSTALLER
echo    AI-Powered Captions in 25+ Languages
echo ============================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

echo ============================================================
echo    STEP 1: Installing PyTorch (CPU version)
echo ============================================================
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo [WARNING] PyTorch installation had issues, continuing...
)
echo.

echo ============================================================
echo    STEP 2: Installing Core Dependencies
echo ============================================================
echo.
pip install transformers>=4.35.0 accelerate>=0.24.0
pip install "gradio>=4.0.0,<6.0.0"
pip install pillow>=10.0.0
pip install "numpy<2.0.0" --force-reinstall
echo.

echo ============================================================
echo    STEP 3: Installing Voice Support
echo ============================================================
echo.
pip install edge-tts>=6.1.0
pip install pygame>=2.5.0
pip install pyttsx3>=2.90
echo.

echo ============================================================
echo    STEP 4: Installing OCR and Translation
echo ============================================================
echo.
pip install easyocr>=1.7.0
pip install deep-translator>=1.11.0
pip install langdetect>=1.0.9
echo.

echo ============================================================
echo    STEP 5: Installing Other Dependencies
echo ============================================================
echo.
pip install nltk>=3.7
pip install tqdm>=4.66.0
pip install requests>=2.28.0
echo.

echo ============================================================
echo    STEP 6: Downloading NLTK Data
echo ============================================================
echo.
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); print('NLTK data downloaded!')"
echo.

echo ============================================================
echo    INSTALLATION COMPLETE!
echo ============================================================
echo.
echo To run the application:
echo    1. Double-click 'run.bat'
echo    2. Or run: python app_blip.py
echo    3. Open http://localhost:7860 in your browser
echo.
echo First run will download AI models (~1GB) - this is normal!
echo.
pause

