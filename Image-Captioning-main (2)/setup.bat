@echo off
REM Image Captioning Setup Script for Windows
REM This script automates the setup process

echo ============================================================
echo   Image Captioning - Automated Setup (Windows)
echo ============================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python is installed
python --version
echo.

REM Create virtual environment (optional but recommended)
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt')"
echo [OK] NLTK data downloaded
echo.

REM Check for dataset
echo Checking for COCO dataset...
if exist "coco_data\val2017" (
    echo [OK] COCO dataset found
) else (
    echo [WARNING] COCO dataset not found
    echo.
    set /p download="Download COCO validation set now? (y/n): "
    if /i "%download%"=="y" (
        echo Downloading COCO dataset...
        python download_coco.py
    )
)
echo.

REM Setup complete
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Download dataset (if not done): python download_coco.py
echo   2. Train model (optional): python train_cpu.py --num_epochs 3 --batch_size 16
echo   3. Run inference: python inference_cpu.py --image path\to\image.jpg
echo   4. Launch web app: python app_gradio.py
echo.
echo For detailed instructions, see SETUP_GUIDE.md
echo.
pause
