@echo off
title Multilingual Image Captioning
color 0B

echo.
echo ============================================================
echo    MULTILINGUAL IMAGE CAPTIONING
echo    AI-Powered Captions in 25+ Languages
echo ============================================================
echo.
echo Starting the application...
echo.
echo Once loaded, open your browser to:
echo    http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
echo.

python app_blip.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application failed to start!
    echo.
    echo Possible solutions:
    echo    1. Run install.bat first
    echo    2. Check if Python is installed
    echo    3. Check error messages above
    echo.
    pause
)

