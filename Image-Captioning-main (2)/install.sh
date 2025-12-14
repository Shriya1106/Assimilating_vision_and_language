#!/bin/bash

# =============================================================================
# MULTILINGUAL IMAGE CAPTIONING - INSTALLER (Linux/macOS)
# =============================================================================

echo ""
echo "============================================================"
echo "   MULTILINGUAL IMAGE CAPTIONING - INSTALLER"
echo "   AI-Powered Captions in 25+ Languages"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed!"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

echo "[OK] Python found"
python3 --version
echo ""

echo "============================================================"
echo "   STEP 1: Installing PyTorch"
echo "============================================================"
echo ""
pip3 install torch torchvision
echo ""

echo "============================================================"
echo "   STEP 2: Installing Core Dependencies"
echo "============================================================"
echo ""
pip3 install transformers>=4.35.0 accelerate>=0.24.0
pip3 install "gradio>=4.0.0,<6.0.0"
pip3 install pillow>=10.0.0
pip3 install "numpy<2.0.0"
echo ""

echo "============================================================"
echo "   STEP 3: Installing Voice Support"
echo "============================================================"
echo ""
pip3 install edge-tts>=6.1.0
pip3 install pygame>=2.5.0
pip3 install pyttsx3>=2.90
echo ""

echo "============================================================"
echo "   STEP 4: Installing OCR and Translation"
echo "============================================================"
echo ""
pip3 install easyocr>=1.7.0
pip3 install deep-translator>=1.11.0
pip3 install langdetect>=1.0.9
echo ""

echo "============================================================"
echo "   STEP 5: Installing Other Dependencies"
echo "============================================================"
echo ""
pip3 install nltk>=3.7 tqdm>=4.66.0 requests>=2.28.0
echo ""

echo "============================================================"
echo "   STEP 6: Downloading NLTK Data"
echo "============================================================"
echo ""
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); print('NLTK data downloaded!')"
echo ""

echo "============================================================"
echo "   INSTALLATION COMPLETE!"
echo "============================================================"
echo ""
echo "To run the application:"
echo "   python3 app_blip.py"
echo ""
echo "Then open http://localhost:7860 in your browser"
echo ""

