# 📥 Installation Guide

Complete installation guide for Multilingual Image Captioning.

---

## 🖥️ System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Windows 10, macOS 10.15, Ubuntu 20.04 | Windows 11, macOS 12+, Ubuntu 22.04 |
| **Python** | 3.10 | 3.11 or 3.12 |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 5 GB | 10 GB |
| **Internet** | Required for first run | Required for first run |

---

## 🚀 Quick Installation (Windows)

### Step 1: Install Python

1. Download Python from [python.org](https://www.python.org/downloads/)
2. **Important**: Check ✅ "Add Python to PATH" during installation
3. Restart your computer

### Step 2: Install the App

1. Download or clone this project
2. Double-click `install.bat`
3. Wait for installation to complete (5-10 minutes)

### Step 3: Run the App

1. Double-click `run.bat`
2. Open http://localhost:7860 in your browser
3. Start captioning images!

---

## 📋 Manual Installation

### For Windows (PowerShell)

```powershell
# Step 1: Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install main dependencies
pip install transformers>=4.35.0
pip install gradio>=4.0.0,<6.0.0
pip install accelerate>=0.24.0

# Step 3: Install voice support
pip install edge-tts>=6.1.0
pip install pygame>=2.5.0
pip install pyttsx3>=2.90

# Step 4: Install OCR and translation
pip install easyocr>=1.7.0
pip install deep-translator>=1.11.0
pip install langdetect>=1.0.9

# Step 5: Install other dependencies
pip install pillow>=10.0.0
pip install numpy<2.0.0
pip install nltk>=3.7

# Step 6: Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Step 7: Run the app
python app_blip.py
```

### For macOS/Linux (Terminal)

```bash
# Step 1: Install PyTorch
pip3 install torch torchvision

# Step 2: Install all dependencies
pip3 install -r requirements.txt

# Step 3: Download NLTK data
python3 -c "import nltk; nltk.download('punkt')"

# Step 4: Run the app
python3 app_blip.py
```

---

## 📦 One-Line Installation

### Windows
```powershell
pip install torch torchvision transformers gradio edge-tts pygame easyocr deep-translator langdetect pyttsx3 pillow "numpy<2.0.0" nltk accelerate && python -c "import nltk; nltk.download('punkt')" && python app_blip.py
```

### macOS/Linux
```bash
pip3 install torch torchvision transformers gradio edge-tts pygame easyocr deep-translator langdetect pyttsx3 pillow "numpy<2.0.0" nltk accelerate && python3 -c "import nltk; nltk.download('punkt')" && python3 app_blip.py
```

---

## 🔧 Troubleshooting

### Issue: "pip is not recognized"

**Solution**: Python is not in PATH
```powershell
# Windows - use full path
C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python312\python.exe -m pip install ...
```

### Issue: "Module not found"

**Solution**: Install the missing module
```bash
pip install <module_name>
```

### Issue: "CUDA out of memory" or GPU errors

**Solution**: Use CPU-only PyTorch
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "NumPy version conflict"

**Solution**: Downgrade NumPy
```bash
pip install "numpy<2.0.0" --force-reinstall
```

### Issue: Voice not working

**Solution**: Reinstall audio packages
```bash
pip install edge-tts pygame pyttsx3 --upgrade --force-reinstall
```

### Issue: OCR not detecting text

**Solution**: Reinstall EasyOCR
```bash
pip install easyocr --upgrade --force-reinstall
```

### Issue: Model download fails

**Solution**: 
1. Check internet connection
2. Try again - Hugging Face servers may be busy
3. Models are ~1GB, ensure enough disk space

### Issue: Gradio interface not loading

**Solution**: Check port availability
```bash
# Use different port
python app_blip.py --port 7861
```

---

## 📥 First Run

On first run, the app will download:

| Model | Size | Time |
|-------|------|------|
| BLIP Base | ~1 GB | 2-5 min |
| OCR Models | ~100 MB | 1-2 min |

These are cached locally and won't be downloaded again.

---

## ✅ Verify Installation

Run this to verify everything is installed correctly:

```python
# Save as test_install.py and run: python test_install.py

print("Testing installation...")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except: print("❌ PyTorch not installed")

try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except: print("❌ Transformers not installed")

try:
    import gradio
    print(f"✅ Gradio {gradio.__version__}")
except: print("❌ Gradio not installed")

try:
    import edge_tts
    print("✅ Edge TTS")
except: print("❌ Edge TTS not installed")

try:
    import easyocr
    print("✅ EasyOCR")
except: print("❌ EasyOCR not installed")

try:
    import deep_translator
    print("✅ Deep Translator")
except: print("❌ Deep Translator not installed")

try:
    import pygame
    print("✅ Pygame")
except: print("❌ Pygame not installed")

print("\nInstallation check complete!")
```

---

## 🆘 Need Help?

If you're still having issues:

1. **Check Python version**: `python --version` (should be 3.10+)
2. **Check pip version**: `pip --version`
3. **Try fresh install**: Uninstall all packages and reinstall
4. **Check disk space**: Need at least 5GB free

---

## 📝 Notes

- **No virtual environment needed**: Install directly with pip
- **Models cache**: Stored in `~/.cache/huggingface/`
- **First run slower**: Models download on first use
- **Subsequent runs**: Much faster (models cached)


