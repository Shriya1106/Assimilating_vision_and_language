"""
Installation Verification Script
Run this to check if all dependencies are properly installed.
"""

import sys

def check_module(name, import_name=None):
    """Check if a module is installed"""
    if import_name is None:
        import_name = name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'installed')
        print(f"✅ {name}: {version}")
        return True
    except ImportError:
        print(f"❌ {name}: NOT INSTALLED")
        return False

def main():
    print("\n" + "=" * 60)
    print("   INSTALLATION VERIFICATION")
    print("=" * 60 + "\n")
    
    print(f"Python Version: {sys.version}\n")
    
    all_ok = True
    
    print("--- Core Libraries ---")
    all_ok &= check_module("PyTorch", "torch")
    all_ok &= check_module("TorchVision", "torchvision")
    all_ok &= check_module("Transformers", "transformers")
    all_ok &= check_module("Accelerate", "accelerate")
    
    print("\n--- Image Processing ---")
    all_ok &= check_module("Pillow", "PIL")
    all_ok &= check_module("NumPy", "numpy")
    
    print("\n--- Web Interface ---")
    all_ok &= check_module("Gradio", "gradio")
    
    print("\n--- Voice Support ---")
    all_ok &= check_module("Edge TTS", "edge_tts")
    all_ok &= check_module("Pygame", "pygame")
    all_ok &= check_module("pyttsx3", "pyttsx3")
    
    print("\n--- OCR ---")
    all_ok &= check_module("EasyOCR", "easyocr")
    
    print("\n--- Translation ---")
    all_ok &= check_module("Deep Translator", "deep_translator")
    all_ok &= check_module("LangDetect", "langdetect")
    
    print("\n--- NLP ---")
    all_ok &= check_module("NLTK", "nltk")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("\nYou can now run: python app_blip.py")
    else:
        print("❌ SOME DEPENDENCIES ARE MISSING!")
        print("\nRun: pip install -r requirements.txt")
    print("=" * 60 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

