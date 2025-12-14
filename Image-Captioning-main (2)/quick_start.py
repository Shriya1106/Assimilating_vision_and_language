"""
Quick Start Script for Image Captioning Project
Automates setup and provides guided workflow
"""

import os
import sys
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"➤ {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"Output: {e.output}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required = ['torch', 'torchvision', 'PIL', 'nltk', 'gradio']
    missing = []
    
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)
    
    return len(missing) == 0, missing


def install_dependencies():
    """Install required dependencies"""
    print_header("Installing Dependencies")
    
    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found!")
        return False
    
    print("Installing packages from requirements.txt...")
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies"
    )
    
    if success:
        # Download NLTK data
        print("\nDownloading NLTK data...")
        run_command(
            f"{sys.executable} -c \"import nltk; nltk.download('punkt')\"",
            "Downloading NLTK punkt tokenizer"
        )
    
    return success


def check_dataset():
    """Check if COCO dataset exists"""
    print_header("Checking COCO Dataset")
    
    paths = [
        './coco_data/val2017',
        './coco_data/annotations/captions_val2017.json'
    ]
    
    all_exist = True
    for path in paths:
        if os.path.exists(path):
            print(f"✓ Found: {path}")
        else:
            print(f"✗ Not found: {path}")
            all_exist = False
    
    return all_exist


def download_dataset():
    """Download COCO dataset"""
    print_header("Downloading COCO Dataset")
    
    print("This will download the COCO validation set (~1 GB)")
    print("This may take 10-30 minutes depending on your internet speed.")
    
    response = input("\nProceed with download? (y/n): ").strip().lower()
    
    if response == 'y':
        return run_command(
            f"{sys.executable} download_coco.py",
            "Downloading COCO dataset"
        )
    else:
        print("Skipping dataset download.")
        return False


def check_models():
    """Check if trained models exist"""
    print_header("Checking Trained Models")
    
    model_files = ['encoder-5.pkl', 'decoder-5.pkl', 'vocab.pkl']
    all_exist = True
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✓ Found: {model_file}")
        else:
            print(f"✗ Not found: {model_file}")
            all_exist = False
    
    return all_exist


def show_next_steps(has_models, has_dataset):
    """Show next steps based on current state"""
    print_header("Next Steps")
    
    if not has_dataset:
        print("📥 1. Download COCO Dataset:")
        print("   python download_coco.py")
        print()
    
    if not has_models:
        print("🎓 2. Train the Model (Optional):")
        print("   python train_cpu.py --num_epochs 3 --batch_size 16")
        print("   Note: Training on CPU takes 1-2 hours per epoch")
        print()
        print("   OR use pre-trained models if available")
        print()
    
    print("🔮 3. Run Inference:")
    print("   python inference_cpu.py --image path/to/image.jpg")
    print()
    
    print("🌐 4. Launch Web Interface:")
    print("   python app_gradio.py")
    print()
    
    print("📖 5. Read the Setup Guide:")
    print("   See SETUP_GUIDE.md for detailed instructions")
    print()


def main():
    """Main quick start function"""
    print("\n" + "="*60)
    print("  🖼️  Image Captioning - Quick Start Setup")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        print(f"  Current version: {sys.version}")
        return
    
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Step 1: Check dependencies
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        response = input("\nInstall missing dependencies? (y/n): ").strip().lower()
        
        if response == 'y':
            if not install_dependencies():
                print("\n✗ Failed to install dependencies")
                print("Please install manually: pip install -r requirements.txt")
                return
            print("\n✓ Dependencies installed successfully!")
        else:
            print("\nPlease install dependencies manually:")
            print("  pip install -r requirements.txt")
            return
    else:
        print("\n✓ All dependencies are installed!")
    
    # Step 2: Check dataset
    has_dataset = check_dataset()
    
    if not has_dataset:
        print("\n⚠ COCO dataset not found")
        response = input("\nDownload dataset now? (y/n): ").strip().lower()
        
        if response == 'y':
            download_dataset()
            has_dataset = check_dataset()
    
    # Step 3: Check models
    has_models = check_models()
    
    if not has_models:
        print("\n⚠ Trained models not found")
        print("You'll need to either:")
        print("  1. Train the model (takes several hours on CPU)")
        print("  2. Download pre-trained models (if available)")
    
    # Show next steps
    show_next_steps(has_models, has_dataset)
    
    # Final summary
    print_header("Setup Summary")
    print(f"✓ Dependencies: {'Installed' if deps_ok else 'Missing'}")
    print(f"{'✓' if has_dataset else '✗'} Dataset: {'Available' if has_dataset else 'Not found'}")
    print(f"{'✓' if has_models else '✗'} Models: {'Available' if has_models else 'Not found'}")
    
    if deps_ok and has_dataset and has_models:
        print("\n🎉 Setup complete! You're ready to generate captions!")
        print("\nQuick start:")
        print("  python app_gradio.py")
    elif deps_ok and has_dataset:
        print("\n⚠ Setup partially complete")
        print("Train the model or download pre-trained weights to continue")
    else:
        print("\n⚠ Setup incomplete")
        print("Follow the steps above to complete setup")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Please check SETUP_GUIDE.md for manual setup instructions")
