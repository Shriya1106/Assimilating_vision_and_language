# Image Captioning Setup Guide (CPU-Optimized)

This guide will help you set up and run the Image Captioning project on CPU without requiring a GPU.

## 📋 Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Download COCO Dataset](#download-coco-dataset)
- [Training](#training)
- [Inference](#inference)
- [Web Interface](#web-interface)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Requirements

- Python 3.8 or higher
- At least 8GB RAM (16GB recommended for training)
- 20GB free disk space (for COCO dataset)
- Windows/Linux/MacOS

---

## 📦 Installation

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

---

## 📥 Download COCO Dataset

The project includes an automated COCO dataset downloader.

### Quick Start (Validation Set Only - Recommended for Testing)

```bash
python download_coco.py
```

This will present you with options:
1. **Option 1**: Download validation set (~1 GB) - **Recommended for testing**
2. **Option 2**: Download training set (~18 GB) - For full training
3. **Option 3**: Download annotations only (~250 MB)
4. **Option 4**: Download everything (~19 GB)

### Manual Download (Alternative)

If the script doesn't work, download manually:

1. **Validation Images**: http://images.cocodataset.org/zips/val2017.zip
2. **Training Images**: http://images.cocodataset.org/zips/train2017.zip
3. **Annotations**: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Extract to:
```
coco_data/
├── train2017/
├── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

---

## 🎓 Training

### Training on CPU (Small Dataset)

For CPU training, it's recommended to use a subset of data:

```bash
python train_cpu.py \
    --num_epochs 3 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --embed_size 256 \
    --hidden_size 512 \
    --train_images ./coco_data/train2017 \
    --train_annotations ./coco_data/annotations/captions_train2017.json \
    --save_dir ./models
```

### Training Parameters

- `--num_epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size (default: 32, reduce to 8-16 for CPU)
- `--learning_rate`: Learning rate (default: 0.001)
- `--embed_size`: Embedding dimension (default: 256)
- `--hidden_size`: LSTM hidden size (default: 512)
- `--save_dir`: Directory to save models (default: ./models)

### Resume Training

```bash
python train_cpu.py --resume ./models/checkpoint-3.pkl
```

### Training Tips for CPU

1. **Reduce batch size** to 8-16 to avoid memory issues
2. **Use fewer epochs** (3-5) for initial testing
3. **Monitor memory usage** - close other applications
4. **Be patient** - CPU training is slower than GPU (expect 1-2 hours per epoch)

---

## 🔮 Inference

### Single Image Inference

```bash
python inference_cpu.py \
    --image path/to/your/image.jpg \
    --encoder ./models/encoder-5.pkl \
    --decoder ./models/decoder-5.pkl \
    --vocab vocab.pkl
```

### Batch Inference (Multiple Images)

```bash
python inference_cpu.py \
    --image_dir path/to/images/ \
    --output results.txt \
    --max_images 10 \
    --encoder ./models/encoder-5.pkl \
    --decoder ./models/decoder-5.pkl
```

### Inference Parameters

- `--image`: Path to single image
- `--image_dir`: Directory with multiple images
- `--encoder`: Path to encoder weights
- `--decoder`: Path to decoder weights
- `--vocab`: Path to vocabulary file (default: vocab.pkl)
- `--max_length`: Maximum caption length (default: 20)
- `--output`: Output file for batch results

---

## 🌐 Web Interface

Launch the Gradio web interface for easy image captioning:

```bash
python app_gradio.py
```

This will start a web server at `http://localhost:7860`

### Features:
- Upload images via drag-and-drop
- Adjust caption length
- Real-time caption generation
- User-friendly interface

### Public Access

To create a public link (accessible from anywhere):

Edit `app_gradio.py` and change:
```python
interface.launch(share=True)  # Creates public URL
```

---

## 🐛 Troubleshooting

### Issue: "Out of Memory" Error

**Solution:**
- Reduce batch size: `--batch_size 8`
- Close other applications
- Use validation set instead of training set

### Issue: "COCO API not found"

**Solution:**
```bash
pip install pycocotools
```

On Windows, if you encounter issues:
```bash
pip install pycocotools-windows
```

### Issue: "Module not found" errors

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Slow Training

**Solution:**
- This is expected on CPU
- Reduce dataset size
- Use smaller model (reduce `--hidden_size` to 256)
- Consider using pre-trained models

### Issue: "Vocabulary file not found"

**Solution:**
The vocabulary is created during first training. Either:
1. Train the model first to create vocab.pkl
2. Use the existing vocab.pkl in the repository

### Issue: Model weights not found

**Solution:**
- Train the model first using `train_cpu.py`
- Or download pre-trained weights (if available)
- The inference script will warn but still run with random weights (for testing)

---

## 📊 Expected Performance

### Training Time (CPU)
- **Per Epoch**: 1-2 hours (depends on CPU and dataset size)
- **Full Training (5 epochs)**: 5-10 hours

### Inference Time (CPU)
- **Single Image**: 2-5 seconds
- **Batch (10 images)**: 20-50 seconds

### Memory Usage
- **Training**: 4-8 GB RAM
- **Inference**: 2-4 GB RAM

---

## 🎯 Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (validation set)
python download_coco.py
# Choose option 1

# 3. Train model (optional - if you want to train)
python train_cpu.py --num_epochs 3 --batch_size 16

# 4. Run inference
python inference_cpu.py --image test_image.jpg

# 5. Or launch web interface
python app_gradio.py
```

---

## 📚 Additional Resources

- **Original Paper**: [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- **COCO Dataset**: [https://cocodataset.org/](https://cocodataset.org/)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)

---

## 🤝 Support

If you encounter any issues:
1. Check the Troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify dataset paths are correct
4. Check that you have enough disk space and RAM

---

## 📝 Notes

- **CPU vs GPU**: This project is optimized for CPU. GPU training would be 10-50x faster.
- **Model Size**: The default model is relatively small to work well on CPU.
- **Dataset**: Using the full COCO dataset on CPU will take significant time. Consider using a subset for testing.
- **Pre-trained Models**: If available, using pre-trained weights will save training time.

---

**Happy Captioning! 🎉**
