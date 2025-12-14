# 🔊 Text-to-Speech Voice Feature Guide

This guide explains how to use the text-to-speech (voice output) feature in the Image Captioning project.

---

## 📋 Overview

The voice feature converts generated image captions into spoken audio, making the application more accessible and interactive. It uses the `pyttsx3` library, which works offline and supports multiple voices.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pyttsx3
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Test Voice Output

```bash
python inference_with_voice.py --test-voice
```

This will play test phrases to verify your audio setup.

### 3. Use Voice with Single Image

```bash
python inference_with_voice.py --image path/to/image.jpg
```

The caption will be displayed AND spoken aloud! 🔊

---

## 💻 Usage Examples

### Command Line Interface

#### Single Image with Voice
```bash
python inference_with_voice.py --image beach.jpg
```
**Output:**
- Displays image with caption
- Speaks: "a person walking on the beach near the ocean"

#### Single Image WITHOUT Voice
```bash
python inference_with_voice.py --image beach.jpg --no-voice
```

#### Batch Processing with Voice
```bash
python inference_with_voice.py --image_dir ./photos/ --output results.txt
```
Each caption will be spoken as it's generated!

#### Batch Processing WITHOUT Voice
```bash
python inference_with_voice.py --image_dir ./photos/ --no-voice
```

#### Adjust Voice Settings
```bash
# Slower speech (100 words per minute)
python inference_with_voice.py --image photo.jpg --voice-rate 100

# Faster speech (200 words per minute)
python inference_with_voice.py --image photo.jpg --voice-rate 200

# Lower volume (50%)
python inference_with_voice.py --image photo.jpg --voice-volume 0.5
```

---

## 🌐 Web Interface (Gradio)

### Launch Web App with Voice
```bash
python app_gradio.py
```

### Using Voice in Web Interface

1. Open browser to `http://localhost:7860`
2. Upload an image
3. ✅ **Check the "🔊 Enable Voice Output" checkbox**
4. Click Submit
5. The caption will be displayed AND spoken!

**Features:**
- Toggle voice on/off with checkbox
- Adjustable caption length
- Real-time voice output
- Works with any image

---

## ⚙️ Voice Settings

### Available Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--voice-rate` | Speech speed (words/min) | 150 | 50-300 |
| `--voice-volume` | Volume level | 1.0 | 0.0-1.0 |
| `--no-voice` | Disable voice output | False | - |

### Examples

```bash
# Slow and quiet
python inference_with_voice.py --image photo.jpg --voice-rate 100 --voice-volume 0.3

# Fast and loud
python inference_with_voice.py --image photo.jpg --voice-rate 200 --voice-volume 1.0

# Normal speed, half volume
python inference_with_voice.py --image photo.jpg --voice-volume 0.5
```

---

## 🎙️ Voice Selection

The system automatically selects the best available voice on your system.

### Windows
- Default voices: Microsoft David, Microsoft Zira
- Female voice preferred if available

### macOS
- Default voices: Alex, Samantha, Victoria
- Multiple high-quality voices available

### Linux
- Uses espeak or festival
- May need additional installation:
  ```bash
  sudo apt-get install espeak
  # or
  sudo apt-get install festival
  ```

---

## 🔧 Troubleshooting

### Issue: "Could not initialize text-to-speech"

**Solution 1: Install pyttsx3**
```bash
pip install pyttsx3
```

**Solution 2 (Windows): Install pywin32**
```bash
pip install pywin32
```

**Solution 3 (Linux): Install espeak**
```bash
sudo apt-get install espeak
```

### Issue: No sound output

**Check:**
1. Volume is not muted
2. Speakers/headphones are connected
3. Other audio applications work
4. Try adjusting `--voice-volume 1.0`

### Issue: Voice is too fast/slow

**Solution:**
```bash
# Adjust speech rate
python inference_with_voice.py --image photo.jpg --voice-rate 120
```

### Issue: Voice sounds robotic

**Note:** This is normal for offline TTS engines. The quality depends on your system's installed voices.

**Improvement options:**
- Windows: Install additional SAPI5 voices
- macOS: Use built-in high-quality voices
- Linux: Install festival or espeak-ng

---

## 📊 Performance Impact

### With Voice Enabled
- **Inference Time**: 2-3 seconds (same as without voice)
- **Voice Output Time**: 1-3 seconds (depends on caption length)
- **Total Time**: 3-6 seconds per image

### Without Voice
- **Total Time**: 2-3 seconds per image

**Note:** Voice output happens AFTER caption generation, so it doesn't slow down the model.

---

## 🎯 Use Cases

### 1. Accessibility
- Helps visually impaired users understand image content
- Audio feedback for screen reader users

### 2. Hands-Free Operation
- Listen to captions while multitasking
- Useful for presentations or demos

### 3. Educational
- Learn vocabulary through audio
- Practice pronunciation
- Language learning applications

### 4. Entertainment
- Fun interactive demos
- Social media content creation
- Automated narration

---

## 🔄 Integration Examples

### Python Script Integration

```python
from inference_with_voice import ImageCaptionerWithVoice

# Initialize with voice enabled
captioner = ImageCaptionerWithVoice(
    encoder_path='encoder-5.pkl',
    decoder_path='decoder-5.pkl',
    vocab_path='vocab.pkl',
    enable_voice=True,
    voice_rate=150,
    voice_volume=1.0
)

# Generate caption with voice
caption, time = captioner.generate_caption(
    'photo.jpg',
    speak_caption=True
)

print(f"Caption: {caption}")
```

### Custom Voice Settings

```python
# Initialize with custom settings
captioner = ImageCaptionerWithVoice(
    encoder_path='encoder-5.pkl',
    decoder_path='decoder-5.pkl',
    vocab_path='vocab.pkl',
    enable_voice=True,
    voice_rate=120,  # Slower
    voice_volume=0.8  # Quieter
)

# Test voice
captioner.test_voice()

# Generate with voice
caption, _ = captioner.generate_caption('image.jpg', speak_caption=True)
```

---

## 📱 Platform-Specific Notes

### Windows
- ✅ Works out of the box
- ✅ Multiple voices available
- ✅ High quality SAPI5 voices
- 📦 Requires: `pywin32` (installed with pyttsx3)

### macOS
- ✅ Works out of the box
- ✅ Excellent voice quality
- ✅ Multiple accents available
- 📦 Uses: NSSpeechSynthesizer

### Linux
- ⚠️ Requires espeak or festival
- ⚠️ Voice quality varies
- ✅ Lightweight and fast
- 📦 Install: `sudo apt-get install espeak`

---

## 🎨 Advanced Features

### Voice Testing
```bash
# Test voice output
python inference_with_voice.py --test-voice
```

### Batch Processing with Voice
```bash
# Process 10 images with voice
python inference_with_voice.py \
    --image_dir ./photos/ \
    --max_images 10 \
    --output results.txt
```

### Silent Mode (No Voice)
```bash
# Disable voice for batch processing
python inference_with_voice.py \
    --image_dir ./photos/ \
    --no-voice \
    --output results.txt
```

---

## 📈 Comparison

| Feature | Without Voice | With Voice |
|---------|--------------|------------|
| Caption Generation | ✅ | ✅ |
| Visual Display | ✅ | ✅ |
| Audio Output | ❌ | ✅ |
| Accessibility | Limited | High |
| Inference Time | 2-3s | 2-3s |
| Total Time | 2-3s | 3-6s |
| Dependencies | Standard | +pyttsx3 |

---

## 🎓 Best Practices

### 1. Testing
- Always test voice before demos
- Use `--test-voice` to verify setup
- Check volume levels

### 2. Performance
- Disable voice for batch processing if not needed
- Use `--no-voice` for faster processing
- Voice adds 1-3 seconds per caption

### 3. User Experience
- Provide option to disable voice
- Set appropriate speech rate (150 recommended)
- Consider environment (quiet vs noisy)

### 4. Accessibility
- Always provide text output alongside voice
- Allow users to control voice settings
- Support keyboard shortcuts for voice toggle

---

## 🔮 Future Enhancements

Potential improvements for the voice feature:

1. **Multiple Language Support**
   - Captions in different languages
   - Voice output in matching language

2. **Voice Selection UI**
   - Choose from available voices
   - Preview different voices

3. **Audio Export**
   - Save voice output to MP3/WAV
   - Batch export for multiple images

4. **Advanced TTS**
   - Use cloud TTS services (Google, AWS)
   - Higher quality voices
   - More natural speech

5. **Voice Effects**
   - Adjust pitch and tone
   - Add emphasis to keywords
   - Emotion in voice

---

## 📚 Resources

### Documentation
- [pyttsx3 Documentation](https://pyttsx3.readthedocs.io/)
- [Python Text-to-Speech Guide](https://realpython.com/python-text-to-speech-pyttsx3/)

### Voice Engines
- **Windows**: SAPI5
- **macOS**: NSSpeechSynthesizer
- **Linux**: espeak, festival

### Additional Voices
- **Windows**: [Microsoft Speech Platform](https://www.microsoft.com/en-us/download/details.aspx?id=27224)
- **macOS**: System Preferences → Accessibility → Speech
- **Linux**: `sudo apt-get install espeak-ng`

---

## ✅ Summary

The text-to-speech feature adds voice output to image captions:

✅ **Easy to use** - Just add `--voice` or check a box
✅ **Cross-platform** - Works on Windows, macOS, Linux
✅ **Offline** - No internet required
✅ **Customizable** - Adjust rate, volume, and more
✅ **Accessible** - Helps visually impaired users
✅ **Fast** - Minimal performance impact

**Get started:**
```bash
pip install pyttsx3
python inference_with_voice.py --image your_image.jpg
```

**Or use the web interface:**
```bash
python app_gradio.py
# Check "Enable Voice Output" checkbox
```

---

**Enjoy your talking image captioning system! 🔊🖼️**
