"""
Gradio Web Interface for Image Captioning
CPU-optimized web app for generating image captions with voice output
"""

import os
import pickle
import torch
import numpy as np
from PIL import Image
import gradio as gr
from torchvision import transforms
import pyttsx3

from model import EncoderCNN, DecoderRNN


class ImageCaptionerApp:
    def __init__(self, encoder_path='encoder-5.pkl', decoder_path='decoder-5.pkl', 
                 vocab_path='vocab.pkl', embed_size=256, hidden_size=512):
        """
        Initialize the Image Captioner App with Voice Support
        """
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Initialize text-to-speech
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 1.0)
            self.voice_enabled = True
            print("✓ Text-to-speech initialized")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize text-to-speech: {e}")
            self.voice_enabled = False
        
        # Load vocabulary
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        vocab_size = len(self.vocab)
        
        # Initialize models
        self.encoder = EncoderCNN(embed_size, device='cpu').to(self.device)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, device='cpu').to(self.device)
        
        # Load weights if available
        if os.path.exists(encoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print(f"✓ Loaded encoder from {encoder_path}")
        else:
            print(f"⚠ Warning: Using untrained encoder (weights not found)")
        
        if os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(f"✓ Loaded decoder from {decoder_path}")
        else:
            print(f"⚠ Warning: Using untrained decoder (weights not found)")
        
        # Set to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✓ Model initialized successfully!")
    
    def speak(self, text):
        """
        Convert text to speech
        
        Args:
            text: Text to speak
        """
        if self.voice_enabled and text:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"⚠ Error speaking text: {e}")
    
    def generate_caption(self, image, max_length=20, enable_voice=False):
        """
        Generate caption for an image
        
        Args:
            image: PIL Image or numpy array
            max_length: Maximum caption length
            enable_voice: Whether to speak the caption
            
        Returns:
            Generated caption string
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Generate caption
        with torch.no_grad():
            features = self.encoder(image_tensor)
            caption_ids = self.decoder.sample(features.unsqueeze(1), max_len=max_length)
        
        # Convert IDs to words
        caption_words = []
        for word_id in caption_ids:
            word = self.vocab.idx2word[word_id]
            if word == '<end>':
                break
            if word != '<start>':
                caption_words.append(word)
        
        caption = ' '.join(caption_words)
        
        if not caption:
            caption = "Unable to generate caption"
        
        # Speak the caption if requested
        if enable_voice:
            self.speak(caption)
        
        return caption


def create_interface():
    """
    Create Gradio interface
    """
    # Initialize the captioner
    try:
        captioner = ImageCaptionerApp()
        model_loaded = True
    except Exception as e:
        print(f"Error initializing model: {e}")
        model_loaded = False
    
    def predict(image, max_length, enable_voice):
        """
        Prediction function for Gradio
        """
        if not model_loaded:
            return "Error: Model not loaded. Please check if model weights and vocab.pkl exist."
        
        try:
            caption = captioner.generate_caption(
                image, 
                max_length=int(max_length),
                enable_voice=enable_voice
            )
            voice_status = " 🔊 (Spoken)" if enable_voice else ""
            return caption + voice_status
        except Exception as e:
            return f"Error generating caption: {str(e)}"
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Slider(minimum=5, maximum=50, value=20, step=1, 
                     label="Maximum Caption Length"),
            gr.Checkbox(label="🔊 Enable Voice Output (Text-to-Speech)", value=False)
        ],
        outputs=gr.Textbox(label="Generated Caption", lines=3),
        title="🖼️ Image Captioning with CNN-RNN + Voice",
        description="""
        Upload an image to generate a descriptive caption using a CNN-RNN model.
        
        **Features:**
        - 🖼️ Image-to-text caption generation
        - 🔊 Text-to-speech voice output (optional)
        - ⚙️ Adjustable caption length
        
        **Note:** This model runs on CPU, so inference may take a few seconds.
        
        **How to use:**
        1. Upload an image (JPG, PNG, etc.)
        2. Adjust maximum caption length if needed
        3. Check "Enable Voice Output" to hear the caption
        4. Click Submit to generate caption
        """,
        examples=[
            # Add example images if available
        ],
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    return interface


def main():
    """
    Launch the Gradio app
    """
    print("\n" + "="*60)
    print("Image Captioning Web App")
    print("="*60)
    print("Starting Gradio interface...")
    
    interface = create_interface()
    
    # Launch the app
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )


if __name__ == "__main__":
    main()
