"""
BLIP-based Image Captioning with Text-to-Speech
Modern image captioning that works on ANY image!

Uses Salesforce BLIP model from Hugging Face Transformers
"""

import os
import torch
from PIL import Image
import pyttsx3
import time
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings("ignore")


class BLIPCaptioner:
    """
    Modern Image Captioner using BLIP (Bootstrapped Language-Image Pre-training)
    
    Features:
    - Works on ANY image (not limited to dataset vocabulary)
    - State-of-the-art caption quality
    - Multiple caption styles (standard, detailed, Q&A)
    - Integrated text-to-speech
    - CPU and GPU support
    """
    
    def __init__(
        self, 
        model_name: str = "Salesforce/blip-image-captioning-large",
        device: str = "auto",
        enable_voice: bool = True,
        voice_rate: int = 150,
        voice_volume: float = 1.0
    ):
        """
        Initialize BLIP Captioner
        
        Args:
            model_name: BLIP model to use. Options:
                - "Salesforce/blip-image-captioning-base" (smaller, faster)
                - "Salesforce/blip-image-captioning-large" (better quality)
            device: Device to run on ("auto", "cpu", or "cuda")
            enable_voice: Enable text-to-speech output
            voice_rate: Speech rate (words per minute)
            voice_volume: Volume level (0.0 to 1.0)
        """
        print("=" * 60)
        print("🚀 Initializing BLIP Image Captioner")
        print("=" * 60)
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"📱 Using device: {self.device}")
        
        # Load BLIP model
        print(f"📥 Loading BLIP model: {model_name}")
        print("   (This may take a minute on first run...)")
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32  # Use float32 for CPU compatibility
            ).to(self.device)
            
            self.model.eval()
            print("✅ BLIP model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading BLIP model: {e}")
            print("\n💡 Try running: pip install transformers accelerate")
            raise
        
        # Initialize text-to-speech
        self.enable_voice = enable_voice
        self.tts_engine = None
        
        if self.enable_voice:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', voice_rate)
                self.tts_engine.setProperty('volume', voice_volume)
                
                # Try to use a pleasant voice
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                
                print("🔊 Text-to-speech initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize text-to-speech: {e}")
                self.enable_voice = False
        
        print("=" * 60)
        print("✨ Ready to caption ANY image!")
        print("=" * 60 + "\n")
    
    def speak(self, text: str) -> None:
        """
        Convert text to speech
        
        Args:
            text: Text to speak aloud
        """
        if self.enable_voice and self.tts_engine and text:
            try:
                print(f"🔊 Speaking: \"{text}\"")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"⚠️  Error speaking: {e}")
    
    def load_image(self, image_source) -> Image.Image:
        """
        Load image from various sources
        
        Args:
            image_source: Can be:
                - File path (str)
                - URL (str starting with http)
                - PIL Image
                - numpy array
        
        Returns:
            PIL Image in RGB format
        """
        import numpy as np
        
        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                # Load from URL
                import requests
                from io import BytesIO
                response = requests.get(image_source, timeout=10)
                image = Image.open(BytesIO(response.content))
            else:
                # Load from file path
                image = Image.open(image_source)
        elif isinstance(image_source, np.ndarray):
            image = Image.fromarray(image_source)
        elif isinstance(image_source, Image.Image):
            image = image_source
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")
        
        return image.convert("RGB")
    
    def generate_caption(
        self,
        image_source,
        max_length: int = 50,
        min_length: int = 5,
        num_beams: int = 5,
        speak_caption: bool = True,
        caption_style: str = "standard"
    ) -> Tuple[str, float]:
        """
        Generate caption for an image
        
        Args:
            image_source: Image path, URL, PIL Image, or numpy array
            max_length: Maximum caption length (tokens)
            min_length: Minimum caption length (tokens)
            num_beams: Beam search width (higher = better but slower)
            speak_caption: Whether to speak the caption aloud
            caption_style: Caption style:
                - "standard": Normal descriptive caption
                - "detailed": More detailed description
                - "brief": Short, concise caption
        
        Returns:
            Tuple of (caption string, inference time in seconds)
        """
        start_time = time.time()
        
        # Load image
        image = self.load_image(image_source)
        
        # Adjust parameters based on style
        if caption_style == "detailed":
            text_prompt = "a detailed photograph of"
            max_length = min(max_length + 30, 100)
        elif caption_style == "brief":
            text_prompt = None
            max_length = min(max_length, 20)
            min_length = 3
        else:  # standard
            text_prompt = None
        
        # Process image
        if text_prompt:
            inputs = self.processor(image, text_prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode caption
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Clean up caption
        caption = caption.strip()
        if caption and not caption[0].isupper():
            caption = caption[0].upper() + caption[1:]
        
        inference_time = time.time() - start_time
        
        # Speak if enabled
        if speak_caption:
            self.speak(caption)
        
        return caption, inference_time
    
    def answer_question(
        self,
        image_source,
        question: str,
        speak_answer: bool = True
    ) -> Tuple[str, float]:
        """
        Answer a question about an image (Visual Question Answering)
        
        Args:
            image_source: Image path, URL, PIL Image, or numpy array
            question: Question to ask about the image
            speak_answer: Whether to speak the answer aloud
        
        Returns:
            Tuple of (answer string, inference time in seconds)
        """
        start_time = time.time()
        
        # Load image
        image = self.load_image(image_source)
        
        # Process with question
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5
            )
        
        # Decode answer
        answer = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        inference_time = time.time() - start_time
        
        # Speak if enabled
        if speak_answer:
            self.speak(answer)
        
        return answer, inference_time
    
    def batch_caption(
        self,
        image_sources: List,
        speak_captions: bool = False,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Generate captions for multiple images
        
        Args:
            image_sources: List of image paths, URLs, or PIL Images
            speak_captions: Whether to speak each caption
            **kwargs: Additional arguments for generate_caption
        
        Returns:
            List of (caption, inference_time) tuples
        """
        results = []
        total = len(image_sources)
        
        print(f"\n📸 Processing {total} images...")
        
        for i, source in enumerate(image_sources, 1):
            print(f"\n[{i}/{total}] Processing...", end=" ")
            try:
                caption, time_taken = self.generate_caption(
                    source, 
                    speak_caption=speak_captions,
                    **kwargs
                )
                results.append((caption, time_taken))
                print(f"✅ ({time_taken:.2f}s)")
                print(f"   Caption: {caption}")
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append((f"Error: {e}", 0))
        
        return results
    
    def test_voice(self) -> None:
        """Test the text-to-speech functionality"""
        if self.enable_voice:
            print("\n🔊 Testing voice output...")
            test_phrases = [
                "Hello! I am your BLIP image captioning assistant.",
                "I can describe any image you show me.",
                "Voice output is working correctly!"
            ]
            for phrase in test_phrases:
                self.speak(phrase)
                time.sleep(0.3)
            print("✅ Voice test complete!")
        else:
            print("⚠️  Voice output is disabled")


def main():
    """Demo and CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🖼️ BLIP Image Captioning with Voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python blip_captioner.py --image photo.jpg
  python blip_captioner.py --image photo.jpg --style detailed
  python blip_captioner.py --image https://example.com/image.jpg
  python blip_captioner.py --image photo.jpg --question "What color is the car?"
  python blip_captioner.py --test-voice
        """
    )
    
    parser.add_argument('--image', type=str, help='Image path or URL')
    parser.add_argument('--question', type=str, help='Question about the image (VQA mode)')
    parser.add_argument('--style', type=str, default='standard',
                       choices=['standard', 'detailed', 'brief'],
                       help='Caption style')
    parser.add_argument('--max-length', type=int, default=50,
                       help='Maximum caption length')
    parser.add_argument('--no-voice', action='store_true',
                       help='Disable voice output')
    parser.add_argument('--voice-rate', type=int, default=150,
                       help='Speech rate (words per minute)')
    parser.add_argument('--model', type=str, 
                       default='Salesforce/blip-image-captioning-large',
                       help='BLIP model to use')
    parser.add_argument('--test-voice', action='store_true',
                       help='Test voice output and exit')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Initialize captioner
    captioner = BLIPCaptioner(
        model_name=args.model,
        device=args.device,
        enable_voice=not args.no_voice,
        voice_rate=args.voice_rate
    )
    
    # Test voice mode
    if args.test_voice:
        captioner.test_voice()
        return
    
    # Need an image for other modes
    if not args.image:
        print("\n💡 Usage examples:")
        print("   python blip_captioner.py --image photo.jpg")
        print("   python blip_captioner.py --image photo.jpg --style detailed")
        print("   python blip_captioner.py --test-voice")
        return
    
    # VQA mode
    if args.question:
        print(f"\n❓ Question: {args.question}")
        answer, time_taken = captioner.answer_question(
            args.image,
            args.question,
            speak_answer=not args.no_voice
        )
        print(f"\n💬 Answer: {answer}")
        print(f"⏱️  Time: {time_taken:.2f}s")
    
    # Caption mode
    else:
        print(f"\n🖼️  Generating {args.style} caption for: {args.image}")
        caption, time_taken = captioner.generate_caption(
            args.image,
            max_length=args.max_length,
            speak_caption=not args.no_voice,
            caption_style=args.style
        )
        print(f"\n📝 Caption: {caption}")
        print(f"⏱️  Time: {time_taken:.2f}s")


if __name__ == "__main__":
    main()

