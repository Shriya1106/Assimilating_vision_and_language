"""
🖼️ Multilingual Image Captioning with OCR
AI-Powered Captions in 25+ Languages with Text Detection!
Now detects quotes, text, and signs in images for more accurate descriptions.
"""

import gradio as gr
import torch
from PIL import Image
import numpy as np
import time
import asyncio
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

# Translation support
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("⚠️ deep-translator not available")

# Language detection
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️ langdetect not available")

# OCR for text detection
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ easyocr not available - text detection disabled")

# Edge TTS for natural voice
try:
    import edge_tts
    import pygame
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("⚠️ edge-tts not available")


# =============================================================================
# LANGUAGE CONFIGURATION
# =============================================================================

# Auto-detect option
AUTO_DETECT = "🔄 Auto Detect (from image text)"

# Voice name to voice ID mapping
VOICE_OPTIONS = {
    "Jenny (US Female)": "en-US-JennyNeural",
    "Aria (US Female)": "en-US-AriaNeural",
    "Emma (US Female)": "en-US-EmmaNeural",
    "Guy (US Male)": "en-US-GuyNeural",
    "Sonia (UK Female)": "en-GB-SoniaNeural",
    "Ryan (UK Male)": "en-GB-RyanNeural",
    "Natasha (AU Female)": "en-AU-NatashaNeural",
    "Swara (Hindi)": "hi-IN-SwaraNeural",
    "Pallavi (Tamil)": "ta-IN-PallaviNeural",
    "Shruti (Telugu)": "te-IN-ShrutiNeural",
    "Tanishaa (Bengali)": "bn-IN-TanishaaNeural",
    "Xiaoxiao (Chinese)": "zh-CN-XiaoxiaoNeural",
    "Nanami (Japanese)": "ja-JP-NanamiNeural",
    "Elvira (Spanish)": "es-ES-ElviraNeural",
    "Denise (French)": "fr-FR-DeniseNeural",
}

# Language code to display name mapping (for auto-detect)
LANG_CODE_TO_NAME = {
    "hi": "🇮🇳 Hindi", "ta": "🇮🇳 Tamil", "te": "🇮🇳 Telugu", "bn": "🇮🇳 Bengali",
    "kn": "🇮🇳 Kannada", "ml": "🇮🇳 Malayalam", "mr": "🇮🇳 Marathi", "gu": "🇮🇳 Gujarati",
    "pa": "🇮🇳 Punjabi", "en": "🇺🇸 English (US)", "es": "🇪🇸 Spanish", "fr": "🇫🇷 French",
    "de": "🇩🇪 German", "it": "🇮🇹 Italian", "pt": "🇵🇹 Portuguese", "ru": "🇷🇺 Russian",
    "nl": "🇳🇱 Dutch", "zh-cn": "🇨🇳 Chinese", "ja": "🇯🇵 Japanese", "ko": "🇰🇷 Korean",
    "th": "🇹🇭 Thai", "vi": "🇻🇳 Vietnamese", "id": "🇮🇩 Indonesian", "ar": "🇸🇦 Arabic",
    "tr": "🇹🇷 Turkish", "he": "🇮🇱 Hebrew"
}

LANGUAGES = {
    # Indian Regional Languages
    "🇮🇳 Hindi": {"code": "hi", "voice": "hi-IN-SwaraNeural", "voice_name": "Swara", "ocr": "hi"},
    "🇮🇳 Tamil": {"code": "ta", "voice": "ta-IN-PallaviNeural", "voice_name": "Pallavi", "ocr": "ta"},
    "🇮🇳 Telugu": {"code": "te", "voice": "te-IN-ShrutiNeural", "voice_name": "Shruti", "ocr": "te"},
    "🇮🇳 Bengali": {"code": "bn", "voice": "bn-IN-TanishaaNeural", "voice_name": "Tanishaa", "ocr": "bn"},
    "🇮🇳 Kannada": {"code": "kn", "voice": "kn-IN-SapnaNeural", "voice_name": "Sapna", "ocr": "kn"},
    "🇮🇳 Malayalam": {"code": "ml", "voice": "ml-IN-SobhanaNeural", "voice_name": "Sobhana", "ocr": "ml"},
    "🇮🇳 Marathi": {"code": "mr", "voice": "mr-IN-AarohiNeural", "voice_name": "Aarohi", "ocr": "mr"},
    "🇮🇳 Gujarati": {"code": "gu", "voice": "gu-IN-DhwaniNeural", "voice_name": "Dhwani", "ocr": "gu"},
    "🇮🇳 Punjabi": {"code": "pa", "voice": "pa-IN-OjasNeural", "voice_name": "Ojas", "ocr": "pa"},
    
    # English Variants
    "🇺🇸 English (US)": {"code": "en", "voice": "en-US-JennyNeural", "voice_name": "Jenny", "ocr": "en"},
    "🇬🇧 English (UK)": {"code": "en", "voice": "en-GB-SoniaNeural", "voice_name": "Sonia", "ocr": "en"},
    "🇦🇺 English (AU)": {"code": "en", "voice": "en-AU-NatashaNeural", "voice_name": "Natasha", "ocr": "en"},
    
    # European Languages
    "🇪🇸 Spanish": {"code": "es", "voice": "es-ES-ElviraNeural", "voice_name": "Elvira", "ocr": "es"},
    "🇫🇷 French": {"code": "fr", "voice": "fr-FR-DeniseNeural", "voice_name": "Denise", "ocr": "fr"},
    "🇩🇪 German": {"code": "de", "voice": "de-DE-KatjaNeural", "voice_name": "Katja", "ocr": "de"},
    "🇮🇹 Italian": {"code": "it", "voice": "it-IT-ElsaNeural", "voice_name": "Elsa", "ocr": "it"},
    "🇵🇹 Portuguese": {"code": "pt", "voice": "pt-BR-FranciscaNeural", "voice_name": "Francisca", "ocr": "pt"},
    "🇷🇺 Russian": {"code": "ru", "voice": "ru-RU-SvetlanaNeural", "voice_name": "Svetlana", "ocr": "ru"},
    "🇳🇱 Dutch": {"code": "nl", "voice": "nl-NL-ColetteNeural", "voice_name": "Colette", "ocr": "nl"},
    
    # Asian Languages
    "🇨🇳 Chinese": {"code": "zh-CN", "voice": "zh-CN-XiaoxiaoNeural", "voice_name": "Xiaoxiao", "ocr": "ch_sim"},
    "🇯🇵 Japanese": {"code": "ja", "voice": "ja-JP-NanamiNeural", "voice_name": "Nanami", "ocr": "ja"},
    "🇰🇷 Korean": {"code": "ko", "voice": "ko-KR-SunHiNeural", "voice_name": "SunHi", "ocr": "ko"},
    "🇹🇭 Thai": {"code": "th", "voice": "th-TH-PremwadeeNeural", "voice_name": "Premwadee", "ocr": "th"},
    "🇻🇳 Vietnamese": {"code": "vi", "voice": "vi-VN-HoaiMyNeural", "voice_name": "HoaiMy", "ocr": "vi"},
    "🇮🇩 Indonesian": {"code": "id", "voice": "id-ID-GadisNeural", "voice_name": "Gadis", "ocr": "id"},
    
    # Middle Eastern Languages
    "🇸🇦 Arabic": {"code": "ar", "voice": "ar-SA-ZariyahNeural", "voice_name": "Zariyah", "ocr": "ar"},
    "🇹🇷 Turkish": {"code": "tr", "voice": "tr-TR-EmelNeural", "voice_name": "Emel", "ocr": "tr"},
    "🇮🇱 Hebrew": {"code": "he", "voice": "he-IL-HilaNeural", "voice_name": "Hila", "ocr": "he"},
}


# =============================================================================
# OCR TEXT DETECTION ENGINE
# =============================================================================

class TextDetector:
    """Detect and extract text from images using OCR with auto language detection"""
    
    def __init__(self):
        self.available = OCR_AVAILABLE
        self.lang_detect_available = LANGDETECT_AVAILABLE
        self.readers = {}  # Cache readers for different languages
        
        if self.available:
            print("🔍 OCR Text Detection initialized")
        else:
            print("⚠️ OCR not available")
        
        if self.lang_detect_available:
            print("🔄 Auto Language Detection initialized")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of given text
        
        Returns:
            Language code (e.g., 'hi', 'en', 'es') or 'en' as fallback
        """
        if not self.lang_detect_available or not text or len(text.strip()) < 3:
            return "en"
        
        try:
            detected = detect(text)
            print(f"🔄 Detected language: {detected}")
            return detected
        except Exception as e:
            print(f"⚠️ Language detection error: {e}")
            return "en"
    
    def get_language_name(self, lang_code: str) -> str:
        """Convert language code to display name"""
        return LANG_CODE_TO_NAME.get(lang_code.lower(), "🇺🇸 English (US)")
    
    def get_reader(self, languages=['en']):
        """Get or create OCR reader for specified languages"""
        key = tuple(sorted(languages))
        if key not in self.readers:
            try:
                self.readers[key] = easyocr.Reader(languages, gpu=False, verbose=False)
            except Exception as e:
                print(f"⚠️ Could not create OCR reader: {e}")
                # Fallback to English only
                if 'en' not in languages:
                    return self.get_reader(['en'])
                return None
        return self.readers[key]
    
    def detect_text(self, image, language='en') -> str:
        """
        Detect text in image
        
        Args:
            image: PIL Image or numpy array
            language: OCR language code
            
        Returns:
            Detected text as string, or empty string if none found
        """
        if not self.available:
            return ""
        
        try:
            # Convert to numpy if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Get appropriate reader
            ocr_langs = ['en']  # Always include English
            if language != 'en' and language in ['hi', 'ta', 'te', 'bn', 'kn', 'ml', 'mr', 
                                                   'ch_sim', 'ja', 'ko', 'ar', 'ru', 'de', 
                                                   'fr', 'es', 'it', 'pt', 'nl', 'tr']:
                ocr_langs = [language, 'en']
            
            reader = self.get_reader(ocr_langs)
            if reader is None:
                return ""
            
            # Detect text
            results = reader.readtext(img_array, detail=0, paragraph=True)
            
            if results:
                # Join all detected text
                detected_text = ' '.join(results)
                # Clean up
                detected_text = detected_text.strip()
                if len(detected_text) > 3:  # Only return if meaningful text
                    return detected_text
            
            return ""
            
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
            return ""


# =============================================================================
# TRANSLATION ENGINE
# =============================================================================

class TranslationEngine:
    """Multi-language translation engine"""
    
    def __init__(self):
        self.available = TRANSLATOR_AVAILABLE
        self.cache = {}
        
        if self.available:
            print("🌐 Translation engine initialized")
    
    def translate(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        """Translate text to target language"""
        if not self.available or not text:
            return text
        
        if target_lang == "en" or target_lang == source_lang:
            return text
        
        cache_key = f"{source_lang}:{target_lang}:{text}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(text)
            self.cache[cache_key] = translated
            return translated
        except Exception as e:
            print(f"⚠️ Translation error: {e}")
            return text
    
    def get_language_code(self, language_name: str) -> str:
        """Get language code from display name"""
        if language_name in LANGUAGES:
            return LANGUAGES[language_name]["code"]
        return "en"


# =============================================================================
# MULTILINGUAL VOICE ENGINE
# =============================================================================

class MultilingualVoice:
    """Natural-sounding Text-to-Speech in multiple languages"""
    
    def __init__(self):
        self.enabled = EDGE_TTS_AVAILABLE
        
        if self.enabled:
            try:
                pygame.mixer.init()
                print("🎙️ Multilingual voice engine initialized")
            except Exception as e:
                print(f"⚠️ Audio init error: {e}")
                self.enabled = False
    
    def get_voice_for_language(self, language_name: str) -> str:
        """Get the appropriate voice for a language"""
        if language_name in LANGUAGES:
            return LANGUAGES[language_name]["voice"]
        return "en-US-JennyNeural"
    
    async def _generate_speech(self, text: str, voice: str, output_file: str):
        """Generate speech using edge-tts"""
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
    
    def speak(self, text: str, language: str = "🇺🇸 English (US)") -> bool:
        """Speak text in the specified language"""
        if not self.enabled or not text:
            return False
        
        try:
            voice = self.get_voice_for_language(language)
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_file = f.name
            
            asyncio.run(self._generate_speech(text, voice, temp_file))
            
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            pygame.mixer.music.unload()
            os.unlink(temp_file)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Voice error: {e}")
            return False
    
    def speak_with_voice(self, text: str, voice_id: str, voice_name: str = "") -> bool:
        """Speak text with a specific voice ID"""
        if not self.enabled or not text:
            return False
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_file = f.name
            
            asyncio.run(self._generate_speech(text, voice_id, temp_file))
            
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            pygame.mixer.music.unload()
            os.unlink(temp_file)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Voice error: {e}")
            return False


# =============================================================================
# ENHANCED CAPTIONING ENGINE WITH OCR
# =============================================================================

class EnhancedCaptioner:
    """BLIP Image Captioning with OCR text detection"""
    
    def __init__(self, use_large_model=False):
        print("\n" + "=" * 60)
        print("🌍 Starting Enhanced Multilingual Image Captioning")
        print("   with Text/Quote Detection (OCR)")
        print("=" * 60)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {self.device}")
        
        # Load BLIP model
        model_name = "Salesforce/blip-image-captioning-large" if use_large_model else "Salesforce/blip-image-captioning-base"
        print(f"📦 Loading: {model_name.split('/')[-1]}")
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            self.model_loaded = True
            print("✅ BLIP model loaded!")
            
        except Exception as e:
            print(f"❌ Model error: {e}")
            self.model_loaded = False
        
        # Initialize components
        self.translator = TranslationEngine()
        self.voice = MultilingualVoice()
        self.ocr = TextDetector()
        
        print("=" * 60 + "\n")
    
    def _create_natural_caption(self, visual_caption: str, detected_text: str) -> str:
        """
        Create a natural, comprehensive caption combining visual description and detected text
        
        Args:
            visual_caption: Caption from BLIP (visual description)
            detected_text: Text detected by OCR
            
        Returns:
            Natural combined caption
        """
        if not detected_text:
            return visual_caption
        
        # Clean up the detected text
        detected_text = detected_text.strip()
        
        # Check if text looks like a quote (has quotation marks or is short enough)
        is_quote = ('"' in detected_text or "'" in detected_text or 
                   len(detected_text.split()) <= 15)
        
        # Create natural combined caption
        if is_quote:
            # Format as quote
            if not (detected_text.startswith('"') or detected_text.startswith("'")):
                detected_text = f'"{detected_text}"'
            combined = f'{visual_caption}. The image contains the text: {detected_text}'
        else:
            # Longer text - might be a sign, document, etc.
            combined = f'{visual_caption}. Text visible in image: "{detected_text}"'
        
        return combined
    
    def generate_caption(
        self,
        image,
        language: str = "🇺🇸 English (US)",
        style: str = "Standard",
        max_length: int = 50,
        enable_voice: bool = False,
        show_english: bool = True,
        detect_text: bool = True,
        auto_detect: bool = False,
        voice_choice: str = "Jenny (US Female)"
    ) -> str:
        """Generate enhanced caption with text detection and auto language detection"""
        
        if not self.model_loaded:
            return "❌ Model not loaded."
        
        if image is None:
            return "⚠️ Please upload an image first."
        
        try:
            start_time = time.time()
            
            # Convert image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            else:
                pil_image = image
            
            # Step 1: Detect text in image using OCR
            detected_text = ""
            detected_language = None
            
            if detect_text and self.ocr.available:
                # First, detect with English to get any text
                detected_text = self.ocr.detect_text(pil_image, "en")
                
                if detected_text:
                    print(f"🔍 Detected text: {detected_text[:50]}...")
                    
                    # Auto-detect language from the text
                    if auto_detect and self.ocr.lang_detect_available:
                        detected_lang_code = self.ocr.detect_language(detected_text)
                        detected_language = self.ocr.get_language_name(detected_lang_code)
                        print(f"🔄 Auto-detected language: {detected_language}")
                        
                        # Use detected language instead of selected
                        if detected_language and detected_language != language:
                            language = detected_language
            
            # Handle auto-detect option from dropdown
            if language == AUTO_DETECT:
                if detected_language:
                    language = detected_language
                else:
                    language = "🇺🇸 English (US)"  # Fallback
            
            # Step 2: Generate visual caption with BLIP
            if style == "Detailed":
                text_prompt = "a detailed photograph showing"
                actual_max_length = min(max_length + 40, 120)
                min_length = 20
            elif style == "Brief":
                text_prompt = None
                actual_max_length = min(max_length, 25)
                min_length = 3
            else:
                text_prompt = None
                actual_max_length = max_length
                min_length = 5
            
            if text_prompt:
                inputs = self.processor(pil_image, text_prompt, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=actual_max_length,
                    min_length=min_length,
                    num_beams=4,  # Increased for better quality
                    early_stopping=True
                )
            
            visual_caption = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
            
            # Capitalize first letter
            if visual_caption and not visual_caption[0].isupper():
                visual_caption = visual_caption[0].upper() + visual_caption[1:]
            
            # Step 3: Combine visual caption with detected text
            english_caption = self._create_natural_caption(visual_caption, detected_text)
            
            # Step 4: Translate if needed
            lang_code = self.translator.get_language_code(language)
            if lang_code != "en":
                translated_caption = self.translator.translate(english_caption, lang_code)
            else:
                translated_caption = english_caption
            
            inference_time = time.time() - start_time
            
            # Step 5: Speak if enabled
            if enable_voice and translated_caption:
                # Use auto-detected language voice or manually selected voice
                if auto_detect and detected_language:
                    voice_name = LANGUAGES.get(language, {}).get("voice_name", "Jenny")
                    print(f"🎙️ Speaking in {language} (auto-detected) with voice: {voice_name}")
                    self.voice.speak(translated_caption, language)
                else:
                    # Use manually selected voice
                    voice_id = VOICE_OPTIONS.get(voice_choice, "en-US-JennyNeural")
                    print(f"🎙️ Speaking with selected voice: {voice_choice}")
                    self.voice.speak_with_voice(translated_caption, voice_id, voice_choice)
            
            # Format output
            lang_info = LANGUAGES.get(language, {})
            
            # Determine voice info for display
            if auto_detect and detected_language:
                voice_display = lang_info.get('voice_name', 'Auto')
            else:
                voice_display = voice_choice.split(" (")[0] if voice_choice else "Jenny"
            
            voice_icon = f" 🎙️ {voice_display}" if enable_voice else ""
            text_detected_icon = " 📝" if detected_text else ""
            auto_detect_icon = " 🔄" if (auto_detect and detected_language) else ""
            
            if show_english and lang_code != "en":
                result = f"**{language}:**{auto_detect_icon}\n{translated_caption}{voice_icon}{text_detected_icon}\n\n"
                result += f"**🇺🇸 English:**\n{english_caption}\n\n"
            else:
                result = f"{translated_caption}{voice_icon}{text_detected_icon}\n\n"
            
            if detected_text:
                result += f"📝 **Detected Text:** \"{detected_text}\"\n"
            
            if auto_detect and detected_language:
                result += f"🔄 **Auto-detected Language:** {detected_language}\n"
            
            if enable_voice:
                result += f"🎙️ **Voice:** {voice_display}\n"
            
            result += f"\n⏱️ Generated in {inference_time:.2f}s"
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}"
    
    def answer_question(
        self,
        image,
        question: str,
        language: str = "🇺🇸 English (US)",
        enable_voice: bool = False
    ) -> str:
        """Answer question about image"""
        
        if not self.model_loaded:
            return "❌ Model not loaded."
        
        if image is None:
            return "⚠️ Please upload an image."
        
        if not question or not question.strip():
            return "⚠️ Please enter a question."
        
        try:
            start_time = time.time()
            
            # Convert image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
            
            # Translate question to English if needed
            lang_code = self.translator.get_language_code(language)
            if lang_code != "en":
                english_question = self.translator.translate(question, "en", lang_code)
            else:
                english_question = question
            
            # Process
            inputs = self.processor(image, english_question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_length=50, num_beams=4)
            
            english_answer = self.processor.decode(output_ids[0], skip_special_tokens=True)
            
            # Translate answer
            if lang_code != "en":
                answer = self.translator.translate(english_answer, lang_code)
            else:
                answer = english_answer
            
            inference_time = time.time() - start_time
            
            # Speak
            if enable_voice and answer:
                self.voice.speak(answer, language)
            
            voice_icon = " 🎙️" if enable_voice else ""
            return f"💬 {answer}{voice_icon}\n\n⏱️ {inference_time:.2f}s"
            
        except Exception as e:
            return f"❌ Error: {str(e)}"


# =============================================================================
# GRADIO WEB INTERFACE
# =============================================================================

def create_app():
    """Create the enhanced Gradio interface"""
    
    # Initialize captioner
    app = EnhancedCaptioner(use_large_model=False)
    
    # Get all language names with auto-detect option first
    all_languages = [AUTO_DETECT] + list(LANGUAGES.keys())
    
    # Build interface
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate"
        ),
        title="Multilingual Image Captioning with OCR"
    ) as interface:
        
        # Header
        gr.Markdown("""
        # 🌍 Multilingual Image Captioning + Auto Detection
        ### AI-Powered Captions in 25+ Languages with Natural Voice 🎙️
        
        **Features:**
        - 📝 **Text Detection:** Reads quotes, signs, and text in images
        - 🔄 **Auto Language Detection:** Automatically detects language from image text
        - 🎙️ **Natural Voice:** Speaks captions in detected language
        
        ---
        """)
        
        # Main tabs
        with gr.Tabs():
            
            # Tab 1: Enhanced Captioning
            with gr.Tab("📝 Caption with Text Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        caption_image = gr.Image(
                            label="📷 Upload Image",
                            type="pil",
                            height=380,
                            sources=["upload", "clipboard", "webcam"]
                        )
                        
                        language_select = gr.Dropdown(
                            choices=all_languages,
                            value=AUTO_DETECT,
                            label="🌐 Output Language",
                            info="Select 'Auto Detect' to detect language from image text"
                        )
                        
                        with gr.Row():
                            caption_style = gr.Dropdown(
                                choices=["Standard", "Detailed", "Brief"],
                                value="Standard",
                                label="Style"
                            )
                            caption_length = gr.Slider(
                                minimum=10, maximum=100, value=50, step=5,
                                label="Max Length"
                            )
                        
                        with gr.Row():
                            enable_voice = gr.Checkbox(
                                label="🎙️ Speak Caption",
                                value=True
                            )
                            detect_text = gr.Checkbox(
                                label="📝 Detect Text/Quotes",
                                value=True,
                                info="OCR to find text in images"
                            )
                        
                        with gr.Row():
                            auto_detect = gr.Checkbox(
                                label="🔄 Auto Detect Language",
                                value=True,
                                info="Detect language from text in image"
                            )
                            show_english = gr.Checkbox(
                                label="📝 Show English",
                                value=True
                            )
                        
                        # Voice selection
                        voice_select = gr.Dropdown(
                            choices=[
                                "Jenny (US Female)", "Aria (US Female)", "Emma (US Female)",
                                "Guy (US Male)", "Sonia (UK Female)", "Ryan (UK Male)",
                                "Natasha (AU Female)", "Swara (Hindi)", "Pallavi (Tamil)",
                                "Shruti (Telugu)", "Tanishaa (Bengali)", "Xiaoxiao (Chinese)",
                                "Nanami (Japanese)", "Elvira (Spanish)", "Denise (French)"
                            ],
                            value="Jenny (US Female)",
                            label="🎙️ Voice (when not auto-detecting)",
                            info="Used when language is manually selected"
                        )
                        
                        caption_btn = gr.Button(
                            "✨ Generate Caption",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        caption_output = gr.Textbox(
                            label="Generated Caption",
                            lines=10,
                            show_copy_button=True
                        )
                        
                        gr.Markdown("### 🚀 Quick Language Select")
                        
                        with gr.Row():
                            gr.Button("🇮🇳 Hindi", size="sm").click(
                                lambda: "🇮🇳 Hindi", outputs=language_select)
                            gr.Button("🇮🇳 Tamil", size="sm").click(
                                lambda: "🇮🇳 Tamil", outputs=language_select)
                            gr.Button("🇮🇳 Telugu", size="sm").click(
                                lambda: "🇮🇳 Telugu", outputs=language_select)
                            gr.Button("🇮🇳 Bengali", size="sm").click(
                                lambda: "🇮🇳 Bengali", outputs=language_select)
                        
                        with gr.Row():
                            gr.Button("🇪🇸 Spanish", size="sm").click(
                                lambda: "🇪🇸 Spanish", outputs=language_select)
                            gr.Button("🇫🇷 French", size="sm").click(
                                lambda: "🇫🇷 French", outputs=language_select)
                            gr.Button("🇩🇪 German", size="sm").click(
                                lambda: "🇩🇪 German", outputs=language_select)
                            gr.Button("🇨🇳 Chinese", size="sm").click(
                                lambda: "🇨🇳 Chinese", outputs=language_select)
                        
                        gr.Markdown("""
                        ### ✨ Features:
                        - 📝 Detects quotes, signs, labels
                        - 🔄 Auto-detects language from text
                        - 🎙️ Speaks in detected language
                        - ✅ Combines visual + text description
                        """)
                
                caption_btn.click(
                    fn=app.generate_caption,
                    inputs=[caption_image, language_select, caption_style, 
                            caption_length, enable_voice, show_english, detect_text, auto_detect, voice_select],
                    outputs=caption_output
                )
            
            # Tab 2: Visual Q&A
            with gr.Tab("❓ Ask Questions"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vqa_image = gr.Image(
                            label="📷 Upload Image",
                            type="pil",
                            height=350,
                            sources=["upload", "clipboard", "webcam"]
                        )
                        
                        vqa_language = gr.Dropdown(
                            choices=all_languages,
                            value="🇮🇳 Hindi",
                            label="🌐 Language"
                        )
                        
                        vqa_question = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask about text in the image: 'What does the sign say?' or 'Read the quote'",
                            lines=2
                        )
                        
                        vqa_voice = gr.Checkbox(
                            label="🎙️ Speak Answer",
                            value=True
                        )
                        
                        vqa_btn = gr.Button(
                            "🔍 Get Answer",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        vqa_output = gr.Textbox(
                            label="Answer",
                            lines=4,
                            show_copy_button=True
                        )
                        
                        gr.Markdown("""
                        ### 💬 Example Questions:
                        - What text is in this image?
                        - What does the sign say?
                        - Read the quote in the image
                        - What is written on the board?
                        """)
                
                vqa_btn.click(
                    fn=app.answer_question,
                    inputs=[vqa_image, vqa_question, vqa_language, vqa_voice],
                    outputs=vqa_output
                )
            
            # Tab 3: Languages
            with gr.Tab("🌐 Languages"):
                gr.Markdown("""
                ## 🌍 Supported Languages & Voices
                
                ### 🇮🇳 Indian Regional Languages
                | Language | Voice | Text Detection |
                |----------|-------|----------------|
                | Hindi | Swara | ✅ |
                | Tamil | Pallavi | ✅ |
                | Telugu | Shruti | ✅ |
                | Bengali | Tanishaa | ✅ |
                | Kannada | Sapna | ✅ |
                | Malayalam | Sobhana | ✅ |
                | Marathi | Aarohi | ✅ |
                | Gujarati | Dhwani | ✅ |
                | Punjabi | Ojas | ✅ |
                
                ### 🌍 International Languages
                | Language | Voice | Text Detection |
                |----------|-------|----------------|
                | English (US/UK/AU) | Jenny/Sonia/Natasha | ✅ |
                | Spanish | Elvira | ✅ |
                | French | Denise | ✅ |
                | German | Katja | ✅ |
                | Chinese | Xiaoxiao | ✅ |
                | Japanese | Nanami | ✅ |
                | Korean | SunHi | ✅ |
                | Arabic | Zariyah | ✅ |
                
                ---
                **📝 Text Detection (OCR)** works for most major scripts!
                """)
            
            # Tab 4: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## 🚀 Enhanced Image Captioning
                
                This app combines **three powerful AI technologies**:
                
                ### 1. 🖼️ Visual Understanding (BLIP)
                - Understands what's IN the image
                - Describes objects, people, scenes
                - Powered by Salesforce BLIP
                
                ### 2. 📝 Text Detection (OCR)
                - Reads text, quotes, signs in images
                - Supports 20+ languages/scripts
                - Powered by EasyOCR
                
                ### 3. 🎙️ Natural Voice (Neural TTS)
                - Human-like speech output
                - 25+ language voices
                - Powered by Microsoft Edge TTS
                
                ---
                
                ### ⚡ How It Works:
                1. Upload an image
                2. AI analyzes visual content (BLIP)
                3. OCR detects any text/quotes
                4. Both are combined into natural caption
                5. Caption is translated to your language
                6. Natural voice speaks the caption
                
                ---
                *Made with ❤️ for accurate, multilingual image understanding*
                """)
    
    return interface


def main():
    """Launch the app"""
    print("\n🌍 Starting Enhanced Multilingual Image Captioning...")
    
    interface = create_app()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
