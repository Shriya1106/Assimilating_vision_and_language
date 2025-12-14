"""
Voice Feature Demo
Demonstrates the text-to-speech functionality
"""

import pyttsx3
import time


def demo_voice_feature():
    """
    Demonstrate the voice feature with examples
    """
    print("\n" + "="*70)
    print("  🔊 TEXT-TO-SPEECH VOICE FEATURE DEMO")
    print("="*70 + "\n")
    
    # Initialize TTS engine
    try:
        engine = pyttsx3.init()
        print("✓ Text-to-speech engine initialized successfully!\n")
    except Exception as e:
        print(f"✗ Error: Could not initialize text-to-speech: {e}")
        print("\nPlease install pyttsx3:")
        print("  pip install pyttsx3")
        return
    
    # Get voice properties
    rate = engine.getProperty('rate')
    volume = engine.getProperty('volume')
    voices = engine.getProperty('voices')
    
    print("Current Voice Settings:")
    print(f"  Speech Rate: {rate} words per minute")
    print(f"  Volume: {volume * 100}%")
    print(f"  Available Voices: {len(voices)}")
    print()
    
    # List available voices
    print("Available Voices:")
    for i, voice in enumerate(voices[:5], 1):  # Show first 5 voices
        print(f"  {i}. {voice.name}")
    if len(voices) > 5:
        print(f"  ... and {len(voices) - 5} more")
    print()
    
    # Demo captions
    demo_captions = [
        "a person walking on the beach near the ocean",
        "a group of people playing soccer on a field",
        "a dog sitting on a couch in a living room",
        "a plate of food with vegetables and meat",
        "a beautiful sunset over the mountains"
    ]
    
    print("="*70)
    print("  DEMO: Sample Image Captions with Voice")
    print("="*70 + "\n")
    
    for i, caption in enumerate(demo_captions, 1):
        print(f"[{i}/{len(demo_captions)}] Caption: \"{caption}\"")
        print(f"         🔊 Speaking...")
        
        try:
            engine.say(caption)
            engine.runAndWait()
            print(f"         ✓ Done")
        except Exception as e:
            print(f"         ✗ Error: {e}")
        
        print()
        time.sleep(0.5)
    
    # Demo different speech rates
    print("="*70)
    print("  DEMO: Different Speech Rates")
    print("="*70 + "\n")
    
    test_caption = "This is a test of different speech rates"
    rates = [100, 150, 200]
    
    for rate in rates:
        print(f"Speech Rate: {rate} words per minute")
        print(f"  🔊 Speaking: \"{test_caption}\"")
        
        engine.setProperty('rate', rate)
        engine.say(test_caption)
        engine.runAndWait()
        
        print(f"  ✓ Done")
        print()
        time.sleep(0.5)
    
    # Demo different volumes
    print("="*70)
    print("  DEMO: Different Volume Levels")
    print("="*70 + "\n")
    
    test_caption = "This is a test of different volume levels"
    volumes = [0.3, 0.6, 1.0]
    
    # Reset rate to normal
    engine.setProperty('rate', 150)
    
    for vol in volumes:
        print(f"Volume: {int(vol * 100)}%")
        print(f"  🔊 Speaking: \"{test_caption}\"")
        
        engine.setProperty('volume', vol)
        engine.say(test_caption)
        engine.runAndWait()
        
        print(f"  ✓ Done")
        print()
        time.sleep(0.5)
    
    # Final message
    print("="*70)
    print("  DEMO COMPLETE")
    print("="*70 + "\n")
    
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    
    final_message = "Voice feature demo complete! You can now use text to speech with image captioning."
    print(f"🔊 {final_message}")
    engine.say(final_message)
    engine.runAndWait()
    
    print("\n" + "="*70)
    print("  HOW TO USE")
    print("="*70)
    print("""
1. Command Line:
   python inference_with_voice.py --image your_image.jpg

2. Web Interface:
   python app_gradio.py
   Then check "Enable Voice Output" checkbox

3. Adjust Settings:
   python inference_with_voice.py --image photo.jpg --voice-rate 150 --voice-volume 0.8

4. Disable Voice:
   python inference_with_voice.py --image photo.jpg --no-voice

For more information, see VOICE_FEATURE_GUIDE.md
""")
    print("="*70 + "\n")


def quick_test():
    """
    Quick voice test
    """
    print("\n🔊 Quick Voice Test\n")
    
    try:
        engine = pyttsx3.init()
        test_message = "Hello! Voice output is working correctly."
        print(f"Speaking: \"{test_message}\"")
        engine.say(test_message)
        engine.runAndWait()
        print("✓ Voice test successful!\n")
    except Exception as e:
        print(f"✗ Voice test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Install pyttsx3: pip install pyttsx3")
        print("  2. Check audio output device")
        print("  3. Verify volume is not muted\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        demo_voice_feature()
