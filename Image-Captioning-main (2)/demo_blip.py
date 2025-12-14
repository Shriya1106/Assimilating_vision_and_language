"""
🎬 BLIP Image Captioning Demo
Test the BLIP captioner with sample images from the web
"""

import os
import sys

def main():
    print("\n" + "=" * 60)
    print("🎬 BLIP Image Captioning Demo")
    print("=" * 60)
    print("\nThis demo will test BLIP on various images from the web.\n")
    
    # Check if transformers is installed
    try:
        from transformers import BlipProcessor
        print("✅ Transformers library found")
    except ImportError:
        print("❌ Transformers not installed!")
        print("\n📦 Installing required packages...")
        os.system(f"{sys.executable} -m pip install transformers accelerate")
        print("\n✅ Packages installed. Please run this script again.")
        return
    
    # Import our captioner
    from blip_captioner import BLIPCaptioner
    
    # Initialize (this downloads the model on first run)
    print("\n📥 Loading BLIP model (first run downloads ~2GB)...")
    captioner = BLIPCaptioner(
        model_name="Salesforce/blip-image-captioning-large",
        enable_voice=True
    )
    
    # Test images from the web (public domain / sample images)
    test_images = [
        {
            "url": "https://images.unsplash.com/photo-1529778873920-4da4926a72c2?w=640",
            "description": "A cat photo"
        },
        {
            "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640",
            "description": "Mountain landscape"
        },
        {
            "url": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=640",
            "description": "Food photography"
        },
        {
            "url": "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=640",
            "description": "Technology/Robot"
        },
        {
            "url": "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?w=640",
            "description": "Cat with glasses"
        }
    ]
    
    print("\n" + "=" * 60)
    print("🖼️  Testing on Sample Images")
    print("=" * 60)
    
    for i, img_info in enumerate(test_images, 1):
        print(f"\n{'─' * 50}")
        print(f"📷 Image {i}: {img_info['description']}")
        print(f"🔗 URL: {img_info['url'][:50]}...")
        print("─" * 50)
        
        try:
            # Generate different caption styles
            print("\n📝 Standard caption:")
            caption, time_taken = captioner.generate_caption(
                img_info['url'],
                caption_style="standard",
                speak_caption=False  # Don't speak during batch demo
            )
            print(f"   \"{caption}\"")
            print(f"   ⏱️  {time_taken:.2f}s")
            
            print("\n📝 Detailed caption:")
            caption, time_taken = captioner.generate_caption(
                img_info['url'],
                caption_style="detailed",
                speak_caption=False
            )
            print(f"   \"{caption}\"")
            print(f"   ⏱️  {time_taken:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test VQA
    print("\n" + "=" * 60)
    print("❓ Testing Visual Question Answering")
    print("=" * 60)
    
    test_url = "https://images.unsplash.com/photo-1529778873920-4da4926a72c2?w=640"
    questions = [
        "What animal is in this image?",
        "What color is the cat?",
        "Is the cat sleeping?"
    ]
    
    print(f"\n📷 Image: Cat photo")
    for q in questions:
        try:
            answer, time_taken = captioner.answer_question(
                test_url, q, speak_answer=False
            )
            print(f"\n❓ Q: {q}")
            print(f"💬 A: {answer} ({time_taken:.2f}s)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test voice
    print("\n" + "=" * 60)
    print("🔊 Testing Voice Output")
    print("=" * 60)
    
    print("\nGenerating caption with voice...")
    caption, _ = captioner.generate_caption(
        test_images[0]['url'],
        speak_caption=True
    )
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print("""
Next steps:
1. Run the web app:     python app_blip.py
2. Caption your images: python blip_captioner.py --image your_photo.jpg
3. With voice:          python blip_captioner.py --image photo.jpg
4. Ask questions:       python blip_captioner.py --image photo.jpg --question "What is this?"
    """)


if __name__ == "__main__":
    main()

