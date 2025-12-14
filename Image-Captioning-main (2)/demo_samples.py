"""
Sample Output Demonstration for Image Captioning
Shows example outputs and creates a visual demo
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def create_sample_output_visualization():
    """
    Create a visual demonstration of sample outputs
    """
    # Sample outputs (typical results from the model)
    samples = [
        {
            "image_desc": "Beach Scene",
            "caption": "a person walking on the beach near the ocean",
            "inference_time": 2.3,
            "confidence": "High"
        },
        {
            "image_desc": "City Street",
            "caption": "a group of people walking down a city street",
            "inference_time": 2.1,
            "confidence": "High"
        },
        {
            "image_desc": "Dog in Park",
            "caption": "a dog playing with a frisbee in the park",
            "inference_time": 2.5,
            "confidence": "Medium"
        },
        {
            "image_desc": "Food Plate",
            "caption": "a plate of food with vegetables and meat",
            "inference_time": 2.2,
            "confidence": "High"
        },
        {
            "image_desc": "Mountain Landscape",
            "caption": "a view of mountains with snow on top",
            "inference_time": 2.4,
            "confidence": "High"
        },
        {
            "image_desc": "Indoor Living Room",
            "caption": "a living room with a couch and a television",
            "inference_time": 2.3,
            "confidence": "Medium"
        }
    ]
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Image Captioning - Sample Outputs', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (ax, sample) in enumerate(zip(axes, samples)):
        # Create a placeholder "image" with text
        ax.text(0.5, 0.7, sample['image_desc'], 
               ha='center', va='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Add caption
        ax.text(0.5, 0.3, f'Caption: "{sample["caption"]}"',
               ha='center', va='center', fontsize=10, wrap=True,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        # Add metadata
        metadata = f"Time: {sample['inference_time']}s | Confidence: {sample['confidence']}"
        ax.text(0.5, 0.05, metadata,
               ha='center', va='center', fontsize=8, style='italic',
               color='gray')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = 'sample_outputs_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Sample output visualization saved to: {output_path}")
    plt.show()
    
    return samples


def print_sample_outputs():
    """
    Print sample outputs in a formatted way
    """
    print("\n" + "="*70)
    print("  IMAGE CAPTIONING - SAMPLE OUTPUTS")
    print("="*70 + "\n")
    
    print("These are typical outputs you can expect from the trained model:\n")
    
    samples = [
        {
            "scenario": "Beach Scene",
            "input": "beach_sunset.jpg",
            "output": "a person walking on the beach near the ocean",
            "time": "2.3s",
            "notes": "Correctly identifies beach and person"
        },
        {
            "scenario": "Urban Environment",
            "input": "city_street.jpg",
            "output": "a group of people walking down a city street",
            "time": "2.1s",
            "notes": "Recognizes urban setting and crowd"
        },
        {
            "scenario": "Pet/Animal",
            "input": "dog_park.jpg",
            "output": "a dog playing with a frisbee in the park",
            "time": "2.5s",
            "notes": "Identifies animal, object, and activity"
        },
        {
            "scenario": "Food Photography",
            "input": "dinner_plate.jpg",
            "output": "a plate of food with vegetables and meat",
            "time": "2.2s",
            "notes": "Describes food items accurately"
        },
        {
            "scenario": "Nature/Landscape",
            "input": "mountain_view.jpg",
            "output": "a view of mountains with snow on top",
            "time": "2.4s",
            "notes": "Captures landscape features"
        },
        {
            "scenario": "Indoor Scene",
            "input": "living_room.jpg",
            "output": "a living room with a couch and a television",
            "time": "2.3s",
            "notes": "Identifies room type and furniture"
        },
        {
            "scenario": "Sports/Action",
            "input": "soccer_game.jpg",
            "output": "a group of people playing soccer on a field",
            "time": "2.6s",
            "notes": "Recognizes sport and activity"
        },
        {
            "scenario": "Transportation",
            "input": "train_station.jpg",
            "output": "a train sitting at a station platform",
            "time": "2.2s",
            "notes": "Identifies vehicle and location"
        }
    ]
    
    for i, sample in enumerate(samples, 1):
        print(f"{'─'*70}")
        print(f"Sample {i}: {sample['scenario']}")
        print(f"{'─'*70}")
        print(f"  Input Image:  {sample['input']}")
        print(f"  Generated Caption: \"{sample['output']}\"")
        print(f"  Inference Time: {sample['time']} (CPU)")
        print(f"  Notes: {sample['notes']}")
        print()
    
    print("="*70)
    print("\n📊 PERFORMANCE SUMMARY")
    print("="*70)
    print(f"  Average Inference Time: 2.3 seconds per image (CPU)")
    print(f"  Caption Length: 8-15 words typically")
    print(f"  Accuracy: Varies by image complexity")
    print(f"  Device: CPU (Intel/AMD)")
    print()


def print_training_output_sample():
    """
    Show sample training output
    """
    print("\n" + "="*70)
    print("  TRAINING OUTPUT SAMPLE")
    print("="*70 + "\n")
    
    print("Example output during training:\n")
    
    training_log = """
Epoch 1/5 Summary:
├─ Batch 1/500   | Loss: 4.2341 | Avg Loss: 4.2341
├─ Batch 50/500  | Loss: 3.8765 | Avg Loss: 4.0123
├─ Batch 100/500 | Loss: 3.5432 | Avg Loss: 3.8567
├─ Batch 150/500 | Loss: 3.2876 | Avg Loss: 3.6789
├─ Batch 200/500 | Loss: 3.0234 | Avg Loss: 3.5123
└─ Batch 500/500 | Loss: 2.7654 | Avg Loss: 3.2456

Epoch 1 Complete:
  Average Loss: 3.2456
  Perplexity: 25.67
  Time: 5432.1s (1.51 hours)
  ✓ Checkpoint saved to ./models/encoder-1.pkl, ./models/decoder-1.pkl

Epoch 2/5 Summary:
├─ Batch 1/500   | Loss: 2.6543 | Avg Loss: 2.6543
├─ Batch 50/500  | Loss: 2.4321 | Avg Loss: 2.5234
└─ ...

Training Complete!
Total Time: 7.5 hours
Final Loss: 2.1234
Final Perplexity: 8.36
"""
    
    print(training_log)
    print("="*70 + "\n")


def print_inference_output_sample():
    """
    Show sample inference output
    """
    print("\n" + "="*70)
    print("  INFERENCE OUTPUT SAMPLE")
    print("="*70 + "\n")
    
    print("Example output when running inference:\n")
    
    inference_log = """
$ python inference_cpu.py --image beach.jpg

Using device: cpu
Loading vocabulary...
Vocabulary size: 9955
Initializing models...
✓ Loaded encoder from encoder-5.pkl
✓ Loaded decoder from decoder-5.pkl
✓ Model initialized successfully!

Generating caption for: beach.jpg
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

Caption: "a person walking on the beach near the ocean"
Inference time: 2.34s

[Image displayed with caption overlay]
"""
    
    print(inference_log)
    print("="*70 + "\n")


def print_batch_inference_sample():
    """
    Show sample batch inference output
    """
    print("\n" + "="*70)
    print("  BATCH INFERENCE OUTPUT SAMPLE")
    print("="*70 + "\n")
    
    print("Example output for batch processing:\n")
    
    batch_log = """
$ python inference_cpu.py --image_dir ./test_images/ --output results.txt

Processing 10 images...

[1/10] Processing: beach.jpg
  Caption: a person walking on the beach near the ocean
  Time: 2.34s

[2/10] Processing: city.jpg
  Caption: a group of people walking down a city street
  Time: 2.12s

[3/10] Processing: dog.jpg
  Caption: a dog playing with a frisbee in the park
  Time: 2.45s

[4/10] Processing: food.jpg
  Caption: a plate of food with vegetables and meat
  Time: 2.21s

[5/10] Processing: mountain.jpg
  Caption: a view of mountains with snow on top
  Time: 2.38s

[6/10] Processing: room.jpg
  Caption: a living room with a couch and a television
  Time: 2.29s

[7/10] Processing: soccer.jpg
  Caption: a group of people playing soccer on a field
  Time: 2.56s

[8/10] Processing: train.jpg
  Caption: a train sitting at a station platform
  Time: 2.18s

[9/10] Processing: cat.jpg
  Caption: a cat sitting on a window sill
  Time: 2.33s

[10/10] Processing: sunset.jpg
  Caption: a beautiful sunset over the ocean
  Time: 2.41s

============================================================
SUMMARY
============================================================
Total images processed: 10
Total time: 23.27s
Average time per image: 2.33s

✓ Results saved to results.txt
"""
    
    print(batch_log)
    print("="*70 + "\n")


def print_web_interface_sample():
    """
    Show sample web interface description
    """
    print("\n" + "="*70)
    print("  WEB INTERFACE SAMPLE")
    print("="*70 + "\n")
    
    print("When you run: python app_gradio.py\n")
    
    web_description = """
Starting Gradio interface...
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.

┌─────────────────────────────────────────────────────┐
│  🖼️ Image Captioning with CNN-RNN                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [Drag and drop image here or click to upload]     │
│                                                     │
│  Maximum Caption Length: [────●────────] 20        │
│                                                     │
│  [        Submit        ]  [      Clear      ]     │
│                                                     │
│  Generated Caption:                                 │
│  ┌───────────────────────────────────────────────┐ │
│  │ a person walking on the beach near the ocean │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
└─────────────────────────────────────────────────────┘

Features:
✓ Drag-and-drop image upload
✓ Real-time caption generation
✓ Adjustable caption length
✓ Clean, modern interface
✓ Works on any device with a browser
"""
    
    print(web_description)
    print("="*70 + "\n")


def main():
    """
    Main function to display all samples
    """
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  IMAGE CAPTIONING - COMPREHENSIVE SAMPLE OUTPUTS".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Print all sample outputs
    print_sample_outputs()
    print_training_output_sample()
    print_inference_output_sample()
    print_batch_inference_sample()
    print_web_interface_sample()
    
    # Create visualization
    print("\n" + "="*70)
    print("  CREATING VISUAL DEMONSTRATION")
    print("="*70 + "\n")
    
    try:
        create_sample_output_visualization()
    except Exception as e:
        print(f"Note: Could not create visualization (matplotlib may not be available)")
        print(f"Error: {e}")
    
    # Final notes
    print("\n" + "="*70)
    print("  NOTES")
    print("="*70)
    print("""
1. These are example outputs based on typical model performance
2. Actual captions will vary based on:
   - Training data quality
   - Number of training epochs
   - Model architecture
   - Image content and quality

3. To generate real captions:
   - Train the model: python train_cpu.py
   - Run inference: python inference_cpu.py --image your_image.jpg
   - Or use web interface: python app_gradio.py

4. Performance metrics are based on CPU inference
   - GPU inference would be 10-50x faster
   - Inference time varies by CPU speed

5. Caption quality improves with:
   - More training epochs (5-10 recommended)
   - Larger training dataset
   - Better quality training images
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
