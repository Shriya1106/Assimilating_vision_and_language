"""
CPU-Optimized Image Captioning Inference Script
Generates captions for images using trained model on CPU
"""

import os
import pickle
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
import time

from model import EncoderCNN, DecoderRNN
from vocabulary import Vocabulary


class ImageCaptioner:
    def __init__(self, encoder_path, decoder_path, vocab_path, embed_size=256, hidden_size=512, device='cpu'):
        """
        Initialize Image Captioner
        
        Args:
            encoder_path: Path to encoder model weights
            decoder_path: Path to decoder model weights
            vocab_path: Path to vocabulary pickle file
            embed_size: Embedding size
            hidden_size: Hidden size for LSTM
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Load vocabulary
        print("Loading vocabulary...")
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        vocab_size = len(self.vocab)
        print(f"Vocabulary size: {vocab_size}")
        
        # Initialize models
        print("Initializing models...")
        self.encoder = EncoderCNN(embed_size, device=device).to(self.device)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, device=device).to(self.device)
        
        # Load model weights
        print("Loading model weights...")
        if os.path.exists(encoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print(f"✓ Loaded encoder from {encoder_path}")
        else:
            print(f"⚠ Warning: Encoder weights not found at {encoder_path}")
            print("  Using randomly initialized weights (for testing only)")
        
        if os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(f"✓ Loaded decoder from {decoder_path}")
        else:
            print(f"⚠ Warning: Decoder weights not found at {decoder_path}")
            print("  Using randomly initialized weights (for testing only)")
        
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
    
    def load_image(self, image_path):
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor and original image
        """
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Transform image
        image_tensor = self.transform(image).unsqueeze(0)
        
        return image_tensor.to(self.device), original_image
    
    def generate_caption(self, image_path, max_length=20, show_image=True):
        """
        Generate caption for an image
        
        Args:
            image_path: Path to image file
            max_length: Maximum length of generated caption
            show_image: Whether to display the image with caption
            
        Returns:
            Generated caption as string
        """
        # Load image
        image_tensor, original_image = self.load_image(image_path)
        
        # Generate caption
        start_time = time.time()
        
        with torch.no_grad():
            # Get image features from encoder
            features = self.encoder(image_tensor)
            
            # Generate caption using decoder
            caption_ids = self.decoder.sample(features.unsqueeze(1), max_len=max_length)
        
        inference_time = time.time() - start_time
        
        # Convert word ids to words
        caption_words = []
        for word_id in caption_ids:
            word = self.vocab.idx2word[word_id]
            if word == '<end>':
                break
            if word != '<start>':
                caption_words.append(word)
        
        caption = ' '.join(caption_words)
        
        # Display image with caption
        if show_image:
            plt.figure(figsize=(10, 6))
            plt.imshow(original_image)
            plt.axis('off')
            plt.title(f"Caption: {caption}\nInference time: {inference_time:.2f}s", 
                     fontsize=12, wrap=True)
            plt.tight_layout()
            plt.show()
        
        return caption, inference_time
    
    def batch_inference(self, image_dir, output_file=None, max_images=None):
        """
        Generate captions for multiple images in a directory
        
        Args:
            image_dir: Directory containing images
            output_file: Optional file to save captions
            max_images: Maximum number of images to process
        """
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(image_dir) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"\nProcessing {len(image_files)} images...")
        
        results = []
        total_time = 0
        
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, image_file)
            print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
            
            try:
                caption, inference_time = self.generate_caption(image_path, show_image=False)
                total_time += inference_time
                
                results.append({
                    'filename': image_file,
                    'caption': caption,
                    'time': inference_time
                })
                
                print(f"  Caption: {caption}")
                print(f"  Time: {inference_time:.2f}s")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total images processed: {len(results)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per image: {total_time/len(results):.2f}s")
        
        # Save results to file
        if output_file:
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(f"{result['filename']}: {result['caption']}\n")
            print(f"\n✓ Results saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Image Captioning Inference (CPU-Optimized)')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Directory containing multiple images')
    parser.add_argument('--encoder', type=str, default='encoder-5.pkl', 
                       help='Path to encoder weights')
    parser.add_argument('--decoder', type=str, default='decoder-5.pkl',
                       help='Path to decoder weights')
    parser.add_argument('--vocab', type=str, default='vocab.pkl',
                       help='Path to vocabulary file')
    parser.add_argument('--embed_size', type=int, default=256,
                       help='Embedding size')
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='Hidden size for LSTM')
    parser.add_argument('--max_length', type=int, default=20,
                       help='Maximum caption length')
    parser.add_argument('--output', type=str, help='Output file for batch results')
    parser.add_argument('--max_images', type=int, help='Maximum number of images to process')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Initialize captioner
    captioner = ImageCaptioner(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        vocab_path=args.vocab,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        device=args.device
    )
    
    # Single image inference
    if args.image:
        print(f"\nGenerating caption for: {args.image}")
        caption, inference_time = captioner.generate_caption(args.image, 
                                                            max_length=args.max_length,
                                                            show_image=True)
        print(f"\nCaption: {caption}")
        print(f"Inference time: {inference_time:.2f}s")
    
    # Batch inference
    elif args.image_dir:
        captioner.batch_inference(args.image_dir, 
                                 output_file=args.output,
                                 max_images=args.max_images)
    
    else:
        print("Please provide either --image or --image_dir argument")
        print("\nExamples:")
        print("  Single image: python inference_cpu.py --image path/to/image.jpg")
        print("  Batch: python inference_cpu.py --image_dir path/to/images/ --output results.txt")


if __name__ == "__main__":
    main()
