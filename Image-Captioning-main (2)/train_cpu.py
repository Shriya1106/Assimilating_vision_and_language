"""
CPU-Optimized Training Script for Image Captioning
Trains CNN-RNN model on COCO dataset using CPU with optimizations
"""

import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm
import time
import math

from model import EncoderCNN, DecoderRNN
from coco_dataset import CoCoDataset
from data_loader import get_loader


def train_epoch(encoder, decoder, data_loader, criterion, optimizer, device, epoch, total_epochs):
    """
    Train for one epoch
    """
    encoder.train()
    decoder.train()
    
    total_loss = 0
    num_batches = len(data_loader)
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}/{total_epochs}")
    
    for i, (images, captions) in enumerate(progress_bar):
        # Move to device
        images = images.to(device)
        captions = captions.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Calculate loss (ignore first <start> token)
        loss = criterion(outputs.view(-1, outputs.size(2)), captions.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })
    
    return total_loss / num_batches


def save_checkpoint(encoder, decoder, optimizer, epoch, loss, save_dir):
    """
    Save model checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    
    encoder_path = os.path.join(save_dir, f'encoder-{epoch}.pkl')
    decoder_path = os.path.join(save_dir, f'decoder-{epoch}.pkl')
    
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    
    # Save training state
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'optimizer_state': optimizer.state_dict()
    }
    checkpoint_path = os.path.join(save_dir, f'checkpoint-{epoch}.pkl')
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✓ Saved checkpoint to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Image Captioning Model (CPU-Optimized)')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='Embedding size')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    
    # Data parameters
    parser.add_argument('--vocab_threshold', type=int, default=5, 
                       help='Minimum word count threshold')
    parser.add_argument('--vocab_file', type=str, default='vocab.pkl',
                       help='Path to vocabulary file')
    parser.add_argument('--train_images', type=str, default='./coco_data/train2017',
                       help='Path to training images')
    parser.add_argument('--train_annotations', type=str, 
                       default='./coco_data/annotations/captions_train2017.json',
                       help='Path to training annotations')
    
    # Checkpoint parameters
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save models')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to train on')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Check if data exists
    if not os.path.exists(args.train_images):
        print(f"⚠ Warning: Training images not found at {args.train_images}")
        print("Please run: python download_coco.py")
        return
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create data loader
    print("Loading dataset...")
    try:
        data_loader = get_loader(
            transform=transform,
            mode='train',
            batch_size=args.batch_size,
            vocab_threshold=args.vocab_threshold,
            vocab_file=args.vocab_file,
            start_word='<start>',
            end_word='<end>',
            unk_word='<unk>',
            vocab_from_file=os.path.exists(args.vocab_file),
            num_workers=args.num_workers,
            cocoapi_loc=args.train_images,
            annotations_file=args.train_annotations
        )
        print(f"✓ Dataset loaded: {len(data_loader.dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure COCO dataset is downloaded and paths are correct")
        return
    
    # Load vocabulary
    with open(args.vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize models
    print("Initializing models...")
    encoder = EncoderCNN(args.embed_size, device=args.device).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, vocab_size, 
                        args.num_layers, device=args.device).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Total epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    training_start = time.time()
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start = time.time()
        
        # Train one epoch
        avg_loss = train_epoch(encoder, decoder, data_loader, criterion, 
                              optimizer, device, epoch, args.num_epochs)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.num_epochs} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Perplexity: {math.exp(avg_loss):.4f}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(encoder, decoder, optimizer, epoch, avg_loss, args.save_dir)
    
    total_time = time.time() - training_start
    
    # Training complete
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Models saved to: {args.save_dir}")
    print("\nTo run inference:")
    print(f"  python inference_cpu.py --image path/to/image.jpg \\")
    print(f"    --encoder {args.save_dir}/encoder-{args.num_epochs}.pkl \\")
    print(f"    --decoder {args.save_dir}/decoder-{args.num_epochs}.pkl")


if __name__ == "__main__":
    main()
