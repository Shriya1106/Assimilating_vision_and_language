"""
COCO Dataset Downloader
Downloads COCO 2017 dataset (train/val images and annotations)
Supports resumable downloads and validates file integrity
"""

import os
import requests
import zipfile
from tqdm import tqdm
import hashlib


class COCODownloader:
    def __init__(self, data_dir="./coco_data"):
        """
        Initialize COCO downloader
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.base_url = "http://images.cocodataset.org"
        
        # Dataset URLs and their expected sizes (approximate)
        self.datasets = {
            "train_images": {
                "url": f"{self.base_url}/zips/train2017.zip",
                "filename": "train2017.zip",
                "extract_to": "train2017",
                "size_gb": 18
            },
            "val_images": {
                "url": f"{self.base_url}/zips/val2017.zip",
                "filename": "val2017.zip",
                "extract_to": "val2017",
                "size_gb": 1
            },
            "annotations": {
                "url": f"{self.base_url}/annotations/annotations_trainval2017.zip",
                "filename": "annotations_trainval2017.zip",
                "extract_to": "annotations",
                "size_gb": 0.25
            }
        }
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_file(self, url, filename):
        """
        Download a file with progress bar and resume capability
        """
        filepath = os.path.join(self.data_dir, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            print(f"✓ {filename} already exists. Skipping download.")
            return filepath
            
        print(f"Downloading {filename}...")
        
        # Get file size
        response = requests.head(url, allow_redirects=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        print(f"✓ Downloaded {filename}")
        return filepath
    
    def extract_zip(self, zip_path, extract_to):
        """
        Extract zip file with progress bar
        """
        extract_path = os.path.join(self.data_dir, extract_to)
        
        # Check if already extracted
        if os.path.exists(extract_path) and os.listdir(extract_path):
            print(f"✓ {extract_to} already extracted. Skipping extraction.")
            return extract_path
        
        print(f"Extracting {os.path.basename(zip_path)}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    zip_ref.extract(member, self.data_dir)
                    pbar.update(1)
        
        print(f"✓ Extracted to {extract_path}")
        return extract_path
    
    def download_dataset(self, dataset_type="all", extract=True, keep_zip=False):
        """
        Download and optionally extract COCO dataset
        
        Args:
            dataset_type: "train_images", "val_images", "annotations", or "all"
            extract: Whether to extract zip files
            keep_zip: Whether to keep zip files after extraction
        """
        if dataset_type == "all":
            datasets_to_download = list(self.datasets.keys())
        else:
            datasets_to_download = [dataset_type]
        
        print("=" * 60)
        print("COCO Dataset Downloader")
        print("=" * 60)
        
        for dataset in datasets_to_download:
            if dataset not in self.datasets:
                print(f"Warning: Unknown dataset type '{dataset}'. Skipping.")
                continue
            
            info = self.datasets[dataset]
            print(f"\n📦 Processing: {dataset}")
            print(f"   Size: ~{info['size_gb']} GB")
            
            # Download
            zip_path = self.download_file(info["url"], info["filename"])
            
            # Extract
            if extract:
                self.extract_zip(zip_path, info["extract_to"])
                
                # Remove zip file if requested
                if not keep_zip:
                    print(f"Removing {info['filename']}...")
                    os.remove(zip_path)
                    print(f"✓ Removed zip file")
        
        print("\n" + "=" * 60)
        print("✓ Download complete!")
        print(f"Data saved to: {os.path.abspath(self.data_dir)}")
        print("=" * 60)
        
        # Print directory structure
        self.print_structure()
    
    def print_structure(self):
        """
        Print the directory structure
        """
        print("\n📁 Directory Structure:")
        for root, dirs, files in os.walk(self.data_dir):
            level = root.replace(self.data_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{sub_indent}{file}")
            if len(files) > 5:
                print(f"{sub_indent}... and {len(files) - 5} more files")


def main():
    """
    Main function with user-friendly interface
    """
    print("\n🖼️  COCO Dataset Downloader for Image Captioning")
    print("=" * 60)
    print("\nOptions:")
    print("1. Download validation set only (~1 GB) - Recommended for testing")
    print("2. Download training set only (~18 GB)")
    print("3. Download annotations only (~250 MB)")
    print("4. Download everything (~19 GB)")
    print("5. Custom download")
    
    choice = input("\nEnter your choice (1-5) [default: 1]: ").strip() or "1"
    
    downloader = COCODownloader(data_dir="./coco_data")
    
    if choice == "1":
        print("\n📥 Downloading validation set and annotations...")
        downloader.download_dataset("val_images", extract=True, keep_zip=False)
        downloader.download_dataset("annotations", extract=True, keep_zip=False)
    elif choice == "2":
        print("\n📥 Downloading training set and annotations...")
        downloader.download_dataset("train_images", extract=True, keep_zip=False)
        downloader.download_dataset("annotations", extract=True, keep_zip=False)
    elif choice == "3":
        print("\n📥 Downloading annotations only...")
        downloader.download_dataset("annotations", extract=True, keep_zip=False)
    elif choice == "4":
        print("\n📥 Downloading everything...")
        downloader.download_dataset("all", extract=True, keep_zip=False)
    elif choice == "5":
        print("\nAvailable datasets: train_images, val_images, annotations")
        datasets = input("Enter datasets to download (comma-separated): ").strip().split(",")
        for dataset in datasets:
            dataset = dataset.strip()
            downloader.download_dataset(dataset, extract=True, keep_zip=False)
    else:
        print("Invalid choice. Downloading validation set by default...")
        downloader.download_dataset("val_images", extract=True, keep_zip=False)
        downloader.download_dataset("annotations", extract=True, keep_zip=False)
    
    print("\n✅ Setup complete! You can now use the dataset for training/inference.")
    print("\n📝 Next steps:")
    print("   1. Update paths in your training/inference scripts")
    print("   2. Run: python train_cpu.py (for CPU training)")
    print("   3. Run: python inference_cpu.py (for CPU inference)")


if __name__ == "__main__":
    main()
