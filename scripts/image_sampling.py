import os
import random
import shutil
from pathlib import Path

# # Paths
dataset_path = "./drone-dataset"
sample_dir = "drone-dataset-sample"

def sample_dataset(dataset_type, size=400):
    """
    Randomly sample files from a dataset, ensuring images and labels are properly paired.
    
    Args:
        dataset_type (str): Type of dataset ('train', 'test', 'valid')
        size (int): Number of samples to select
    """
    # Define folders
    image_dir = os.path.join(dataset_path, dataset_type, "images")
    label_dir = os.path.join(dataset_path, dataset_type, "labels")
    sample_img_dir = os.path.join(sample_dir, dataset_type, "images")
    sample_lbl_dir = os.path.join(sample_dir, dataset_type, "labels")
    
    # Create sample directories
    os.makedirs(sample_img_dir, exist_ok=True)
    os.makedirs(sample_lbl_dir, exist_ok=True)
    
    # Validate source directories exist
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    
    # Get all image files
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
    if len(all_images) == 0:
        raise ValueError(f"No image files found in {image_dir}")
    
    # Filter images that have corresponding label files
    valid_pairs = []
    missing_labels = []
    
    for img_file in all_images:
        base = os.path.splitext(img_file)[0]
        lbl_file = base + ".txt"
        
        if os.path.exists(os.path.join(label_dir, lbl_file)):
            valid_pairs.append(img_file)
        else:
            missing_labels.append(img_file)
    
    # Report missing labels
    if missing_labels:
        print(f"Warning: Found {len(missing_labels)} images without corresponding labels in {dataset_type}")
        print(f"First few missing labels: {missing_labels[:5]}")
    
    # Check if we have enough valid pairs
    if len(valid_pairs) < size:
        print(f"Warning: Only {len(valid_pairs)} valid pairs available, but {size} requested for {dataset_type}")
        size = len(valid_pairs)
    
    # Randomly sample from valid pairs
    # random.seed(42)  # For reproducible results - remove or change seed as needed
    random.shuffle(valid_pairs)
    sample_files = random.sample(valid_pairs, size)
    
    # Copy sampled files
    copied_count = 0
    failed_copies = []
    
    for img_file in sample_files:
        try:
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            
            # Copy image
            src_img = os.path.join(image_dir, img_file)
            dst_img = os.path.join(sample_img_dir, img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            src_lbl = os.path.join(label_dir, lbl_file)
            dst_lbl = os.path.join(sample_lbl_dir, lbl_file)
            shutil.copy2(src_lbl, dst_lbl)
            
            copied_count += 1
            
        except Exception as e:
            failed_copies.append((img_file, str(e)))
    
    # Report results
    print(f"Successfully sampled {copied_count} pairs for {dataset_type}")
    if failed_copies:
        print(f"Failed to copy {len(failed_copies)} files:")
        for file, error in failed_copies[:3]:  # Show first 3 errors
            print(f"  {file}: {error}")
    
    return copied_count

def cleanup_sample_directory():
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir, exist_ok=True)

    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir, exist_ok=True)

cleanup_sample_directory()
sample_dataset("train", 2100)
sample_dataset("test", 400)
sample_dataset("valid", 400)