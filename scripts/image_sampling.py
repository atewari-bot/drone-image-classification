import zipfile
import os
import random
import shutil

# Paths
dataset_path = "./drone-dataset"
sample_dir = "drone-dataset-sample"


def sample_dataset(dataset_type, size=400):
    # Define folders
    image_dir = os.path.join(dataset_path, dataset_type, "images")
    label_dir = os.path.join(dataset_path, dataset_type, "labels")
    sample_img_dir = os.path.join(sample_dir, dataset_type, "images")
    sample_lbl_dir = os.path.join(sample_dir, dataset_type, "labels")
    os.makedirs(sample_img_dir, exist_ok=True)
    os.makedirs(sample_lbl_dir, exist_ok=True)

    # Sample 2000 files
    all_images = os.listdir(image_dir)
    sample_files = random.sample(all_images, size)

    for img_file in sample_files:
        base = os.path.splitext(img_file)[0]
        lbl_file = base + ".txt"

        shutil.copy(os.path.join(image_dir, img_file), os.path.join(sample_img_dir, img_file))
        shutil.copy(os.path.join(label_dir, lbl_file), os.path.join(sample_lbl_dir, lbl_file))

sample_dataset("train", 2000)
sample_dataset("test")
sample_dataset("valid")