# Albedo and Shading Generator for Full Images
# This script processes full images to generate albedo and shading maps using a pre-trained model.
# It saves the results in a specified output directory and optionally displays a preview of the results.
from chrislib.data_util import load_image
from intrinsic.pipeline import load_models, run_pipeline
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def process_full_image(input_path, output_dir="output_full", model_version="v2", show_preview=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare image
    full_img = load_image(input_path)
    full_img_uint8 = (full_img * 255).astype(np.uint8)
    full_img_float = full_img.astype(np.float32)

    # Load model
    models = load_models(model_version)

    # Run model on the whole image
    res = run_pipeline(models, full_img_float)

    alb = (res['hr_alb'] * 255).astype(np.uint8)
    shd = (res['hr_shd'] * 255).astype(np.uint8)

    # Save albedo and shading
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"albedo_{base_name}.png"), cv2.cvtColor(alb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"shading_{base_name}.png"), cv2.cvtColor(shd, cv2.COLOR_RGB2BGR))

    print(f"[INFO] Saved albedo and shading for {base_name}.")

    # Optional: Show preview
    if show_preview:
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1); plt.imshow(full_img_uint8); plt.title("Original Image"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(alb); plt.title("Albedo"); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(shd); plt.title("Shading"); plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    input_folder = "Pitch_Analysis\\Lords Images"  # Change to your folder path
    output_folder = "output_full"
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]

    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        process_full_image(img_path, output_dir=output_folder, show_preview=False)