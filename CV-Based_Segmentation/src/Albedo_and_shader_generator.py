from chrislib.data_util import load_image
from intrinsic.pipeline import load_models, run_pipeline
import numpy as np
import cv2
import os
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

def preprocess_image(img, reference_img):
    # Match histogram to reference image
    matched = match_histograms(img, reference_img, channel_axis=-1).astype(np.float32)

    # CLAHE on the V channel of HSV
    matched_uint8 = (matched * 255).astype(np.uint8)
    hsv = cv2.cvtColor(matched_uint8, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv_clahe = cv2.merge((h, s, v))
    enhanced = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2RGB)

    return enhanced.astype(np.float32) / 255.0

def normalize_albedo_to_reference(albedo, reference_albedo):
    normalized = albedo.astype(np.float32).copy()
    for c in range(3):  # channel-wise normalization
        mean_src = np.mean(albedo[:, :, c])
        std_src = np.std(albedo[:, :, c])
        mean_ref = np.mean(reference_albedo[:, :, c])
        std_ref = np.std(reference_albedo[:, :, c])
        if std_src > 0:
            normalized[:, :, c] = ((albedo[:, :, c] - mean_src) / std_src) * std_ref + mean_ref
    return np.clip(normalized, 0, 255).astype(np.uint8)

def process_full_image(input_path, albedo_output_dir, shading_output_dir, models, reference_img, reference_albedo, show_preview=False):
    os.makedirs(albedo_output_dir, exist_ok=True)
    os.makedirs(shading_output_dir, exist_ok=True)

    full_img = load_image(input_path)
    enhanced_img = preprocess_image(full_img, reference_img)
    enhanced_uint8 = (enhanced_img * 255).astype(np.uint8)

    # Run decomposition
    result = run_pipeline(models, enhanced_img)
    raw_albedo = (result['hr_alb'] * 255).astype(np.uint8)
    shading = (result['hr_shd'] * 255).astype(np.uint8)

    # Normalize albedo to reference albedo
    normalized_albedo = normalize_albedo_to_reference(raw_albedo, reference_albedo)

    # Save with consistent naming
    fname = os.path.basename(input_path)
    albedo_save_path = os.path.join(albedo_output_dir, fname)
    shading_save_path = os.path.join(shading_output_dir, fname)

    cv2.imwrite(albedo_save_path, cv2.cvtColor(normalized_albedo, cv2.COLOR_RGB2BGR))
    cv2.imwrite(shading_save_path, cv2.cvtColor(shading, cv2.COLOR_RGB2BGR))

    print(f"[SAVED] Albedo:  {albedo_save_path}")
    print(f"[SAVED] Shading: {shading_save_path}")

    if show_preview:
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1); plt.imshow(enhanced_uint8); plt.title("Preprocessed Image"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(normalized_albedo); plt.title("Albedo (Normalized)"); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(shading); plt.title("Shading"); plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 🔧 INPUTS
    input_folder = "D:\\Quidich\\Pitch_Analysis\\Lords Images"
    albedo_output_dir = "D:\\Quidich\\output_full_albedo"
    shading_output_dir = "D:\\Quidich\\output_full_shader"

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]

    # Load model once
    models = load_models("v2")

    # Load reference image for preprocessing
    reference_path = os.path.join(input_folder, "DAY 1_1.png")
    if not os.path.exists(reference_path):
        raise FileNotFoundError("Reference image 'DAY 1_1.png' not found in input folder.")
    reference_img = load_image(reference_path)

    # Decompose DAY 1_1 once to get reference albedo for normalization
    print("[INFO] Generating reference albedo from DAY 1_1...")
    enhanced_day1 = preprocess_image(reference_img, reference_img)
    result_ref = run_pipeline(models, enhanced_day1)
    reference_albedo = (result_ref['hr_alb'] * 255).astype(np.uint8)

    for fname in image_files:
        img_path = os.path.join(input_folder, fname)
        process_full_image(
            input_path=img_path,
            albedo_output_dir=albedo_output_dir,
            shading_output_dir=shading_output_dir,
            models=models,
            reference_img=reference_img,
            reference_albedo=reference_albedo,
            show_preview=False
        )