from chrislib.data_util import load_image
from intrinsic.pipeline import load_models, run_pipeline
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# --- Strip splitter --- #
def split_image(img, strip_w=512, overlap=64):
    h, w, _ = img.shape
    stride = strip_w - overlap
    strips, coords = [], []
    for x in range(0, w, stride):
        x1 = min(x + strip_w, w)
        x0 = max(0, x1 - strip_w)
        strips.append(img[:, x0:x1])
        coords.append((x0, x1))
    return strips, coords

# --- Main logic --- #
def process_stripwise_local(input_path, output_dir="output_strips", strip_w=512, overlap=64, model_version="v2"):
    os.makedirs(output_dir, exist_ok=True)
    
    full_img = load_image(input_path)
    full_img = (full_img * 255).astype(np.uint8)

    strips, coords = split_image(full_img, strip_w, overlap)
    models = load_models(model_version)

    for i, (st, (x0, x1)) in enumerate(zip(strips, coords)):
        st_float = st.astype(np.float32) / 255.0
        res = run_pipeline(models, st_float)

        alb = (res['hr_alb'] * 255).astype(np.uint8)
        shd = (res['hr_shd'] * 255).astype(np.uint8)

        # Save albedo and shading of each strip
        cv2.imwrite(os.path.join(output_dir, f"albedo_strip_{i:02d}.png"),
                    cv2.cvtColor(alb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"shading_strip_{i:02d}.png"),
                    cv2.cvtColor(shd, cv2.COLOR_RGB2BGR))

        print(f"[INFO] Saved strip {i+1}/{len(strips)} at x={x0}-{x1}")

        # Optional: Show preview
        if i == 0:
            plt.figure(figsize=(15, 4))
            plt.subplot(1, 3, 1); plt.imshow(st); plt.title("Original Strip"); plt.axis('off')
            plt.subplot(1, 3, 2); plt.imshow(alb); plt.title("Albedo"); plt.axis('off')
            plt.subplot(1, 3, 3); plt.imshow(shd); plt.title("Shading"); plt.axis('off')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    process_stripwise_local("DAY 3 LUNCH.jpg")
