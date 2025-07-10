#The script detects grass and cracks in albedo and shading images, respectively, and overlays the results on the original image.
# It calculates the percentage coverage of grass and cracks, saves the masks and overlay images, and writes the statistics to a text file.
# It processes images in pairs from specified directories and saves the results in an output directory.
import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern

def sharpen_image(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

def detect_cracks_from_shading(shading_path):
    shading = cv2.imread(shading_path)
    gray = cv2.cvtColor(shading, cv2.COLOR_BGR2GRAY)
    sharpened = sharpen_image(gray)

    sobelx = cv2.Sobel(sharpened, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(sharpened, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag = cv2.convertScaleAbs(sobel_mag)

    thresh_val = np.percentile(sobel_mag, 90)
    _, crack_mask = cv2.threshold(sobel_mag, thresh_val, 255, cv2.THRESH_BINARY)
    return crack_mask, shading

def detect_grass_from_albedo(albedo_path):
    albedo = cv2.imread(albedo_path)
    albedo_gray = cv2.cvtColor(albedo, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(albedo_gray, P=8, R=1, method='uniform')
    lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, grass_mask = cv2.threshold(lbp_norm, 140, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel)

    return grass_mask, albedo

def analyze_and_overlay(base_img, crack_mask, grass_mask, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    total_pixels = crack_mask.shape[0] * crack_mask.shape[1]
    crack_pixels = np.sum(crack_mask == 255)
    grass_pixels = np.sum(grass_mask == 255)

    gray_base = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    overlay = cv2.cvtColor(gray_base, cv2.COLOR_GRAY2BGR)

    overlay[crack_mask == 255] = [0, 0, 255]
    overlay[grass_mask == 255] = [0, 255, 0]

    cv2.imwrite(os.path.join(output_dir, "crack_mask.png"), crack_mask)
    cv2.imwrite(os.path.join(output_dir, "grass_mask.png"), grass_mask)
    cv2.imwrite(os.path.join(output_dir, "overlay_output.png"), overlay)

    crack_pct = 100 * crack_pixels / total_pixels
    grass_pct = 100 * grass_pixels / total_pixels
    print(f"[{os.path.basename(output_dir)}] Crack coverage: {crack_pct:.2f}%")
    print(f"[{os.path.basename(output_dir)}] Grass coverage: {grass_pct:.2f}%")

    with open(os.path.join(output_dir, "stats.txt"), "w") as f:
        f.write(f"Crack coverage: {crack_pct:.2f}%\n")
        f.write(f"Grass coverage: {grass_pct:.2f}%\n")

def process_named_pairs(albedo_dir="output_full_albedo", shading_dir="output_full_shader", output_dir="output_full_results"):
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(albedo_dir)):
        if not fname.endswith(".png"):
            continue

        albedo_path = os.path.join(albedo_dir, fname)
        shading_path = os.path.join(shading_dir, fname)

        if not os.path.isfile(shading_path):
            print(f"[WARN] Missing shading for {fname}, skipping.")
            continue

        crack_mask, shading_img = detect_cracks_from_shading(shading_path)
        grass_mask, _ = detect_grass_from_albedo(albedo_path)

        name_without_ext = os.path.splitext(fname)[0]
        img_output_dir = os.path.join(output_dir, name_without_ext)
        analyze_and_overlay(shading_img, crack_mask, grass_mask, img_output_dir)

if __name__ == "__main__":
    process_named_pairs()
