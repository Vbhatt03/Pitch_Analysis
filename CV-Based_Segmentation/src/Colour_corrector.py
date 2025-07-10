import cv2
import numpy as np

def Colour_correct(img):
    target_brightness=126
    min_gamma=0.5
    max_gamma=2.5
    # CLAHE (local contrast enhancement)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_corrected = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Gamma correction (global brightness normalization)
    gray = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)
    current = np.mean(gray)
    if abs(current - target_brightness) < 2:
        return img_corrected

    gamma = np.log(target_brightness / 255.0) / np.log(current / 255.0)
    gamma = np.clip(gamma, min_gamma, max_gamma)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_corrected, table)
