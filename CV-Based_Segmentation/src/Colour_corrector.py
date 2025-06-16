import cv2
import numpy as np


def Colour_correct(img_path):
    img = cv2.imread(img_path)
    assert img is not None, "Image not found"
    # Color correction (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_corrected = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img_corrected
