import cv2
import numpy as np

def get_green_confidence(img_strip, rough_mask):
    hsv = cv2.cvtColor(img_strip, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)
    h_vals = h[rough_mask > 0]
    s_vals = s[rough_mask > 0]

    if len(h_vals) == 0 or len(s_vals) == 0:
        return 0

    h_score = np.mean((h_vals > 25) & (h_vals < 95))
    s_score = np.mean(s_vals) / 255.0
    return h_score * s_score

def Segmentation(img, img_corrected, strip_height=200, min_crack_area=600):
    output = np.ones_like(img) * 127  # Gray background

    for y in range(0, img.shape[0], strip_height):
        y_end = min(y + strip_height, img.shape[0])
        img_strip = img_corrected[y:y_end]

        # Rough mask using ExG for confidence check
        b, g, r = cv2.split(img_strip.astype(np.int16))
        exg = 2.8 * g - r - b
        exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, rough_mask = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        green_confidence = get_green_confidence(img_strip, rough_mask)

        # HSV for grass detection with adaptive bounds
        hsv = cv2.cvtColor(img_strip, cv2.COLOR_BGR2HSV)
        base_lower = np.array([15, 20, 20])
        base_upper = np.array([110, 255, 255])
        scale = np.interp(green_confidence, [0.0, 1.0], [0.5, 1.0])
        low_offset = np.array([10, 10, 10])
        high_offset = np.array([0, 0, 0])
        lower_green = np.clip(base_lower + (1 - scale) * low_offset, 0, 255).astype(np.uint8)
        upper_green = np.clip(base_upper - (1 - scale) * high_offset, 0, 255).astype(np.uint8)
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Adaptive morphology based on green confidence
        kernel_size = int(np.interp(green_confidence, [0.0, 1.0], [9, 5]))
        kernel_size = max(3, kernel_size | 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)
        grass_mask = cv2.erode(grass_mask, kernel, iterations=1)

        # Non-grass grayscale area
        gray = cv2.cvtColor(img_strip, cv2.COLOR_BGR2GRAY)
        non_grass = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(grass_mask))

        # Enhance cracks
        blackhat = cv2.morphologyEx(non_grass, cv2.MORPH_BLACKHAT, np.ones((7, 7), np.uint8))
        _, cracks = cv2.threshold(blackhat, 60, 255, cv2.THRESH_BINARY)

        # Closing to connect cracks
        cracks_closed = cv2.morphologyEx(cracks, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # Filter cracks by area and shape
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cracks_closed, connectivity=8)
        crack_mask = np.zeros_like(cracks_closed)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if area >= min_crack_area and 0.2 < w / (h + 1e-5) < 6:
                crack_mask[labels == i] = 255

        # Final coloring
        grass_only = (grass_mask > 0) & (crack_mask == 0)
        output[y:y_end][grass_only] = [255, 255, 255]  # white
        output[y:y_end][crack_mask > 0] = [0, 0, 255]  # red

    return output
