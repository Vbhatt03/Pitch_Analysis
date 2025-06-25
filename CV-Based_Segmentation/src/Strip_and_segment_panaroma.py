import cv2
import numpy as np

def Segmentation(img, img_corrected, strip_height, min_crack_area, 
                 lower_green, upper_green, lower_yellow, upper_yellow):
    output = np.ones_like(img) * 127  # Gray background

    for y in range(0, img.shape[0], strip_height):
        y_end = min(y + strip_height, img.shape[0])
        img_strip = img_corrected[y:y_end]

        hsv = cv2.cvtColor(img_strip, cv2.COLOR_BGR2HSV)
        #mask green grass.
        grass_mask_green = cv2.inRange(hsv, lower_green, upper_green)
        # Dead grass (yellow/brown)
    
        grass_mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Combine both masks
        grass_mask = cv2.bitwise_or(grass_mask_green, grass_mask_yellow)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        gray = cv2.cvtColor(img_strip, cv2.COLOR_BGR2GRAY)
        non_grass = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(grass_mask))

        # Canny for cracks, no dilation/closing to preserve shape
        cracks = cv2.Canny(non_grass, 80, 180)

        # Use connected components to filter by area, but keep thin shape
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cracks, connectivity=8)
        crack_mask = np.zeros_like(cracks)
        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            component = (labels == i)
            if area >= min_crack_area:
                crack_mask[component] = 255
            # else:
            #     # For small cracks, only keep if in grass region
            #     if np.any((grass_mask > 0) & component):
            #         crack_mask[component] = 255

        # Set grass (green or dead) to white (not crack)
        grass_only = (grass_mask > 0) & (crack_mask == 0)
        output[y:y_end][grass_only] = [255, 255, 255]
        # Set cracks to red for visibility
        output[y:y_end][crack_mask > 0] = [0, 0, 255]
    return output