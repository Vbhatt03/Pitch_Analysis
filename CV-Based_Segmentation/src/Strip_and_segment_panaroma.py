import cv2
import numpy as np

def Segmentation(img, img_corrected, strip_height, min_crack_area):
    output = np.ones_like(img) * 127  # Gray background
    height = img.shape[0]
    overlap = strip_height // 2

    for y in range(0, height, overlap):
        y_end = min(y + strip_height, height)
        img_strip = img_corrected[y:y_end]

        # --- LAB-based Grass Mask ---
        lab = cv2.cvtColor(img_strip, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        mask_A = cv2.inRange(A, 95, 130)
        mask_B = cv2.inRange(B, 120, 160)
        grass_mask = cv2.bitwise_and(mask_A, mask_B)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # --- Crack Detection ---
        gray = cv2.cvtColor(img_strip, cv2.COLOR_BGR2GRAY)
        non_grass = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(grass_mask))
        non_grass_blur = cv2.GaussianBlur(non_grass, (5, 5), 0)

        # Adaptive Canny (clipped)
        med = np.median(non_grass_blur[non_grass_blur > 0]) if np.count_nonzero(non_grass_blur) > 0 else 100
        lower = int(np.clip(0.66 * med, 120, 180))
        upper = int(np.clip(1.33 * med, lower + 20, 280))

        cracks = cv2.Canny(non_grass_blur, lower, upper)
        cracks = cv2.morphologyEx(cracks, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # Filter small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cracks, connectivity=8)
        crack_mask = np.zeros_like(cracks)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_crack_area:
                crack_mask[labels == i] = 255

        # Paint final output
        slice_out = output[y:y_end]
        slice_out[grass_mask > 0] = [255, 255, 255]
        slice_out[(crack_mask > 0) & (grass_mask == 0)] = [0, 0, 255]
        output[y:y_end] = np.maximum(output[y:y_end], slice_out)

        # --- Optional Debug ---
        # if debug:
        #     cv2.imshow("Strip", img_strip)
        #     cv2.imshow("Grass Mask", grass_mask)
        #     cv2.imshow("Non-Grass", non_grass)
        #     cv2.imshow("Canny Output", cracks)
        #     cv2.imshow("Crack Mask", crack_mask)
        #     cv2.waitKey(0)

    return output