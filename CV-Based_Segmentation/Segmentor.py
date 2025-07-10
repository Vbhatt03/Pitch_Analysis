import cv2
import numpy as np

def apply_grayworld_white_balance(img):
    wb = cv2.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    return wb.balanceWhite(img)

def segment_grass_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Tuned for cricket pitch grass tones
    lower_green = np.array([15, 5, 5])
    upper_green = np.array([110, 255, 255])
    return cv2.inRange(hsv, lower_green, upper_green)

def detect_cracks(img_gray, grass_mask):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    pitch_mask = cv2.bitwise_not(grass_mask)
    cracks = cv2.bitwise_and(edges, edges, mask=pitch_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cracks = cv2.dilate(cracks, kernel, iterations=1)
    return cracks

def create_segmented_output(img, grass_mask, crack_mask):
    h, w = img.shape[:2]
    output = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)  # gray background
    output[grass_mask > 0] = [255, 255, 255]  # grass → white
    output[crack_mask > 0] = [0, 0, 255]      # cracks → red
    return output

def segment_cricket_pitch(image_path, save_path=None, strip_height=1000):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    h, w = img.shape[:2]
    if h > strip_height:  # If image is large, process in vertical strips
        result = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, strip_height):
            y_end = min(y + strip_height, h)
            strip = img[y:y_end]

            # === Process strip ===
            wb_strip = apply_grayworld_white_balance(strip)
            grass = segment_grass_hsv(wb_strip)
            gray_strip = cv2.cvtColor(wb_strip, cv2.COLOR_BGR2GRAY)
            cracks = detect_cracks(gray_strip, grass)
            segment = create_segmented_output(strip, grass, cracks)
            result[y:y_end] = segment
    else:
        wb_img = apply_grayworld_white_balance(img)
        grass_mask = segment_grass_hsv(wb_img)
        gray = cv2.cvtColor(wb_img, cv2.COLOR_BGR2GRAY)
        crack_mask = detect_cracks(gray, grass_mask)
        result = create_segmented_output(img, grass_mask, crack_mask)

    if save_path:
        cv2.imwrite(save_path, result)
        print(f"[✔] Saved segmented output to: {save_path}")
    return result

if __name__ == "__main__":
    INPUT = "DAY_1_START.png"     # Replace with your input file
    OUTPUT = "lords_day1_segmented.png"     # Output file
    segmented = segment_cricket_pitch(INPUT, OUTPUT)
    
    # Optional: Show result
    # cv2.imshow("Segmented Grass and Cracks", segmented)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
