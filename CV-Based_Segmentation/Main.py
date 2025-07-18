import cv2
import numpy as np
from src.Colour_corrector import Colour_correct
from src.Strip_and_segment_panaroma import Segmentation


def main():
    path = 'DAY 3 LUNCH.jpg'
    img = cv2.imread(path)
    assert img is not None, "Image not found"

    # Color correction (CLAHE)
    img_corrected = Colour_correct(img)
    cv2.imwrite("Strip-wise CLAHE + Gamma Corrected_3.png", img_corrected)
    # Define parameters for segmentation
    strip_height = 2000
    # Minimum area for cracks to be considered significant
    min_crack_area = 5000
    #change the values here for changing the sensitivity of grass detection.
    lower_green = np.array([15, 5, 5])
    upper_green = np.array([110, 255, 255])
    lower_yellow = np.array([20, 30, 80])
    upper_yellow = np.array([30, 255, 255])

#Use this for original segementation fn
    #output = Segmentation(img, img_corrected, strip_height, min_crack_area,
       #                 lower_green, upper_green, lower_yellow, upper_yellow)
    output = Segmentation(img, img_corrected, strip_height, min_crack_area)

    # Save the output image
    cv2.imwrite("lords_day3_segmented(area=5000-nogc_changed).png", output)
    print("Done!! exiting script...")
if __name__ == "__main__":
    main()
# This script processes a pitch image to detect grass and cracks, applying color correction and segmentation techniques.