from src.Visualizer import Pitch_Visualizer
from src.Contour_vis import Contour_Visualizer
from src.Analyser import grass_crack_detection
import laspy
import numpy as np
def main():
    # Load the LAZ file.
    las = laspy.read("D:\\Quidich\\Point_Cloud\\1st_Time_Scan_3,4,5,6.laz")
    points = np.vstack((las.x, las.y, las.z)).transpose()
    # Visualize the pitch area.
    Pitch_Visualizer(points)
    
    # Visualize the contour map.
    Contour_Visualizer(points)
    
    # Visualize cracks and grass areas and print height statistics.
    grass_crack_detection(points)
if __name__ == "__main__":
    main()
