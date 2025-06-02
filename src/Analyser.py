# This Script is used to detect grass and cracks in a point cloud using LAZ files.
# This script uses a constant ground height (median Z) and visualizes the results with Open3D.
import laspy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
def grass_crack_detection(points):
    # Define Region of Interest
    x_min, x_max = 379400.71, 379430.49
    y_min, y_max = 1702083.45, 1702120.13

    # Filter points within region and Z below 101
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] <= 100.9)
    )
    region_points = points[mask]
    XY = region_points[:, :2]
    Z = region_points[:, 2]
    # Calculate and print point density
    area = (x_max - x_min) * (y_max - y_min)  # in square meters
    density = len(region_points) / area if area > 0 else 0
    print(f"Point density: {density:.2f} points/m²")
    # Use a constant ground height (median)
    ground_height = np.median(Z)
    print(f"Constant ground height = {ground_height:.4f} m")

    # Set thresholds in meters (e.g., ±1.5 cm)
    crack_thresh = -0.015  # -1.5 cm
    grass_thresh = 0.015   # +1.5 cm
    max_defect = 0.8       # 80 cm

    # Residuals from constant ground
    residuals = Z - ground_height

    # Print stats for debugging
    print("Z min:", np.min(Z), "Z max:", np.max(Z))
    print("Residuals min:", np.min(residuals), "max:", np.max(residuals))

    # Plot histogram of residuals
    plt.hist(residuals, bins=100, color='gray')
    plt.axvline(grass_thresh, color='g', linestyle='--', label='Grass Thresh')
    plt.axvline(crack_thresh, color='r', linestyle='--', label='Crack Thresh')
    plt.legend()
    plt.title("Residuals Distribution (Ground = median Z)")
    plt.xlabel("Residual (meters)")
    plt.ylabel("Count")
    plt.show()

    # Classify
    grass_mask = (residuals > grass_thresh) & (residuals < max_defect)
    crack_mask = (residuals < crack_thresh) & (residuals > -max_defect)
    normal_mask = (~grass_mask) & (~crack_mask)

    # Split point groups
    grass_pts = region_points[grass_mask]
    crack_pts = region_points[crack_mask]
    normal_pts = region_points[normal_mask]

    grass_heights = residuals[grass_mask]
    crack_depths = residuals[crack_mask]
    # Clip heights and depths to a maximum defect size
    grass_heights = np.clip(grass_heights, 0, max_defect)
    crack_depths = np.clip(crack_depths, -max_defect, 0)

    # Output sample defects
    print(f"Total Points: {len(region_points)}")
    print(f"Grass Points Detected: {len(grass_pts)}")
    print(f"Crack Points Detected: {len(crack_pts)}\n")

    print("Sample Grass (x, y, height):")
    for i in range(min(5, len(grass_pts))):
        x, y = grass_pts[i][:2]
        h = grass_heights[i] * 100
        print(f"({x:.2f}, {y:.2f}) -> +{h:.1f} cm")

    print("\nSample Cracks (x, y, depth):")
    for i in range(min(5, len(crack_pts))):
        x, y = crack_pts[i][:2]
        d = crack_depths[i] * 100
        print(f"({x:.2f}, {y:.2f}) -> {d:.1f} cm")

    # Assign RGB colors: gray for normal, green for grass, red for crack
    color_normal = np.tile(np.array([[0.6, 0.6, 0.6]]), (len(normal_pts), 1))  # gray
    color_grass = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(grass_pts), 1))    # green
    color_crack = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(crack_pts), 1))    # red

    # Combine all points and colors
    all_pts = np.vstack((normal_pts, grass_pts, crack_pts))
    all_colors = np.vstack((color_normal, color_grass, color_crack))

    # Create and visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.visualization.draw_geometries([pcd], window_name="Grass & Crack Detection (Constant Ground)", width=1280, height=720)