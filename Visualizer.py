import laspy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load LAZ file
las = laspy.read("D:\\Quidich\\Point_Cloud\\1st_Time_Scan_3,4,5,6.laz")
points = np.vstack((las.x, las.y, las.z)).transpose()

# Define pitch bounding box based on orientation
x_min, x_max = 379399.6, 379425.6  # 26m along X (length)
y_min, y_max = 1702097.8, 1702102.8  # 5m along Y (width)

# Mask the pitch region
mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
       (points[:, 1] >= y_min) & (points[:, 1] <= y_max)

pitch_points = points[mask]

# Remove Z outliers within pitch region
z = pitch_points[:, 2]
z_lower = np.percentile(z, 1)
z_upper = np.percentile(z, 99)
z_clipped = np.clip(z, z_lower, z_upper)

# Linear color mapping: low Z = light, high Z = red (use 'Reds' colormap)
norm_z = (z_clipped - z_lower) / (z_upper - z_lower + 1e-8)
colors = plt.cm.Reds(norm_z)[:, :3]  # Reds: light for low, red for high

# Create point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pitch_points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd], window_name="Cricket Pitch: Linear Z Color (Red High)")