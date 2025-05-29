import laspy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load the LAZ file
las = laspy.read("D:\\Quidich\\Point_Cloud\\1st_Time_Scan_3,4,5,6.laz")
points = np.vstack((las.x, las.y, las.z)).transpose()

# Region of interest (pitch area)
x_min, x_max = 379407.71, 379425.49
y_min, y_max = 1702083.45, 1702116.13

# Filter points in region + remove extreme Z outliers
mask = (
    (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
    (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
    (points[:, 2] <= 101)  # Z upper bound for removing spikes
)
region_points = points[mask]
z = region_points[:, 2]

# ----- Optional: Percentile clipping for better contrast -----
z_clip_min = np.percentile(z, 1)
z_clip_max = np.percentile(z, 99)
z_clipped = np.clip(z, z_clip_min, z_clip_max)

# ----- Normalize for colormap -----
norm_z = (z_clipped - z_clip_min) / (z_clip_max - z_clip_min + 1e-8)
colors = plt.cm.viridis(norm_z)[:, :3]

# Debug stats
z_vals = np.sort(np.unique(z))
z_diffs = np.diff(z_vals)
print(f"Z range: {z.min():.4f} to {z.max():.4f}")
print(f"Clipped Z range: {z_clip_min:.4f} to {z_clip_max:.4f}")
print(f"Number of unique Z: {z_vals.size}")
print("Smallest Z diff (nonzero):", np.min(z_diffs[z_diffs > 0]))

# ----- Open3D visualization -----
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(region_points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Show the point cloud
o3d.visualization.draw_geometries([pcd], window_name="Contrast-Stretched Pitch Z Colors")
