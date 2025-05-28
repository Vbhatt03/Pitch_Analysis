import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import open3d as o3d

# Load the LAZ file
las = laspy.read("D:\\Quidich\\Point_Cloud\\1st_Time_Scan_3,4,5,6.laz")
points = np.vstack((las.x, las.y, las.z)).transpose()

# Define pitch bounds (approx 22m x 3.05m)
x_center, y_center = 379412.6, 1702100.29
x_min, x_max = x_center - 11, x_center + 11
y_min, y_max = y_center - 1.525, y_center + 1.525

# Filter points within the pitch
mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
       (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
pitch_points = points[mask]
x, y, z = pitch_points[:, 0], pitch_points[:, 1], pitch_points[:, 2]

# --- 1. Interpolated Z-Height Heatmap ---
grid_x, grid_y = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

plt.figure(figsize=(10, 3))
heatmap = plt.imshow(grid_z.T, extent=(x.min(), x.max(), y.min(), y.max()),
                     origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(label='Z Height (m)')
plt.title('Interpolated Z-Height Heatmap of Cricket Pitch')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("pitch_heatmap.png", dpi=300)
plt.show()

# --- 2. Z-Averaged Grid Heatmap (1m x 1m) ---
x_bins = np.arange(x.min(), x.max(), 1)
y_bins = np.arange(y.min(), y.max(), 1)
z_matrix = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)

for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        bin_mask = (x >= x_bins[i]) & (x < x_bins[i+1]) & \
                   (y >= y_bins[j]) & (y < y_bins[j+1])
        if np.any(bin_mask):
            z_matrix[i, j] = np.mean(z[bin_mask])

plt.figure(figsize=(10, 3))
plt.imshow(z_matrix.T, extent=(x.min(), x.max(), y.min(), y.max()),
           origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Avg Z Height (m)')
plt.title('Z-Averaged Grid Heatmap (1m x 1m)')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("pitch_grid_average.png", dpi=300)
plt.show()

# --- 3. 3D Surface Mesh from Point Cloud ---
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pitch_points)
# pcd.estimate_normals()

# radii = [0.5, 1.0, 1.5]
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd, o3d.utility.DoubleVector(radii))

# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], window_name="3D Pitch Surface")
# o3d.io.write_triangle_mesh("pitch_surface_mesh.ply", mesh)
