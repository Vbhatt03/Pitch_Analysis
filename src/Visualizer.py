#This script is used to visualize a point cloud from a LAZ file, applying percentile-based clipping to the Z values and displaying the results with Open3D and Matplotlib.
#Does not include grass/crack detection logic, but focuses on visualizing the point cloud with color mapping based on Z values.
import laspy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
def Pitch_Visualizer(points):

    # Region of interest (pitch area)
    x_min, x_max = 379407.71, 379425.49
    y_min, y_max = 1702083.45, 1702116.13

    # Filter points in region + remove extreme Z outliers
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] <= 101)
    )
    region_points = points[mask]
    z = region_points[:, 2]

    # Percentile clipping
    z_clip_min = np.percentile(z, 1)
    z_clip_max = np.percentile(z, 99)
    z_clipped = np.clip(z, z_clip_min, z_clip_max)

    # Normalize and apply colormap
    norm_z = (z_clipped - z_clip_min) / (z_clip_max - z_clip_min + 1e-8)
    colors = plt.cm.viridis(norm_z)[:, :3]

    # Debug stats
    z_vals = np.sort(np.unique(z))
    z_diffs = np.diff(z_vals)
    print(f"Z range: {z.min():.4f} to {z.max():.4f}")
    print(f"Clipped Z range: {z_clip_min:.4f} to {z_clip_max:.4f}")
    print(f"Number of unique Z: {z_vals.size}")
    print("Smallest Z diff (nonzero):", np.min(z_diffs[z_diffs > 0]))

    # Set up Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(region_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ----- Add colorbar with Matplotlib -----
    fig, ax = plt.subplots(figsize=(6, 1.2))
    fig.subplots_adjust(bottom=0.5)

    norm = mpl.colors.Normalize(vmin=z_clip_min, vmax=z_clip_max)
    cbar = mpl.colorbar.ColorbarBase(
        ax, cmap='viridis', norm=norm, orientation='horizontal'
    )
    cbar.set_label('Elevation (Z, meters)')
    plt.title('Elevation Color Mapping')
    plt.show()

    # ----- Show Open3D point cloud -----
    o3d.visualization.draw_geometries([pcd], window_name="Contrast-Stretched Pitch Z Colors")
