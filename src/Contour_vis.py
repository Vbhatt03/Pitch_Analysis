import laspy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import griddata
def Contour_Visualizer(points):
    mask = (points[:,2]<=105)
    height_cap_points = points[mask] 
    x = height_cap_points[:, 0]
    y = height_cap_points[:, 1]
    z_grid = height_cap_points[:, 2]

    # Create grid
    num_grid = 300
    xi = np.linspace(x.min(), x.max(), num_grid)
    yi = np.linspace(y.min(), y.max(), num_grid)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z_grid, (Xi, Yi), method='linear')

    # Plot contour
    fig, ax = plt.subplots(figsize=(10, 6))
    # Choose number of contour levels
    num_levels = 10
    levels = np.linspace(np.nanmin(Zi), np.nanmax(Zi), num_levels)

    contours = ax.contour(Xi, Yi, Zi, levels=levels, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=10, fmt="%.2f", colors='black')
    contourf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='viridis', alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=10, fmt="%.2f", colors='black')

    # Add filled contours for colorbar
    contourf = ax.contourf(Xi, Yi, Zi, levels=10, cmap='viridis', alpha=0.5)
    fig.colorbar(contourf, ax=ax, label="Elevation (Z, meters)")

    ax.set_title("Elevation Contour Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    plt.show()