import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pyvista as pv

# === Step 1: Load LiDAR ===
las = laspy.read("D:\\Quidich\\Point_Cloud\\1st_Time_Scan_3,4,5,6.laz")
points = np.vstack((las.x, las.y, las.z)).T

# === Step 2: Filter ROI ===
x_min, x_max = 379400.71, 379430.49
y_min, y_max = 1702083.45, 1702120.13
z_max = 100.1

mask = (
    (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
    (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
    (points[:, 2] <= z_max)
)
region_points = points[mask]
XY = region_points[:, :2]
Z = region_points[:, 2]

# === Step 3: Fit Ground Plane (Robust via Least Squares + Percentile Clipping) ===
z_lo, z_hi = np.percentile(Z, [20, 80])
inliers = (Z >= z_lo) & (Z <= z_hi)

A = np.c_[XY[inliers], np.ones(np.sum(inliers))]
C, _, _, _ = np.linalg.lstsq(A, Z[inliers], rcond=None)

Z_fit = (np.c_[XY, np.ones(XY.shape[0])] @ C)
residuals = Z - Z_fit  # Elevation deviation from pitch plane

# === Step 4: Interpolate to Grid ===
grid_x, grid_y = np.mgrid[
    XY[:, 0].min():XY[:, 0].max():500j,
    XY[:, 1].min():XY[:, 1].max():500j
]
grid_z = griddata(XY, residuals, (grid_x, grid_y), method='cubic')

# # === Step 5: Visualize as Heatmap ===
# plt.figure(figsize=(12, 8))
# plt.imshow(
#     grid_z.T,
#     extent=(XY[:, 0].min(), XY[:, 0].max(), XY[:, 1].min(), XY[:, 1].max()),
#     origin='lower',
#     cmap='coolwarm',
#     vmin=-0.03, vmax=0.03  # ±3 cm defect range
# )
# plt.title("Residual Elevation Map of Cricket Pitch (LiDAR)")
# plt.xlabel("X Coordinate (m)")
# plt.ylabel("Y Coordinate (m)")
# plt.colorbar(label="Deviation from Pitch Plane (m)")
# plt.show()
# plt.close("all")
# === Step 6 (Optional): Create PyVista Mesh for 3D Visual ===
pv.set_plot_theme("document")
cloud = pv.PolyData(region_points)
cloud["residuals"] = residuals

surf = cloud.delaunay_2d()
p = pv.Plotter()
p.add_mesh(surf, scalars="residuals", cmap="coolwarm", show_edges=False, clim=[-0.03, 0.03])
p.enable_eye_dome_lighting()
p.show()
