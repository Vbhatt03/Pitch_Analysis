# === Step 6 (Optional): Create PyVista Mesh for 3D Visual ===
# pv.set_plot_theme("document")
# cloud = pv.PolyData(region_points)
# cloud["residuals"] = residuals

# surf = cloud.delaunay_2d()
# p = pv.Plotter()
# p.add_mesh(surf, scalars="residuals", cmap="coolwarm", show_edges=False, clim=[-0.03, 0.03])
# p.enable_eye_dome_lighting()
# p.show()
