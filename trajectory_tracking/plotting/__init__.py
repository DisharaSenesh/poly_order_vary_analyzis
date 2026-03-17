import matplotlib as _mpl
_mpl.use("Agg")  # non-interactive backend — safe for headless / CI

from .plot_geometry import plot_camera_ray_geometry, plot_camera_ray_geometry_plotly

__all__ = [
    "plot_camera_ray_geometry",
    "plot_camera_ray_geometry_plotly",
]
