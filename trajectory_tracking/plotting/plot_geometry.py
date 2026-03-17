"""Camera–ray–trajectory 3-D geometry plot (primary debugging tool)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from trajectory_tracking.core.measurement import Measurement


def plot_camera_ray_geometry(
    measurements: List[Measurement],
    output_path: Optional[str] = None,
    ray_length: float = 300.0,
    figsize: tuple = (12, 9),
    dpi: int = 150,
    title: Optional[str] = None,
    save_pdf: bool = False,
) -> None:
    """3-D plot showing camera centres, viewing rays, and trajectory.

    This is the **most important debugging plot**: if the rays do not
    roughly converge at the estimated trajectory, the geometry transforms
    are wrong.

    Parameters
    ----------
    measurements : list[Measurement]
    output_path : str | None
    ray_length : float
        Length (mm) to draw each viewing ray.
    title : str | None
        Custom plot title.
    save_pdf : bool
        If True and output_path is given, also save a PDF next to the PNG.
    """
    valid = [m for m in measurements if m.has_geometry()]
    if not valid:
        return

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Camera centres
    cam_pos = np.array([m.camera_position for m in valid])
    ax.plot(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
            "b^", markersize=5, label="Camera centres")

    # Viewing rays
    for m in valid:
        start = m.camera_position
        end = start + m.ray_direction * ray_length
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            "c-", alpha=0.4, linewidth=0.7,
        )

    # Ground-truth trajectory (if available)
    has_gt = [m for m in valid if m.gt_position is not None]
    if has_gt:
        gt = np.array([m.gt_position for m in has_gt])
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2],
                "g.-", linewidth=2, markersize=6, label="Actual trajectory (GT)")

    # Estimated trajectory
    has_pos = [m for m in valid if m.position is not None]
    if has_pos:
        pos = np.array([m.position for m in has_pos])
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                "r.-", linewidth=2, markersize=6, label="Estimated trajectory")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title or "Camera–Ray–Trajectory Geometry")
    ax.legend()

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved camera–ray geometry → {output_path}")
        if save_pdf:
            pdf_path = str(Path(output_path).with_suffix(".pdf"))
            fig.savefig(pdf_path)
            print(f"[plot] Saved camera–ray geometry → {pdf_path}")
    plt.close(fig)


def show_camera_ray_geometry_interactive(
    measurements: List[Measurement],
    ray_length: float = 300.0,
    figsize: tuple = (12, 9),
) -> None:
    """Interactive 3-D plot showing camera centres, viewing rays, and trajectory.

    This opens a matplotlib window that allows you to rotate and zoom 
    in 3D space using the mouse.

    Parameters
    ----------
    measurements : list[Measurement]
    ray_length : float
        Length (mm) to draw each viewing ray.
    """
    valid = [m for m in measurements if m.has_geometry()]
    if not valid:
        print("[plot] No valid geometry to plot.")
        return

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Camera centres
    cam_pos = np.array([m.camera_position for m in valid])
    ax.plot(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
            "b^", markersize=5, label="Camera centres")

    # Viewing rays
    for m in valid:
        start = m.camera_position
        end = start + m.ray_direction * ray_length
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            "c-", alpha=0.4, linewidth=0.7,
        )

    # Estimated trajectory
    has_pos = [m for m in valid if m.position is not None]
    if has_pos:
        pos = np.array([m.position for m in has_pos])
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                "r.-", linewidth=2, markersize=6, label="Estimated trajectory")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Interactive Camera–Ray–Trajectory Geometry")
    ax.legend()

    plt.tight_layout()
    print("[plot] Opening interactive 3D plot. Use mouse to rotate, pan, and zoom.")
    plt.show()


def plot_camera_ray_geometry_plotly(
    measurements: List[Measurement],
    ray_length: float = 300.0,
    output_html: Optional[str] = None
) -> None:
    """Interactive 3-D plot using Plotly (smooth mouse rotation/zoom).
    
    This will open the plot in your default web browser for a much smoother
    interactive 3D experience compared to Matplotlib.
    
    Parameters
    ----------
    measurements : list[Measurement]
    ray_length : float
        Length (mm) to draw each viewing ray.
    output_html : str | None
        Optional path to save the interactive HTML file.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[plot] Plotly is not installed. Please install it using 'pip install plotly'")
        return

    valid = [m for m in measurements if m.has_geometry()]
    if not valid:
        print("[plot] No valid geometry to plot.")
        return

    fig = go.Figure()

    # Camera centres
    cam_pos = np.array([m.camera_position for m in valid])
    fig.add_trace(go.Scatter3d(
        x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
        mode='markers',
        marker=dict(size=4, symbol='square', color='blue'),
        name='Camera centres'
    ))

    # Viewing rays
    ray_x = []
    ray_y = []
    ray_z = []
    for m in valid:
        start = m.camera_position
        end = start + m.ray_direction * ray_length
        # Add None to break the line segments for efficiency
        ray_x.extend([start[0], end[0], None])
        ray_y.extend([start[1], end[1], None])
        ray_z.extend([start[2], end[2], None])

    fig.add_trace(go.Scatter3d(
        x=ray_x, y=ray_y, z=ray_z,
        mode='lines',
        line=dict(color='cyan', width=2),
        opacity=0.4,
        name='Viewing rays'
    ))

    # Estimated trajectory
    has_pos = [m for m in valid if m.position is not None]
    if has_pos:
        pos = np.array([m.position for m in has_pos])
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='lines+markers',
            marker=dict(size=5, color='red'),
            line=dict(color='red', width=4),
            name='Estimated trajectory'
        ))

    fig.update_layout(
        title="Interactive Camera-Ray-Trajectory Geometry",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode='data'
        ),
        legend=dict(x=0, y=1)
    )

    if output_html:
        Path(output_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_html)
        print(f"[plot] Saved interactive Plotly geometry → {output_html}")
    
    # Opens in browser automatically
    print("[plot] Opening interactive Plotly 3D viewer in browser...")
    fig.show()

