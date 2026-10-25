"""Closed paths used by the Fourier epicycle animation."""

from __future__ import annotations

import numpy as np


def _resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon to n roughly equal-arc-length samples.
    
    Args:
        points: Nx2 array of (x, y) points forming a closed or open polygon
        n: Number of output samples (uniformly distributed by arc length)
    
    Returns:
        nx2 array of resampled points (does NOT duplicate the closing point)
    
    Raises:
        ValueError: If the polygon is degenerate (zero perimeter) or too short
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("polygon must have at least 3 points")
    
    # Ensure the polygon is closed for arc-length calculation
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    
    diffs = np.diff(pts, axis=0)
    seg = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    
    if total < 1e-9:
        raise ValueError("degenerate polygon (zero perimeter)")
    
    # Sample n points uniformly by arc length (endpoint=False avoids duplicate)
    targets = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(targets, cum, pts[:, 0])
    y = np.interp(targets, cum, pts[:, 1])
    return np.column_stack([x, y])


def hummingbird_path(n_points: int = 512) -> np.ndarray:
    """
    Side-view hummingbird silhouette (beak right, wing up, tail left).
    Tuned so ~50 harmonics recover a recognizable outline.
    """
    raw = np.array(
        [
            # beak tip → head
            [3.20, 0.55],
            [2.70, 0.58],
            [2.20, 0.65],
            [1.75, 0.78],
            [1.40, 0.95],
            [1.20, 1.15],
            [1.10, 1.35],
            [0.95, 1.48],
            [0.75, 1.52],
            [0.55, 1.45],
            [0.42, 1.28],
            # neck → upper back / wing root
            [0.35, 1.10],
            [0.15, 1.20],
            [-0.10, 1.40],
            [-0.35, 1.65],
            [-0.55, 1.95],
            [-0.60, 2.25],
            [-0.45, 2.40],  # wing tip
            [-0.25, 2.25],
            [-0.10, 1.95],
            [0.00, 1.60],
            [0.05, 1.30],
            # back → tail
            [-0.15, 1.05],
            [-0.45, 0.90],
            [-0.85, 0.80],
            [-1.30, 0.75],
            [-1.75, 0.78],
            [-2.15, 0.90],
            [-2.45, 1.10],
            [-2.65, 1.20],  # upper tail fork
            [-2.80, 1.05],
            [-2.70, 0.80],
            [-2.45, 0.55],
            [-2.15, 0.35],
            [-1.80, 0.22],
            [-1.40, 0.15],
            [-1.00, 0.10],
            # belly → chest → throat → beak
            [-0.55, 0.08],
            [-0.10, 0.08],
            [0.35, 0.12],
            [0.75, 0.22],
            [1.10, 0.32],
            [1.45, 0.40],
            [1.90, 0.45],
            [2.40, 0.50],
            [2.90, 0.53],
            [3.20, 0.55],
        ],
        dtype=float,
    )
    return _resample_closed(raw, n_points)


def heart_path(n_points: int = 512) -> np.ndarray:
    """Classic heart curve — good for a quick sanity check."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return np.column_stack([x, y])


def star_path(n_points: int = 512, spikes: int = 5) -> np.ndarray:
    """Simple star for debugging."""
    m = spikes * 2
    angles = np.linspace(0, 2 * np.pi, m, endpoint=False) - np.pi / 2
    radii = np.where(np.arange(m) % 2 == 0, 1.0, 0.42)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _resample_closed(pts, n_points)


SHAPES = {
    "hummingbird": hummingbird_path,
    "heart": heart_path,
    "star": star_path,
}
