"""
fourier_drawing — shared core for Fourier-series epicycle animations.

Submodules:
  series          Real two-axis Fourier series: x_N(t), y_N(t)
  complex_series  Single complex-plane Fourier series: z(t)
  schedule        Progressive N scheduling (1 → 2 → 3 → … → N) + path normalization
  shapes          Built-in closed-path shapes (hummingbird, heart, star)
  image_path      Image → closed outline extraction (contours, masks)
  debug_frames    Optional per-N PNG dump to samples/_debug_frames/ for inspecting complex shapes
"""

__all__ = ["series", "complex_series", "schedule", "shapes", "image_path", "debug_frames"]
