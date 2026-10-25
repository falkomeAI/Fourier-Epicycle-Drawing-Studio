"""
fourier_drawing — Backward-compatible alias for falkourier.
"""

from __future__ import annotations

from falkourier import (
    fit_fourier,
    chain_xy,
    fit_complex_fourier,
    sorted_harmonics,
    extract_path_from_image,
    extract_boundary_contour_path,
    get_n_schedule,
    normalize_path,
    sample_points_for_n,
    SHAPES,
    __version__,
)

__all__ = [
    "fit_fourier",
    "chain_xy",
    "fit_complex_fourier",
    "sorted_harmonics",
    "extract_path_from_image",
    "extract_boundary_contour_path",
    "get_n_schedule",
    "normalize_path",
    "sample_points_for_n",
    "SHAPES",
    "__version__",
]
