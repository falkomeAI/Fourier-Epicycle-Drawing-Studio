"""
falkourier — High-Precision Fourier Series Epicycle Drawing Engine & Studio.

Created by @falkomeai
"""

from __future__ import annotations

__version__ = "1.0.1"
__author__ = "falkomeai"

from .series import fit_fourier, chain_xy
from .complex_series import fit_complex_fourier, sorted_harmonics
from .image_path import extract_path_from_image, extract_boundary_contour_path
from .schedule import get_n_schedule, normalize_path, sample_points_for_n
from .shapes import SHAPES

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
