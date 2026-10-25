"""Complex Fourier series decomposition with sorted frequency harmonics.

Transforms the complete complex path, sorts all frequency terms by radius
(amplitude), and treats N as the number of active epicycles.
"""

from __future__ import annotations

import numpy as np


def fit_complex_fourier(z: np.ndarray, max_arms: int | None = None) -> dict[int, complex]:
    """Return the complete complex DFT, keyed by signed frequency.

    ``max_arms`` is accepted for compatibility with existing callers; it is
    applied when selecting terms, not while fitting. The FFT computes every
    frequency in O(M log M), including small but potentially useful terms.
    """
    values = np.asarray(z, dtype=np.complex128)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("the path must be a one-dimensional array with at least 2 points")
    if not np.all(np.isfinite(values.real) & np.isfinite(values.imag)):
        raise ValueError("path points must be finite")
    if max_arms is not None and int(max_arms) < 1:
        raise ValueError("max_arms must be at least 1")

    spectrum = np.fft.fft(values) / values.size
    m = values.size
    coeffs: dict[int, complex] = {}
    for index, coefficient in enumerate(spectrum):
        freq = index if index <= m // 2 else index - m
        coeffs[int(freq)] = complex(coefficient)
    return coeffs


def sorted_harmonics(coeffs: dict[int, complex], n: int | None = None) -> list[int]:
    """Frequency keys ordered by coefficient magnitude, largest first.

    ``n`` means the number of epicycle terms to keep (top n highest amplitudes).
    """
    order = sorted(coeffs, key=lambda k: -abs(coeffs[k]))
    return order if n is None else order[: max(0, int(n))]


def evaluate_fourier(
    coeffs: dict[int, complex], t: float | np.ndarray, n: int | None = None
) -> complex | np.ndarray:
    """Evaluate the N strongest terms at time ``t``."""
    ks = np.asarray(sorted_harmonics(coeffs, n), dtype=int)
    cs = np.asarray([coeffs[int(k)] for k in ks], dtype=np.complex128)
    if isinstance(t, np.ndarray):
        return np.exp(1j * np.outer(t, ks)) @ cs
    return complex(np.sum(cs * np.exp(1j * ks * t)))


def arm_chain(
    coeffs: dict[int, complex], t: float, n: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Epicycle joint positions and radii for the N strongest terms."""
    order = sorted_harmonics(coeffs, n)
    pts = [0j]
    radii = []
    z = 0j
    for k in order:
        c = coeffs[k]
        z += c * np.exp(1j * k * t)
        pts.append(z)
        radii.append(abs(c))
    return np.asarray(pts), np.asarray(radii)
