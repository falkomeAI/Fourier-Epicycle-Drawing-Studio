"""
Real two-axis Fourier series.

Two independent series reconstruct a closed path:

    x_N(t) = a0/2 + Σ_{k=1..N} (a_k cos kt + b_k sin kt)
    y_N(t) = c0/2 + Σ_{k=1..N} (c_k cos kt + d_k sin kt)

Fit one `FourierSeries` per axis with `fit_fourier`. Each harmonic k is one
rotating vector of length `hypot(a_k, b_k)` — i.e. one epicycle circle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FourierSeries:
    """a0/2 + sum_{k=1..N} (a_k cos kt + b_k sin kt)."""

    a0: float
    a: np.ndarray
    b: np.ndarray

    @property
    def n_max(self) -> int:
        return len(self.a)

    def evaluate(self, t: float, n: int | None = None) -> float:
        n = self.n_max if n is None else min(n, self.n_max)
        if n <= 0:
            return float(self.a0 / 2)
        k = np.arange(1, n + 1)
        return float(
            self.a0 / 2
            + np.sum(self.a[:n] * np.cos(k * t) + self.b[:n] * np.sin(k * t))
        )

    def epicycle_radii(self, n: int | None = None) -> np.ndarray:
        n = self.n_max if n is None else min(n, self.n_max)
        return np.hypot(self.a[:n], self.b[:n])


def fit_fourier(signal: np.ndarray, n_harmonics: int) -> FourierSeries:
    """Fit one coordinate of a closed path using FFT coefficients.
    
    Uses the standard DFT coefficient estimation:
    - Stored a0 = 2 * mean(signal); evaluation uses the conventional a0/2
    - a_k = 2 * mean(signal * cos(k*t))
    - b_k = 2 * mean(signal * sin(k*t))

    Thus, if the series is written with the constant term as `A0` (without
    `/2`), then `A0 = mean(signal)` and `A0 = a0/2` in this stored form.
    
    The factor of 2 accounts for the periodic extension assumption.
    The FFT computes all coefficients in O(m log m) time and O(m) memory,
    avoiding a large harmonics-by-samples matrix for high-detail drawings.
    """
    values = np.asarray(signal, dtype=float)
    m = len(values)
    if m == 0:
        raise ValueError("signal must contain at least one sample")
    if n_harmonics < 0:
        raise ValueError("n_harmonics must be non-negative")
    if m < 2 * n_harmonics + 1:
        # Warn if we might have aliasing issues (Nyquist criterion)
        import warnings
        warnings.warn(
            f"Signal has only {m} samples for {n_harmonics} harmonics. "
            f"Consider using at least {2 * n_harmonics + 1} samples to avoid aliasing."
        )
    
    a0 = 2.0 * float(np.mean(values))
    spectrum = np.fft.rfft(values)
    available = min(n_harmonics, len(spectrum) - 1)
    a = np.zeros(n_harmonics, dtype=float)
    b = np.zeros(n_harmonics, dtype=float)
    if available:
        # np.fft uses exp(-i*k*t), so negate its imaginary part to match
        # the positive-sine coefficient convention used by this module.
        a[:available] = 2.0 * spectrum[1 : available + 1].real / m
        b[:available] = -2.0 * spectrum[1 : available + 1].imag / m
        # The Nyquist term of an even-length signal has no sine component
        # and is not doubled in the real Fourier series.
        if m % 2 == 0 and available == m // 2:
            a[available - 1] *= 0.5
            b[available - 1] = 0.0

    return FourierSeries(a0=a0, a=a, b=b)


def chain_xy(
    series: FourierSeries,
    t: float,
    n: int,
    origin: complex,
    rotate90: bool = False,
    max_vectors: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Largest-radius-first chain of the first n harmonics, as complex points
    starting at `origin`. Set rotate90=True to draw the chain vertically
    (used for the y-axis series so the two chains don't overlap visually).
    """
    radii = series.epicycle_radii(n)
    order = np.argsort(-radii)
    if max_vectors is not None:
        order = order[:max(0, int(max_vectors))]
    pts = [origin]
    used_r = []
    z = origin
    rot = 1j if rotate90 else 1 + 0j
    for idx in order:
        k = int(idx) + 1
        ak, bk = float(series.a[idx]), float(series.b[idx])
        z = z + rot * (ak - 1j * bk) * np.exp(1j * k * t)
        pts.append(z)
        used_r.append(float(radii[idx]))
    return np.asarray(pts), np.asarray(used_r)
