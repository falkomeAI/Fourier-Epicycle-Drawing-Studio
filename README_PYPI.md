# 🌀 falkourier

[![PyPI Version](https://img.shields.io/pypi/v/falkourier.svg?color=blue)](https://pypi.org/project/falkourier/)
[![Python Versions](https://img.shields.io/pypi/pyversions/falkourier.svg)](https://pypi.org/project/falkourier/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/falkomeAI/Fourier-Epicycle-Drawing-Studio/blob/main/LICENSE)
[![Creator](https://img.shields.io/badge/Creator-@falkomeai-rose.svg)](https://instagram.com/falkomeai)
[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fourier-studio.streamlit.app/)
[![GitHub Stars](https://img.shields.io/github/stars/falkomeAI/Fourier-Epicycle-Drawing-Studio?style=social)](https://github.com/falkomeAI/Fourier-Epicycle-Drawing-Studio)

![falkourier Epicycle Preview](https://raw.githubusercontent.com/falkomeAI/Fourier-Epicycle-Drawing-Studio/main/assets/preview_epicycle_studio.png)

> **High-precision Fourier Series Epicycle Drawing Engine & Animation Studio** by [@falkomeai](https://instagram.com/falkomeai).
>
> Transform any portrait, sketch, or mathematical curve into a continuous closed tour drawn dynamically by rotating epicycle chains with exact real-time projection formulas, harmonic counters, and video export.

---

### 🌐 Live Interactive Demos
- 🚀 **Full Python Web Studio**: [https://fourier-studio.streamlit.app](https://fourier-studio.streamlit.app/) *(Up to 4096 harmonics, video export & audio sync)*
- ⚡ **In-Browser DFT Canvas**: [https://falkomeAI.github.io/Fourier-Epicycle-Drawing-Studio/](https://falkomeAI.github.io/Fourier-Epicycle-Drawing-Studio/) *(Real-time client-side drawing)*
- 📂 **GitHub Source**: [falkomeAI/Fourier-Epicycle-Drawing-Studio](https://github.com/falkomeAI/Fourier-Epicycle-Drawing-Studio)

---

## ⚡ Installation

Install the core library and CLI tools directly from PyPI:

```bash
pip install falkourier
```

For AI-based background removal and segmentation features:
```bash
pip install "falkourier[ai]"
```

---

## 🚀 Instant Quickstart

### 1. Launch the Interactive Studio UI (No Code Required)
With `falkourier` installed, run one command in your terminal to start the full graphical studio:

```bash
falkourier-studio
```
*Opens an interactive web application in your browser with real-time harmonic sliders, live previews, audio synchronization, and 60fps MP4 video rendering.*

---

### 2. Generate Epicycle Videos from Terminal (CLI)
Generate high-resolution Fourier drawing videos directly from your command line:

```bash
# Render from an image (square format, 500 harmonics)
falkourier --image portrait.jpg --n 500 --save output/portrait.mp4

# Render vertical 9:16 Reel / TikTok format (1080x1920)
falkourier --shape hummingbird --n 300 --format reels --save output/reel.mp4

# Render from built-in curves (hummingbird, heart, star, butterfly, etc.)
falkourier --shape heart --n 200 --save output/heart.mp4
```

---

### 3. Python API Quickstart

Use `falkourier` programmatically in your Python scripts, Jupyter notebooks, or generative art pipelines:

```python
import falkourier as fk

# 1. Load a built-in mathematical curve or extract from an image
path = fk.SHAPES["hummingbird"]()
# Or from an image:
# path = fk.extract_boundary_contour_path("portrait.jpg", target_points=4096)

# 2. Fit Dual-Chain Real Fourier Series (X & Y independent axes)
fit = fk.fit_fourier(path[:, 0], path[:, 1], n_harmonics=500)
print(f"X Fourier Coefficients: {len(fit['x_coeffs']['an'])} harmonics")
print(f"Y Fourier Coefficients: {len(fit['y_coeffs']['cn'])} harmonics")

# 3. Fit Single-Chain Complex Fourier Series (Rotating Vectors)
complex_fit = fk.fit_complex_fourier(path[:, 0], path[:, 1], n_harmonics=500)
print(f"Computed {len(complex_fit['harmonics'])} rotating epicycles ranked by radius.")
```

---

## 📐 Mathematical Decomposition

![Fourier Mathematical Formulation](https://raw.githubusercontent.com/falkomeAI/Fourier-Epicycle-Drawing-Studio/main/assets/math_formula_banner.png)

A continuous closed parametric curve $\gamma(t) = (x(t), y(t))$ for $t \in [0, 2\pi]$ is decomposed into harmonic frequencies:

### Real Fourier Series
$$x_N(t) = \frac{a_0}{2} + \sum_{k=1}^N \left( a_k \cos(kt) + b_k \sin(kt) \right)$$
$$y_N(t) = \frac{c_0}{2} + \sum_{k=1}^N \left( c_k \cos(kt) + d_k \sin(kt) \right)$$

### Complex Exponential Series
$$z_N(t) = \sum_{k=-N}^N c_k e^{i k t}, \quad c_k = \frac{1}{2\pi} \int_0^{2\pi} z(t) e^{-i k t} dt$$

Rotating circle tips trace the curve with **> 99.99% mathematical accuracy** at high harmonic counts ($N \ge 1024$).

---

## ✨ Core Capabilities

- **Ultra-High Harmonic Range**: Compute up to $N = 4096$ harmonics for photo-accurate detail.
- **Continuous Closed Tour Chaining**: Intelligent KD-tree algorithms connect disconnected contours into a single continuous drawing path.
- **Dual X/Y & Complex Single-Chain Modes**: Visualize independent horizontal/vertical projection chains or single rotating vector arms.
- **Video Outro Comparison**: Synthesizes end cards comparing original input against the reconstructed Fourier drawing with exact fit metrics.
- **Audio Looping**: Automatic background soundtrack synchronization during video creation.

![Outro Comparison Frame](https://raw.githubusercontent.com/falkomeAI/Fourier-Epicycle-Drawing-Studio/main/assets/preview_comparison_endcard.png)

---

## 👤 Creator & Attribution

Created & maintained by **[@falkomeai](https://instagram.com/falkomeai)**.

- 📸 Instagram: [https://instagram.com/falkomeai](https://instagram.com/falkomeai)
- 🐙 GitHub: [https://github.com/falkomeAI/Fourier-Epicycle-Drawing-Studio](https://github.com/falkomeAI/Fourier-Epicycle-Drawing-Studio)

Licensed under the [MIT License](https://github.com/falkomeAI/Fourier-Epicycle-Drawing-Studio/blob/main/LICENSE).
