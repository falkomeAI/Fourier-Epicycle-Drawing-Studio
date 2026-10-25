# falkourier

[![PyPI](https://img.shields.io/pypi/v/falkourier.svg?color=blue)](https://pypi.org/project/falkourier/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://falkourier.streamlit.app/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Canvas-2ea44f.svg?logo=github)](https://falkomeAI.github.io/falkourier/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Creator](https://img.shields.io/badge/Creator-@falkomeai-rose.svg)](https://instagram.com/falkomeai)

![falkourier Preview](./assets/preview_epicycle_studio.png)

**falkourier** is a Fourier series epicycle drawing studio and Python library. It decomposes photos, sketches, and mathematical curves into harmonic frequencies and animates the drawing process using synchronized rotating circles.

You can use it through an interactive browser studio, render MP4 videos from the terminal, or import it into your own Python projects.

Created by [@falkomeai](https://instagram.com/falkomeai).

---

## Live Deployments

- **Web Studio**: [https://falkourier.streamlit.app/](https://falkourier.streamlit.app/) *(Full studio with live epicycle simulation, audio looping, and video export)*
- **Interactive Canvas**: [https://falkomeAI.github.io/falkourier/](https://falkomeAI.github.io/falkourier/) *(Real-time in-browser discrete Fourier transform)*
- **PyPI Package**: [https://pypi.org/project/falkourier/](https://pypi.org/project/falkourier/)

---

## Installation

### From PyPI
```bash
pip install falkourier
```

### From Source (for Development)
```bash
git clone https://github.com/falkomeAI/falkourier.git
cd falkourier
pip install -e .
```

*Note: Video export uses `imageio-ffmpeg`, which downloads `ffmpeg` automatically. If running in a minimal Linux container, install `sudo apt install ffmpeg`.*

---

## How It Works

Any closed 2D outline $(x(t), y(t))$ for $t \in [0, 2\pi]$ is decomposed into trigonometric Fourier series:

$$x_N(t) = \frac{a_0}{2} + \sum_{k=1}^N \left( a_k \cos(kt) + b_k \sin(kt) \right)$$

$$y_N(t) = \frac{c_0}{2} + \sum_{k=1}^N \left( c_k \cos(kt) + d_k \sin(kt) \right)$$

Two epicycle chains rotate independently in real time:
- **Top chain (Horizontal, Cyan)**: Computes the $x$-coordinate over time $t$.
- **Left chain (Vertical, Amber)**: Computes the $y$-coordinate over time $t$.
- **Pen Intersection**: Guide lines from both chain tips meet at the drawing pen to trace the path.

```
                  ┌──────────────────────┐
                  │ X-Epicycle Chain (t) │  ──┐
                  └──────────────────────┘    │
                             │                ▼
                             ▼          [ Vertical Guide ]
                   [ Horizontal Guide ]       │
                             │                ▼
┌──────────────────────┐     │          ┌───────────┐
│ Y-Epicycle Chain (t) │ ────┴────────► │  Pen Tip  │ ──► [ Complete Drawing ]
└──────────────────────┘                └───────────┘
```

---

## Quick Start

### 1. Interactive Web Studio
Launch the browser UI with zero configuration:
```bash
falkourier-studio
```
*(Or run `streamlit run apps/web_app.py` if working from a source clone).*

In the studio you can:
- Upload any image (PNG, JPG, WEBP) or pick a preset curve.
- Select contour extraction methods (boundary tracing, adaptive threshold, edge detection).
- Adjust harmonics from $N = 100$ (fast preview) up to $N = 4096$ (high detail).
- Export 60fps MP4 video with background music and side-by-side comparison outro.

---

### 2. Command Line (CLI)
Render videos directly from your terminal:

```bash
# Render from an image (square 1080x1080, 500 harmonics)
falkourier --image assets/sample_portrait.jpg --n 500 --save output/portrait.mp4

# Render vertical 9:16 format for Instagram Reels or TikTok (1080x1920)
falkourier --shape hummingbird --n 300 --format reels --save output/hummingbird_reel.mp4

# Render from built-in mathematical curves
falkourier --shape heart --n 200 --save output/heart.mp4
```

---

### 3. Python API
Use `falkourier` in Python scripts or Jupyter notebooks:

```python
import falkourier as fk

# 1. Use a built-in curve or extract from an image
path = fk.SHAPES["hummingbird"]()
# path = fk.extract_boundary_contour_path("assets/sample_portrait.jpg", target_points=2048)

# 2. Fit real Fourier series for X and Y coordinates
fx = fk.fit_fourier(path[:, 0], n_harmonics=500)
fy = fk.fit_fourier(path[:, 1], n_harmonics=500)
print(f"X series: {len(fx.a)} harmonics, Y series: {len(fy.a)} harmonics")

# 3. Fit single-chain complex Fourier series (rotating vectors)
z = path[:, 0] + 1j * path[:, 1]
complex_coeffs = fk.fit_complex_fourier(z)
print(f"Computed {len(complex_coeffs)} complex frequency harmonics.")
```

---

## Built-in Curves

The library includes pre-computed parametric shapes under `fk.SHAPES`:
- `hummingbird` — Continuous silhouette of a flying hummingbird
- `heart` — Classical algebraic heart curve
- `star` — 5-point star polygon
- `butterfly` — Temple H. Fay's transcendental butterfly curve
- `infinity` — Lemniscate of Bernoulli
- `rose` — Rhodonea rose curve

```python
import falkourier as fk

curve = fk.SHAPES["butterfly"](n_points=1024)
```

---

## Project Structure

```
falkourier/
├── apps/
│   ├── web_app.py              # Streamlit Web Studio UI
│   ├── cli_instagram_xy.py     # Core video rendering engine
│   └── cli_xy.py               # Side-by-side desktop video tool
├── src/
│   ├── falkourier/             # Core pip package
│   │   ├── series.py           # Real two-axis Fourier series
│   │   ├── complex_series.py   # Complex exponential series
│   │   ├── image_path.py       # Boundary tracing & contour chaining
│   │   ├── shapes.py           # Built-in curves
│   │   └── cli.py              # CLI entry points
│   └── fourier_drawing/        # Backwards-compatibility alias
├── assets/                     # Hero banners, sample portrait, presets
├── audio/                      # Audio tracks for video loop
├── tests/                      # Unit tests
├── pyproject.toml              # Packaging metadata (PEP 621)
└── setup.py                    # Build shim
```

---

## Running Tests

Run the test suite using Python's built-in `unittest`:

```bash
python -m unittest discover tests
```

---

## Author & License

- Designed & developed by **[@falkomeai](https://instagram.com/falkomeai)**
- Open source under the [MIT License](LICENSE)
