# falkourier

[![PyPI](https://img.shields.io/pypi/v/falkourier.svg)](https://pypi.org/project/falkourier/)
[![Python](https://img.shields.io/pypi/pyversions/falkourier.svg)](https://pypi.org/project/falkourier/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/falkomeAI/falkourier/blob/main/LICENSE)
[![Creator](https://img.shields.io/badge/creator-@falkomeai-purple.svg)](https://instagram.com/falkomeai)

![falkourier Preview](https://raw.githubusercontent.com/falkomeAI/falkourier/main/assets/preview_epicycle_studio.png)

`falkourier` is a Python library that draws images, sketches, and curves using Fourier series epicycles (rotating circles). It converts any 2D outline into mathematical harmonic frequencies and animates the drawing process.

Created by [@falkomeai](https://instagram.com/falkomeai).

- **Live Web Studio**: [https://falkourier.streamlit.app/](https://falkourier.streamlit.app/)
- **GitHub Repository**: [falkomeAI/falkourier](https://github.com/falkomeAI/falkourier)

---

## Installation

```bash
pip install falkourier
```

---

## Quick Start

### 1. Web Studio UI
The easiest way to use it is through the interactive web interface:

```bash
falkourier-studio
```

This launches a local web app in your browser where you can upload an image, tune the number of harmonics, preview the epicycles live, and export MP4 videos.

### 2. Command Line
You can also generate videos directly from your terminal:

```bash
# Render a video from an image
falkourier --image portrait.jpg --n 500 --save output.mp4

# Render vertical 9:16 format (for Instagram Reels or TikTok)
falkourier --shape hummingbird --n 300 --format reels --save reel.mp4

# Render from built-in curves
falkourier --shape heart --n 200 --save heart.mp4
```

### 3. Python Code

```python
import falkourier as fk

# 1. Use a built-in curve or extract from an image
path = fk.SHAPES["hummingbird"]()
# path = fk.extract_boundary_contour_path("portrait.jpg", target_points=2048)

# 2. Fit Fourier series
fit = fk.fit_fourier(path[:, 0], path[:, 1], n_harmonics=500)

print(f"X harmonics: {len(fit['x_coeffs']['an'])}")
print(f"Y harmonics: {len(fit['y_coeffs']['cn'])}")
```

---

## Built-in Shapes

You can try out any of the built-in mathematical curves:
`hummingbird`, `heart`, `star`, `butterfly`, `infinity`, `spiral`, `rose`.

```python
import falkourier as fk

curve = fk.SHAPES["butterfly"]()
```

---

## Author & License

- Built by **[@falkomeai](https://instagram.com/falkomeai)**
- Distributed under the [MIT License](https://github.com/falkomeAI/falkourier/blob/main/LICENSE)
