#!/usr/bin/env python3
"""
Fourier Drawing Web App — Streamlit UI

Clean two-panel layout:
  LEFT  → upload image + settings
  RIGHT → generated video

Run:  streamlit run apps/web_app.py
"""

import _bootstrap  # noqa: F401  (adds ../src to sys.path)

import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Fourier Drawing Generator",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon, Rectangle

from fourier_drawing.debug_frames import DebugFrameRecorder
from fourier_drawing.image_path import (
    _resample_closed, _mask_grabcut, _mask_otsu, _mask_rembg, _mask_canny,
    extract_boundary_contour_path, path_from_mask_contours,
)
from fourier_drawing.schedule import get_n_schedule, normalize_path, sample_points_for_n
from fourier_drawing.series import fit_fourier, chain_xy
from fourier_drawing.complex_series import fit_complex_fourier, sorted_harmonics
from fourier_drawing.shapes import SHAPES

# Hard ceiling for harmonics. Higher values (1000+) produce more detail but
# take longer to render. 4096 provides ultra-high precision.
N_MAX_CAP = 4096
N_QUICK = 250      # Quick mode default
N_DETAILED = 1024  # Detailed mode
FOURIER_PRESET = Path(__file__).resolve().parents[1] / "assets" / "fourier-points.json"


def load_fourier_preset() -> np.ndarray:
    """Load the built-in preset outline point set."""
    if not FOURIER_PRESET.is_file():
        raise FileNotFoundError(f"Built-in preset outline not found: {FOURIER_PRESET}")
    points = np.asarray(json.loads(FOURIER_PRESET.read_text()), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Built-in preset outline must contain [x, y] pairs")
    if not np.all(np.isfinite(points)):
        raise ValueError("Built-in preset outline contains invalid coordinates")
    return points


AUDIO_DIR = Path(__file__).resolve().parents[1] / "audio"


def get_available_audio_tracks() -> list[Path]:
    """Scan the audio folder for available background music tracks."""
    if not AUDIO_DIR.is_dir():
        return []
    tracks = []
    for ext in ("*.mp3", "*.wav", "*.m4a", "*.aac", "*.ogg"):
        tracks.extend(AUDIO_DIR.glob(ext))
    return sorted(tracks, key=lambda p: p.name)


# ──────────────────────────────────────────────────────────────────────────────
# Theme — light, clean, minimal
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Theme — Modern Studio Dark Glassmorphism
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Theme — Ultra-Clean Minimal Monochromatic (Black & White, Zero-Scroll)
# ──────────────────────────────────────────────────────────────────────────────

ACCENT = "#0f172a"       # Deep Obsidian
ACCENT_2 = "#1e293b"     # Slate Charcoal
BG_0 = "#ffffff"         # Pure Clean White
BG_1 = "#fafafa"         # Subtle Warm White
BG_2 = "#f1f5f9"         # Slate Light Tint
BORDER = "#e2e8f0"       # Delicate Crisp Gray
BORDER_STRONG = "#cbd5e1"
TEXT = "#0f172a"         # Deep Crisp Ink
TEXT_DIM = "#64748b"     # Muted Neutral Slate

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    /* Reset & Viewport Lock */
    html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        color: {TEXT};
    }}
    
    .stApp {{
        background-color: {BG_0};
        color: {TEXT};
    }}

    #MainMenu, header, footer {{ display: none !important; }}

    /* Centered clean workspace container */
    .block-container {{
        padding: 0.8rem 1.8rem 1rem 1.8rem !important;
        max-width: 1340px !important;
        margin: 0 auto !important;
    }}

    /* Minimal Compact Top Header */
    .top-navbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: {BG_0};
        border-bottom: 1px solid {BORDER};
        padding: 0.35rem 0 0.5rem 0;
        margin-bottom: 0.8rem;
    }}
    .nav-brand {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .nav-brand-title {{
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: {TEXT};
    }}
    .nav-brand-title span {{
        font-weight: 400;
        color: {TEXT_DIM};
    }}
    .nav-badge {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.18rem 0.5rem;
        border-radius: 4px;
        background: {BG_2};
        color: {TEXT_DIM};
        border: 1px solid {BORDER};
    }}

    /* Balanced Two-Panel Columns */
    div[data-testid="stHorizontalBlock"] {{
        gap: 1.5rem !important;
        align-items: flex-start !important;
    }}

    /* Compact Section Header */
    .section-title {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {TEXT_DIM};
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin: 0.6rem 0 0.35rem 0;
    }}
    .section-title .badge-num {{
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: {TEXT};
        color: #ffffff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.64rem;
        font-weight: 700;
    }}

    /* Form Controls: Clean natural spacing with no overlapping */
    div[data-testid="stSelectbox"] {{
        margin-bottom: 0.5rem !important;
    }}
    div[data-testid="stSelectbox"] > div > div {{
        background: {BG_1} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        color: {TEXT} !important;
        font-size: 0.88rem !important;
        transition: border 0.15s ease !important;
    }}
    div[data-testid="stSelectbox"] > div > div:hover {{
        border-color: {TEXT} !important;
    }}

    /* Compact File Uploader */
    div[data-testid="stFileUploader"] {{
        background: {BG_1};
        border: 1px dashed {BORDER_STRONG};
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }}
    div[data-testid="stFileUploader"] section {{
        padding: 0.4rem 0 !important;
    }}

    /* Expander */
    div[data-testid="stExpander"] {{
        background: {BG_1} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
        box-shadow: none !important;
    }}
    div[data-testid="stExpander"] summary {{
        font-size: 0.82rem !important;
        padding: 0.4rem 0.6rem !important;
        font-weight: 600 !important;
    }}

    /* Primary and Secondary Buttons */
    .stButton {{
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {TEXT} !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: 1px solid {TEXT} !important;
        border-radius: 8px !important;
        padding: 0.45rem 1rem !important;
        min-height: 2.35rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
        transition: all 0.15s ease !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: #1e293b !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1) !important;
    }}

    .stButton > button:not([kind="primary"]) {{
        background: {BG_1} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 7px !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        min-height: 2.1rem !important;
        padding: 0.3rem 0.65rem !important;
        transition: all 0.15s ease !important;
    }}
    .stButton > button:not([kind="primary"]):hover {{
        background: {BG_2} !important;
        border-color: {TEXT_DIM} !important;
    }}

    /* Chips */
    .chip-row {{
        display: flex;
        gap: 0.35rem;
        margin: 0.35rem 0 0.5rem;
        flex-wrap: wrap;
    }}
    .chip {{
        background: {BG_1};
        border: 1px solid {BORDER};
        border-radius: 5px;
        padding: 0.2rem 0.5rem;
        font-size: 0.72rem;
        color: {TEXT_DIM};
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }}
    .chip b {{
        color: {TEXT};
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* Status Indicator */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.2rem 0.6rem;
        border: 1px solid {BORDER};
        background: {BG_1};
        border-radius: 9999px;
        color: {TEXT};
        font-size: 0.72rem;
        font-weight: 600;
    }}
    .status-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: {TEXT};
    }}

    /* Workspace / Preview State - Perfectly fitted to viewport */
    .empty-state {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 280px;
        max-height: 290px;
        color: {TEXT_DIM};
        text-align: center;
        gap: 0.35rem;
        background: {BG_1};
        border: 1px dashed {BORDER};
        border-radius: 8px;
    }}
    .empty-state .glyph {{
        font-size: 1.5rem;
        color: {TEXT_DIM};
        opacity: 0.6;
    }}
    .empty-state .msg {{
        font-size: 0.86rem;
        color: {TEXT};
        max-width: 300px;
        font-weight: 600;
    }}
    .empty-state .sub-msg {{
        font-size: 0.74rem;
        color: {TEXT_DIM};
        max-width: 300px;
        line-height: 1.35;
    }}

    /* Workspace Preview & Video - Crisp Portrait Display */
    div[data-testid="stImage"] img,
    div[data-testid="stImage"] {{
        max-height: 520px !important;
        object-fit: contain !important;
        margin: 0 auto !important;
    }}
    div[data-testid="stPlotlyChart"],
    div.element-container:has(> iframe),
    div[data-testid="element-container"]:has(img) {{
        max-height: 540px !important;
    }}
    div[data-testid="element-container"] img {{
        max-height: 520px !important;
        object-fit: contain !important;
        display: block;
        margin: 0 auto;
    }}

    /* Video player container - sleek dark backdrop for orthogonal view */
    div[data-testid="stVideo"] {{
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.35rem;
        background: #0b0f14;
        display: flex;
        justify-content: center;
    }}
    video {{
        border-radius: 6px;
        max-height: 520px !important;
        object-fit: contain;
        margin: 0 auto;
    }}

    div[data-testid="stDownloadButton"] button {{
        background: {TEXT} !important;
        color: #ffffff !important;
        border: 1px solid {TEXT} !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 0.8rem !important;
    }}

    .stSlider {{
        margin-top: 0.2rem !important;
        margin-bottom: 0.4rem !important;
    }}
    div[data-testid="stVerticalBlock"] {{
        gap: 0.65rem !important;
    }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Image processing
# ──────────────────────────────────────────────────────────────────────────────

MAX_IMAGE_DIMENSION = 4096  # Maximum allowed image dimension
MAX_IMAGE_PIXELS = 16_000_000  # Maximum total pixels (~4000x4000)
SOURCE_IMAGE_MAX_PX = 1400
SOURCE_IMAGE_DATA_BUDGET = 2_500_000


def prepare_uploaded_image(image_bytes: bytes) -> tuple[bytes, bool]:
    """Return raw uploaded bytes without resizing so all fine edges remain preserved."""
    return image_bytes, False


def make_preview_copy(image: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    """Create a high-fidelity display copy of an image, preserving crisp visible edges."""
    if image is None or image.size == 0:
        raise ValueError("preview image is empty")
    copy_img = image.copy()
    h, w = copy_img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        copy_img = cv2.resize(
            copy_img,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_LINEAR,
        )
    return copy_img


def render_connected_contour_image(
    points: np.ndarray,
    width: int = 1200,
    height: int = 1200,
    dark_mode: bool = True,
) -> np.ndarray:
    """Render high-resolution connected contour image: ONLY contour, NOT color photo."""
    pts = np.asarray(points, dtype=float).copy()
    min_x, min_y = pts[:, 0].min(), pts[:, 1].min()
    max_x, max_y = pts[:, 0].max(), pts[:, 1].max()
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    span = max(span_x, span_y)

    margin = 80
    scale = (min(width, height) - 2 * margin) / span
    cx = (width - span_x * scale) / 2
    cy = (height - span_y * scale) / 2

    px = np.rint(cx + (pts[:, 0] - min_x) * scale).astype(np.int32)
    py = np.rint(height - (cy + (pts[:, 1] - min_y) * scale)).astype(np.int32)
    pixel_pts = np.column_stack([px, py]).reshape(-1, 1, 2)

    if dark_mode:
        canvas = np.full((height, width, 3), (20, 15, 11), dtype=np.uint8)  # #0b0f14 in BGR
        line_color = (248, 189, 56)  # Glowing cyan #38bdf8 in BGR
    else:
        canvas = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
        line_color = (42, 23, 15)  # Dark slate #0f172a in BGR

    cv2.polylines(canvas, [pixel_pts], isClosed=True, color=line_color, thickness=2, lineType=cv2.LINE_AA)
    return canvas


@st.cache_resource(show_spinner=False)
def load_lightweight_segmentation_model():
    """Load the small COCO instance-segmentation model once per app process."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Lightweight AI segmentation needs Ultralytics. Install it with: pip install ultralytics"
        ) from exc
    # Lightweight pretrained instance-segmentation model.
    return YOLO("yolov8n-seg.pt")


def lightweight_ai_mask(bgr: np.ndarray, confidence: float = 0.25) -> tuple[np.ndarray, str]:
    """Segment the largest recognized object, returning its mask and label."""
    model = load_lightweight_segmentation_model()
    # This environment's torchvision wheel exposes the NMS operator schema but
    # has no CPU kernel. Ultralytics ships an equivalent PyTorch NMS fallback;
    # use it when torchvision is present so CPU inference works with this build.
    try:
        import torchvision
        from ultralytics.utils.nms import TorchNMS

        torchvision.ops.nms = TorchNMS.nms
    except (ImportError, AttributeError):
        # Ultralytics itself falls back to TorchNMS when torchvision is absent.
        pass
    result = model.predict(
        source=bgr, imgsz=640, conf=confidence, device="cpu",
        retina_masks=True, verbose=False,
    )[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        raise ValueError(
            "The lightweight model found no COCO object. Try a lower confidence or another extraction method."
        )

    masks = result.masks.data.detach().cpu().numpy()
    h, w = bgr.shape[:2]
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    index = int(np.argmax(areas))
    mask = masks[index]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (mask >= 0.5).astype(np.uint8) * 255
    class_id = int(result.boxes.cls[index].item())
    confidence_score = float(result.boxes.conf[index].item())
    class_name = result.names.get(class_id, str(class_id))
    return binary, f"{class_name} ({confidence_score:.0%} confidence)"


@st.cache_data(show_spinner=False, max_entries=8)
def extract_contour_from_bytes(
    img_bytes: bytes, method: str, n_points: int = 512,
    invert: bool = False, min_contour_len: int = 8, max_contours: int = 500,
    ai_confidence: float = 0.25,
):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image")

    # High-detail boundary contour extraction with nearest-contour chaining and arc resampling.
    # Keep the original alpha channel until the helper has composited it.
    if method == "boundary":
        points, preview_bgr, mask, contour_preview, contour_count = extract_boundary_contour_path(
            img, n_points=n_points, invert=invert, line_art=True,
            min_len=min_contour_len, max_shapes=max_contours,
        )
        return points, preview_bgr, mask, contour_preview, contour_count, ""

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = None
    elif img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        alpha = img[:, :, 3]
    else:
        bgr = img
        alpha = None

    # AI path (neural lineart + closed tour)
    if method in ("ai", "ai_anime"):
        import tempfile
        from fourier_drawing.image_path import extract_path_from_image
        suffix = ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            cv2.imwrite(tmp.name, bgr)
            tmp_path = tmp.name
        try:
            pts = extract_path_from_image(tmp_path, n_points=n_points, method=method)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        # Fake mask for UI preview (AI path has no binary mask)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )
        # Draw on copy_and_resize image, taking reference always from original unresized image
        h_orig, w_orig = bgr.shape[:2]
        scale = min(1.0, 1200.0 / max(h_orig, w_orig))
        disp_w = max(1, int(round(w_orig * scale)))
        disp_h = max(1, int(round(h_orig * scale)))
        preview = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        scaled_cnts = [np.rint(c.astype(float) * scale).astype(np.int32) for c in contours]
        cv2.drawContours(preview, scaled_cnts, -1, (0, 220, 120), 2, lineType=cv2.LINE_AA)
        return pts, bgr, mask, preview, len(contours), "Neural line-art path"

    detected_label = ""
    if method == "ai_lite":
        try:
            mask, detected_label = lightweight_ai_mask(bgr, confidence=ai_confidence)
        except Exception:
            mask = _mask_otsu(bgr, invert=True)
            detected_label = "Fallback: Binary threshold"
    elif method == "alpha" and alpha is not None:
        mask = alpha
    elif method == "grabcut":
        mask = _mask_grabcut(bgr)
    elif method == "rembg":
        try:
            mask = _mask_rembg(bgr)
        except Exception:
            mask = _mask_otsu(bgr, invert=True)
    elif method == "canny":
        mask = _mask_canny(bgr)
    elif method == "adaptive":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )
    else:
        mask = _mask_otsu(bgr, invert=True)

    if invert:
        mask = 255 - np.asarray(mask, dtype=np.uint8)

    pts, traced_contours = path_from_mask_contours(
        mask, n_points, min_len=min_contour_len, max_contours=max_contours,
    )
    # Draw on copy_and_resize image, taking reference always from original unresized image
    h, w = bgr.shape[:2]
    scale = min(1.0, 1200.0 / max(h, w))
    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))
    preview = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    cv_contours = [np.rint(c * scale).astype(np.int32).reshape(-1, 1, 2) for c in traced_contours]
    cv2.drawContours(preview, cv_contours, -1, (0, 220, 120), 2, lineType=cv2.LINE_AA)
    return pts, bgr, mask, preview, len(traced_contours), detected_label


# ──────────────────────────────────────────────────────────────────────────────
# Video generation
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Fourier math logic (complex FFT & harmonics)
# ──────────────────────────────────────────────────────────────────────────────

def compute_fourier_terms(shape: np.ndarray, max_arms: int | None = None) -> list[dict]:
    """Computes complex FFT on centered points z = x + 1j * y, extracts amplitude,
    phase, and signed frequency, then sorts descending by amplitude.
    """
    pts = np.asarray(shape, dtype=float)
    n = len(pts)
    z = pts[:, 0] + 1j * pts[:, 1]
    fft_vals = np.fft.fft(z) / n
    terms = []
    for k in range(n):
        c = fft_vals[k]
        amp = float(np.hypot(c.real, c.imag))
        phase = float(np.arctan2(c.imag, c.real))
        freq = int(k if k <= n // 2 else k - n)
        terms.append({"amp": amp, "phase": phase, "freq": freq})
    terms.sort(key=lambda t: t["amp"], reverse=True)
    if max_arms is not None:
        terms = terms[:max_arms]
    return terms


def get_arm_chain(terms: list[dict], t: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns cumulative joint positions (shape: (active_n + 1, 2)) and radii."""
    active_n = min(n, len(terms))
    joints = np.zeros((active_n + 1, 2), dtype=float)
    radii = np.zeros(active_n, dtype=float)
    x, y = 0.0, 0.0
    for i in range(active_n):
        tm = terms[i]
        ang = 2.0 * np.pi * tm["freq"] * t + tm["phase"]
        x += tm["amp"] * np.cos(ang)
        y += tm["amp"] * np.sin(ang)
        joints[i + 1, 0] = x
        joints[i + 1, 1] = y
        radii[i] = tm["amp"]
    return joints, radii


def get_pen_at(terms: list[dict], t: float, n: int) -> tuple[float, float]:
    """Evaluate pen tip position at time t with n harmonics."""
    active_n = min(n, len(terms))
    x, y = 0.0, 0.0
    for i in range(active_n):
        tm = terms[i]
        ang = 2.0 * np.pi * tm["freq"] * t + tm["phase"]
        x += tm["amp"] * np.cos(ang)
        y += tm["amp"] * np.sin(ang)
    return x, y


def plot_epicycle_preview(points: np.ndarray, n: int = 50) -> tuple[plt.Figure, float]:
    """Generate the static epicycle preview in dual X/Y projection style."""
    pts = normalize_path(points, target_span=2.0)
    effective_n = min(int(n), len(pts))
    if effective_n < 1:
        effective_n = 1

    fx = fit_fourier(pts[:, 0], effective_n)
    fy = fit_fourier(pts[:, 1], effective_n)

    draw_cx, draw_cy = 0.8, -0.6
    x0, y0 = fx.a0 / 2, fy.a0 / 2
    x_origin = complex(draw_cx, 2.4)   # Top X-chain
    y_origin = complex(-2.2, draw_cy)  # Left Y-chain

    BG_COLOR = "#0d1117"
    X_CHAIN_COLOR = "#5eead4"
    Y_CHAIN_COLOR = "#fbbf24"
    TRAIL_COLOR = "#ef4444"
    GUIDE_X_COLOR = "#5eead4"
    GUIDE_Y_COLOR = "#fbbf24"
    SILHOUETTE_COLOR = "#1e293b"
    SILHOUETTE_EDGE = "#475569"
    TIP_COLOR = "#ffffff"
    N_COLOR = "#f97316"

    fig, ax = plt.subplots(figsize=(5.4, 9.6), facecolor=BG_COLOR, dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-5.5, 5.5)
    ax.axis("off")

    # Silhouette in bottom-right drawing zone
    sil_pts = pts.copy()
    sil_pts[:, 0] += draw_cx
    sil_pts[:, 1] += draw_cy
    ax.add_patch(Polygon(
        sil_pts, closed=True,
        facecolor=SILHOUETTE_COLOR, edgecolor=SILHOUETTE_EDGE,
        lw=1.5, alpha=0.6, zorder=1
    ))

    # Formulas on left (cyan for X, gold for Y)
    ax.text(
        -3.2, 5.0,
        r"$x_N(t) = \sum_{k=1}^N (a_k \cos kt + b_k \sin kt)$",
        color=X_CHAIN_COLOR, fontsize=10.5, family="serif", va="top", ha="left"
    )
    ax.text(
        -3.2, 4.3,
        r"$y_N(t) = \sum_{k=1}^N (c_k \cos kt + d_k \sin kt)$",
        color=Y_CHAIN_COLOR, fontsize=10.5, family="serif", va="top", ha="left"
    )

    # Instagram handle on top right
    ax.text(
        3.2, 5.0,
        "@falkomeai",
        color="#cbd5e1", fontsize=12, fontweight="bold",
        ha="right", va="top", alpha=0.9
    )

    # Calculate reconstruction error (subsampling uniformly across the closed loop)
    step = max(1, len(pts) // 1024)
    pts_sub = pts[::step]
    t_eval = np.linspace(0, 2 * np.pi, len(pts), endpoint=False)[::step]
    rx_eval = np.array([fx.evaluate(ti, effective_n) for ti in t_eval])
    ry_eval = np.array([fy.evaluate(ti, effective_n) for ti in t_eval])
    max_dev = float(np.max(np.hypot(rx_eval - pts_sub[:, 0], ry_eval - pts_sub[:, 1])))
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-6)
    err_pct = (max_dev / span) * 100.0

    # Large N counter and Error at bottom
    err_text = "Error: < 0.01%" if err_pct < 0.01 else f"Error: {err_pct:.2f}%"
    ax.text(
        0.0, -4.55, f"N = {effective_n}",
        color=N_COLOR, fontsize=42, fontweight="bold",
        ha="center", va="center"
    )
    ax.text(
        0.0, -5.15, err_text,
        color="#94a3b8", fontsize=15, fontweight="bold",
        ha="center", va="center"
    )

    t = 0.75 * 2 * np.pi
    xc, rx = chain_xy(fx, t, effective_n, x_origin, rotate90=False)
    yc, ry = chain_xy(fy, t, effective_n, y_origin, rotate90=True)
    pen_x = fx.evaluate(t, effective_n) - x0 + draw_cx
    pen_y = fy.evaluate(t, effective_n) - y0 + draw_cy

    # Draw X circles and arm line (top, cyan)
    max_vis = min(effective_n, 32)
    for i in range(max_vis):
        if rx[i] > 0.01:
            ax.add_patch(Circle((xc[i].real, xc[i].imag), rx[i], fill=False, ec=X_CHAIN_COLOR, lw=0.9, alpha=0.55))
    ax.plot(xc[:max_vis + 1].real, xc[:max_vis + 1].imag, color=X_CHAIN_COLOR, lw=1.2, alpha=0.9)

    # Draw Y circles and arm line (left, gold)
    for i in range(max_vis):
        if ry[i] > 0.01:
            ax.add_patch(Circle((yc[i].real, yc[i].imag), ry[i], fill=False, ec=Y_CHAIN_COLOR, lw=0.9, alpha=0.55))
    ax.plot(yc[:max_vis + 1].real, yc[:max_vis + 1].imag, color=Y_CHAIN_COLOR, lw=1.2, alpha=0.9)

    # Dashed guide lines projecting to pen
    ax.plot([pen_x, pen_x], [xc[-1].imag, pen_y], "--", color=GUIDE_X_COLOR, lw=1.2, alpha=0.7)
    ax.plot([yc[-1].real, pen_x], [pen_y, pen_y], "--", color=GUIDE_Y_COLOR, lw=1.2, alpha=0.7)

    # Complete reconstructed loop in subtle glow
    t_full = np.linspace(0, 2 * np.pi, 600)
    full_x = [fx.evaluate(ti, effective_n) - x0 + draw_cx for ti in t_full]
    full_y = [fy.evaluate(ti, effective_n) - y0 + draw_cy for ti in t_full]
    ax.plot(full_x, full_y, color=TRAIL_COLOR, lw=1.4, alpha=0.3, zorder=5)

    # Drawn trail up to t
    t_draw = np.linspace(0, t, 400)
    trail_x = [fx.evaluate(ti, effective_n) - x0 + draw_cx for ti in t_draw]
    trail_y = [fy.evaluate(ti, effective_n) - y0 + draw_cy for ti in t_draw]
    ax.plot(trail_x, trail_y, color=TRAIL_COLOR, lw=2.6, solid_capstyle="round", zorder=10)

    # Glowing tip dot at pen
    ax.plot([pen_x], [pen_y], "o", color=TIP_COLOR, ms=7, zorder=20)
    ax.plot([pen_x], [pen_y], "o", color=TRAIL_COLOR, ms=14, alpha=0.4, zorder=19)

    return fig, err_pct


def plot_comparison_endcard(
    points: np.ndarray,
    input_image: np.ndarray | None,
    n: int = 50,
) -> tuple[plt.Figure, float]:
    """Generate the comparison end-card: up side sketch image, down side drawing image."""
    pts = normalize_path(points, target_span=2.0)
    effective_n = min(int(n), len(pts))
    if effective_n < 1:
        effective_n = 1

    fx = fit_fourier(pts[:, 0], effective_n)
    fy = fit_fourier(pts[:, 1], effective_n)
    x0, y0 = fx.a0 / 2, fy.a0 / 2

    # Subsampled error calculation
    step = max(1, len(pts) // 1024)
    pts_sub = pts[::step]
    t_eval = np.linspace(0, 2 * np.pi, len(pts), endpoint=False)[::step]
    rx_eval = np.array([fx.evaluate(ti, effective_n) for ti in t_eval])
    ry_eval = np.array([fy.evaluate(ti, effective_n) for ti in t_eval])
    max_dev = float(np.max(np.hypot(rx_eval - pts_sub[:, 0], ry_eval - pts_sub[:, 1])))
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-6)
    err_pct = (max_dev / span) * 100.0
    err_text = "< 0.01%" if err_pct < 0.01 else f"{err_pct:.2f}%"
    fit_text = "> 99.99%" if err_pct < 0.01 else f"{100.0 - err_pct:.2f}%"

    BG_COLOR = "#0d1117"
    SILHOUETTE_COLOR = "#1e293b"
    SILHOUETTE_EDGE = "#475569"
    TRAIL_COLOR = "#ef4444"

    fig, ax = plt.subplots(figsize=(5.4, 9.6), facecolor=BG_COLOR, dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-5.5, 5.5)
    ax.axis("off")

    # Watermark on top right
    ax.text(3.2, 5.15, "@falkomeai", color="#cbd5e1", fontsize=12, fontweight="bold", ha="right", va="top", alpha=0.9)

    # Subtle divider
    ax.plot([-3.1, 3.1], [0.0, 0.0], color="#334155", lw=1.2, linestyle="--", alpha=0.5)

    # ─── TOP: UP SIDE SKETCH IMAGE ───
    ax.text(0.0, 4.95, "ORIGINAL SKETCH", color="#94a3b8", fontsize=14, fontweight="bold", ha="center", va="top")

    if input_image is not None and input_image.size > 0:
        h, w = input_image.shape[:2]
        aspect = w / h
        max_box_w, max_box_h = 4.8, 3.8
        if aspect > (max_box_w / max_box_h):
            disp_w = max_box_w
            disp_h = max_box_w / aspect
        else:
            disp_h = max_box_h
            disp_w = max_box_h * aspect

        cx, cy = 0.0, 2.5
        extent = [cx - disp_w / 2, cx + disp_w / 2, cy - disp_h / 2, cy + disp_h / 2]
        rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb, extent=extent, aspect="auto", zorder=2)
        rect = Rectangle((extent[0], extent[2]), disp_w, disp_h, fill=False, edgecolor="#475569", lw=1.5, zorder=3)
        ax.add_patch(rect)
    else:
        # Plot original ground-truth contour as line sketch
        top_pts = pts.copy()
        top_span = max(float(np.ptp(top_pts[:, 0])), float(np.ptp(top_pts[:, 1])), 1e-6)
        scale_top = 3.6 / top_span
        top_pts[:, 0] = (top_pts[:, 0] - float(np.mean(top_pts[:, 0]))) * scale_top
        top_pts[:, 1] = (top_pts[:, 1] - float(np.mean(top_pts[:, 1]))) * scale_top + 2.5
        ax.add_patch(Polygon(top_pts, closed=True, facecolor="#161f2e", edgecolor="#64748b", lw=1.8, alpha=0.8, zorder=2))
        ax.plot(top_pts[:, 0], top_pts[:, 1], color="#38bdf8", lw=1.5, alpha=0.9, zorder=3)

    # ─── BOTTOM: DOWN SIDE DRAWING IMAGE ───
    ax.text(0.0, -0.25, f"FOURIER DRAWING  (N = {effective_n})", color="#f97316", fontsize=14, fontweight="bold", ha="center", va="top")

    t_full = np.linspace(0, 2 * np.pi, 1000)
    draw_x = np.array([fx.evaluate(ti, effective_n) - x0 for ti in t_full])
    draw_y = np.array([fy.evaluate(ti, effective_n) - y0 for ti in t_full])
    bot_span = max(float(np.ptp(draw_x)), float(np.ptp(draw_y)), 1e-6)
    scale_bot = 3.6 / bot_span
    draw_x = (draw_x - float(np.mean(draw_x))) * scale_bot
    draw_y = (draw_y - float(np.mean(draw_y))) * scale_bot - 2.8

    bot_poly = np.column_stack([draw_x, draw_y])
    ax.add_patch(Polygon(bot_poly, closed=True, facecolor=SILHOUETTE_COLOR, edgecolor=SILHOUETTE_EDGE, lw=1.2, alpha=0.6, zorder=2))
    ax.plot(draw_x, draw_y, color=TRAIL_COLOR, lw=2.4, solid_capstyle="round", zorder=4)
    ax.plot(draw_x, draw_y, color=TRAIL_COLOR, lw=5.5, alpha=0.25, solid_capstyle="round", zorder=3)

    # Footer metrics
    ax.text(0.0, -5.05, f"Reconstruction: {fit_text} Fit  •  Error: {err_text}", color="#94a3b8", fontsize=12, fontweight="bold", ha="center", va="center")

    return fig, err_pct


def generate_sidebyside_video(
    points: np.ndarray,
    input_image: np.ndarray | None,
    n_max: int,
    fps: int,
    frames_per_n: int,
    final_hold_loops: int = 2,
    progress_callback=None,
    debug_frames: bool = False,
    debug_name: str = "run",
    progressive: bool = False,
    audio_path: str | Path | None = None,
):
    """Render the epicycle drawing video in dual X/Y projection style."""
    import subprocess
    import shutil
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"

    pts = normalize_path(points, target_span=2.0)
    effective_n = min(int(n_max), len(pts))
    if effective_n < 1:
        raise ValueError("No Fourier arms are available for this path")

    fx = fit_fourier(pts[:, 0], effective_n)
    fy = fit_fourier(pts[:, 1], effective_n)

    draw_cx, draw_cy = 0.8, -0.6
    x0, y0 = fx.a0 / 2, fy.a0 / 2
    x_origin = complex(draw_cx, 2.4)   # Top X-chain
    y_origin = complex(-2.2, draw_cy)  # Left Y-chain

    if progressive:
        schedule = get_n_schedule(effective_n)
        n_of_frame, t_of_frame = [], []
        for i, n in enumerate(schedule):
            laps = final_hold_loops if i == len(schedule) - 1 else 1
            for _ in range(laps):
                for t in np.linspace(0, 2 * np.pi, frames_per_n, endpoint=False):
                    n_of_frame.append(n)
                    t_of_frame.append(float(t))
    else:
        # Draw directly at chosen N arms from t=0 to 1
        schedule = [effective_n]
        draw_frames = max(240, frames_per_n * 8)
        n_of_frame = [effective_n] * draw_frames
        t_of_frame = list(np.linspace(0, 2 * np.pi, draw_frames, endpoint=True))
        hold_frames = fps * max(1, final_hold_loops)
        n_of_frame += [effective_n] * hold_frames
        t_of_frame += [2 * np.pi] * hold_frames

    n_of_frame = np.array(n_of_frame)
    t_of_frame = np.array(t_of_frame)
    total_frames = len(n_of_frame)

    temp_dir = tempfile.mkdtemp(prefix="fourier_xy_")

    try:
        BG_COLOR = "#0d1117"
        X_CHAIN_COLOR = "#5eead4"
        Y_CHAIN_COLOR = "#fbbf24"
        TRAIL_COLOR = "#ef4444"
        GUIDE_X_COLOR = "#5eead4"
        GUIDE_Y_COLOR = "#fbbf24"
        SILHOUETTE_COLOR = "#1e293b"
        SILHOUETTE_EDGE = "#475569"
        TIP_COLOR = "#ffffff"
        N_COLOR = "#f97316"

        fig, ax = plt.subplots(figsize=(5.4, 9.6), facecolor=BG_COLOR, dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_facecolor(BG_COLOR)
        ax.set_aspect("equal")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-5.5, 5.5)
        ax.axis("off")

        # Silhouette
        sil_pts = pts.copy()
        sil_pts[:, 0] += draw_cx
        sil_pts[:, 1] += draw_cy
        ax.add_patch(Polygon(
            sil_pts, closed=True,
            facecolor=SILHOUETTE_COLOR, edgecolor=SILHOUETTE_EDGE,
            lw=1.5, alpha=0.6, zorder=1
        ))

        # Formulas
        ax.text(
            -3.2, 5.0,
            r"$x_N(t) = \sum_{k=1}^N (a_k \cos kt + b_k \sin kt)$",
            color=X_CHAIN_COLOR, fontsize=10.5, family="serif", va="top", ha="left"
        )
        ax.text(
            -3.2, 4.3,
            r"$y_N(t) = \sum_{k=1}^N (c_k \cos kt + d_k \sin kt)$",
            color=Y_CHAIN_COLOR, fontsize=10.5, family="serif", va="top", ha="left"
        )

        # Instagram handle on top right
        ax.text(
            3.2, 5.0,
            "@falkomeai",
            color="#cbd5e1", fontsize=12, fontweight="bold",
            ha="right", va="top", alpha=0.9
        )

        # Precompute error for each unique N in schedule (subsampling uniformly across the closed loop)
        span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-6)
        step = max(1, len(pts) // 1024)
        pts_sub = pts[::step]
        t_eval = np.linspace(0, 2 * np.pi, len(pts), endpoint=False)[::step]
        error_by_n = {}
        for un in np.unique(n_of_frame):
            rx_eval = np.array([fx.evaluate(ti, int(un)) for ti in t_eval])
            ry_eval = np.array([fy.evaluate(ti, int(un)) for ti in t_eval])
            dev = float(np.max(np.hypot(rx_eval - pts_sub[:, 0], ry_eval - pts_sub[:, 1])))
            error_by_n[int(un)] = (dev / span) * 100.0

        # N counter and Error
        init_n = int(n_of_frame[0])
        init_err = error_by_n.get(init_n, 0.0)
        init_err_str = "Error: < 0.01%" if init_err < 0.01 else f"Error: {init_err:.2f}%"
        n_label = ax.text(
            0.0, -4.55,
            f"N = {init_n}",
            color=N_COLOR, fontsize=42, fontweight="bold",
            ha="center", va="center"
        )
        err_label = ax.text(
            0.0, -5.15,
            init_err_str,
            color="#94a3b8", fontsize=15, fontweight="bold",
            ha="center", va="center"
        )

        max_vis = min(effective_n, 32)
        x_circles = [
            Circle((0, 0), 0.01, fill=False, ec=X_CHAIN_COLOR, lw=0.9, alpha=0.55, visible=False)
            for _ in range(max_vis)
        ]
        y_circles = [
            Circle((0, 0), 0.01, fill=False, ec=Y_CHAIN_COLOR, lw=0.9, alpha=0.55, visible=False)
            for _ in range(max_vis)
        ]
        for c in x_circles: ax.add_patch(c)
        for c in y_circles: ax.add_patch(c)

        x_arm_line, = ax.plot([], [], color=X_CHAIN_COLOR, lw=1.2, alpha=0.9)
        y_arm_line, = ax.plot([], [], color=Y_CHAIN_COLOR, lw=1.2, alpha=0.9)
        v_guide, = ax.plot([], [], "--", color=GUIDE_X_COLOR, lw=1.2, alpha=0.7)
        h_guide, = ax.plot([], [], "--", color=GUIDE_Y_COLOR, lw=1.2, alpha=0.7)
        trail_line, = ax.plot([], [], color=TRAIL_COLOR, lw=2.6, solid_capstyle="round", zorder=10)
        tip_dot, = ax.plot([], [], "o", color=TIP_COLOR, ms=7, zorder=20)
        tip_glow, = ax.plot([], [], "o", color=TRAIL_COLOR, ms=14, alpha=0.4, zorder=19)

        trail_x, trail_y = [], []
        prev_n = -1
        prev_t = 0.0
        substeps = 16

        for frame in range(total_frames):
            n = int(n_of_frame[frame])
            t = float(t_of_frame[frame])

            if n != prev_n:
                trail_x.clear()
                trail_y.clear()
                prev_n = n
                prev_t = 0.0
                n_label.set_text(f"N = {n}")
                curr_err = error_by_n.get(n, 0.0)
                err_label.set_text("Error: < 0.01%" if curr_err < 0.01 else f"Error: {curr_err:.2f}%")

            xc, rx = chain_xy(fx, t, n, x_origin, rotate90=False)
            yc, ry = chain_xy(fy, t, n, y_origin, rotate90=True)
            pen_x = fx.evaluate(t, n) - x0 + draw_cx
            pen_y = fy.evaluate(t, n) - y0 + draw_cy

            v_count = min(n, max_vis)
            for i in range(max_vis):
                if i < n and rx[i] > 0.01:
                    x_circles[i].set_visible(True)
                    x_circles[i].center = (xc[i].real, xc[i].imag)
                    x_circles[i].set_radius(rx[i])
                else:
                    x_circles[i].set_visible(False)

                if i < n and ry[i] > 0.01:
                    y_circles[i].set_visible(True)
                    y_circles[i].center = (yc[i].real, yc[i].imag)
                    y_circles[i].set_radius(ry[i])
                else:
                    y_circles[i].set_visible(False)

            x_arm_line.set_data(xc[:v_count + 1].real, xc[:v_count + 1].imag)
            y_arm_line.set_data(yc[:v_count + 1].real, yc[:v_count + 1].imag)
            v_guide.set_data([pen_x, pen_x], [xc[-1].imag, pen_y])
            h_guide.set_data([yc[-1].real, pen_x], [pen_y, pen_y])

            # Sub-steps for ultra-smooth continuous trail
            if t > prev_t:
                sub_t = np.linspace(prev_t, t, substeps + 1)[1:]
                sub_x = [fx.evaluate(st, n) - x0 + draw_cx for st in sub_t]
                sub_y = [fy.evaluate(st, n) - y0 + draw_cy for st in sub_t]
                trail_x.extend(sub_x)
                trail_y.extend(sub_y)
                prev_t = t
            elif frame == 0 or not trail_x:
                trail_x.append(pen_x)
                trail_y.append(pen_y)
                prev_t = t

            trail_line.set_data(trail_x, trail_y)
            tip_dot.set_data([pen_x], [pen_y])
            tip_glow.set_data([pen_x], [pen_y])

            frame_path = Path(temp_dir) / f"frame_{frame:06d}.png"
            fig.savefig(frame_path, facecolor=BG_COLOR, dpi=100)

            if progress_callback and (frame % 15 == 0 or frame == total_frames - 1):
                progress_callback(min(frame / (total_frames + int(fps * 2.5)) * 0.90, 0.90))

        plt.close(fig)

        # ── End comparison card (up side sketch image, down side drawing image) ──
        endcard_frames = int(fps * 2.5)  # 2.5 second comparison card
        fig_end, _ = plot_comparison_endcard(points, input_image, effective_n)
        endcard_path = Path(temp_dir) / "endcard.png"
        fig_end.savefig(endcard_path, facecolor=BG_COLOR, dpi=100)
        plt.close(fig_end)

        for ec_idx in range(endcard_frames):
            frame_num = total_frames + ec_idx
            shutil.copyfile(endcard_path, Path(temp_dir) / f"frame_{frame_num:06d}.png")

        all_total_frames = total_frames + endcard_frames

        output_path = Path(temp_dir) / "output.mp4"
        frame_pattern = str(Path(temp_dir) / "frame_%06d.png")

        # Resolve audio track to loop
        actual_audio = None
        if audio_path is not None and Path(audio_path).is_file():
            actual_audio = Path(audio_path).resolve()
        elif audio_path is not False:  # None means default to first available track in audio/
            avail = get_available_audio_tracks()
            if avail:
                actual_audio = avail[0].resolve()

        video_dur = all_total_frames / max(1, fps)
        fade_start = max(0.0, video_dur - 1.2)
        af_filter = f"afade=t=out:st={fade_start:.2f}:d=1.2"

        if actual_audio is not None and actual_audio.is_file():
            ffmpeg_cmd = [
                ffmpeg_exe, "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-stream_loop", "-1",
                "-i", str(actual_audio),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-af", af_filter,
                "-shortest",
                "-movflags", "+faststart",
                str(output_path)
            ]
        else:
            ffmpeg_cmd = [
                ffmpeg_exe, "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path)
            ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0 and actual_audio is not None:
            # Fallback without audio if audio stream failed
            fallback_cmd = [
                ffmpeg_exe, "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path)
            ]
            result = subprocess.run(fallback_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed (exit code {result.returncode}): {result.stderr}")

        if progress_callback:
            progress_callback(0.98)

        data = output_path.read_bytes()

        if progress_callback:
            progress_callback(1.0)

        return data, schedule, all_total_frames, error_by_n.get(effective_n, 0.0)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)




# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    col_left, col_right = st.columns([1.0, 1.45], gap="large")

    # ── Persistent session state ──────────────────────────────────────────
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state["uploaded_image_bytes"] = None
    if "generated_video" not in st.session_state:
        st.session_state["generated_video"] = None

    # ── LEFT: INPUT ──────────────────────────────────────────────────────
    with col_left:
        st.markdown(
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">'
            '<span class="section-title" style="margin:0;">SOURCE</span>'
            '<a href="https://instagram.com/falkomeai" target="_blank" style="text-decoration:none;font-size:0.75rem;font-weight:600;color:#e11d48;background:#fff1f2;border:1px solid #ffe4e6;padding:0.15rem 0.5rem;border-radius:9999px;">📸 @falkomeai</a>'
            '</div>',
            unsafe_allow_html=True,
        )

        source = st.selectbox(
            "Source", ["Preset outline", "Upload image", "Built-in shape"],
            label_visibility="collapsed",
            key="source_selector",
        )

        points = None
        input_img = None
        img_bytes = None
        contour_preview = None
        method = "boundary"
        shape_name = None
        invert = False
        min_contour_len = 8
        max_contours = 120
        ai_confidence = 0.25

        if source == "Upload image":
            uploaded = st.file_uploader(
                "Drop an image here",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
                key="image_file_uploader",
            )
            if uploaded is not None:
                try:
                    st.session_state["uploaded_image_bytes"] = uploaded.getvalue()
                except Exception as exc:
                    st.error(f"Could not read this image: {exc}")

            img_bytes = st.session_state.get("uploaded_image_bytes")

            if img_bytes is not None:
                # Use a display copy for thumbnail preview in the sidebar
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    thumb = pil_img.copy()
                    thumb.thumbnail((260, 95), Image.Resampling.LANCZOS)
                    c_thumb, c_btn = st.columns([3, 1])
                    with c_thumb:
                        st.image(thumb, use_container_width=False)
                    with c_btn:
                        if st.button("✕ Remove", help="Clear current image"):
                            st.session_state["uploaded_image_bytes"] = None
                            st.session_state["generated_video"] = None
                            st.rerun()

                method_options = [
                    "boundary", "adaptive", "canny", "otsu", "grabcut", "alpha"
                ]
                method_labels = {
                    "boundary": "⚫ High-Detail Boundary Contours",
                    "adaptive": "✏️ Adaptive Threshold (Line Art)",
                    "canny": "🔍 Edge Detection (Fine Detail)",
                    "otsu": "🌓 Binary Threshold (High Contrast)",
                    "grabcut": "📸 Foreground Subject Isolation",
                    "alpha": "🔲 PNG Alpha Channel",
                }
                try:
                    import ultralytics
                    method_options.append("ai_lite")
                    method_labels["ai_lite"] = "🤖 AI Object Segmentation"
                except ImportError:
                    pass
                try:
                    import rembg
                    method_options.append("rembg")
                    method_labels["rembg"] = "🌐 AI Background Removal"
                except ImportError:
                    pass

                method = st.selectbox(
                    "Extraction method",
                    method_options,
                    index=0,
                    format_func=lambda x: method_labels.get(x, x),
                )

                if method not in ("ai", "ai_anime"):
                    if method != "ai_lite":
                        invert = st.checkbox("Invert foreground/background", value=False)
                    else:
                        ai_confidence = st.slider(
                            "AI detection confidence", 0.05, 0.90, 0.25, 0.05,
                            help="Lower values allow less-certain detections; the largest detected object is traced.",
                        )
                    min_contour_len = st.slider(
                        "Minimum contour length", min_value=3, max_value=100, value=8,
                        help="Lower values retain smaller details and components.",
                    )
                    max_contours = st.slider(
                        "Maximum contours to include", min_value=1, max_value=500, value=120,
                        help="More contours preserve more separate details.",
                    )
            else:
                st.markdown(
                    '<div class="empty-state" style="height:120px;">'
                    '<div class="glyph">🖼️</div>'
                    '<div class="msg">Upload a PNG, JPG, or WEBP</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                sample_path = Path(__file__).resolve().parents[1] / "assets" / "sample_portrait.jpg"
                if sample_path.is_file():
                    if st.button("📷 Load sample photo", use_container_width=True, help="Load sample photo"):
                        st.session_state["uploaded_image_bytes"] = sample_path.read_bytes()
                        st.session_state["generated_video"] = None
                        st.rerun()

        elif source == "Built-in shape":
            shape_name = st.selectbox("Shape", list(SHAPES.keys()), label_visibility="collapsed")
        else:
            try:
                points = load_fourier_preset()
            except Exception as exc:
                st.error(f"Could not load the preset outline: {exc}")

        # Uploaded image workflow: cached extraction
        if source == "Upload image" and img_bytes is not None:
            try:
                preview_n_points = 4096
                with st.spinner("Processing image contours…"):
                    points, bgr, mask, contour_preview, contour_count, detected_label = extract_contour_from_bytes(
                        img_bytes, method, n_points=preview_n_points, invert=invert,
                        min_contour_len=min_contour_len, max_contours=max_contours,
                        ai_confidence=ai_confidence,
                    )
                input_img = bgr
            except Exception as exc:
                points = None
                st.error(f"Contour extraction error: {exc}")

        # Shapes and Presets
        if source != "Upload image":
            preview_n_points = sample_points_for_n(N_MAX_CAP)
            if source == "Built-in shape":
                points = SHAPES[shape_name](preview_n_points)
            elif source == "Preset outline" and points is not None:
                points = _resample_closed(points, preview_n_points)

        st.markdown(
            '<div class="section-title" style="margin-top:0.75rem;">SETTINGS</div>',
            unsafe_allow_html=True,
        )

        quality_preset = st.selectbox(
            "Quality preset",
            ["⚡ Quick (N=100)", "✨ Detailed (N=500)", "🎯 High Detail (N=1024)", "🔥 Ultra (N=4000)"],
            index=3,
            label_visibility="collapsed",
            help="Higher N = more detail but longer render time",
        )

        preset_map = {
            "⚡ Quick (N=100)": (100, 40, 2, 30),
            "✨ Detailed (N=500)": (500, 45, 2, 30),
            "🎯 High Detail (N=1024)": (1024, 45, 3, 30),
            "🔥 Ultra (N=4000)": (4000, 35, 3, 30),
        }
        preset_n, preset_fpn, preset_hold, preset_fps = preset_map[quality_preset]

        anim_mode = st.radio(
            "Animation style",
            ["🎯 Complete Drawing at N", "📈 Progressive Evolution (1 → N)"],
            index=0,
            help="Complete Drawing traces the full image directly using N arms from start to end (fully fitting the drawing). Progressive ramps N in steps.",
        )
        is_progressive = "Progressive" in anim_mode

        with st.expander("⚙️ Advanced parameters"):
            c1, c2 = st.columns(2)
            with c1:
                n_max = st.slider("Fourier arms (N)", 10, N_MAX_CAP, preset_n, 10)
            with c2:
                fps = st.select_slider("FPS", [15, 20, 24, 30], value=preset_fps)
            frames_per_n = st.slider("Frames per N step", 20, 60, preset_fpn)
            final_hold_loops = st.slider("Final hold (laps)", 1, 5, preset_hold)

        available_audio = get_available_audio_tracks()
        enable_audio = st.checkbox("🎵 Background music (in loop)", value=True, help="Loop audio from audio directory to match video length")
        selected_audio_path = None
        if enable_audio and available_audio:
            if len(available_audio) == 1:
                selected_audio_path = available_audio[0]
                st.caption(f"Audio track: **{selected_audio_path.name}**")
            else:
                track_map = {t.name: t for t in available_audio}
                chosen_track = st.selectbox("Audio track", list(track_map.keys()), index=0)
                selected_audio_path = track_map[chosen_track]

        n_points = sample_points_for_n(n_max)

        if source == "Upload image" and points is not None and len(points) != n_points:
            points = _resample_closed(points, n_points)
        elif source == "Built-in shape":
            points = SHAPES[shape_name](n_points)
        elif source == "Preset outline" and points is not None:
            points = _resample_closed(points, n_points)

        display_points = len(points) if points is not None else n_points
        st.caption(f"Outline: **{display_points:,}** sample points")

        generate_clicked = st.button("▶  Generate video", type="primary", use_container_width=True)

    # ── RIGHT: SINGLE ALIGNED WORKSPACE PANEL ─────────────────────────────
    with col_right:
        # Generation Trigger
        if generate_clicked:
            if points is None:
                st.error("No valid shape or contour found to draw.")
            else:
                effective_n_max = min(n_max, len(points))
                if is_progressive:
                    schedule_preview = get_n_schedule(effective_n_max)
                    est_frames = len(schedule_preview) * frames_per_n + (final_hold_loops - 1) * frames_per_n
                else:
                    schedule_preview = [effective_n_max]
                    draw_frames = max(240, frames_per_n * 8)
                    est_frames = draw_frames + fps * max(1, final_hold_loops)
                est_duration = est_frames / fps

                progress = st.progress(0.0)
                status = st.empty()

                def on_progress(p):
                    progress.progress(min(p, 1.0))
                    status.markdown(
                        f'<span style="color:{TEXT_DIM};font-size:0.78rem;">Rendering frames… {int(p*100)}%</span>',
                        unsafe_allow_html=True,
                    )

                try:
                    with st.spinner("Generating Fourier drawing video…"):
                        video_bytes, schedule, total_frames, final_err = generate_sidebyside_video(
                            points, input_image=input_img, n_max=effective_n_max,
                            fps=fps, frames_per_n=frames_per_n,
                            final_hold_loops=final_hold_loops,
                            progress_callback=on_progress,
                            progressive=is_progressive,
                            audio_path=selected_audio_path if enable_audio else False,
                        )
                    progress.empty()
                    status.empty()
                    st.session_state["generated_video"] = {
                        "bytes": video_bytes,
                        "schedule": schedule,
                        "frames": total_frames,
                        "fps": fps,
                        "n_max": effective_n_max,
                        "duration": round(total_frames / fps, 1),
                        "error": final_err,
                        "audio": selected_audio_path.name if (enable_audio and selected_audio_path) else None,
                    }
                except Exception as e:
                    progress.empty()
                    status.empty()
                    st.error(f"Generation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Workspace Display
        has_video = st.session_state.get("generated_video") is not None
        status_label = "Complete" if has_video else ("Ready" if points is not None else "Empty")
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">'
            f'<div class="section-title" style="margin:0;">PREVIEW & VIDEO</div>'
            f'<div class="status-badge"><span class="status-dot"></span>{status_label}</div></div>',
            unsafe_allow_html=True,
        )

        if has_video:
            vid_data = st.session_state["generated_video"]
            st.video(vid_data["bytes"])
            c_dl, c_reset = st.columns([3, 1])
            with c_dl:
                st.download_button(
                    "⬇  Download MP4",
                    data=vid_data["bytes"],
                    file_name="fourier_drawing.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
            with c_reset:
                if st.button("✕ Reset", help="Clear generated video and return to preview", use_container_width=True):
                    st.session_state["generated_video"] = None
                    st.rerun()

            v_err = vid_data.get("error", 0.0)
            err_str = "< 0.01%" if v_err < 0.01 else f"{v_err:.2f}%"
            fit_str = "> 99.99%" if v_err < 0.01 else f"{100.0 - v_err:.2f}%"
            st.markdown(
                f'<div class="chip-row">'
                f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                f'Fourier arms: <b style="color:#f97316;font-size:0.95rem;">N = {vid_data["n_max"]}</b></div>'
                f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                f'Error: <b style="color:#38bdf8;font-size:0.95rem;">{err_str}</b></div>'
                f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                f'Fit: <b style="color:#4ade80;font-size:0.95rem;">{fit_str}</b></div>'
                f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                f'Instagram: <b style="color:#f43f5e;font-size:0.95rem;">@falkomeai</b></div>'
                f'<div class="chip">Frames: <b>{vid_data["frames"]}</b></div>'
                f'<div class="chip">FPS: <b>{vid_data["fps"]}</b></div>'
                f'<div class="chip">Duration: <b>{vid_data["duration"]}s</b></div>'
                f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                f'Audio: <b style="color:#a855f7;font-size:0.95rem;">{"Looped 🎵" if vid_data.get("audio") else "Muted 🔇"}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif points is not None:
            effective_n_prev = min(n_max, len(points))
            preview_n = effective_n_prev
            if source == "Upload image" and contour_preview is not None:
                tab_connected, tab_preview, tab_endcard, tab_contour = st.tabs([
                    "🔗 Connected Contour",
                    "🔭 Epicycle Projection Preview",
                    "🖼️ End Comparison Frame",
                    "📷 Original Photo Overlay",
                ])
                with tab_connected:
                    c_style, _ = st.columns([1.2, 1.8])
                    with c_style:
                        canvas_theme = st.radio(
                            "Canvas style",
                            ["🌙 Dark Canvas", "☀️ Light Canvas"],
                            index=0,
                            horizontal=True,
                            label_visibility="collapsed",
                            key="connected_canvas_theme",
                        )
                    dark_mode = "Dark" in canvas_theme
                    connected_img = render_connected_contour_image(points, dark_mode=dark_mode)
                    st.image(cv2.cvtColor(connected_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption(f"Continuous connected contour path • {len(points):,} points chained in a single loop.")
                with tab_preview:
                    fig_p, p_err = plot_epicycle_preview(points, n=preview_n)
                    st.pyplot(fig_p, use_container_width=True)
                    plt.close(fig_p)
                    p_err_str = "< 0.01%" if p_err < 0.01 else f"{p_err:.2f}%"
                    p_fit_str = "> 99.99%" if p_err < 0.01 else f"{100.0 - p_err:.2f}%"
                    st.markdown(
                        f'<div class="chip-row" style="margin-top:0.5rem;justify-content:center;">'
                        f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                        f'Fourier arms: <b style="color:#f97316;font-size:0.95rem;">N = {preview_n}</b></div>'
                        f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                        f'Reconstruction Error: <b style="color:#38bdf8;font-size:0.95rem;">{p_err_str}</b></div>'
                        f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                        f'Fit: <b style="color:#4ade80;font-size:0.95rem;">{p_fit_str}</b></div>'
                        f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                        f'Instagram: <b style="color:#f43f5e;font-size:0.95rem;">@falkomeai</b></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with tab_endcard:
                    fig_ec, _ = plot_comparison_endcard(points, input_img, n=preview_n)
                    st.pyplot(fig_ec, use_container_width=True)
                    plt.close(fig_ec)
                    st.caption("End comparison frame: Up side sketch image, down side drawing image.")
                with tab_contour:
                    cp = make_preview_copy(contour_preview)
                    st.image(cv2.cvtColor(cp, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                tab_connected, tab_preview, tab_endcard = st.tabs([
                    "🔗 Connected Contour",
                    "🔭 Epicycle Projection Preview",
                    "🖼️ End Comparison Frame",
                ])
                with tab_connected:
                    connected_img = render_connected_contour_image(points, dark_mode=True)
                    st.image(cv2.cvtColor(connected_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption(f"Connected contour path • {len(points):,} points.")
                with tab_preview:
                    fig_p, p_err = plot_epicycle_preview(points, n=preview_n)
                    st.pyplot(fig_p, use_container_width=True)
                    plt.close(fig_p)
                    p_err_str = "< 0.01%" if p_err < 0.01 else f"{p_err:.2f}%"
                    p_fit_str = "> 99.99%" if p_err < 0.01 else f"{100.0 - p_err:.2f}%"
                    st.markdown(
                        f'<div class="chip-row" style="margin-top:0.5rem;justify-content:center;">'
                        f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                        f'Fourier arms: <b style="color:#f97316;font-size:0.95rem;">N = {preview_n}</b></div>'
                        f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                        f'Reconstruction Error: <b style="color:#38bdf8;font-size:0.95rem;">{p_err_str}</b></div>'
                        f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                        f'Fit: <b style="color:#4ade80;font-size:0.95rem;">{p_fit_str}</b></div>'
                        f'<div class="chip" style="font-size:0.8rem;padding:0.3rem 0.65rem;background:#0f172a;color:#f8fafc;border-color:#334155;">'
                        f'Instagram: <b style="color:#f43f5e;font-size:0.95rem;">@falkomeai</b></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with tab_endcard:
                    fig_ec, _ = plot_comparison_endcard(points, None, n=preview_n)
                    st.pyplot(fig_ec, use_container_width=True)
                    plt.close(fig_ec)
                    st.caption("End comparison frame: Up side sketch image, down side drawing image.")

            st.markdown(
                '<div style="margin-top:0.3rem;padding:0.4rem 0.65rem;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;color:#475569;font-size:0.76rem;display:flex;align-items:center;gap:0.4rem;">'
                '<span>✓</span> Path loaded. Click <b>Generate video</b> to synthesize epicycles.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="empty-state">'
                '<div class="glyph">＋</div>'
                '<div class="msg">No Contour or Shape Selected</div>'
                '<div class="sub-msg">Select a preset outline, upload an image, or pick a geometric curve from the left panel.</div>'
                '</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
