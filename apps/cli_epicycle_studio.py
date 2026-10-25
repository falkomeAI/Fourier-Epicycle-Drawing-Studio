#!/usr/bin/env python3
"""
Dual-chain portrait X/Y Fourier Drawing Generator
Clean high-resolution epicycle drawing animation.

Features:
- Two epicycle chains: X (top-right) and Y (center-left)
- Multiple nested visible circles per chain
- L-shaped dashed guides meeting at the pen
- Approximation error percentage displayed
- Filled silhouette background
- 720x1280 portrait format (Instagram Reels)
- Final frame: just drawing + large formula

Usage:
  python apps/cli_epicycle_studio.py --image character.png --n 256 --save output/epicycle_studio.mp4
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from fourier_drawing.series import fit_fourier, chain_xy
from fourier_drawing.image_path import extract_path_from_image
from fourier_drawing.schedule import get_n_schedule, normalize_path, sample_points_for_n
from fourier_drawing.shapes import SHAPES


# ── Colour theme ─────────────────────────────────────────────────────────────
BG       = "#000000"
GOLD     = "#c9a227"
TRAIL    = "#d97757"
GUIDE    = "#666666"
SIL_FILL = "#1a1a2e"
SIL_EDGE = "#d97757"
TXT      = "#e0e0e0"
N_CLR    = "#ef4444"
ERR_CLR  = "#909090"

MAX_VIS = 8            # visible circles per chain
MAX_CIRCLE_R = 0.45    # cap display radius so one circle can't dominate


def get_ffmpeg_executable() -> str:
    """Resolve ffmpeg executable from imageio_ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _encoder(ffmpeg_exe: str = "ffmpeg"):
    for enc, args in [("libx264", ["-preset", "slow", "-crf", "18"]),
                      ("libopenh264", ["-b:v", "5000k"]),
                      ("mpeg4", ["-q:v", "3"])]:
        try:
            r = subprocess.run([ffmpeg_exe, "-encoders"], capture_output=True, text=True, timeout=10)
            if enc in r.stdout:
                return enc, args
        except Exception:
            pass
    return "libx264", ["-preset", "slow", "-crf", "18"]


def _approx_error(n, n_max):
    if n >= n_max:
        return 0.1
    return max(0.1, min(99.9, 60.0 * np.exp(-0.03 * n)))


def render(
    points: np.ndarray,
    n_harmonics: int = 256,
    fps: int = 30,
    frames_per_n: int = 50,
    final_hold_seconds: int = 5,
    output_path: str = "epicycle_studio.mp4",
    show_formula: bool = True,
    show_silhouette: bool = True,
    watermark: str = "",
):
    # ── Fit ────────────────────────────────────────────────────────────────
    pts = normalize_path(points, target_span=1.6)
    fx = fit_fourier(pts[:, 0], n_harmonics)
    fy = fit_fourier(pts[:, 1], n_harmonics)
    x0, y0 = fx.a0 / 2, fy.a0 / 2

    # ── Layout (portrait 9:16) ──────
    #   Drawing right-of-centre, mid-height.
    #   X-chain (fx) at TOP — rotate90=True so circles stack vertically.
    #     Origin shares X with drawing → tip drifts near pen_x.
    #     Vertical guide: (pen_x, x_tip.imag) → (pen_x, pen_y).
    #   Y-chain (fy) at LEFT — rotate90=False so circles spread horizontally.
    #     Origin shares Y with drawing → tip drifts near pen_y.
    #     Horizontal guide: (y_tip.real, pen_y) → (pen_x, pen_y).
    #   Circle centres stay FIXED (no per-frame shifting).
    draw_cx =  0.50       # drawing centre X
    draw_cy = -0.15       # drawing centre Y

    # Fixed chain origins (never moved)
    x_origin = complex(draw_cx, 1.95)   # X-chain at top
    y_origin = complex(-1.20, draw_cy)  # Y-chain at left

    # ── Schedule ──────────────────────────────────────────────────────────
    schedule = get_n_schedule(n_harmonics)
    n_of, t_of = [], []
    for i, n in enumerate(schedule):
        laps = max(1, int(final_hold_seconds * fps / frames_per_n)) if i == len(schedule) - 1 else 1
        for _ in range(laps):
            for t in np.linspace(0, 2 * np.pi, frames_per_n, endpoint=True):
                n_of.append(n)
                t_of.append(float(t))
    n_of = np.array(n_of, dtype=int)
    t_of = np.array(t_of, dtype=float)
    total = len(n_of)
    end_card = fps * 3

    print(f"Schedule: {schedule[0]} -> {schedule[-1]}")
    print(f"Frames: {total} + {end_card} end-card = {(total+end_card)/fps:.1f}s @ {fps}fps")

    # ── Figure ────────────────────────────────────────────────────────────
    W, H, dpi = 7.2, 12.8, 100
    fig, ax = plt.subplots(figsize=(W, H), facecolor=BG, dpi=dpi)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-3.2, 3.2)
    ax.axis("off")

    # Silhouette
    if show_silhouette:
        sp = pts.copy()
        sp[:, 0] += draw_cx
        sp[:, 1] += draw_cy
        ax.add_patch(Polygon(sp, closed=True,
                             facecolor=SIL_FILL, edgecolor=SIL_EDGE,
                             lw=0.8, alpha=0.45, zorder=1))

    # Formula
    if show_formula:
        ftxt = ax.text(
            -1.65, 2.7,
            r"$x_N(t)=\sum_{k=1}^{N}(a_k\cos kt+b_k\sin kt)$" + "\n\n" +
            r"$y_N(t)=\sum_{k=1}^{N}(c_k\cos kt+d_k\sin kt)$",
            color=TXT, fontsize=10, family="serif",
            va="top", ha="left", linespacing=1.6, alpha=0.85)

    if watermark:
        ax.text(1.65, 2.9, watermark, color=TXT, fontsize=9, ha="right", va="top", alpha=0.6)

    n_pre = ax.text(0.0, -2.55, "N = ", color=TXT, fontsize=30,
                    fontweight="bold", ha="right", va="center")
    n_lbl = ax.text(0.0, -2.55, "1", color=N_CLR, fontsize=30,
                    fontweight="bold", ha="left", va="center")
    err_lbl = ax.text(0.0, -2.90, "", color=ERR_CLR, fontsize=11, ha="center")

    # Circles
    x_circs = [Circle((0,0), .01, fill=False, ec=GOLD, lw=1.3, alpha=.7, visible=False) for _ in range(MAX_VIS)]
    y_circs = [Circle((0,0), .01, fill=False, ec=GOLD, lw=1.3, alpha=.7, visible=False) for _ in range(MAX_VIS)]
    for c in x_circs + y_circs:
        ax.add_patch(c)

    x_arm, = ax.plot([], [], color=GOLD, lw=1.0, alpha=0.85)
    y_arm, = ax.plot([], [], color=GOLD, lw=1.0, alpha=0.85)
    v_guide, = ax.plot([], [], "--", color=GUIDE, lw=0.9, alpha=0.55)
    h_guide, = ax.plot([], [], "--", color=GUIDE, lw=0.9, alpha=0.55)
    trail, = ax.plot([], [], color=TRAIL, lw=2.2, solid_capstyle="round", zorder=10)

    tx_i, = ax.plot([], [], "o", color="#fff", ms=5, zorder=22)
    tx_o, = ax.plot([], [], "o", color=GOLD, ms=9, alpha=.45, zorder=21)
    ty_i, = ax.plot([], [], "o", color="#fff", ms=5, zorder=22)
    ty_o, = ax.plot([], [], "o", color=GOLD, ms=9, alpha=.45, zorder=21)
    tp_i, = ax.plot([], [], "o", color="#fff", ms=5, zorder=25)
    tp_o, = ax.plot([], [], "o", color=TRAIL, ms=9, alpha=.45, zorder=24)

    tr_x, tr_y = [], []
    prev_n = -1

    tmp = tempfile.mkdtemp(prefix="epicycle_")
    try:
        print("Rendering...")
        for frame in range(total):
            n = int(n_of[frame])
            t = float(t_of[frame])

            if n != prev_n:
                tr_x.clear(); tr_y.clear()
                prev_n = n
                n_lbl.set_text(str(n))

            # Pen = Fourier evaluation placed at drawing centre
            pen_x = fx.evaluate(t, n) - x0 + draw_cx
            pen_y = fy.evaluate(t, n) - y0 + draw_cy

            # X-chain (TOP, vertical stack) — FIXED origin, no shifting
            xc, xr = chain_xy(fx, t, n, x_origin, rotate90=True)
            # Y-chain (LEFT, horizontal spread) — FIXED origin, no shifting
            yc, yr = chain_xy(fy, t, n, y_origin, rotate90=False)

            x_tip = xc[-1]   # tip of X-chain
            y_tip = yc[-1]   # tip of Y-chain

            # Cap radii for display
            xr_d = np.minimum(xr, MAX_CIRCLE_R)
            yr_d = np.minimum(yr, MAX_CIRCLE_R)

            # X circles (fixed centres — only arm rotates)
            ns = min(n, MAX_VIS, len(xc) - 1)
            for i, c in enumerate(x_circs):
                if i < ns and xr_d[i] > 0.004:
                    c.set_visible(True)
                    c.center = (xc[i].real, xc[i].imag)
                    c.set_radius(float(xr_d[i]))
                    c.set_alpha(min(0.8, 0.35 + float(xr_d[i])))
                else:
                    c.set_visible(False)

            # Y circles (fixed centres)
            ns2 = min(n, MAX_VIS, len(yc) - 1)
            for i, c in enumerate(y_circs):
                if i < ns2 and yr_d[i] > 0.004:
                    c.set_visible(True)
                    c.center = (yc[i].real, yc[i].imag)
                    c.set_radius(float(yr_d[i]))
                    c.set_alpha(min(0.8, 0.35 + float(yr_d[i])))
                else:
                    c.set_visible(False)

            x_arm.set_data(xc.real, xc.imag)
            y_arm.set_data(yc.real, yc.imag)

            # L-shaped guides:
            #   vertical at x=pen_x from X-tip height → pen
            #   horizontal at y=pen_y from Y-tip position → pen
            v_guide.set_data([pen_x, pen_x], [x_tip.imag, pen_y])
            h_guide.set_data([y_tip.real, pen_x], [pen_y, pen_y])

            tr_x.append(pen_x); tr_y.append(pen_y)
            trail.set_data(tr_x, tr_y)

            # Tip dots — X tip, Y tip, pen intersection
            tx_i.set_data([x_tip.real], [x_tip.imag])
            tx_o.set_data([x_tip.real], [x_tip.imag])
            ty_i.set_data([y_tip.real], [y_tip.imag])
            ty_o.set_data([y_tip.real], [y_tip.imag])
            tp_i.set_data([pen_x], [pen_y])
            tp_o.set_data([pen_x], [pen_y])

            err_lbl.set_text(f"approximation error = {_approx_error(n, n_harmonics):.2f}%")

            fig.savefig(Path(tmp) / f"f_{frame:06d}.png", facecolor=BG, dpi=dpi)

            if frame % 100 == 0:
                print(f"  {frame}/{total} ({100*frame//total}%)")

        # ── End card ──────────────────────────────────────────────────────
        print("End card...")
        for c in x_circs + y_circs:
            c.set_visible(False)
        for art in [x_arm, y_arm, v_guide, h_guide,
                    tx_i, tx_o, ty_i, ty_o, tp_i, tp_o]:
            art.set_data([], [])
        for txt in [n_pre, n_lbl, err_lbl]:
            txt.set_visible(False)
        if show_formula:
            ftxt.set_visible(False)

        ax.text(0, -2.2, r"$x(t)=\sum_{k}(a_k\cos kt+b_k\sin kt)$",
                color=TXT, fontsize=15, family="serif", ha="center", alpha=0.9)
        ax.text(0, -2.65, r"$y(t)=\sum_{k}(c_k\cos kt+d_k\sin kt)$",
                color="#808080", fontsize=13, family="serif", ha="center", alpha=0.7)

        for i in range(end_card):
            fig.savefig(Path(tmp) / f"f_{total+i:06d}.png", facecolor=BG, dpi=dpi)

        plt.close(fig)

        # ── Encode ────────────────────────────────────────────────────────
        print("Encoding...")
        ffmpeg_exe = get_ffmpeg_executable()
        enc, ea = _encoder(ffmpeg_exe)
        cmd = [ffmpeg_exe, "-y", "-framerate", str(fps),
               "-i", str(Path(tmp) / "f_%06d.png"),
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", enc, *ea, "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(f"ffmpeg error: {r.stderr}")
            raise RuntimeError("ffmpeg failed")
        print(f"Saved: {output_path}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description="Dual-chain Fourier drawing animation")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", default="")
    src.add_argument("--shape", choices=sorted(SHAPES), default="hummingbird")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--points", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames-per-n", type=int, default=50)
    p.add_argument("--hold", type=int, default=5)
    p.add_argument("--method",
                   choices=["auto","boundary","adaptive","otsu","grabcut","canny","rembg","alpha",
                            "lineart","allcontours","ai","ai_anime"],
                   default="ai")
    p.add_argument("--save", default="output/epicycle_studio.mp4")
    p.add_argument("--no-formula", action="store_true")
    p.add_argument("--no-silhouette", action="store_true")
    p.add_argument("--watermark", default="")
    args = p.parse_args()

    n_pts = args.points or sample_points_for_n(args.n)

    if args.image:
        pts = extract_path_from_image(args.image, n_points=n_pts, method=args.method)
        print(f"Extracted {len(pts)} pts from {args.image}")
    else:
        pts = SHAPES[args.shape](n_pts)
        print(f"Shape '{args.shape}' — {n_pts} pts")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    render(pts, n_harmonics=args.n, fps=args.fps, frames_per_n=args.frames_per_n,
           final_hold_seconds=args.hold, output_path=args.save,
           show_formula=not args.no_formula, show_silhouette=not args.no_silhouette,
           watermark=args.watermark)


if __name__ == "__main__":
    main()
