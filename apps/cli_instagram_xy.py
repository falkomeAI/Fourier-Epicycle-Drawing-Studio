#!/usr/bin/env python3
"""
Instagram-optimized X/Y Fourier Drawing Generator
Dual-chain epicycle animation with X and Y projection chains.

Features:
- Two epicycle chains: X (top, cyan) and Y (left, gold)
- Dashed guide lines from chain tips to pen
- Filled silhouette of the original shape
- Both formulas: x_N(t) and y_N(t)
- N counter
- 1080x1080 or 1080x1920 resolution

Usage:
  python apps/cli_instagram_xy.py --image flower.png --n 200 --save output/instagram_xy.mp4
  python apps/cli_instagram_xy.py --shape hummingbird --n 100 --save output/hummingbird_xy.mp4
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


# Dark studio color theme
BG_COLOR = "#0d1117"       # Deep dark blue-black
X_CHAIN_COLOR = "#5eead4"  # Teal/cyan for X chain
Y_CHAIN_COLOR = "#fbbf24"  # Gold/amber for Y chain
TRAIL_COLOR = "#ef4444"    # Red trail
GUIDE_X_COLOR = "#5eead4"  # Teal dashed line
GUIDE_Y_COLOR = "#fbbf24"  # Gold dashed line
SILHOUETTE_COLOR = "#1e293b"  # Dark blue-gray fill
SILHOUETTE_EDGE = "#475569"   # Lighter edge
TIP_COLOR = "#ffffff"
TEXT_COLOR = "#e2e8f0"
N_COLOR = "#f97316"        # Orange for N counter


def get_ffmpeg_executable() -> str:
    """Resolve ffmpeg executable from imageio_ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def get_encoder(ffmpeg_exe: str = "ffmpeg"):
    """Detect available video encoder."""
    encoders = [
        ("libx264", ["-preset", "slow", "-crf", "18"]),
        ("libopenh264", ["-b:v", "5000k"]),
        ("mpeg4", ["-q:v", "3"]),
    ]
    try:
        result = subprocess.run([ffmpeg_exe, "-encoders"], capture_output=True, text=True, timeout=10)
        for encoder, args in encoders:
            if encoder in result.stdout:
                return encoder, args
    except:
        pass
    return "libx264", ["-preset", "slow", "-crf", "18"]


def render_instagram_xy_video(
    points: np.ndarray,
    n_harmonics: int = 100,
    fps: int = 30,
    frames_per_n: int = 50,
    final_hold_seconds: int = 3,
    output_path: str = "instagram_xy.mp4",
    format: str = "square",
    show_formula: bool = True,
    show_silhouette: bool = True,
):
    """Generate Instagram-optimized X/Y Fourier animation."""
    
    # Normalize and fit both X and Y series
    pts = normalize_path(points, target_span=2.0)
    fx = fit_fourier(pts[:, 0], n_harmonics)
    fy = fit_fourier(pts[:, 1], n_harmonics)
    
    # Drawing region center and chain origins
    draw_cx, draw_cy = 0.8, -0.2
    x0 = fx.a0 / 2
    y0 = fy.a0 / 2
    x_origin = complex(draw_cx, 2.2)   # X chain at top
    y_origin = complex(-2.2, draw_cy)  # Y chain at left
    
    # Build schedule
    schedule = get_n_schedule(n_harmonics)
    
    # Build frame timeline
    n_of_frame, t_of_frame = [], []
    for i, n in enumerate(schedule):
        laps = 1
        if i == len(schedule) - 1:
            laps = max(1, int(final_hold_seconds * fps / frames_per_n))
        
        for lap in range(laps):
            for t in np.linspace(0, 2 * np.pi, frames_per_n, endpoint=True):
                n_of_frame.append(n)
                t_of_frame.append(float(t))
    
    n_of_frame = np.array(n_of_frame)
    t_of_frame = np.array(t_of_frame)
    total_frames = len(n_of_frame)
    
    print(f"Schedule: {schedule[0]} -> {schedule[-1]}")
    print(f"Total frames: {total_frames} ({total_frames/fps:.1f}s at {fps}fps)")
    
    # Figure setup
    if format == "reels":
        figsize = (10.8, 19.2)
        dpi = 100
    else:
        figsize = (10.8, 10.8)
        dpi = 100
    
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR, dpi=dpi)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    
    # Axis limits
    if format == "reels":
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-5.5, 5.5)
    else:
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
    ax.axis("off")
    
    # Draw silhouette of original shape (filled)
    if show_silhouette:
        silhouette_pts = pts.copy()
        silhouette_pts[:, 0] += draw_cx
        silhouette_pts[:, 1] += draw_cy
        silhouette = Polygon(
            silhouette_pts, closed=True,
            facecolor=SILHOUETTE_COLOR, edgecolor=SILHOUETTE_EDGE,
            linewidth=1.5, alpha=0.6, zorder=1
        )
        ax.add_patch(silhouette)
    
    # Formula text
    if show_formula:
        formula_y = 3.0 if format == "square" else 5.0
        ax.text(
            -3.2, formula_y,
            r"$x_N(t)=\sum_{k=1}^{N}(a_k\cos kt+b_k\sin kt)$" + "\n\n" +
            r"$y_N(t)=\sum_{k=1}^{N}(c_k\cos kt+d_k\sin kt)$",
            color=TEXT_COLOR, fontsize=11, family="serif",
            va="top", ha="left", linespacing=1.8, alpha=0.9
        )

    # Watermark on top right
    ax.text(
        3.2, 3.0 if format == "square" else 5.0,
        "@falkomeai",
        color="#cbd5e1", fontsize=13, fontweight="bold",
        ha="right", va="top", alpha=0.9
    )

    # N counter and Error - large and prominent
    init_n = int(n_of_frame[0])
    n_label = ax.text(
        0.0, -2.8 if format == "square" else -4.55,
        f"N = {init_n}",
        color=N_COLOR, fontsize=42, fontweight="bold",
        ha="center", va="center"
    )
    err_label = ax.text(
        0.0, -3.2 if format == "square" else -5.15,
        "Error: < 0.01%",
        color="#94a3b8", fontsize=15, fontweight="bold",
        ha="center", va="center"
    )
    
    # Create circles for X chain
    x_circles = []
    for _ in range(n_harmonics):
        c = Circle((0, 0), 0.01, fill=False, ec=X_CHAIN_COLOR, lw=1.2, alpha=0.7, visible=False)
        ax.add_patch(c)
        x_circles.append(c)
    
    # Create circles for Y chain
    y_circles = []
    for _ in range(n_harmonics):
        c = Circle((0, 0), 0.01, fill=False, ec=Y_CHAIN_COLOR, lw=1.2, alpha=0.7, visible=False)
        ax.add_patch(c)
        y_circles.append(c)
    
    # Arm lines
    x_arm_line, = ax.plot([], [], color=X_CHAIN_COLOR, lw=1.5, alpha=0.9)
    y_arm_line, = ax.plot([], [], color=Y_CHAIN_COLOR, lw=1.5, alpha=0.9)
    
    # Guide lines (dashed)
    v_guide, = ax.plot([], [], "--", color=GUIDE_X_COLOR, lw=1.2, alpha=0.6)
    h_guide, = ax.plot([], [], "--", color=GUIDE_Y_COLOR, lw=1.2, alpha=0.6)
    
    # Trail line
    trail_line, = ax.plot([], [], color=TRAIL_COLOR, lw=3.0, solid_capstyle="round", zorder=10)
    
    # Tip dot
    tip_dot, = ax.plot([], [], "o", color=TIP_COLOR, ms=8, zorder=20)
    tip_glow, = ax.plot([], [], "o", color=TRAIL_COLOR, ms=12, alpha=0.4, zorder=19)
    
    trail_x, trail_y = [], []
    prev_n = -1
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="instagram_xy_")
    
    try:
        print("Rendering frames...")
        for frame in range(total_frames):
            n = int(n_of_frame[frame])
            t = float(t_of_frame[frame])
            
            # Clear trail when N changes
            if n != prev_n:
                trail_x.clear()
                trail_y.clear()
                prev_n = n
                n_label.set_text(f"N = {n}")
            
            # Compute X chain (horizontal, at top)
            xc, radii_x = chain_xy(fx, t, n, x_origin, rotate90=False)
            
            # Compute Y chain (vertical, at left)
            yc, radii_y = chain_xy(fy, t, n, y_origin, rotate90=True)
            
            # Get chain tips
            x_tip = xc[-1]
            y_tip = yc[-1]
            
            # Compute pen position
            pen_x = fx.evaluate(t, n) - x0 + draw_cx
            pen_y = fy.evaluate(t, n) - y0 + draw_cy
            
            # Update X circles
            for i in range(n_harmonics):
                if i < n and i < len(radii_x):
                    x_circles[i].set_visible(True)
                    x_circles[i].center = (xc[i].real, xc[i].imag)
                    x_circles[i].set_radius(max(float(radii_x[i]), 1e-6))
                    x_circles[i].set_alpha(max(0.3, min(0.8, float(radii_x[i]) * 1.5)))
                else:
                    x_circles[i].set_visible(False)
            
            # Update Y circles
            for i in range(n_harmonics):
                if i < n and i < len(radii_y):
                    y_circles[i].set_visible(True)
                    y_circles[i].center = (yc[i].real, yc[i].imag)
                    y_circles[i].set_radius(max(float(radii_y[i]), 1e-6))
                    y_circles[i].set_alpha(max(0.3, min(0.8, float(radii_y[i]) * 1.5)))
                else:
                    y_circles[i].set_visible(False)
            
            # Update arm lines
            x_arm_line.set_data(xc.real, xc.imag)
            y_arm_line.set_data(yc.real, yc.imag)
            
            # Update guide lines (from chain tips to pen)
            v_guide.set_data([pen_x, pen_x], [x_tip.imag, pen_y])
            h_guide.set_data([y_tip.real, pen_x], [pen_y, pen_y])
            
            # Update trail
            trail_x.append(pen_x)
            trail_y.append(pen_y)
            trail_line.set_data(trail_x, trail_y)
            
            # Update tip
            tip_dot.set_data([pen_x], [pen_y])
            tip_glow.set_data([pen_x], [pen_y])
            
            # Save frame
            frame_path = Path(temp_dir) / f"frame_{frame:06d}.png"
            fig.savefig(frame_path, facecolor=BG_COLOR, dpi=dpi)
            
            if frame % 50 == 0:
                pct = 100 * frame // total_frames
                print(f"  Frame {frame}/{total_frames} ({pct}%)")
        
        plt.close(fig)

        # End comparison card: Up side sketch image, down side drawing image
        endcard_frames = int(fps * 2.5)
        try:
            from web_app import plot_comparison_endcard
            fig_end, _ = plot_comparison_endcard(points, None, n_harmonics)
            endcard_path = Path(temp_dir) / "endcard.png"
            fig_end.savefig(endcard_path, facecolor=BG_COLOR, dpi=dpi)
            plt.close(fig_end)
            for ec_idx in range(endcard_frames):
                shutil.copyfile(endcard_path, Path(temp_dir) / f"frame_{total_frames + ec_idx:06d}.png")
        except Exception as e:
            print(f"Note: endcard generation skipped ({e})")
        print("Encoding video...")
        ffmpeg_exe = get_ffmpeg_executable()
        encoder, encoder_args = get_encoder(ffmpeg_exe)
        
        # Check for audio in audio directory
        audio_dir = Path(__file__).resolve().parents[1] / "audio"
        audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
        audio_args = []
        if audio_files:
            audio_args = [
                "-stream_loop", "-1",
                "-i", str(audio_files[0].resolve()),
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest"
            ]

        frame_pattern = str(Path(temp_dir) / "frame_%06d.png")
        ffmpeg_cmd = [
            ffmpeg_exe, "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            *audio_args,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", encoder,
            *encoder_args,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            raise RuntimeError("ffmpeg failed")
        
        print(f"Saved: {output_path}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Instagram X/Y Fourier Epicycle Drawing Generator"
    )
    
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", type=str, default="", help="Input image file")
    src.add_argument("--shape", choices=sorted(SHAPES), default="hummingbird")
    
    parser.add_argument("--n", type=int, default=100, help="Max harmonics")
    parser.add_argument("--points", type=int, default=0, help="Sample points (0 = auto)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames-per-n", type=int, default=50)
    parser.add_argument("--hold", type=int, default=3, help="Final hold seconds")
    parser.add_argument("--format", choices=["square", "reels"], default="square")
    parser.add_argument("--method", choices=["auto", "boundary", "adaptive", "otsu", "grabcut", "canny", "rembg", "alpha", "lineart", "ai", "ai_anime"],
                       default="grabcut")
    parser.add_argument("--save", type=str, default="output/instagram_xy.mp4")
    parser.add_argument("--no-formula", action="store_true")
    parser.add_argument("--no-silhouette", action="store_true")
    
    args = parser.parse_args()
    
    # Load points
    n_points = args.points if args.points > 0 else sample_points_for_n(args.n)
    
    if args.image:
        points = extract_path_from_image(
            args.image, n_points=n_points, method=args.method
        )
        print(f"Extracted {len(points)} points from {args.image}")
    else:
        points = SHAPES[args.shape](n_points)
        print(f"Using shape '{args.shape}' with {n_points} points")
    
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    
    render_instagram_xy_video(
        points,
        n_harmonics=args.n,
        fps=args.fps,
        frames_per_n=args.frames_per_n,
        final_hold_seconds=args.hold,
        output_path=args.save,
        format=args.format,
        show_formula=not args.no_formula,
        show_silhouette=not args.no_silhouette,
    )


if __name__ == "__main__":
    main()
