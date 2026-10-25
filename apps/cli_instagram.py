#!/usr/bin/env python3
"""
Instagram-optimized Fourier Drawing Generator

Creates high-quality, Instagram-ready Fourier series animations with:
- 1080x1080 resolution (Instagram square) or 1080x1920 (Reels)
- Smooth 30fps animation
- Professional dark theme
- Visible rotating circles with glow effect
- Clean trail line
- Final hold for completed drawing

Usage:
  python apps/cli_instagram.py --image flower.png --n 500 --save output/instagram.mp4
  python apps/cli_instagram.py --shape heart --n 200 --format reels --save output/reels.mp4
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
from matplotlib.patches import Circle

from falkourier.complex_series import fit_complex_fourier, sorted_harmonics
from falkourier.image_path import extract_path_from_image
from falkourier.schedule import get_n_schedule, normalize_path, sample_points_for_n
from falkourier.shapes import SHAPES


# Instagram color theme - professional dark
BG_COLOR = "#0a0a0a"
CIRCLE_COLOR = "#00d4ff"  # Cyan glow
TRAIL_COLOR = "#ff4757"   # Vibrant red
ARM_COLOR = "#00d4ff"
TIP_COLOR = "#ffffff"
TEXT_COLOR = "#ffffff"
N_COLOR = "#ffa502"       # Orange for N counter


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


def render_instagram_video(
    points: np.ndarray,
    n_harmonics: int = 300,
    fps: int = 30,
    frames_per_n: int = 40,
    final_hold_seconds: int = 3,
    output_path: str = "instagram_fourier.mp4",
    format: str = "square",  # "square" (1080x1080) or "reels" (1080x1920)
    show_formula: bool = True,
):
    """Generate Instagram-optimized Fourier animation."""
    
    # Normalize and fit
    pts = normalize_path(points, target_span=2.2)
    z = pts[:, 0] + 1j * pts[:, 1]
    coeffs = fit_complex_fourier(z, n_harmonics)
    ranked_freqs = sorted_harmonics(coeffs)
    
    # Build schedule with smooth progression
    schedule = get_n_schedule(n_harmonics)
    
    # Add more intermediate steps for smoother video
    full_schedule = []
    for i, n in enumerate(schedule):
        full_schedule.append(n)
    
    # Build frame timeline
    n_of_frame, t_of_frame = [], []
    for i, n in enumerate(full_schedule):
        laps = 1
        if i == len(full_schedule) - 1:
            # Extra laps at the end for final hold
            laps = max(1, int(final_hold_seconds * fps / frames_per_n))
        
        for lap in range(laps):
            for t in np.linspace(0, 2 * np.pi, frames_per_n, endpoint=True):
                n_of_frame.append(n)
                t_of_frame.append(float(t))
    
    n_of_frame = np.array(n_of_frame)
    t_of_frame = np.array(t_of_frame)
    total_frames = len(n_of_frame)
    
    print(f"Schedule: {full_schedule[0]} -> {full_schedule[-1]}")
    print(f"Total frames: {total_frames} ({total_frames/fps:.1f}s at {fps}fps)")
    
    # Figure setup based on format
    if format == "reels":
        figsize = (10.8, 19.2)  # 1080x1920 at 100 DPI
        dpi = 100
    else:  # square
        figsize = (10.8, 10.8)  # 1080x1080 at 100 DPI
        dpi = 100
    
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR, dpi=dpi)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    
    # Adjust limits based on format
    if format == "reels":
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-5.0, 5.0)
    else:
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
    ax.axis("off")
    
    # Title/formula
    if show_formula:
        ax.text(
            0.0, 2.6 if format == "square" else 4.5,
            r"$z_N(t)=\sum_{j=1}^{N} c_{k_j} e^{i k_j t}$",
            color=TEXT_COLOR, fontsize=16, family="serif",
            ha="center", va="center", alpha=0.8
        )
    
    # N counter
    n_label = ax.text(
        0.0, -2.6 if format == "square" else -4.5,
        "N = 1",
        color=N_COLOR, fontsize=24, fontweight="bold",
        ha="center", va="center"
    )
    
    # Create circles (max needed)
    max_circles = n_harmonics
    circles = []
    for _ in range(max_circles):
        c = Circle((0, 0), 0.01, fill=False, ec=CIRCLE_COLOR, lw=1.2, alpha=0.6, visible=False)
        ax.add_patch(c)
        circles.append(c)
    
    # Arm line
    arm_line, = ax.plot([], [], color=ARM_COLOR, lw=1.5, alpha=0.8)
    
    # Trail line - thicker for Instagram
    trail_line, = ax.plot([], [], color=TRAIL_COLOR, lw=3.0, solid_capstyle="round")
    
    # Tip dot - larger and glowing
    tip_dot, = ax.plot([], [], "o", color=TIP_COLOR, ms=8, zorder=20)
    tip_glow, = ax.plot([], [], "o", color=TRAIL_COLOR, ms=12, alpha=0.5, zorder=19)
    
    trail_x, trail_y = [], []
    prev_n = -1
    
    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="instagram_fourier_")
    
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
            
            # Compute arm chain
            order = ranked_freqs[:n]
            arm_pts = [0j]
            radii = []
            z_cur = 0j
            for k in order:
                c = coeffs[k]
                z_cur = z_cur + c * np.exp(1j * k * t)
                arm_pts.append(z_cur)
                radii.append(abs(c))
            
            # Update circles
            for i, circ in enumerate(circles):
                if i < len(order):
                    circ.set_visible(True)
                    circ.center = (arm_pts[i].real, arm_pts[i].imag)
                    circ.set_radius(max(radii[i], 1e-6))
                    # Fade smaller circles
                    circ.set_alpha(max(0.3, min(0.8, radii[i] * 2)))
                else:
                    circ.set_visible(False)
            
            # Update arm line
            arm_arr = np.array(arm_pts)
            arm_line.set_data(arm_arr.real, arm_arr.imag)
            
            # Compute pen position
            pen = z_cur
            trail_x.append(pen.real)
            trail_y.append(pen.imag)
            
            # Update trail and tip
            trail_line.set_data(trail_x, trail_y)
            tip_dot.set_data([pen.real], [pen.imag])
            tip_glow.set_data([pen.real], [pen.imag])
            
            # Save frame
            frame_path = Path(temp_dir) / f"frame_{frame:06d}.png"
            fig.savefig(frame_path, facecolor=BG_COLOR, dpi=dpi,
                       bbox_inches='tight', pad_inches=0)
            
            if frame % 50 == 0:
                pct = 100 * frame // total_frames
                print(f"  Frame {frame}/{total_frames} ({pct}%)")
        
        plt.close(fig)
        
        # Encode video with ffmpeg
        print("Encoding video...")
        ffmpeg_exe = get_ffmpeg_executable()
        encoder, encoder_args = get_encoder(ffmpeg_exe)
        
        frame_pattern = str(Path(temp_dir) / "frame_%06d.png")
        
        ffmpeg_cmd = [
            ffmpeg_exe, "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
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
        description="Instagram-optimized Fourier Drawing Generator"
    )
    
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", type=str, default="", help="Input image file")
    src.add_argument("--shape", choices=sorted(SHAPES), default="heart")
    
    parser.add_argument("--n", type=int, default=300, help="Max harmonics (higher = more detail)")
    parser.add_argument("--points", type=int, default=0, help="Sample points (0 = auto)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--frames-per-n", type=int, default=40, help="Frames per N level")
    parser.add_argument("--hold", type=int, default=3, help="Final hold seconds")
    parser.add_argument("--format", choices=["square", "reels"], default="square",
                       help="square (1080x1080) or reels (1080x1920)")
    parser.add_argument("--method", choices=["auto", "boundary", "adaptive", "otsu", "grabcut", "canny", "rembg", "alpha", "lineart", "ai", "ai_anime"],
                       default="canny", help="Contour extraction method")
    parser.add_argument("--save", type=str, default="output/instagram_fourier.mp4")
    parser.add_argument("--no-formula", action="store_true", help="Hide formula text")
    
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
    
    # Ensure output directory exists
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    
    # Render
    render_instagram_video(
        points,
        n_harmonics=args.n,
        fps=args.fps,
        frames_per_n=args.frames_per_n,
        final_hold_seconds=args.hold,
        output_path=args.save,
        format=args.format,
        show_formula=not args.no_formula,
    )


if __name__ == "__main__":
    main()
