#!/usr/bin/env python3
"""
Complex Fourier Series animation — single-arm epicycle drawing.

Single complex-plane Fourier series: one chain of rotating circles in
the complex plane draws the entire path.

Examples:
  python apps/cli_complex.py --image bird.jpg --n 100 --save bird.mp4
  python apps/cli_complex.py --shape hummingbird --progressive --save out.mp4
  python apps/cli_complex.py --svg logo.svg --n 80 --save logo.mp4
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path)

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle
from PIL import Image

from falkourier.complex_series import fit_complex_fourier, sorted_harmonics
from falkourier.image_path import extract_path_from_image, save_silhouette_preview
from falkourier.schedule import build_n_timeline, get_n_schedule, normalize_path, sample_points_for_n
from falkourier.shapes import SHAPES


def build_animation(
    points: np.ndarray,
    n_harmonics: int = 80,
    frames: int = 600,
    fps: int = 30,
    source_image: str | Path | None = None,
    side_by_side: bool = False,
    progressive: bool = True,
    frames_per_n: int | None = None,
    offline: bool = False,
    save_path: str | Path | None = None,
):
    """
    Build a complex Fourier series animation.
    
    Args:
        offline: If True, use offline frame-by-frame rendering (better for
                 complex shapes, lower memory usage, higher quality output)
        save_path: Required when offline=True. Path to save the video.
    """
    pts = normalize_path(points, target_span=2.5)
    z = pts[:, 0] + 1j * pts[:, 1]
    coeffs = fit_complex_fourier(z, n_harmonics)
    # Rank the complete DFT once, then N is a count of arms.
    ranked_freqs = sorted_harmonics(coeffs)

    if progressive:
        schedule = get_n_schedule(n_harmonics)
        if frames_per_n is None:
            frames_per_n = max(24, int(0.7 * fps))
        n_of_frame, t_of_frame = build_n_timeline(
            schedule,
            frames_per_n,
            final_hold_loops=1,
            nyquist_safe_all=False,
            nyquist_safe_final=False,
        )
        total_frames = len(n_of_frame)
        print(f"Progressive N schedule: {schedule}")
        print(
            f"  {total_frames} frames total; the final step draws all selected arms "
            "and keeps a fully sampled trace"
        )
    else:
        total_frames = frames
        n_of_frame = np.full(total_frames, n_harmonics, dtype=int)
        # Use endpoint=True to complete the full cycle and close the shape
        t_of_frame = np.linspace(0, 2 * np.pi, total_frames, endpoint=True)

    bg = "#111111"

    if side_by_side:
        fig = plt.figure(figsize=(19, 10), facecolor=bg)
        grid = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.6, 0.8], wspace=0.12)
        ax_left = fig.add_subplot(grid[0, 0])
        ax = fig.add_subplot(grid[0, 1])
        wave_grid = grid[0, 2].subgridspec(3, 1, hspace=0.38)
        ax_left.set_facecolor(bg)
        ax_left.axis("off")
        ax_left.set_title("Input", color="#e0e0e0", fontsize=14, pad=8)
        if source_image and Path(source_image).is_file():
            ax_left.imshow(Image.open(source_image).convert("RGB"))
        else:
            tmp = Path("_sil_tmp.png")
            save_silhouette_preview(points, tmp)
            ax_left.imshow(Image.open(tmp))
            tmp.unlink(missing_ok=True)
    else:
        fig = plt.figure(figsize=(16, 9), facecolor=bg)
        grid = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.12)
        ax = fig.add_subplot(grid[0, 0])
        wave_grid = grid[0, 1].subgridspec(3, 1, hspace=0.38)

    fig.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.05)

    wave_axes = [fig.add_subplot(wave_grid[i, 0]) for i in range(3)]
    wave_names = ["x(t)", "y(t)", "x(t) + y(t)"]
    wave_colors = ["#50c0f0", "#ffd166", "#ff6644"]
    max_freq = max((abs(k) for k in ranked_freqs[:n_harmonics]), default=1)
    # The plot is screen-resolution limited; 8193 samples remain ample for
    # display while avoiding enormous per-frame path rendering.
    wave_count = min(max(801, 2 * max_freq + 2), 8193)
    wave_times = np.linspace(0, 2 * np.pi, wave_count, endpoint=True)
    wave_lines, wave_cursors = [], []
    for wave_ax, name, color in zip(wave_axes, wave_names, wave_colors):
        wave_ax.set_facecolor(bg)
        wave_ax.set_xlim(0, 2 * np.pi)
        wave_ax.set_ylim(-1, 1)
        wave_ax.set_title(name, loc="left", color=color, fontsize=11, pad=4)
        wave_ax.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
        wave_ax.tick_params(colors="#b0b0b0", labelsize=8, length=2)
        wave_ax.grid(axis="y", color="#444444", alpha=0.65, linewidth=0.7)
        for spine in wave_ax.spines.values():
            spine.set_color("#444444")
        (line,) = wave_ax.plot([], [], color=color, lw=1.5)
        cursor = wave_ax.axvline(0, color="#eeeeee", lw=0.8, alpha=0.8)
        (dot,) = wave_ax.plot([], [], "o", color=color, ms=4, zorder=5)
        wave_lines.append(line)
        wave_cursors.append((cursor, dot))

    def sample_terms(order, t_values):
        coeff_values = np.asarray([coeffs[k] for k in order], dtype=np.complex128)
        sampled = np.empty(len(t_values), dtype=np.complex128)
        frequencies = np.asarray(order, dtype=float)
        for start in range(0, len(t_values), 512):
            end = min(start + 512, len(t_values))
            sampled[start:end] = np.exp(1j * np.outer(t_values[start:end], frequencies)) @ coeff_values
        return sampled

    ax.set_facecolor(bg)
    ax.set_aspect("equal")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.axis("off")

    ax.text(
        -3.3, 3.2,
        r"$z_N(t)=\sum_{j=1}^{N} c_{k_j}\, e^{i k_j t}$",
        color="#b0b0b0", fontsize=12, family="serif", va="top",
    )

    n_label = ax.text(
        0.0, -3.2, "1", color="#f07050", fontsize=38, fontweight="bold",
        ha="center", va="center",
    )

    # Tiny later circles are below pixel resolution; draw the largest 256.
    # The arm chain and pen still use every selected term.
    max_circles = min(n_harmonics, 256)
    circles = [
        Circle((0, 0), 0.01, fill=False, ec="#30a0d0", lw=0.8, alpha=0.7, visible=False)
        for _ in range(max_circles)
    ]
    for c in circles:
        ax.add_patch(c)

    (arm_line,) = ax.plot([], [], color="#50c0f0", lw=1.0, alpha=0.85)
    (trail_line,) = ax.plot([], [], color="#ff6644", lw=2.2, solid_capstyle="round")
    (tip_dot,) = ax.plot([], [], "o", color="#ffddcc", ms=5, zorder=20)

    trail_x: list[float] = []
    trail_y: list[float] = []
    prev_n = -1
    prev_t: float | None = None
    cached_wave_n = -1

    def update(frame: int):
        nonlocal prev_n, prev_t, trail_x, trail_y, cached_wave_n

        n = int(n_of_frame[frame])
        t = float(t_of_frame[frame])

        if n != prev_n:
            trail_x, trail_y = [], []
            prev_n = n
            prev_t = None
            n_label.set_text(str(n))

        order = ranked_freqs[:n]
        if n != cached_wave_n:
            signal = sample_terms(order, wave_times)
            waves = (signal.real, signal.imag, signal.real + signal.imag)
            for wave_ax, line, values in zip(wave_axes, wave_lines, waves):
                line.set_data(wave_times, values)
                span = max(float(np.max(np.abs(values))), 1e-9)
                wave_ax.set_ylim(-1.15 * span, 1.15 * span)
            cached_wave_n = n

        arm_pts = [0j]
        radii = []
        z_cur = 0j
        for k in order:
            c = coeffs[k]
            z_cur = z_cur + c * np.exp(1j * k * t)
            arm_pts.append(z_cur)
            radii.append(abs(c))

        for i, circ in enumerate(circles):
            if i < len(order):
                circ.set_visible(True)
                circ.center = (arm_pts[i].real, arm_pts[i].imag)
                circ.set_radius(max(radii[i], 1e-6))
            else:
                circ.set_visible(False)

        arm_arr = np.array(arm_pts)
        arm_line.set_data(arm_arr.real, arm_arr.imag)

        pen = z_cur
        if prev_t is None or t <= prev_t:
            trail_x.append(pen.real)
            trail_y.append(pen.imag)
        else:
            max_frequency = max((abs(k) for k in order), default=0)
            subdivisions = max(1, int(np.ceil(4 * max_frequency * (t - prev_t))))
            sample_times = np.linspace(prev_t, t, subdivisions + 1)[1:]
            samples = sample_terms(order, sample_times)
            trail_x.extend(samples.real.tolist())
            trail_y.extend(samples.imag.tolist())
        prev_t = t
        trail_line.set_data(trail_x, trail_y)
        tip_dot.set_data([pen.real], [pen.imag])

        signal_values = (pen.real, pen.imag, pen.real + pen.imag)
        for wave_ax, (cursor, dot), value in zip(wave_axes, wave_cursors, signal_values):
            cursor.set_xdata([t, t])
            dot.set_data([t], [value])

        return (*circles, arm_line, trail_line, tip_dot, n_label, *wave_lines,
                *(item for pair in wave_cursors for item in pair))

    # ─────────────────────────────────────────────────────────────────────────
    # OFFLINE RENDERING: Save frames to temp folder, then combine with ffmpeg
    # ─────────────────────────────────────────────────────────────────────────
    if offline and save_path:
        from falkourier.offline_render import OfflineRenderer
        
        print(f"Using offline rendering ({total_frames} frames)...")
        with OfflineRenderer("complex_fourier", auto_cleanup=True) as renderer:
            for frame in range(total_frames):
                update(frame)
                renderer.save_frame(fig, frame, dpi=100)
                if frame % 50 == 0:
                    print(f"  Frame {frame}/{total_frames} ({100*frame//total_frames}%)")
            
            print("Encoding video...")
            renderer.create_video(save_path, fps=fps, crf=18)
        
        plt.close(fig)
        return None, None, fps
    
    # Standard matplotlib animation (for preview or smaller animations)
    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps, blit=False, repeat=True,
    )
    return fig, anim, fps


def load_svg_paths(svg_path: str | Path, n_points: int = 512) -> list[np.ndarray]:
    """Load paths from SVG file using svgpathtools (if installed)."""
    try:
        from svgpathtools import svg2paths
    except ImportError:
        raise ImportError("svgpathtools not installed. Run: pip install svgpathtools")

    paths, _ = svg2paths(str(svg_path))
    result = []
    for path in paths:
        if path.length() < 1:
            continue
        ts = np.linspace(0, 1, n_points, endpoint=False)
        pts = np.array([path.point(t) for t in ts])
        xy = np.column_stack([pts.real, -pts.imag])  # SVG y-down -> math y-up
        result.append(xy)
    return result


def load_points(args: argparse.Namespace, n_harmonics: int) -> tuple[np.ndarray, str | None]:
    """Load points from SVG, image, or built-in shape.
    
    Auto-scales sample points based on n_harmonics if --points is not specified.
    """
    # Auto-scale sample points if not explicitly set
    n_points = args.points if args.points > 0 else sample_points_for_n(n_harmonics)
    
    def read_json_points(path: str | Path) -> np.ndarray:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = raw.get("points")
        pts = np.asarray(raw, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError("points JSON must be an array of at least 3 [x, y] pairs")
        if not np.all(np.isfinite(pts)):
            raise ValueError("points JSON contains a non-finite coordinate")
        print(f"Loaded {len(pts)} points from {path}")
        return pts

    if args.points_json:
        return read_json_points(args.points_json), None

    if args.svg:
        paths = load_svg_paths(args.svg, n_points)
        if not paths:
            raise ValueError("No valid paths found in SVG")
        pts = max(paths, key=len)
        print(f"Loaded SVG with {len(paths)} path(s), using longest ({len(pts)} pts)")
        return pts, None

    if args.image:
        pts = extract_path_from_image(
            args.image, n_points=n_points, method=args.method,
            invert=args.invert, debug_path=args.debug_contour or None,
        )
        print(f"Extracted outline from {args.image} ({len(pts)} samples)")
        return pts, args.image

    if args.shape:
        pts = SHAPES[args.shape](n_points)
        print(f"Using shape '{args.shape}' with {n_points} sample points (auto-scaled for N={n_harmonics})")
        return pts, None

    # Use built-in preset outline as the no-argument default.
    preset = Path("assets/fourier-points.json")
    if not preset.is_file():
        raise FileNotFoundError(
            f"default preset outline not found: {preset}. "
            "Pass --image, --svg, --shape, or --points-json instead."
        )
    return read_json_points(preset), None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complex Fourier epicycle animation (single-arm style)"
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", type=str, default="", help="input photo/silhouette")
    src.add_argument("--svg", type=str, default="", help="input SVG file")
    src.add_argument("--points-json", type=str, default="", help="JSON array of [x, y] points")
    src.add_argument("--shape", choices=sorted(SHAPES), default="")

    parser.add_argument("--n", type=int, default=80, help="number of strongest Fourier arms to use")
    parser.add_argument("--progressive", action="store_true", default=True)
    parser.add_argument("--fixed", action="store_true", help="disable progressive")
    parser.add_argument("--frames-per-n", type=int, default=0)
    parser.add_argument("--points", type=int, default=0, 
                        help="sample points (0 = auto-scale based on N)")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--method",
                        choices=["auto", "boundary", "adaptive", "otsu", "grabcut", "rembg", "alpha", "canny",
                                 "lineart", "allcontours", "ai", "ai_anime"],
                        default="auto",
                        help="use 'ai' or 'ai_anime' for complex drawings")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--debug-contour", type=str, default="")
    parser.add_argument("--offline", action="store_true", 
                        help="Use offline rendering (saves frames to temp folder, then combines)")
    args = parser.parse_args()
    
    # Validate inputs
    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    
    args.invert = True if args.invert else None
    progressive = not args.fixed
    frames_per_n = args.frames_per_n if args.frames_per_n > 0 else None

    points, source = load_points(args, n_harmonics=args.n)
    
    # Use offline rendering for complex shapes or when explicitly requested
    use_offline = args.offline or (args.save and args.n >= 100)
    
    if args.save and use_offline:
        print(f"Rendering {args.save} (offline mode)...")
        build_animation(
            points,
            n_harmonics=args.n,
            frames=args.frames,
            fps=args.fps,
            source_image=source,
            side_by_side=args.side_by_side,
            progressive=progressive,
            frames_per_n=frames_per_n,
            offline=True,
            save_path=args.save,
        )
        print("Done.")
        return
    
    fig, anim, fps = build_animation(
        points,
        n_harmonics=args.n,
        frames=args.frames,
        fps=args.fps,
        source_image=source,
        side_by_side=args.side_by_side,
        progressive=progressive,
        frames_per_n=frames_per_n,
    )

    if args.save:
        print(f"Rendering {args.save} ...")
        if args.save.lower().endswith(".gif"):
            anim.save(args.save, writer=animation.PillowWriter(fps=fps))
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
            anim.save(args.save, writer=writer)
        print("Done.")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
