#!/usr/bin/env python3
"""
Fourier-series epicycle animation — dual X/Y two-chain style.

Upload any image → extract outline → animate with two epicycle chains.
By default N grows 1 → 2 → 3 → … → max so the drawing gets sharper each step.

Examples:
  python apps/cli_xy.py --image bird.jpg --n 50 --save out.mp4
  python apps/cli_xy.py --shape hummingbird --side-by-side --save out.mp4
  python apps/cli_xy.py --shape heart --n 30 --fixed   # fixed N, no ramp
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds ../src to sys.path)

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle
from PIL import Image

from fourier_drawing.image_path import extract_path_from_image, save_silhouette_preview
from fourier_drawing.schedule import build_n_timeline, get_n_schedule, normalize_path, sample_points_for_n
from fourier_drawing.series import chain_xy, fit_fourier
from fourier_drawing.shapes import SHAPES


def build_animation(
    points: np.ndarray,
    n_harmonics: int = 50,
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
    Build an X/Y Fourier series animation.
    
    Args:
        offline: If True, use offline frame-by-frame rendering (better for
                 complex shapes, lower memory usage, higher quality output)
        save_path: Required when offline=True. Path to save the video.
    """
    pts = normalize_path(points)
    fx = fit_fourier(pts[:, 0], n_harmonics)
    fy = fit_fourier(pts[:, 1], n_harmonics)

    draw_cx, draw_cy = 1.15, -0.15
    x0 = fx.a0 / 2
    y0 = fy.a0 / 2
    x_origin = complex(draw_cx, 2.05)
    y_origin = complex(-1.85, draw_cy)
    visual_limit = min(n_harmonics, 128)

    if progressive:
        schedule = get_n_schedule(n_harmonics)
        if frames_per_n is None:
            frames_per_n = max(24, int(0.7 * fps))
        n_of_frame, t_of_frame = build_n_timeline(
            schedule,
            frames_per_n,
            final_hold_loops=2,
            nyquist_safe_all=False,
            nyquist_safe_final=False,
        )
        total_frames = len(n_of_frame)
        print(f"Progressive N schedule: {schedule}")
        print(
            f"  {total_frames} frames total; pen motion is interpolated between frames, "
            f"with up to {visual_limit} visible epicycles per axis"
        )
    else:
        schedule = [n_harmonics]
        total_frames = frames
        n_of_frame = np.full(total_frames, n_harmonics, dtype=int)
        # Use endpoint=True to complete the full cycle and close the shape
        t_of_frame = np.linspace(0, 2 * np.pi, total_frames, endpoint=True)

    bg = "#0d3b2e"

    # Keep coordinate signals visible beside the drawing: x(t), y(t), and x(t)+y(t).
    if side_by_side:
        fig = plt.figure(figsize=(18, 10), facecolor=bg)
        grid = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.65, 0.8], wspace=0.12)
        ax_left = fig.add_subplot(grid[0, 0])
        ax = fig.add_subplot(grid[0, 1])
        wave_grid = grid[0, 2].subgridspec(3, 1, hspace=0.38)
        ax_left.set_facecolor(bg)
        ax_left.axis("off")
        ax_left.set_title("Input", color="#e8f5e9", fontsize=14, pad=8)

        if source_image is not None and Path(source_image).is_file():
            ax_left.imshow(Image.open(source_image).convert("RGB"))
        else:
            tmp = Path("_sil_preview.png")
            save_silhouette_preview(points, tmp)
            ax_left.imshow(Image.open(tmp))
            tmp.unlink(missing_ok=True)
    else:
        fig = plt.figure(figsize=(16, 9), facecolor=bg)
        grid = fig.add_gridspec(1, 2, width_ratios=[1.65, 1], wspace=0.12)
        ax = fig.add_subplot(grid[0, 0])
        wave_grid = grid[0, 1].subgridspec(3, 1, hspace=0.38)

    fig.subplots_adjust(left=0.03, right=0.98, top=0.96, bottom=0.05)

    wave_axes = [fig.add_subplot(wave_grid[i, 0]) for i in range(3)]
    wave_names = ["x(t)", "y(t)", "x(t) + y(t)"]
    wave_colors = ["#80cbc4", "#ffe082", "#e85d4c"]
    # Include at least 2N+1 unique samples so the highest requested harmonic
    # is represented in the signal plots instead of folding into lower ones.
    wave_sample_count = max(801, 2 * n_harmonics + 2)  # duplicate endpoint closes plot
    wave_times = np.linspace(0, 2 * np.pi, wave_sample_count, endpoint=True)
    wave_lines = []
    wave_cursors = []
    for wave_ax, name, color in zip(wave_axes, wave_names, wave_colors):
        wave_ax.set_facecolor(bg)
        wave_ax.set_xlim(0, 2 * np.pi)
        wave_ax.set_ylim(-1, 1)
        wave_ax.set_title(name, loc="left", color=color, fontsize=11, pad=4)
        wave_ax.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
        wave_ax.tick_params(colors="#b7c9c1", labelsize=8, length=2)
        wave_ax.grid(axis="y", color="#416b5d", alpha=0.55, linewidth=0.7)
        for spine in wave_ax.spines.values():
            spine.set_color("#416b5d")
        (line,) = wave_ax.plot([], [], color=color, lw=1.5)
        cursor = wave_ax.axvline(0, color="#f5f5f5", lw=0.8, alpha=0.8)
        (dot,) = wave_ax.plot([], [], "o", color=color, ms=4, zorder=5)
        wave_lines.append(line)
        wave_cursors.append((cursor, dot))

    def sample_series(series, t_values, n):
        """Evaluate a real Fourier series in bounded-memory time chunks."""
        count = min(int(n), series.n_max)
        values = np.full(len(t_values), series.a0 / 2, dtype=float)
        if count:
            k = np.arange(1, count + 1, dtype=float)
            for start in range(0, len(t_values), 512):
                end = min(start + 512, len(t_values))
                phase = np.outer(k, t_values[start:end])
                values[start:end] += series.a[:count] @ np.cos(phase)
                values[start:end] += series.b[:count] @ np.sin(phase)
        return values

    cached_wave_n = -1

    ax.set_facecolor(bg)
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 3.4)
    ax.set_ylim(-2.6, 3.4)
    ax.axis("off")

    ax.text(
        -3.0, 3.15,
        r"$x_N(t)=\frac{a_0}{2}+\sum_{k=1}^{N}(a_k\cos kt+b_k\sin kt)$" + "\n"
        r"$y_N(t)=\frac{c_0}{2}+\sum_{k=1}^{N}(c_k\cos kt+d_k\sin kt)$" + "\n"
        r"$P_N(t)=(x_N(t),y_N(t))=x_N(t)+\mathrm{i}\,y_N(t)$",
        color="#e8f5e9", fontsize=10 if side_by_side else 11,
        family="serif", va="top", linespacing=1.6,
    )

    n_label = ax.text(
        0.0, -2.35, "1", color="#e07a5f",
        fontsize=36 if side_by_side else 42, fontweight="bold",
        ha="center", va="center",
    )

    (trail_line,) = ax.plot([], [], color="#e85d4c", lw=2.2, solid_capstyle="round")
    (tip_dot,) = ax.plot([], [], "o", color="#ffccbc", ms=5, zorder=20)
    (v_guide,) = ax.plot([], [], "--", color="#80cbc4", lw=0.9, alpha=0.7)
    (h_guide,) = ax.plot([], [], "--", color="#ffe082", lw=0.9, alpha=0.7)

    x_circles = [
        Circle((0, 0), 0.01, fill=False, ec="#4db6ac", lw=1.0, alpha=0.85, visible=False)
        for _ in range(visual_limit)
    ]
    y_circles = [
        Circle((0, 0), 0.01, fill=False, ec="#ffd54f", lw=1.0, alpha=0.85, visible=False)
        for _ in range(visual_limit)
    ]
    for c in x_circles + y_circles:
        ax.add_patch(c)

    (x_arms,) = ax.plot([], [], color="#80cbc4", lw=1.1, alpha=0.9)
    (y_arms,) = ax.plot([], [], color="#ffe082", lw=1.1, alpha=0.9)

    trail_x: list[float] = []
    trail_y: list[float] = []
    prev_n = -1
    prev_t: float | None = None

    def update(frame: int):
        nonlocal prev_n, prev_t, trail_x, trail_y, cached_wave_n

        n = int(n_of_frame[frame])
        t = float(t_of_frame[frame])

        if n != prev_n:
            trail_x = []
            trail_y = []
            prev_n = n
            prev_t = None
            n_label.set_text(str(n))

        if n != cached_wave_n:
            wave_x = sample_series(fx, wave_times, n)
            wave_y = sample_series(fy, wave_times, n)
            wave_values = (wave_x, wave_y, wave_x + wave_y)
            for wave_ax, line, values in zip(wave_axes, wave_lines, wave_values):
                line.set_data(wave_times, values)
                span = max(float(np.max(np.abs(values))), 1e-6)
                wave_ax.set_ylim(-1.15 * span, 1.15 * span)
            cached_wave_n = n

        # Keep the arms and final tips mathematically complete at high N;
        # only the number of drawn circles is capped for rendering speed.
        xc, radii_x = chain_xy(fx, t, n, x_origin, rotate90=False)
        yc, radii_y = chain_xy(fy, t, n, y_origin, rotate90=True)
        pen_x = fx.evaluate(t, n) - x0 + draw_cx
        pen_y = fy.evaluate(t, n) - y0 + draw_cy
        x_tip = xc[-1]
        y_tip = yc[-1]

        for i in range(visual_limit):
            if i < min(n, visual_limit):
                x_circles[i].set_visible(True)
                y_circles[i].set_visible(True)
                x_circles[i].center = (xc[i].real, xc[i].imag)
                x_circles[i].set_radius(max(float(radii_x[i]), 1e-6))
                y_circles[i].center = (yc[i].real, yc[i].imag)
                y_circles[i].set_radius(max(float(radii_y[i]), 1e-6))
            else:
                x_circles[i].set_visible(False)
                y_circles[i].set_visible(False)

        x_arms.set_data(xc.real, xc.imag)
        y_arms.set_data(yc.real, yc.imag)

        v_guide.set_data([pen_x, pen_x], [x_tip.imag, pen_y])
        h_guide.set_data([y_tip.real, pen_x], [pen_y, pen_y])

        if prev_t is None or t <= prev_t:
            # The second final lap restarts at t=0; its first point coincides
            # with the closed endpoint from the preceding lap.
            trail_x.append(pen_x)
            trail_y.append(pen_y)
        else:
            # A high harmonic can complete several turns between video frames.
            # Add evaluated points along that interval so the visible trail
            # still follows every detail instead of drawing a straight chord.
            subdivisions = max(1, int(np.ceil((n + 1) * (t - prev_t) / np.pi)))
            for sample_t in np.linspace(prev_t, t, subdivisions + 1)[1:]:
                trail_x.append(fx.evaluate(float(sample_t), n) - x0 + draw_cx)
                trail_y.append(fy.evaluate(float(sample_t), n) - y0 + draw_cy)
        prev_t = t
        trail_line.set_data(trail_x, trail_y)
        tip_dot.set_data([pen_x], [pen_y])

        signal_x = fx.evaluate(t, n)
        signal_y = fy.evaluate(t, n)
        signal_values = (signal_x, signal_y, signal_x + signal_y)
        for wave_ax, (cursor, dot), value in zip(wave_axes, wave_cursors, signal_values):
            cursor.set_xdata([t, t])
            dot.set_data([t], [value])

        return (*x_circles, *y_circles, x_arms, y_arms, v_guide, h_guide,
                trail_line, tip_dot, n_label, *wave_lines,
                *(item for pair in wave_cursors for item in pair))

    # ─────────────────────────────────────────────────────────────────────────
    # OFFLINE RENDERING: Save frames to temp folder, then combine with ffmpeg
    # ─────────────────────────────────────────────────────────────────────────
    if offline and save_path:
        from fourier_drawing.offline_render import OfflineRenderer
        
        print(f"Using offline rendering ({total_frames} frames)...")
        with OfflineRenderer("xy_fourier", auto_cleanup=True) as renderer:
            for frame in range(total_frames):
                update(frame)
                renderer.save_frame(fig, frame, dpi=100)
                if frame % 50 == 0:
                    print(f"  Frame {frame}/{total_frames} ({100*frame//total_frames}%)")
            
            print("Encoding video...")
            renderer.create_video(save_path, fps=fps, crf=18)
        
        plt.close(fig)
        return None, None, fps

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps, blit=False, repeat=True,
    )
    return fig, anim, fps


def load_points(args: argparse.Namespace, n_harmonics: int) -> tuple[np.ndarray, str | None]:
    """Load points from image or built-in shape.
    
    Auto-scales sample points based on n_harmonics if --points is not specified.
    """
    # Auto-scale sample points if not explicitly set
    n_points = args.points if args.points > 0 else sample_points_for_n(n_harmonics)
    
    if args.image:
        debug = args.debug_contour or None
        points = extract_path_from_image(
            args.image, n_points=n_points, method=args.method,
            invert=args.invert, debug_path=debug,
        )
        print(f"Extracted outline from {args.image} ({len(points)} samples)")
        return points, args.image

    points = SHAPES[args.shape](n_points)
    print(f"Using shape '{args.shape}' with {n_points} sample points (auto-scaled for N={n_harmonics})")
    return points, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Image -> X/Y Fourier epicycle animation")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", type=str, default="", help="input photo / silhouette")
    src.add_argument("--shape", choices=sorted(SHAPES), default="hummingbird")
    parser.add_argument("--n", type=int, default=50, help="max harmonics N (progressive goes 1...N)")
    parser.add_argument("--progressive", action="store_true", default=True)
    parser.add_argument("--fixed", action="store_true", help="disable progressive ramp")
    parser.add_argument("--frames-per-n", type=int, default=0, help="0 = auto")
    parser.add_argument("--points", type=int, default=0,
                        help="sample points (0 = auto-scale based on N)")
    parser.add_argument("--frames", type=int, default=600, help="frames per loop when --fixed")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--method",
                        choices=["auto", "boundary", "adaptive", "otsu", "grabcut", "rembg", "alpha", "canny",
                                 "lineart", "allcontours", "ai", "ai_anime"],
                        default="auto")
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
            points, n_harmonics=args.n, frames=args.frames, fps=args.fps,
            source_image=source, side_by_side=args.side_by_side,
            progressive=progressive, frames_per_n=frames_per_n,
            offline=True, save_path=args.save,
        )
        print("Done.")
        return
    
    fig, anim, fps = build_animation(
        points, n_harmonics=args.n, frames=args.frames, fps=args.fps,
        source_image=source, side_by_side=args.side_by_side,
        progressive=progressive, frames_per_n=frames_per_n,
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
