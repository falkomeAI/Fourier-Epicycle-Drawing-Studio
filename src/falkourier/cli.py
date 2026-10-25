"""
falkourier CLI — Command-line interface for Fourier Epicycle Drawing Studio.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .series import fit_fourier, chain_xy
from .image_path import extract_path_from_image
from .schedule import sample_points_for_n
from .shapes import SHAPES


def main():
    parser = argparse.ArgumentParser(
        prog="falkourier",
        description="falkourier — High-Precision Fourier Series Epicycle Drawing Engine by @falkomeai",
    )
    
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", type=str, default="", help="Path to input photo or sketch")
    src.add_argument("--shape", choices=sorted(SHAPES), default="hummingbird", help="Built-in mathematical shape")
    
    parser.add_argument("--n", type=int, default=100, help="Max Fourier harmonics (e.g. 100, 500, 1024, 4000)")
    parser.add_argument("--points", type=int, default=0, help="Sample point count (0 = auto-calculate)")
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")
    parser.add_argument("--frames-per-n", type=int, default=50, help="Frames rendered per harmonic step")
    parser.add_argument("--hold", type=int, default=3, help="Duration (seconds) to hold final completed outline")
    parser.add_argument("--format", choices=["square", "reels"], default="square", help="Video aspect ratio")
    parser.add_argument("--method", choices=["auto", "boundary", "adaptive", "otsu", "grabcut", "canny", "rembg", "alpha", "lineart", "ai", "ai_anime"],
                        default="boundary", help="Contour extraction algorithm")
    parser.add_argument("--save", type=str, default="output/falkourier_drawing.mp4", help="Output MP4 file path")
    parser.add_argument("--no-formula", action="store_true", help="Hide mathematical formula overlay")
    parser.add_argument("--no-silhouette", action="store_true", help="Omit background silhouette fill")

    args = parser.parse_args()

    # Import video rendering pipeline
    import apps.cli_instagram_xy as renderer
    
    n_points = args.points if args.points > 0 else sample_points_for_n(args.n)

    if args.image:
        points = extract_path_from_image(args.image, n_points=n_points, method=args.method)
        print(f"Extracted {len(points)} outline points from {args.image}")
    else:
        points = SHAPES[args.shape](n_points)
        print(f"Using built-in shape '{args.shape}' with {n_points} points")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    renderer.render_instagram_xy_video(
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


def studio_main():
    """Launch the interactive Streamlit Web Studio UI."""
    web_app_path = Path(__file__).resolve().parents[2] / "apps" / "web_app.py"
    if not web_app_path.is_file():
        # Fallback for installed package
        import apps.web_app
        web_app_path = Path(apps.web_app.__file__)
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(web_app_path)]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
