"""
Offline frame-by-frame rendering for Fourier animations.

Generates individual PNG frames to a temp folder, then uses ffmpeg to combine
them into a high-quality video. This approach is more memory-efficient and
produces better results for complex animations compared to matplotlib's
built-in animation.save().

Benefits:
- Lower memory usage (frames written to disk immediately)
- Higher quality output (direct PNG → ffmpeg encoding)
- Resumable rendering (frames persist until video is created)
- Better progress tracking
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def create_temp_frames_dir(prefix: str = "fourier_frames_") -> Path:
    """Create a temporary directory for storing frames."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def save_frame(fig: plt.Figure, frame_dir: Path, frame_num: int, dpi: int = 100) -> Path:
    """Save a single frame to the frames directory."""
    frame_path = frame_dir / f"frame_{frame_num:06d}.png"
    fig.savefig(frame_path, facecolor=fig.get_facecolor(), dpi=dpi)
    return frame_path


@lru_cache(maxsize=1)
def _get_available_encoder() -> tuple[str, list[str]]:
    """Detect available video encoder and return (encoder_name, extra_args).
    
    Result is cached for performance.
    """
    if not check_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not installed or not in PATH."
        )
    
    # Try encoders in order of preference
    encoders = [
        ("libx264", ["-preset", "medium", "-crf", "18"]),
        ("libopenh264", ["-b:v", "3000k"]),
        ("libvpx-vp9", ["-crf", "30", "-b:v", "0"]),
        ("mpeg4", ["-q:v", "5"]),
    ]
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        encoder_list = result.stdout
    except subprocess.TimeoutExpired:
        encoder_list = ""
    
    for encoder, args in encoders:
        if encoder in encoder_list:
            return encoder, args
    
    # Fallback to mpeg4 which is usually available
    return "mpeg4", ["-q:v", "5"]


def frames_to_video(
    frame_dir: Path,
    output_path: Path | str,
    fps: int = 30,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """
    Combine PNG frames into a video (MP4 or GIF) using ffmpeg.
    
    Args:
        frame_dir: Directory containing frame_NNNNNN.png files
        output_path: Output video file path (.mp4 or .gif)
        fps: Frames per second
        crf: Constant Rate Factor (lower = higher quality, 18 is visually lossless)
        preset: ffmpeg encoding preset (slower = better compression)
    
    Returns:
        Path to the created video file
    """
    output_path = Path(output_path)
    frame_pattern = str(frame_dir / "frame_%06d.png")
    
    # Check if output is GIF
    is_gif = output_path.suffix.lower() == ".gif"
    
    if is_gif:
        # GIF encoding with palette generation for better quality
        palette_path = frame_dir / "palette.png"
        
        # Generate palette
        palette_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-vf", f"fps={fps},scale=640:-1:flags=lanczos,palettegen",
            str(palette_path)
        ]
        result = subprocess.run(palette_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg palette generation failed: {result.stderr}")
        
        # Create GIF using palette
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-i", str(palette_path),
            "-lavfi", f"fps={fps},scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(output_path)
        ]
    else:
        # MP4 encoding
        encoder, encoder_args = _get_available_encoder()
        
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", encoder,
            *encoder_args,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]
    
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    return output_path


def cleanup_frames_dir(frame_dir: Path) -> None:
    """Remove the temporary frames directory."""
    shutil.rmtree(frame_dir, ignore_errors=True)


class OfflineRenderer:
    """
    Context manager for offline frame rendering.
    
    Usage:
        with OfflineRenderer("my_animation") as renderer:
            for frame_num in range(100):
                # Update matplotlib figure...
                renderer.save_frame(fig, frame_num)
            video_path = renderer.create_video(fps=30)
    """
    
    def __init__(self, name: str = "render", auto_cleanup: bool = True):
        self.name = name
        self.auto_cleanup = auto_cleanup
        self.frame_dir: Path | None = None
        self.frame_count = 0
    
    def __enter__(self) -> "OfflineRenderer":
        self.frame_dir = create_temp_frames_dir(prefix=f"{self.name}_")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_cleanup and self.frame_dir:
            cleanup_frames_dir(self.frame_dir)
    
    def save_frame(self, fig: plt.Figure, frame_num: int | None = None, dpi: int = 100) -> Path:
        """Save a frame. If frame_num is None, auto-increment."""
        if self.frame_dir is None:
            raise RuntimeError("Renderer not initialized. Use 'with' statement.")
        
        if frame_num is None:
            frame_num = self.frame_count
            self.frame_count += 1
        
        return save_frame(fig, self.frame_dir, frame_num, dpi)
    
    def create_video(
        self,
        output_path: Path | str | None = None,
        fps: int = 30,
        crf: int = 18,
        preset: str = "medium",
    ) -> Path:
        """Create video from saved frames."""
        if self.frame_dir is None:
            raise RuntimeError("Renderer not initialized. Use 'with' statement.")
        
        if output_path is None:
            output_path = self.frame_dir / "output.mp4"
        
        return frames_to_video(self.frame_dir, output_path, fps, crf, preset)
    
    def get_video_bytes(self, fps: int = 30, crf: int = 18) -> bytes:
        """Create video and return its contents as bytes."""
        video_path = self.create_video(fps=fps, crf=crf)
        return video_path.read_bytes()


def render_animation_offline(
    update_func: Callable[[int], None],
    fig: plt.Figure,
    total_frames: int,
    output_path: Path | str,
    fps: int = 30,
    dpi: int = 100,
    crf: int = 18,
    progress_callback: Callable[[float], None] | None = None,
) -> Path:
    """
    Render an animation offline by calling update_func for each frame.
    
    Args:
        update_func: Function that updates the figure for frame number
        fig: Matplotlib figure to render
        total_frames: Total number of frames to render
        output_path: Output video file path
        fps: Frames per second
        dpi: Resolution for PNG frames
        crf: Video quality (lower = better)
        progress_callback: Optional callback(progress: 0.0-1.0)
    
    Returns:
        Path to the created video file
    """
    with OfflineRenderer("animation", auto_cleanup=True) as renderer:
        for frame in range(total_frames):
            update_func(frame)
            renderer.save_frame(fig, frame, dpi)
            
            if progress_callback and frame % 5 == 0:
                progress_callback(frame / total_frames * 0.9)
        
        if progress_callback:
            progress_callback(0.95)
        
        video_path = renderer.create_video(output_path, fps=fps, crf=crf)
        
        if progress_callback:
            progress_callback(1.0)
        
        return video_path
