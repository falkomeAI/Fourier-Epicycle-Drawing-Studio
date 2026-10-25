"""
Optional debugging aid for complex shapes.

Instead of rendering a full .mp4 and extracting a frame with ffmpeg every
time a drawing looks wrong (slow, manual), a `DebugFrameRecorder` saves one
PNG per distinct harmonic count N it sees — straight into

    samples/_debug_frames/<run_name>_<timestamp>/frame_N####.png

So you can open any step (N=1, N=12, N=250, ...) directly as an image and
see exactly what the extracted outline / epicycle chains looked like, without
re-rendering or decoding video.

Usage (inside any animation's `update(frame)` callback):

    recorder = DebugFrameRecorder("my_shape", enabled=debug_frames)
    ...
    def update(frame):
        ...
        recorder.maybe_save(fig, n)
        return artists
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# src/fourier_drawing/debug_frames.py -> parents[2] is the project root,
# regardless of whether the caller lives in apps/ or scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "samples"
DEBUG_FRAMES_DIR = SAMPLES_DIR / "_debug_frames"


def _safe_slug(name: str, max_len: int = 60) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return slug[:max_len] or "run"


class DebugFrameRecorder:
    """Saves one PNG per distinct N value seen during an animation render."""

    def __init__(self, run_name: str, enabled: bool = True):
        self.enabled = enabled
        self._last_n: int | None = None
        self.out_dir: Path | None = None
        if self.enabled:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.out_dir = DEBUG_FRAMES_DIR / f"{_safe_slug(run_name)}_{ts}"
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(self, fig, n: int):
        """Save `fig` as the reference frame for N, only on the first time
        this N is seen (skips every other frame at the same N)."""
        if not self.enabled or self.out_dir is None or n == self._last_n:
            return None
        self._last_n = n
        path = self.out_dir / f"frame_N{n:04d}.png"
        fig.savefig(path, facecolor=fig.get_facecolor())
        return path

    def saved_frames(self) -> list[Path]:
        if self.out_dir is None:
            return []
        return sorted(self.out_dir.glob("frame_N*.png"))
