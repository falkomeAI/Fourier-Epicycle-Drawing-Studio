#!/usr/bin/env python3
"""Create a sample silhouette PNG so you can try --image without a photo."""

import _bootstrap  # noqa: F401  (adds ../src to sys.path)

from pathlib import Path

from falkourier.image_path import save_silhouette_preview
from falkourier.shapes import hummingbird_path

OUT = Path(__file__).resolve().parent.parent / "samples" / "hummingbird.png"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_silhouette_preview(hummingbird_path(512), OUT, size=640)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
