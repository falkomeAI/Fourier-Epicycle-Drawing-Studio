"""
Adds ../src to sys.path so scripts/*.py can `import fourier_drawing` without
an editable pip install. Import this module first, before any
fourier_drawing import.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
