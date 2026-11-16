from __future__ import annotations

import sys
from pathlib import Path

_ext_dir = str(Path(__file__).resolve().parent / "_ext")
if _ext_dir not in sys.path:
    sys.path.insert(0, _ext_dir)

from . import optimizer_checks  # noqa: F401
