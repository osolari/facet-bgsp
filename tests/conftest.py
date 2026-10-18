"""Make ``experiments`` and ``configs`` importable from the repo root so the
smoke tests can ``from experiments import m9_scrna`` etc. without needing
``pip install -e .``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
