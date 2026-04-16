from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

_pkg_dir = Path(__file__).resolve().parent
if (_pkg_dir.parent / "hf_space").resolve() == _pkg_dir:
    repo_root = str(_pkg_dir.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from ui.components import create_demo

if __name__ == "__main__":
    demo = create_demo()
    launch_kwargs = getattr(demo, "_launch_kwargs", {})
    demo.launch(share=True, **launch_kwargs)
