"""Python CLUBB standalone driver package.

This package lives at the repository root, while the compiled Fortran/Python
API package lives under ``clubb_release/clubb_python_api/``. Add that directory
to ``sys.path`` so ``clubb_f2py`` and ``clubb_python`` imports work when tools
are launched directly from this checkout.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
_API_ROOT = _REPO_ROOT / "clubb_release" / "clubb_python_api"

if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))
