"""Make ``src`` and ``tests`` importable when running scripts directly."""
import os
import sys
from pathlib import Path

# one BLAS thread per process: scripts parallelise over Monte-Carlo samples, and the master plan (P12)
# measures runtime on a single CPU thread
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
