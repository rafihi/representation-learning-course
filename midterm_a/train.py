"""Backward-compatible shim to the course repository's lec3/train.py entry point.

New Midterm-A runs should be launched with:
    python ../lec3/train.py --config configs/<name>.yaml

This wrapper remains so older commands (`python train.py --config ...`) still
work, but immediately delegates to the course entry point.
"""
from pathlib import Path
import runpy
import sys

course_train = Path(__file__).resolve().parents[1] / "lec3" / "train.py"
if not course_train.exists():
    raise FileNotFoundError(
        f"Course train.py not found at {course_train}. "
        "The repository must include both lec3/ and midterm_a/."
    )

sys.argv[0] = str(course_train)
runpy.run_path(str(course_train), run_name="__main__")
