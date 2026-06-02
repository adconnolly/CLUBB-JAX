#!/usr/bin/env python3
"""run_all_tests.py — run the whole clubb_jax unit-test suite and report pass/fail/skip.

Generalises testing beyond the single-file invocations (DESIGN "How to Test"): discovers every
`clubb_jax/tests/test_*.py`, runs each as a subprocess with the repo root on PYTHONPATH, and prints a
one-line-per-file result + a summary. Tests that compare against an UNAVAILABLE oracle (f2py `clubb_f2py`,
the `fortran_oracle` exe, or stored Fortran stats) SKIP that part gracefully and still return 0 — so the
suite is a clean, portable regression gate in any environment (no compiled Fortran needed for the JAX-only
assertions).

Usage:
    python clubb_jax/run_scripts/run_all_tests.py            # run all
    python clubb_jax/run_scripts/run_all_tests.py -k solver  # only files whose name contains "solver"
    python clubb_jax/run_scripts/run_all_tests.py --timeout 600

Exit code 0 iff every test file returned 0 (passed or skipped cleanly), else 1.
"""
from __future__ import annotations
import argparse
import glob
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))   # contains clubb_jax/
_TESTS_DIR = os.path.join(_REPO_ROOT, "clubb_jax", "tests")


def _result_line(out: str) -> str:
    """Pick the most informative trailing line (pass/fail/skip summary) from a test's output."""
    keys = ("passed", "failed", "PASS", "FAIL", "SKIP", "Error", "Traceback",
            "All ", "tests PASSED", "Results:")
    lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
    for l in reversed(lines):
        if any(k in l for k in keys):
            return l[:140]
    return (lines[-1][:140] if lines else "(no output)")


def main() -> int:
    p = argparse.ArgumentParser(description="Run the clubb_jax unit-test suite")
    p.add_argument("-k", default=None, help="only run test files whose name contains this substring")
    p.add_argument("--timeout", type=int, default=600, help="per-file timeout [s] (default 600)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(_TESTS_DIR, "test_*.py")))
    if args.k:
        files = [f for f in files if args.k in os.path.basename(f)]
    if not files:
        print("No test files matched.")
        return 1

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    print(f"Running {len(files)} test file(s) from clubb_jax/tests/ (timeout {args.timeout}s each):\n")
    failures = []
    for f in files:
        name = os.path.basename(f)[:-3]
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, f], env=env, capture_output=True, text=True,
                               timeout=args.timeout)
            rc, out = r.returncode, (r.stdout + r.stderr)
        except subprocess.TimeoutExpired:
            rc, out = 124, "TIMEOUT"
        dt = time.time() - t0
        tag = "PASS" if rc == 0 else ("TIMEOUT" if rc == 124 else "FAIL")
        if rc != 0:
            failures.append(name)
        print(f"  [{tag:4s}] {name:34s} {dt:6.1f}s | {_result_line(out)}")

    print(f"\n{'='*70}")
    n = len(files)
    print(f"{n - len(failures)}/{n} test files OK"
          + (f"  —  FAILED: {', '.join(failures)}" if failures else "  —  ALL GREEN"))
    print(f"{'='*70}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
