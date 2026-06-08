#!/usr/bin/env python3
"""run_all_tests.py — run the whole clubb_jax unit-test suite and report pass/fail/skip.

Generalises testing beyond the single-file invocations (DESIGN "How to Test"): discovers every
`clubb_jax/tests/test_*.py`, runs each as a subprocess with the repo root on PYTHONPATH, and prints a
one-line-per-file result + a summary. Tests that compare against an UNAVAILABLE oracle (f2py `clubb_f2py`,
the `fortran_oracle` exe, or stored Fortran stats) SKIP that part gracefully and still return 0 — so the
suite is a clean, portable regression gate in any environment (no compiled Fortran needed for the JAX-only
assertions).

Usage:
    python clubb_jax/run_scripts/run_all_tests.py            # run all (serial, live 'running...' marker)
    python clubb_jax/run_scripts/run_all_tests.py -j 8       # 8 files concurrently — much faster wall-clock
    python clubb_jax/run_scripts/run_all_tests.py -k solver  # only files whose name contains "solver"
    python clubb_jax/run_scripts/run_all_tests.py --timeout 600

Exit code 0 iff every test file returned 0 (passed or skipped cleanly), else 1.

Output is line-buffered + flushed per file (with an `(i/N) name running...` marker before each subprocess), so a
redirected long run — `python clubb_jax/run_scripts/run_all_tests.py > out.txt 2>&1 &` — streams live and shows which
slow file (e.g. bugsrad/standalone) is currently running, rather than appearing to stall until completion.
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


def _run_one(f: str, env: dict, timeout: int):
    """Run a single test file as a subprocess; return (name, rc, dt_seconds, result_line)."""
    name = os.path.basename(f)[:-3]
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, f], env=env, capture_output=True, text=True, timeout=timeout)
        rc, out = r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        rc, out = 124, "TIMEOUT"
    return name, rc, time.time() - t0, _result_line(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Run the clubb_jax unit-test suite")
    p.add_argument("-k", default=None, help="only run test files whose name contains this substring")
    p.add_argument("--timeout", type=int, default=600, help="per-file timeout [s] (default 600)")
    p.add_argument("-j", "--jobs", type=int, default=1,
                   help="run up to N test files concurrently (default 1 = serial with a live 'running...' "
                        "marker). Tests are isolated subprocesses, so parallelism is safe and cuts the wall-clock "
                        "dominated by per-file JAX/XLA warmup; in -j>1 mode results print as each file finishes.")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(_TESTS_DIR, "test_*.py")))
    if args.k:
        files = [f for f in files if args.k in os.path.basename(f)]
    if not files:
        print("No test files matched.")
        return 1

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    n = len(files)
    jobs = max(1, args.jobs)

    print(f"Running {n} test file(s) from clubb_jax/tests/ "
          f"(timeout {args.timeout}s each{'' if jobs == 1 else f', {jobs} parallel'}):\n", flush=True)
    failures = []

    if jobs == 1:
        for i, f in enumerate(files, 1):
            name = os.path.basename(f)[:-3]
            # Pre-run marker, flushed, so a redirected log shows WHICH test is currently running (e.g. the slow
            # bugsrad/standalone files) rather than appearing to stall — stdout is block-buffered to a file, so
            # without flushing nothing is visible until the whole run completes.
            print(f"  [....] ({i:>2}/{n}) {name:34s} running...", end="", flush=True)
            name, rc, dt, line = _run_one(f, env, args.timeout)
            tag = "PASS" if rc == 0 else ("TIMEOUT" if rc == 124 else "FAIL")
            if rc != 0:
                failures.append(name)
            # Overwrite the marker line (\r) with the completed result, flushed so it streams to the log live.
            print(f"\r  [{tag:4s}] ({i:>2}/{n}) {name:34s} {dt:6.1f}s | {line}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        done = 0
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_run_one, f, env, args.timeout): f for f in files}
            for fut in as_completed(futs):
                name, rc, dt, line = fut.result()
                done += 1
                tag = "PASS" if rc == 0 else ("TIMEOUT" if rc == 124 else "FAIL")
                if rc != 0:
                    failures.append(name)
                print(f"  [{tag:4s}] ({done:>2}/{n}) {name:34s} {dt:6.1f}s | {line}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"{n - len(failures)}/{n} test files OK"
          + (f"  —  FAILED: {', '.join(failures)}" if failures else "  —  ALL GREEN"), flush=True)
    print(f"{'='*70}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
