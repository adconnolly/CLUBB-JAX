#!/usr/bin/env python3
"""compare_grad.py — whole-driver DIFFERENTIABILITY dashboard (REFACTOR B5 gate).

The grad analogue of compare_cases.py: for each case it runs the whole-driver `jax.grad` probe
(`tests/probe_driver_grad.py <case>`) in a SUBPROCESS (isolating JAX memory per case — BUGSrad/large
domains would otherwise accumulate) and reports one line per case:

    case               thlm        um          worst-FD   status
    arm                133/133     133/133      4.0e-10    COMPLETE

* grad-finite N/M  — reverse-mode `jax.grad` of a one-step loss is finite at N of M levels (the
  differentiability criterion: a finite gradient everywhere it should be smooth).
* worst-FD         — max finite-difference vs autodiff relative error over the sampled levels
  (validates the gradient VALUE; a single-level kink at a hard physical threshold — e.g. the 8e-3 `rtm`
  inversion — can exceed the gate tol legitimately, flagged KINK, not a failure).

Exit 0 iff every case is at least grad-finite (the differentiability gate). Use --strict to also require
FD-correctness (no kinks). This does NOT run the Fortran oracle — faithfulness is compare_cases.py's job.

Usage:
    python clubb_jax/run_scripts/compare_grad.py                 # all cases with a working-dir namelist
    python clubb_jax/run_scripts/compare_grad.py --cases arm,bomex,gabls3_night
    python clubb_jax/run_scripts/compare_grad.py --strict        # FD-correct required (kinks fail)
"""
import argparse
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_JAX_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))
_PROBE = os.path.join(_JAX_ROOT, "clubb_jax", "tests", "probe_driver_grad.py")
_OUT = os.path.join(_JAX_ROOT, "clubb_jax", "output")
_FD_KINK_TOL = 1.0e-3   # worst-FD above this with grad finite => a (single-level) physical KINK, not a bug


def _all_cases():
    """Cases that have a working-dir namelist (clubb_jax/output/<case>_compare_jax/<case>.in)."""
    cases = []
    for d in sorted(os.listdir(_OUT)):
        if d.endswith("_compare_jax"):
            case = d[: -len("_compare_jax")]
            if os.path.isfile(os.path.join(_OUT, d, f"{case}.in")):
                cases.append(case)
    return cases


def _parse(text):
    """Pull (thlm_finite, um_finite, worst_fd, blocked_frame) from a probe run's stdout."""
    fin = {}
    for key in ("thlm", "um"):
        m = re.search(rf"d\(\.5\*sum {key}\^2\)/d {key}: grad finite (\d+/\d+)", text)
        fin[key] = m.group(1) if m else None
    worst = None
    for m in re.finditer(r"worst FD rel = ([0-9.eE+-]+)", text):
        worst = max(worst or 0.0, float(m.group(1)))
    blocked = None
    if "grad BLOCKED" in text:
        bm = re.search(r"blocker frame:.*?([^/\s]+\.py)\", line (\d+), in (\S+)", text)
        blocked = f"{bm.group(1)}:{bm.group(2)} {bm.group(3)}" if bm else "BLOCKED"
    return fin, worst, blocked


def _run_case(case, env):
    try:
        r = subprocess.run([sys.executable, _PROBE, case], env=env, cwd=_JAX_ROOT,
                           capture_output=True, text=True, timeout=900)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "grad BLOCKED: TIMEOUT (probe exceeded 900s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None, help="comma-separated; default = all with a namelist")
    ap.add_argument("--strict", action="store_true", help="require FD-correct (kinks fail the gate)")
    args = ap.parse_args()
    cases = args.cases.split(",") if args.cases else _all_cases()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [_JAX_ROOT, _JAX_ROOT + "/clubb_release", _JAX_ROOT + "/clubb_release/clubb_python_api",
         env.get("PYTHONPATH", "")])

    print(f"=== whole-driver differentiability dashboard ({len(cases)} cases) ===\n")
    print(f"{'case':<24}{'thlm':<12}{'um':<12}{'worst-FD':<11}status")
    print("-" * 74)
    n_finite = n_complete = 0
    bad = []
    for case in cases:
        fin, worst, blocked = _parse(_run_case(case, env))
        if blocked:
            status = f"BLOCKED {blocked}"
            bad.append(case)
        else:
            finite = fin["thlm"] and fin["um"] and all(
                a == b for a, b in (p.split("/") for p in (fin["thlm"], fin["um"])))
            n_finite += bool(finite)
            fd_ok = worst is not None and worst < _FD_KINK_TOL
            if finite and fd_ok:
                status = "COMPLETE"; n_complete += 1
            elif finite:
                status = "KINK (grad finite, FD kink)"
                if args.strict:
                    bad.append(case)
            else:
                status = "PARTIAL (some levels nan)"; bad.append(case)
        w = f"{worst:.1e}" if worst is not None else "-"
        print(f"{case:<24}{fin['thlm'] or '-':<12}{fin['um'] or '-':<12}{w:<11}{status}")
    print("-" * 74)
    print(f"grad-finite: {n_finite}/{len(cases)}   FD-correct (COMPLETE): {n_complete}/{len(cases)}")
    gate = "PASS" if not bad else "FAIL"
    print(f"\nDifferentiability gate [{'strict' if args.strict else 'finite'}]: {gate}"
          + (f" — {bad}" if bad else " — all cases differentiable"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
