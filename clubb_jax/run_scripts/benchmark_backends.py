#!/usr/bin/env python3
"""Benchmark CLUBB throughput across backends: JAX-GPU, JAX-CPU, Fortran-CPU.

Separates the one-time JIT-compile cost (step 1) from steady-state per-step
throughput (steps 2..N), and sweeps the column dimension (``-multicol``) — the
natural data-parallel axis for a single-column model and the axis on which the
GPU is expected to win.

The JAX timing comes from the in-loop ``CLUBB_JAX_BENCH=1`` instrumentation in
``advance_clubb_to_end`` (per-step wall time; the per-step host round-trip forces
a device sync, so the deltas are accurate). The Fortran timing is the wall time
of the standalone subprocess (init is negligible for Fortran).

Backend selection is by environment, so this script must run under the jaxenv
python with the GPU libs reachable. Set that up once from the repo root via the
tracked template (``cp jaxenv.sh.example jaxenv.sh``, edit ``_JAXENV``), then
``source jaxenv.sh``:
  * jax-gpu : default (GPU backend)
  * jax-cpu : JAX_PLATFORMS=cpu
  * fortran : run_scm.py -legacy

Example:
  source jaxenv.sh
  python clubb_jax/run_scripts/benchmark_backends.py \
      --case arm --steps 12 --ngrdcol 1 8 64 256 \
      --backends jax-gpu jax-cpu fortran
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JAX_ROOT = HERE.parent.parent                     # CLUBB-JAX/
RUN_SCM = HERE / "run_scm.py"
RESULTS_DIR = JAX_ROOT / "jax_driver_test_results"

BENCH_RE = re.compile(r"BENCH_JSON\s+(\{.*\})")
FTIMER_RE = re.compile(r"CLUBB-TIMER\s+(\w+)\s*=\s*([0-9.eE+-]+)")


def _base_env() -> dict:
    env = os.environ.copy()
    env["CLUBB_JAX_BENCH"] = "1"
    return env


def run_jax(case: str, ngrdcol: int, steps: int, gpu: bool,
            precision: str = "double") -> dict:
    """Run one JAX case; return the parsed BENCH_JSON dict (+ wall time)."""
    env = _base_env()
    env["CLUBB_JAX_PRECISION"] = precision
    if gpu:
        env.pop("JAX_PLATFORMS", None)
    else:
        env["JAX_PLATFORMS"] = "cpu"
    # -stats none → no per-step sampling, so l_sample is constant False and the
    # whole-step jit compiles exactly one variant; measures pure compute
    # throughput (the production/inference path).
    cmd = [sys.executable, str(RUN_SCM), case, "-jax",
           "-max_iters", str(steps), "-multicol", str(ngrdcol), "-stats", "none"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(JAX_ROOT), env=env,
                       capture_output=True, text=True)
    wall = time.time() - t0
    m = BENCH_RE.search(p.stdout)
    if not m:
        return {"error": True, "wall_s": wall,
                "tail": (p.stdout + p.stderr)[-800:]}
    d = json.loads(m.group(1))
    d["wall_s"] = wall
    d["error"] = False
    return d


def run_fortran(case: str, ngrdcol: int, steps: int) -> dict:
    """Run one Fortran case; wall-time the subprocess (init is negligible)."""
    env = os.environ.copy()
    # No backend flag → run_scm uses install/latest/clubb_standalone (the
    # compiled intel oracle); -legacy points at bin/ which is not populated.
    cmd = [sys.executable, str(RUN_SCM), case,
           "-max_iters", str(steps), "-multicol", str(ngrdcol), "-stats", "none"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(JAX_ROOT), env=env,
                       capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        return {"error": True, "wall_s": wall,
                "tail": (p.stdout + p.stderr)[-800:]}
    # Fortran prints its own internal timers (init-time/binary-load excluded):
    # time_total = whole program loop; time_clubb_advance = closure only.
    timers = {k: float(v) for k, v in FTIMER_RE.findall(p.stdout)}
    t_total = timers.get("time_total", wall)
    return {"error": False, "wall_s": wall, "n_steps": steps,
            "timers": timers,
            "t_total_s": t_total,
            "t_advance_s": timers.get("time_clubb_advance"),
            "t_steady_mean_s": t_total / max(steps, 1)}


def _configs(backends, precisions):
    """Expand (backend × precision) into labeled run configs.

    JAX backends run once per precision (label `jax-gpu/f32`); Fortran runs once
    at its compiled precision (label `fortran`)."""
    out = []
    for be in backends:
        if be == "fortran":
            out.append(("fortran", be, None))
        else:
            for prec in precisions:
                tag = {"double": "f64", "single": "f32"}.get(prec, prec)
                out.append((f"{be}/{tag}", be, prec))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="arm")
    ap.add_argument("--steps", type=int, default=12,
                    help="timesteps to run (step 1 = compile; 2..N = steady)")
    ap.add_argument("--ngrdcol", type=int, nargs="+", default=[1, 8, 64, 256])
    ap.add_argument("--backends", nargs="+",
                    default=["jax-gpu", "jax-cpu", "fortran"],
                    choices=["jax-gpu", "jax-cpu", "fortran"])
    ap.add_argument("--precision", nargs="+", default=["double"],
                    choices=["double", "single"],
                    help="float precision(s) for the JAX backends (Fortran uses "
                         "its compiled precision)")
    ap.add_argument("--out", default=str(RESULTS_DIR / "backend_benchmark.json"))
    args = ap.parse_args()

    configs = _configs(args.backends, args.precision)
    labels = [c[0] for c in configs]

    results = []
    for nc in args.ngrdcol:
        for label, be, prec in configs:
            print(f"[bench] {args.case} ngrdcol={nc} {label} ...", flush=True)
            if be == "fortran":
                r = run_fortran(args.case, nc, args.steps)
            else:
                r = run_jax(args.case, nc, args.steps,
                            gpu=(be == "jax-gpu"), precision=prec)
            r.update({"case": args.case, "ngrdcol": nc, "backend": be,
                      "label": label, "precision": prec or "compiled"})
            results.append(r)
            if r.get("error"):
                print(f"    ERROR: {r.get('tail','')[:300]}", flush=True)
            else:
                steady = r.get("t_steady_mean_s", float("nan"))
                comp = r.get("t_step1_s")
                comp_s = f" compile(step1)={comp:.1f}s" if comp else ""
                mem = r.get("peak_mem_bytes")
                mem_s = f" peakmem={mem/2**20:.0f}MiB" if mem else ""
                print(f"    steady={steady*1e3:.1f} ms/step wall={r['wall_s']:.1f}s"
                      f"{comp_s}{mem_s}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"case": args.case, "steps": args.steps, "results": results}, f,
                  indent=2)
    print(f"\n[bench] wrote {args.out}")

    _print_table(results, args.ngrdcol, labels, "t_steady_mean_s", 1e3,
                 "Steady-state ms/step (lower is better)")
    # Peak device memory only meaningful for GPU configs.
    if any("gpu" in l for l in labels):
        _print_table(results, args.ngrdcol,
                     [l for l in labels if "gpu" in l],
                     "peak_mem_bytes", 1 / 2**20,
                     "Peak GPU memory (MiB)")


def _print_table(results, ngrdcols, labels, key, scale, title):
    by = {(r["ngrdcol"], r["label"]): r for r in results}
    print(f"\n### {title}\n")
    print("| ngrdcol | " + " | ".join(labels) + " |")
    print("|" + "---|" * (len(labels) + 1))
    for nc in ngrdcols:
        cells = []
        for lab in labels:
            r = by.get((nc, lab))
            val = r.get(key) if r and not r.get("error") else None
            cells.append(f"{val*scale:.1f}" if val else "—")
        print(f"| {nc} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
