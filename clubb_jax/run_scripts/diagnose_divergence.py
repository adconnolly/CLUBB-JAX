#!/usr/bin/env python3
"""diagnose_divergence.py — per-step onset/growth classifier for a non-bit-faithful case.

Generalises the per-step decomposition done by hand across many investigations
(rico, dycoms2_rf02_morr, nov11, coriolis_test): instead of a single final-step
rel error (what compare_runs.py reports), this reads the per-step compare stats and,
for every PROGNOSTIC variable, reports

  * the FIRST timestep it exceeds the gate (ABS_TOL + REL_TOL*|ref|, the compare_runs rule),
  * its per-step max|Δ| growth, and
  * a CLASSIFICATION of the onset:
      bit-faithful  — never exceeds the gate over the run,
      FP-growth     — |Δ| is already above the abs floor and grows gradually before
                      the gate cross (the undamped-oscillator / variance-floor signature),
      JUMP@N        — |Δ| jumps from ~bit-exact (<= ABS_TOL) straight past the gate in one
                      step: a DISCRETE onset → a threshold/term bug is worth ruling out.

This is the single most reused diagnostic in the project's history; the JUMP-vs-FP
distinction is exactly what separated real bugs (the two Iter307 KK fixes; the Iter233
sed axis bug) from genuine FP limits.

Prereq: run `compare_runs.py --case CASE --max-iters N` first to generate the per-step
stats under clubb_release/output/{case}_compare_{fort,jax}/. This tool only reads them.

Usage:
    python run_scripts/diagnose_divergence.py CASE [--var upwp] [--all] [--jump-ratio 1e3]

  --var V       restrict to one variable (else every failing prognostic).
  --all         include variables that pass the gate (else only failing ones).
  --jump-ratio  the |Δ| step-to-step factor (from <=ABS_TOL) that counts as a JUMP (default 1e3).
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

from compare_runs import CLUBB_ROOT, PROGNOSTIC, REL_TOL, ABS_TOL


def _stats(case: str):
    fp = os.path.join(CLUBB_ROOT, f"output/{case}_compare_fort/{case}_stats.nc")
    jp = os.path.join(CLUBB_ROOT, f"output/{case}_compare_jax/{case}_stats.nc")
    for p in (fp, jp):
        if not os.path.isfile(p):
            sys.exit(f"missing {p} — run compare_runs.py --case {case} first")
    import netCDF4 as nc
    return nc.Dataset(fp), nc.Dataset(jp)


def _per_step(fv: np.ndarray, jv: np.ndarray):
    """Return (max|Δ| per step, fails per step). Record axis is 0.

    The gate denominator is the GLOBAL max|ref| (over all steps/levels), matching
    compare_runs.py's verdict — so a variable flagged here is one compare_runs also
    fails on; only the onset STEP is new information."""
    fv = fv.reshape(fv.shape[0], -1)
    jv = jv.reshape(jv.shape[0], -1)
    md = np.nanmax(np.abs(fv - jv), axis=1)
    global_ref = float(np.nanmax(np.abs(fv)))
    fails = md > (ABS_TOL + REL_TOL * global_ref)
    return md, fails


def _classify(md: np.ndarray, fails: np.ndarray, jump_ratio: float) -> str:
    """Classify the divergence by its PHYSICAL onset (first departure from the bit-exact
    floor ABS_TOL), independent of the ref-based gate — so the onset isn't masked when a
    field's global |ref| makes the gate threshold high. The gate-cross step (the compare_runs
    'failure' point) is reported separately."""
    if not fails.any():
        return "bit-faithful (never exceeds gate)"
    gate_step = int(np.argmax(fails)) + 1
    left_floor = md > ABS_TOL          # departures from bit-exact
    onset = int(np.argmax(left_floor)) if left_floor.any() else len(md) - 1
    prev = md[onset - 1] if onset > 0 else 0.0
    cur = md[onset]
    # JUMP: bit-exact (<= ABS_TOL) up to onset-1, then a >=jump_ratio leap past the floor —
    # a DISCRETE onset (threshold switch / term first activating) worth ruling out as a bug.
    if prev <= ABS_TOL and cur > ABS_TOL * jump_ratio:
        kind = f"JUMP@step{onset + 1} (|Δ| {prev:.1e} -> {cur:.1e}; discrete — rule out a term/threshold bug)"
    else:
        kind = f"FP-growth (onset step{onset + 1}, |Δ| {prev:.1e} -> {cur:.1e}, gradual)"
    return f"{kind}; gate-cross step{gate_step}"


def _sign_tally(fv: np.ndarray, jv: np.ndarray, gate_step: int) -> str:
    """jax>fort vs jax<fort level-counts at the gate-cross step and two before it.
    Balanced (~50/50) → FP / chaotic decorrelation; strongly one-sided → a systematic bias
    (a term/coefficient bug). Masked to levels where |fort| > 1e-10 (real-signal levels)."""
    fv = fv.reshape(fv.shape[0], -1)
    jv = jv.reshape(jv.shape[0], -1)
    out = []
    for r in sorted({max(0, gate_step - 4), max(0, gate_step - 2), gate_step}):
        m = np.abs(fv[r]) > 1e-10
        d = (jv[r] - fv[r])[m]
        out.append(f"s{r + 1}:>{int((d > 0).sum())}/<{int((d < 0).sum())}")
    return "  ".join(out) + "   (balanced→FP/chaos | one-sided→systematic bias)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--var", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jump-ratio", type=float, default=1e3)
    a = ap.parse_args()

    F, J = _stats(a.case)
    names = ({a.var} if a.var else PROGNOSTIC) & set(F.variables) & set(J.variables)
    if not names:
        sys.exit("no matching prognostic variables in the stats")

    rows = []
    for v in sorted(names):
        fv = np.asarray(F[v][:], np.float64)
        jv = np.asarray(J[v][:], np.float64)
        if fv.shape != jv.shape:
            continue
        md, fails = _per_step(fv, jv)
        rows.append((v, md, fails, _classify(md, fails, a.jump_ratio), fv, jv))

    nstep = len(rows[0][1]) if rows else 0
    print(f"case {a.case}: {nstep} steps, gate = ABS_TOL({ABS_TOL:.0e}) + REL_TOL({REL_TOL:.0e})*|ref|\n")
    for v, md, fails, cls, fv, jv in sorted(rows, key=lambda r: (not r[2].any(), r[0])):
        if not (a.all or a.var) and not fails.any():
            continue
        # compact per-step |Δ| trace (cap to first 16 for readability)
        trace = " ".join(f"{x:.1e}" for x in md[:16])
        print(f"{v:<10} {cls}")
        print(f"           |Δ|/step: {trace}{' ...' if len(md) > 16 else ''}")
        if fails.any():
            print(f"           sign: {_sign_tally(fv, jv, int(np.argmax(fails)))}")
    F.close()
    J.close()


if __name__ == "__main__":
    main()
