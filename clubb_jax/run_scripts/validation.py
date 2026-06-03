#!/usr/bin/env python3
"""validation.py — field-class numerical-accuracy tolerances for the tiered correctness standard.

This is the single source of truth for the **Tier-C** (physical-fidelity vs Fortran) tolerances
described in REFACTOR.md §2. It is intentionally small and importable so that `compare_runs.py`
(and later `validate_case.py` / `compare_cases.py`) consume the SAME classification instead of each
re-implementing it (adversarial-review refinement R1).

Tier-C rule per variable `v` in a *gated* class:  PASS iff  max|Δ| <= ABS_FLOOR + tol_class·max|ref|.
Variables not in a gated class are **report-only** (never gate) — budgets, spurious-source, positive-
definite-correction, and other timing-confounded diagnostics.
"""
from __future__ import annotations

# Absolute floor shared with the bit gate: protects fields that are physically ~0 in a given case
# (e.g. moisture moments in a dry case) from enormous relative error off pure roundoff.
ABS_FLOOR = 1e-12

# ── Field classes and their Tier-C relative tolerances (REFACTOR.md §2 starting points) ──
# Rationale: means are best-conditioned; fluxes looser; second moments are variance-floor /
# cancellation sensitive; microphysics rates have FP-sensitive sharp edges.
MEANS = {"thlm", "rtm", "um", "vm", "sclrm"}
FLUXES = {"wpthlp", "wprtp", "upwp", "vpwp"}
MOMENTS = {"wp2", "wp3", "rtp2", "thlp2", "rtpthlp", "up2", "vp2", "em"}
# Microphysics PROGNOSTIC hydrometeor fields (means/numbers). NOTE: the `*_mc` *tendency/source*
# fields (rcm_mc, thlm_mc, rtm_mc, …) are deliberately NOT here — calibration (iter4, rico) showed they
# behave like timing-confounded budget diagnostics (rel≈1 on tiny noisy tendencies), exactly the fields
# the bit-gate excluded from its PROGNOSTIC set. They are classified 'diagnostic' (report-only); the
# microphysics RATES are validated by the unit tests + Tier-A conservation, and the prognostic
# hydrometeors below carry the accuracy signal.
MICROPHYS = {"rrm", "Nrm", "rim", "rsm", "rgm", "Nim", "Nsm", "Ngm", "rcm", "Ncm"}

CLASS_TOL = {
    "mean": 1e-4,
    "flux": 1e-3,
    "moment": 3e-3,
    "microphys": 1e-2,
}


def classify(var: str) -> str:
    """Return the field class of `var`: 'mean'|'flux'|'moment'|'microphys'|'diagnostic'.

    'diagnostic' is report-only (not gated)."""
    if var in MEANS:
        return "mean"
    if var in FLUXES:
        return "flux"
    if var in MOMENTS:
        return "moment"
    if var in MICROPHYS:
        return "microphys"
    # `*_mc` microphysics tendencies are timing-confounded sources, not prognostic state → report-only.
    return "diagnostic"


def tier_c_tol(var: str):
    """Tier-C relative tolerance for `var`, or None if the variable is report-only."""
    return CLASS_TOL.get(classify(var))


def tier_c_pass(max_diff: float, max_ref: float, tol: float) -> bool:
    """The Tier-C PASS predicate: max|Δ| <= ABS_FLOOR + tol·max|ref|."""
    return max_diff <= ABS_FLOOR + tol * max_ref


def tiered_verdict(rows):
    """Compute the Tier-C verdict over comparison rows.

    `rows` is an iterable of (var, max_diff, max_ref, rel) tuples (as produced by compare_runs).
    Returns a dict with: all_pass (bool over gated vars), per-class counts, and the list of gated
    failures [(var, class, rel, tol), ...].
    """
    per_class = {}          # class -> [n_pass, n_fail]
    worst = {}              # class -> (var, rel)  worst rel error in the class (the binding margin)
    failures = []
    for var, max_diff, max_ref, rel in rows:
        cls = classify(var)
        if cls == "diagnostic":
            continue
        tol = CLASS_TOL[cls]
        ok = tier_c_pass(max_diff, max_ref, tol)
        slot = per_class.setdefault(cls, [0, 0])
        slot[0 if ok else 1] += 1
        if cls not in worst or rel > worst[cls][1]:
            worst[cls] = (var, rel)
        if not ok:
            failures.append((var, cls, rel, tol))
    all_pass = len(failures) == 0
    return {"all_pass": all_pass, "per_class": per_class, "worst": worst, "failures": failures}


def format_verdict(verdict) -> str:
    """One-block human-readable Tier-C summary."""
    lines = ["  Tier-C (physical fidelity vs Fortran):"]
    worst = verdict.get("worst", {})
    for cls in ("mean", "flux", "moment", "microphys"):
        if cls in verdict["per_class"]:
            npass, nfail = verdict["per_class"][cls]
            tol = CLASS_TOL[cls]
            wv, wr = worst.get(cls, ("-", 0.0))
            # margin = how many× below tol the worst field is (>1 means passing with headroom)
            margin = (tol / wr) if wr > 0 else float("inf")
            lines.append(f"    {cls:<10} tol={tol:.0e}  pass={npass}  fail={nfail}  "
                         f"worst={wr:.2e} ({wv}, {margin:.0f}x margin)")
    for var, cls, rel, tol in verdict["failures"]:
        lines.append(f"    ** {var:<22} class={cls:<10} rel={rel:.2e} > tol={tol:.0e}")
    lines.append(f"  Tier-C verdict: {'PASS' if verdict['all_pass'] else 'FAIL'}")
    return "\n".join(lines)
