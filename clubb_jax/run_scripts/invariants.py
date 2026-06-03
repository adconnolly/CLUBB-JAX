#!/usr/bin/env python3
"""invariants.py — Tier-A oracle-free physical invariants for a CLUBB stats file (REFACTOR.md §2).

Tier A is the *strict* gate that replaces the bug-catching value of bit-faithfulness: it does not
need the Fortran oracle, only a JAX run's stats NetCDF. It checks the properties that must hold for
any physically-valid CLUBB state:

  * FINITENESS    — no NaN/Inf in the prognostic fields.
  * POSITIVITY    — variances (wp2, up2, vp2, rtp2, thlp2, em) and hydrometeors/mixing-ratios
                    (rtm, rrm, Nrm, rcm, Ncm, …) are >= 0 (within a small clipping slack).
  * CAUCHY–SCHWARZ — every resolved covariance respects |cov| <= sqrt(var1·var2): a correlation
                    cannot exceed 1. (CLUBB enforces this via clip_covar; a violation is a real bug.)

Usage:
    from invariants import check_all
    v = check_all(nc.Dataset(path)); assert v["ok"], v["violations"]
  or:  python invariants.py <stats.nc>
"""
from __future__ import annotations
import numpy as np

# Fields that must be finite if present.
_FINITE = ["thlm", "rtm", "um", "vm", "wp2", "wp3", "rtp2", "thlp2", "rtpthlp",
           "wpthlp", "wprtp", "upwp", "vpwp", "up2", "vp2", "em"]
# Fields that must be non-negative if present.
_NONNEG = ["wp2", "up2", "vp2", "rtp2", "thlp2", "em",
           "rtm", "rrm", "Nrm", "rcm", "Ncm", "rim", "rsm", "rgm", "Nim", "Nsm", "Ngm"]
# Cauchy–Schwarz pairs: (covariance, variance1, variance2). All on the zm grid (same shape).
_CS_PAIRS = [
    ("wprtp",   "wp2",  "rtp2"),
    ("wpthlp",  "wp2",  "thlp2"),
    ("rtpthlp", "rtp2", "thlp2"),
    ("upwp",    "up2",  "wp2"),
    ("vpwp",    "vp2",  "wp2"),
]


def _arr(ds, name):
    if name not in ds.variables:
        return None
    return np.asarray(ds[name][:], dtype=np.float64)


def check_finiteness(ds):
    out = []
    for v in _FINITE:
        a = _arr(ds, v)
        if a is not None and not np.all(np.isfinite(a)):
            out.append((v, "non-finite", int(np.sum(~np.isfinite(a)))))
    return out


def check_positivity(ds, rel=1e-9, floor=1e-12):
    """Non-negativity within a small slack (clip leaves variances >= a positive tol)."""
    out = []
    for v in _NONNEG:
        a = _arr(ds, v)
        if a is None:
            continue
        scale = float(np.nanmax(np.abs(a))) if a.size else 0.0
        tol = rel * scale + floor
        mn = float(np.nanmin(a))
        if mn < -tol:
            out.append((v, "negative", mn, -tol))
    return out


def check_cauchy_schwarz(ds, slack=1e-3, floor=1e-30):
    """|cov| <= sqrt(var1·var2)·(1+slack) + floor at every level (correlation <= 1)."""
    out = []
    for cov, v1, v2 in _CS_PAIRS:
        c, a1, a2 = _arr(ds, cov), _arr(ds, v1), _arr(ds, v2)
        if c is None or a1 is None or a2 is None:
            continue
        if not (c.shape == a1.shape == a2.shape):
            continue  # different grids — skip rather than mis-compare
        bound = np.sqrt(np.maximum(a1, 0.0) * np.maximum(a2, 0.0)) * (1.0 + slack) + floor
        viol = np.abs(c) > bound
        if np.any(viol):
            # worst offender: largest |cov|/bound
            ratio = np.abs(c) / np.maximum(bound, floor)
            idx = np.unravel_index(np.nanargmax(ratio), ratio.shape)
            out.append((cov, f"|corr|>1 vs {v1},{v2}", float(ratio[idx]), int(np.sum(viol))))
    return out


def check_all(ds, *, cs_slack=1e-3, pos_rel=1e-9):
    """Run all Tier-A checks; return {'ok': bool, 'violations': {...}}."""
    viols = {
        "finiteness":     check_finiteness(ds),
        "positivity":     check_positivity(ds, rel=pos_rel),
        "cauchy_schwarz": check_cauchy_schwarz(ds, slack=cs_slack),
    }
    ok = all(len(v) == 0 for v in viols.values())
    return {"ok": ok, "violations": viols}


def format_result(res) -> str:
    lines = ["Tier-A invariants:"]
    for kind, vs in res["violations"].items():
        lines.append(f"  {kind:<14} {'OK' if not vs else 'VIOLATIONS (' + str(len(vs)) + ')'}")
        for v in vs:
            lines.append(f"      {v}")
    lines.append(f"  Tier-A verdict: {'PASS' if res['ok'] else 'FAIL'}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import netCDF4 as nc
    if len(sys.argv) < 2:
        sys.exit("usage: python invariants.py <stats.nc>")
    res = check_all(nc.Dataset(sys.argv[1]))
    print(format_result(res))
    sys.exit(0 if res["ok"] else 1)
