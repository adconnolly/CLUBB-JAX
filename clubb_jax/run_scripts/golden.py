#!/usr/bin/env python3
"""golden.py — Tier-B golden-trajectory regression (REFACTOR.md §2, Phase 0 P0.3).

Tier B replaces "bit-faithful to Fortran" with "bit-faithful to our own validated JAX": a stored
golden JAX trajectory per case, compared at a tight rel (~1e-9) to catch UNINTENDED self-regressions
during refactors that should be numerically neutral (e.g. the bounded-scan / state-dict changes).
Intended accuracy changes (Phase 1) are handled by a deliberate re-baseline via `update_golden.py`.

Storage convention (R5): the golden `.nc` live under `clubb_jax/output/golden/<case>/` (gitignored),
and a small **manifest** (`clubb_jax/golden_manifest.json`, tracked) records provenance + per-variable
checksums. Checksums are of the float64 CONTENT of the gated variables (not raw HDF5 bytes, which carry
nondeterministic metadata). They detect same-machine drift; the real rel-1e-9 comparison uses the local
golden arrays (XLA/hardware differences mean checksums are not portable across machines — regenerate
the golden on a fresh clone before relying on Tier B).
"""
from __future__ import annotations
import hashlib
import json
import os

import numpy as np

import validation  # gated-field classification (shared single source of truth)

RUN_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
JAX_ROOT = os.path.normpath(os.path.join(RUN_SCRIPTS, "../.."))           # CLUBB-JAX/
GOLDEN_DIR = os.path.join(JAX_ROOT, "clubb_jax", "output", "golden")       # gitignored (.nc)
MANIFEST_PATH = os.path.join(JAX_ROOT, "clubb_jax", "golden_manifest.json")  # tracked


def golden_stats_path(case: str) -> str:
    return os.path.join(GOLDEN_DIR, case, f"{case}_stats.nc")


def gated_vars(ds):
    """Gated (non-diagnostic) variables present in the dataset, sorted."""
    return sorted(v for v in ds.variables if validation.classify(v) != "diagnostic")


def checksums(ds) -> dict:
    """Per-variable sha256 of the float64 content (+ shape) for the gated variables."""
    out = {}
    for v in gated_vars(ds):
        a = np.ascontiguousarray(np.asarray(ds[v][:], dtype=np.float64))
        out[v] = {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape)}
    return out


def load_manifest() -> dict:
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(m: dict):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)
        f.write("\n")


def compare_to_golden(case: str, current_nc: str, rel: float = 1e-9, floor: float = 1e-30):
    """Tier-B: compare a current stats file to the stored golden for `case`.

    Returns {'ok', 'fails': [(var, relerr)], 'n_checked', 'reason'}. ok=False with reason set if the
    golden is missing (so the caller can prompt a baseline)."""
    import netCDF4 as nc
    gp = golden_stats_path(case)
    if not os.path.isfile(gp):
        return {"ok": False, "fails": [], "n_checked": 0,
                "reason": f"no golden for '{case}' — run update_golden.py --case {case}"}
    dg = nc.Dataset(gp)
    dc = nc.Dataset(current_nc)
    fails, n = [], 0
    for v in gated_vars(dg):
        if v not in dc.variables:
            continue
        g = np.asarray(dg[v][:], dtype=np.float64)
        c = np.asarray(dc[v][:], dtype=np.float64)
        if g.shape != c.shape:
            fails.append((v, float("inf")))
            continue
        n += 1
        denom = float(np.nanmax(np.abs(g))) + floor
        relerr = float(np.nanmax(np.abs(g - c))) / denom
        if relerr > rel:
            fails.append((v, relerr))
    dg.close(); dc.close()
    return {"ok": len(fails) == 0, "fails": sorted(fails, key=lambda x: -x[1]),
            "n_checked": n, "reason": ""}
