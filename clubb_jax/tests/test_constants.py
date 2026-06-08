#!/usr/bin/env python3
"""test_constants.py — the physical/numerical constants mirror constants_clubb.F90 value-for-value.

The constants (Cp, Lv, Ls, Lf, Rd, Rv, grav, p0, T_freeze_K, pi, sqrt_2, …) underpin ALL thermodynamics and PDF math;
a single fat-fingered value would mis-physics the whole model. The bit-faithful cases catch a wrong constant only via a
slow full run — this is the fast, direct guard.

Source-grounded (not transcription): it parses the LITERAL `name = <number>_core_rknd` assignments straight out of the
Fortran `constants_clubb.F90` and compares each to the JAX `constants_clubb.<name>`. Comments are stripped first (so the
parser ignores `! ep = 0.622` approximations and commented-out `! grav = 9.80665` lines), and only single-literal
assignments match (so fraction expressions like `three_halves = 3.0/2.0` are skipped, not mis-read as 3.0). The literal
pattern also auto-skips the `#ifdef CLUBB_CAM` branch, whose constants are `shr_const_*` references, not literals — so the
first occurrence is always the standalone-build value the JAX mirrors. SKIPs if the Fortran oracle source is absent.
(iter 469)
"""
import os
import re
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from clubb_jax.src.CLUBB_core import constants_clubb as cc

_F90 = os.path.join(_ROOT, "clubb_release", "src", "CLUBB_core", "constants_clubb.F90")
_LITERAL = re.compile(
    r"\s*([A-Za-z]\w*)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eEdD][-+]?[0-9]+)?)_core_rknd[\s,&]*$")


def _fortran_literal_constants():
    fort = {}
    for raw in open(_F90):
        line = raw.split("!", 1)[0]                      # strip trailing comment
        m = _LITERAL.match(line)
        if m and m.group(1) not in fort:                 # first occurrence = standalone branch
            fort[m.group(1)] = float(m.group(2).replace("d", "e").replace("D", "e"))
    return fort


def test_constants_match_fortran_source():
    if not os.path.exists(_F90):
        print("  constants_clubb.F90 oracle source not present — SKIP")
        return
    fort = _fortran_literal_constants()
    assert len(fort) > 20, f"parsed only {len(fort)} literal constants — the F90 extraction broke"
    checked, mism = 0, []
    for name, fv in fort.items():
        if not hasattr(cc, name):
            continue
        jv = getattr(cc, name)
        if not isinstance(jv, (int, float)) or isinstance(jv, bool):
            continue
        checked += 1
        if abs(float(jv) - fv) > abs(fv) * 1e-12 + 1e-30:
            mism.append(f"{name}: JAX {jv} vs Fortran {fv}")
    # The critical thermodynamic constants must be among those compared (guards against a parser regression
    # that silently checks nothing).
    must = {"Cp", "Lv", "Rd", "Rv", "grav", "p0", "T_freeze_K"}
    present = {n for n in fort if hasattr(cc, n)}
    missing = sorted(must - present)
    assert not missing, f"key constants not found in the F90 parse (extraction regressed): {missing}"
    assert not mism, "JAX constants diverge from constants_clubb.F90:\n  " + "\n  ".join(mism)
    print(f"  {checked} literal constants (Cp/Lv/Rd/Rv/grav/p0/T_freeze_K/…) match constants_clubb.F90 exactly  PASS")


def main():
    print("test_constants:")
    test_constants_match_fortran_source()
    print("All constants checks PASSED")


if __name__ == "__main__":
    main()
