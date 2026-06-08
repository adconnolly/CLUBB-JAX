#!/usr/bin/env python3
"""test_gabls3_night_stability.py — pin the GABLS3-night Businger-Dyer surface-layer stability functions.

`gm1/gh1/fm1/fh1` (gabls3_night.py:26-31 ↔ gabls3_night.F90:163-217) are the Monin-Obukhov stability functions
that the `landflx` surface scheme composes; they are bit-validated end-to-end (gabls3_night IS in the bit-faithful
regression set) but never unit-pinned in ISOLATION, so a coefficient typo would only surface as a full-case
divergence. This pins each against an INDEPENDENT transcription of the Fortran formulas (so the 15 / 9 / 0.74 / π/2
coefficients are checked directly):
    gm1(x) = (1 − 15x)^0.25
    gh1(x) = sqrt(1 − 9x) / 0.74
    fm1(x) = 2·log((1+x)/2) + log((1+x²)/2) − 2·atan(x) + π/2
    fh1(x) = 2·log((1+0.74x)/2)
gm1/gh1 are evaluated over the unstable regime x<0 where `landflx` calls them (so 1−15x>1 and 1−9x>1 — the JAX's
defensive `abs()` in gh1 is a no-op there, forward-identical to the Fortran); fm1/fh1 over the x>1 range of their
gm1/gh1-output arguments. The JAX uses double `log` (vs the Fortran single `alog`), ~1e-7 — the intended, more
accurate computation, and gabls3_night still passes the 1e-6 bit gate. Oracle-independent; never SKIPs. (iter 533)
"""
import os
import sys
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from clubb_jax.src.Benchmark_cases.gabls3_night import gm1, gh1, fm1, fh1, psi_h


def _ref_gm1(x): return (1.0 - 15.0 * x) ** 0.25
def _ref_gh1(x): return math.sqrt(1.0 - 9.0 * x) / 0.74
def _ref_fm1(x): return 2.0 * math.log((1.0 + x) / 2.0) + math.log((1.0 + x * x) / 2.0) \
    - 2.0 * math.atan(x) + math.pi / 2.0
def _ref_fh1(x): return 2.0 * math.log((1.0 + 0.74 * x) / 2.0)


def test_gm1_gh1_unstable_regime():
    worst = 0.0
    for x in np.linspace(-2.0, -1.0e-3, 50):     # unstable: z/L < 0 (where landflx calls gm1/gh1)
        worst = max(worst, abs(float(gm1(float(x))) - _ref_gm1(x)),
                    abs(float(gh1(float(x))) - _ref_gh1(x)))
    assert worst < 1e-13, f"gm1/gh1 mismatch vs F90 formula {worst:.2e}"
    print(f"  gm1=(1−15x)^¼, gh1=√(1−9x)/0.74 over x∈[−2,0): match F90 formula (worst {worst:.1e})  PASS")


def test_fm1_fh1_formula():
    worst = 0.0
    for x in np.linspace(1.0, 6.0, 50):          # gm1/gh1 outputs are > 1 in the unstable regime
        worst = max(worst, abs(float(fm1(float(x))) - _ref_fm1(x)),
                    abs(float(fh1(float(x))) - _ref_fh1(x)))
    assert worst < 1e-12, f"fm1/fh1 mismatch vs F90 formula {worst:.2e}"
    print(f"  fm1 (log+atan+π/2), fh1=2·log((1+0.74x)/2): match F90 formula (worst {worst:.1e})  PASS")


def test_gh1_abs_is_noop_in_domain():
    """The JAX gh1 uses sqrt(abs(1−9x)) (grad-safety); the Fortran has sqrt(1−9x). In the called domain (x<0 →
    1−9x>1>0) the abs is a no-op, so the two are forward-identical — pin that they agree there."""
    for x in (-0.5, -0.01, -1.5, 0.0):
        assert abs(float(gh1(x)) - math.sqrt(1.0 - 9.0 * x) / 0.74) < 1e-13
    print("  gh1 abs() is forward-identical to the Fortran sqrt(1−9x)/0.74 in the x≤0 call domain  PASS")


def test_psi_h_stable():
    """psi_h(x, xlmo) = −5·x/xlmo — the stable-case integrated heat stability function (gabls3_night.F90:150).
    Linear in x; pin the −5 coefficient and the 1/xlmo dependence (added iter 574, completing the gabls3_night
    surface stability functions gm1/gh1/fm1/fh1/psi_h)."""
    for x, xlmo in ((0.25, 50.0), (10.0, 200.0), (2.0, 30.0)):
        assert abs(float(psi_h(x, xlmo)) - (-5.0 * x) / xlmo) < 1e-14, f"psi_h({x},{xlmo})"
    # the landflx combination psi_h(0.25,xlmo) − psi_h(h,xlmo) telescopes linearly in (0.25−h)
    assert abs((psi_h(0.25, 100.0) - psi_h(5.0, 100.0)) - (-5.0 * (0.25 - 5.0) / 100.0)) < 1e-14
    print("  psi_h = −5·x/xlmo (stable-case heat stability function)  PASS")


def main():
    print("test_gabls3_night_stability:")
    test_gm1_gh1_unstable_regime()
    test_fm1_fh1_formula()
    test_gh1_abs_is_noop_in_domain()
    test_psi_h_stable()
    print("All gabls3_night stability-function checks PASSED")


if __name__ == "__main__":
    main()
