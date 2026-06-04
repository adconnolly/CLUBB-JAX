#!/usr/bin/env python3
"""test_silhs_surface_schemes.py — validate the SILHS-blocked cases' surface schemes (mpace_b, arm_97, twp_ice).

Each is bit-exact vs a literal NumPy transcription of the Fortran (the bulk/MOST machinery — diag_ustar and the
drag law — reuses already-validated routines), plus physical checks and a finite jax.grad. twp_ice is verified
to equal cloud_feedback_sfclyr (algebraically identical drag law).
"""
import os
import sys
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clubb_jax.src.Benchmark_cases.mpace_b import mpace_b_sfclyr
from clubb_jax.src.Benchmark_cases.arm_97 import arm_97_sfclyr, _Z0 as _ARM97_Z0
from clubb_jax.src.Benchmark_cases.twp_ice import twp_ice_sfclyr
from clubb_jax.src.Benchmark_cases.cloud_feedback import cloud_feedback_sfclyr
from clubb_jax.src.Benchmark_cases.arm import _diag_ustar
from clubb_jax.src.CLUBB_core.constants_clubb import Cp, Lv, grav

_SAT = 3


def test_mpace_b_sfclyr():
    rng = np.random.default_rng(1)
    H = rng.uniform(-50, 200, 100); LH = rng.uniform(0, 300, 100); rho = rng.uniform(1.0, 1.3, 100)
    wth, wrt, ust = mpace_b_sfclyr(jnp.asarray(H), jnp.asarray(LH), jnp.asarray(rho))
    assert np.max(np.abs(np.asarray(wth) - H / (rho * Cp))) < 1e-15
    assert np.max(np.abs(np.asarray(wrt) - LH / (rho * Lv))) < 1e-15
    assert np.all(np.asarray(ust) == 0.25), "mpace_b ustar should be 0.25"
    print("  mpace_b_sfclyr vs literal (kinematic fluxes + ustar=0.25): exact  PASS")


def test_arm_97_sfclyr():
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(100):
        H, LH = rng.uniform(-30, 250), rng.uniform(0, 350)
        z, rho, thlm, ubar = rng.uniform(10, 40), rng.uniform(1.0, 1.3), rng.uniform(295, 305), rng.uniform(1, 8)
        wth, wrt, ust = arm_97_sfclyr(H, LH, z, rho, thlm, ubar)
        rwth = H / (rho * Cp); rwrt = LH / (rho * Lv)
        rust = _diag_ustar(z, grav / thlm * rwth, ubar, _ARM97_Z0)
        worst = max(worst, abs(float(wth) - rwth), abs(float(wrt) - rwrt), abs(float(ust) - rust))
    assert worst < 1e-12, f"arm_97_sfclyr vs literal {worst:.2e}"
    print(f"  arm_97_sfclyr vs literal (kinematic fluxes + diag_ustar): max diff {worst:.1e}  PASS")


def test_twp_ice_equals_cloud_feedback():
    args = (jnp.asarray([295.0, 290.0]), jnp.asarray([0.012, 0.015]), jnp.asarray([25.0, 40.0]),
            jnp.asarray([5.0, 7.0]), jnp.asarray([1.0e5, 0.98e5]), jnp.asarray([298.0, 300.0]), _SAT)
    a = twp_ice_sfclyr(*args)
    b = cloud_feedback_sfclyr(*args)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y)), "twp_ice != cloud_feedback drag law"
    print("  twp_ice_sfclyr == cloud_feedback_sfclyr (identical RICO drag law)  PASS")


def test_arm_3year_arm_0003_equal_arm_97():
    from clubb_jax.src.Benchmark_cases.arm_3year import arm_3year_sfclyr
    from clubb_jax.src.Benchmark_cases.arm_0003 import arm_0003_sfclyr
    args = (200.0, 150.0, 25.0, 1.1, 300.0, 4.0)
    ref = tuple(float(x) for x in arm_97_sfclyr(*args))
    for fn, name in ((arm_3year_sfclyr, "arm_3year"), (arm_0003_sfclyr, "arm_0003")):
        got = tuple(float(x) for x in fn(*args))
        assert got == ref, f"{name}_sfclyr != arm_97_sfclyr"
    print("  arm_3year_sfclyr == arm_0003_sfclyr == arm_97_sfclyr (identical scheme)  PASS")


def test_differentiable():
    g = jax.grad(lambda T: arm_97_sfclyr(200.0, 150.0, 25.0, 1.1, T, 4.0)[2])(jnp.asarray(300.0))
    g2 = jax.grad(lambda H: mpace_b_sfclyr(H, 150.0, jnp.asarray(1.1))[0])(jnp.asarray(200.0))
    assert np.isfinite(float(g)) and np.isfinite(float(g2)), "non-finite grad"
    print(f"  jax.grad: arm_97 d ustar/d thlm = {float(g):+.3e}, mpace_b d wpthlp/d H = {float(g2):+.3e}: finite  PASS")


def main():
    print("test_silhs_surface_schemes:")
    for t in (test_mpace_b_sfclyr, test_arm_97_sfclyr, test_twp_ice_equals_cloud_feedback,
              test_arm_3year_arm_0003_equal_arm_97, test_differentiable):
        t()
    print("All SILHS-blocked surface-scheme checks PASSED")


if __name__ == "__main__":
    main()
