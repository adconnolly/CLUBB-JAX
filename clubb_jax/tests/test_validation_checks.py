#!/usr/bin/env python3
"""test_validation_checks.py — validate the ported CLUBB check routines (assert_corr_symmetric, sfc_varnce_check).

These are the last in-scope check routines (corr_varnce_module.F90 / numerical_check.F90). Their f2py wrappers
set err_code on `stored_err_info` and `return` (no error-stop, and the f2py does NOT expose err_code), so there is
no observable f2py oracle — validation is by the Fortran source logic (behavioral) plus an f2py no-crash
cross-check on the valid case.
  assert_corr_symmetric: True iff symmetric within 1e-6 AND unit diagonal within eps (1e-10).
  sfc_varnce_check:       True iff every surface variance/covariance is finite.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for p in (_ROOT + "/clubb_release", _ROOT + "/clubb_release/clubb_python_api"):
    if p not in sys.path:
        sys.path.append(p)

import numpy as np

from clubb_jax.src.CLUBB_core.corr_varnce_module import assert_corr_symmetric
from clubb_jax.src.CLUBB_core.numerical_check import sfc_varnce_check


def test_assert_corr_symmetric_behavior():
    n = 5
    rng = np.random.default_rng(2)
    lower = rng.uniform(-0.5, 0.5, (n, n))
    sym = np.tril(lower, -1) + np.tril(lower, -1).T + np.eye(n)   # symmetric, unit diagonal
    assert assert_corr_symmetric(sym) is True, "valid symmetric unit-diagonal matrix rejected"
    # Asymmetric (perturb one off-diagonal beyond tol)
    asym = sym.copy(); asym[0, 1] += 1e-3
    assert assert_corr_symmetric(asym) is False, "asymmetric matrix accepted"
    # Within tolerance (perturb below tol) → still accepted
    near = sym.copy(); near[0, 1] += 5e-7; near[1, 0] += 0.0
    assert assert_corr_symmetric(near) is True, "within-tol asymmetry wrongly rejected"
    # Non-unit diagonal
    nd = sym.copy(); nd[2, 2] = 0.9
    assert assert_corr_symmetric(nd) is False, "non-unit-diagonal matrix accepted"
    print("  assert_corr_symmetric: symmetric+unit-diag PASS / asym + non-unit-diag FAIL  PASS")


def test_assert_corr_symmetric_f2py_nocrash():
    try:
        import clubb_f2py
    except Exception as e:
        print(f"  f2py assert_corr_symmetric (no-crash valid case): SKIP ({type(e).__name__})")
        return
    n = 6
    rng = np.random.default_rng(7)
    lower = rng.uniform(-0.4, 0.4, (n, n))
    sym = np.tril(lower, -1) + np.tril(lower, -1).T + np.eye(n)
    # The f2py sets err_code on stored_err_info and returns (no error-stop). A valid matrix must not raise.
    clubb_f2py.f2py_assert_corr_symmetric(np.asfortranarray(sym))
    assert assert_corr_symmetric(sym) is True
    print("  f2py assert_corr_symmetric: valid matrix runs without error + JAX agrees  PASS")


def test_sfc_varnce_check_behavior():
    ng = 1
    ok = dict(wp2_sfc=0.3, up2_sfc=0.2, vp2_sfc=0.2, thlp2_sfc=0.5, rtp2_sfc=1e-6, rtpthlp_sfc=1e-4)
    assert sfc_varnce_check(0, **ok) is True, "all-finite surface variances rejected"
    bad = dict(ok); bad['thlp2_sfc'] = np.nan
    assert sfc_varnce_check(0, **bad) is False, "NaN thlp2_sfc accepted"
    inf = dict(ok); inf['wp2_sfc'] = np.inf
    assert sfc_varnce_check(0, **inf) is False, "Inf wp2_sfc accepted"
    # scalar fields finite but a NaN passive scalar with sclr_dim>0
    assert sfc_varnce_check(1, sclrp2_sfc=np.array([np.nan]), sclrprtp_sfc=np.array([0.0]),
                            sclrpthlp_sfc=np.array([0.0]), **ok) is False, "NaN scalar variance accepted"
    print("  sfc_varnce_check: all-finite PASS / NaN+Inf FAIL (incl. passive scalars)  PASS")


def main():
    print("test_validation_checks:")
    for t in (test_assert_corr_symmetric_behavior, test_assert_corr_symmetric_f2py_nocrash,
              test_sfc_varnce_check_behavior):
        t()
    print("All validation-check ports PASSED")


if __name__ == "__main__":
    main()
