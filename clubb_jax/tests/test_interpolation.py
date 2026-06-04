#!/usr/bin/env python3
"""test_interpolation.py — validate the JAX interpolation.py port (lin_interpolate_two_points, mono_cubic_interp).

Oracles:
  1. f2py bit-shadow: clubb_f2py.f2py_lin_interpolate_two_points and f2py_mono_cubic_interp on the same args
     (the compiled default uses the Steffen cubic, l_quintic_poly_interp=False). SKIPs if clubb_f2py is unbuilt.
  2. Closed-form linear interpolation identity.
  3. Steffen monotonicity: the cubic stays within [f00, fp1] for monotone data on [z00, zp1].
  4. A finite jax.grad.
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
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.interpolation import (
    lin_interpolate_two_points, mono_cubic_interp, linear_interp_factor, zlinterp_fnc)


def _binary_search(array, var):
    """Literal transcription of interpolation.F90:binary_search (1-based index of the >= bracket, -1 if out)."""
    n = len(array)
    low, high = 2, n
    if var < array[0] or var > array[n - 1] or n < 2:
        return -1
    if array[0] <= var <= array[1]:
        return 2
    while low <= high:
        i = (low + high) // 2
        if array[i - 2] < var <= array[i - 1]:
            return i
        elif var < array[i - 1]:
            high = i - 1
        else:
            low = i + 1
    return -1


def _zlinterp_ref(grid_out, grid_src, var_src):
    """Literal Fortran zlinterp_fnc (binary_search + lin_interpolate_two_points, zero outside range)."""
    out = np.zeros(len(grid_out))
    for kint, go in enumerate(grid_out):
        if go < grid_src[0]:
            continue
        k = _binary_search(grid_src, go)
        if k == -1:
            break
        km1 = max(1, k - 1)
        out[kint] = ((go - grid_src[km1 - 1]) / (grid_src[k - 1] - grid_src[km1 - 1])
                     * (var_src[k - 1] - var_src[km1 - 1]) + var_src[km1 - 1])
    return out

# Branch configurations: (km1, k00, kp1, kp2) exercising km1==k00 / kp1==kp2 / interior / extrapolate.
_CONFIGS = [(0, 0, 1, 2), (0, 1, 2, 2), (0, 1, 2, 3), (2, 1, 2, 3)]
_Z = (0.0, 100.0, 250.0, 450.0)        # zm1, z00, zp1, zp2 (monotone increasing)
_F = (1.0, 2.5, 3.2, 3.9)              # fm1, f00, fp1, fp2 (monotone increasing)


def test_lin_interp_identity():
    val = float(lin_interpolate_two_points(150.0, 200.0, 100.0, 5.0, 1.0))
    assert abs(val - ((150.0 - 100.0) / (200.0 - 100.0) * (5.0 - 1.0) + 1.0)) < 1e-14
    # Endpoints reproduce the known values.
    assert abs(float(lin_interpolate_two_points(100.0, 200.0, 100.0, 5.0, 1.0)) - 1.0) < 1e-14
    assert abs(float(lin_interpolate_two_points(200.0, 200.0, 100.0, 5.0, 1.0)) - 5.0) < 1e-14
    print("  lin_interpolate_two_points: closed-form + endpoints  PASS")


def test_f2py_oracle():
    try:
        import clubb_f2py
    except Exception as e:
        print(f"  f2py interpolation oracle: SKIP ({type(e).__name__})")
        return
    # lin_interpolate_two_points
    r = float(clubb_f2py.f2py_lin_interpolate_two_points(150.0, 200.0, 100.0, 5.0, 1.0))
    j = float(lin_interpolate_two_points(150.0, 200.0, 100.0, 5.0, 1.0))
    assert abs(j - r) < 1e-13, f"lin_interp f2py mismatch {abs(j-r):.2e}"
    # mono_cubic_interp over all branch configs + a few interpolation altitudes.
    worst = 0.0
    zm1, z00, zp1, zp2 = _Z
    fm1, f00, fp1, fp2 = _F
    for (km1, k00, kp1, kp2) in _CONFIGS:
        for z_in in (120.0, 175.0, 240.0):
            ref = float(clubb_f2py.f2py_mono_cubic_interp(
                z_in, km1, k00, kp1, kp2, zm1, z00, zp1, zp2, fm1, f00, fp1, fp2))
            got = float(mono_cubic_interp(
                z_in, km1, k00, kp1, kp2, zm1, z00, zp1, zp2, fm1, f00, fp1, fp2))
            worst = max(worst, abs(got - ref))
    assert worst < 1e-11, f"mono_cubic_interp f2py mismatch {worst:.2e}"
    print(f"  f2py lin + mono_cubic_interp: bit-match over 4 branches x 3 altitudes, worst {worst:.2e}  PASS")


def test_monotonicity():
    # Steffen's method keeps the interpolant within [f00, fp1] for monotone data between z00 and zp1.
    zm1, z00, zp1, zp2 = _Z
    fm1, f00, fp1, fp2 = _F
    for km1, k00, kp1, kp2 in ((0, 1, 2, 3),):
        for z_in in np.linspace(z00, zp1, 21):
            v = float(mono_cubic_interp(z_in, km1, k00, kp1, kp2, zm1, z00, zp1, zp2, fm1, f00, fp1, fp2))
            assert f00 - 1e-12 <= v <= fp1 + 1e-12, f"non-monotone at z={z_in}: {v}"
    print("  Steffen monotonicity: interpolant stays within [f00, fp1]  PASS")


def test_differentiable():
    zm1, z00, zp1, zp2 = _Z
    def loss(f):
        fm1, f00, fp1, fp2 = f
        return mono_cubic_interp(175.0, 0, 1, 2, 3, zm1, z00, zp1, zp2, fm1, f00, fp1, fp2)
    g = np.asarray(jax.grad(loss)(jnp.array(_F)))
    assert np.isfinite(g).all(), "non-finite grad through mono_cubic_interp"
    print(f"  jax.grad through mono_cubic_interp: finite ({g.size} entries)  PASS")


def test_linear_interp_factor():
    assert abs(float(linear_interp_factor(0.25, 8.0, 4.0)) - (0.25 * (8.0 - 4.0) + 4.0)) < 1e-14
    assert abs(float(linear_interp_factor(0.0, 8.0, 4.0)) - 4.0) < 1e-14
    assert abs(float(linear_interp_factor(1.0, 8.0, 4.0)) - 8.0) < 1e-14
    print("  linear_interp_factor: closed-form + endpoints  PASS")


def test_zlinterp():
    rng = np.random.default_rng(13)
    grid_src = np.sort(rng.uniform(0.0, 10000.0, 30))
    var_src = rng.standard_normal(30)
    grid_out = np.sort(rng.uniform(-500.0, 11000.0, 50))   # straddles both ends -> zero-fill exercised
    got = np.asarray(zlinterp_fnc(grid_out, grid_src, var_src))
    ref = _zlinterp_ref(grid_out, grid_src, var_src)
    assert np.max(np.abs(got - ref)) < 1e-12, f"zlinterp mismatch {np.max(np.abs(got-ref)):.2e}"
    # Zero-fill below/above the source range.
    assert got[grid_out < grid_src[0]].tolist() == [0.0] * int((grid_out < grid_src[0]).sum())
    assert np.all(got[grid_out > grid_src[-1]] == 0.0)
    print("  zlinterp_fnc: matches literal binary_search+lin_interp, zero-fill outside range  PASS")


def test_zlinterp_differentiable():
    grid_src = np.linspace(0.0, 1000.0, 12)
    grid_out = np.linspace(50.0, 950.0, 20)
    def loss(v):
        return jnp.sum(zlinterp_fnc(grid_out, grid_src, v) ** 2)
    g = np.asarray(jax.grad(loss)(jnp.asarray(np.random.default_rng(1).standard_normal(12))))
    assert np.isfinite(g).all(), "non-finite grad through zlinterp_fnc"
    print(f"  jax.grad through zlinterp_fnc: finite ({g.size} entries)  PASS")


def main():
    print("test_interpolation:")
    for t in (test_lin_interp_identity, test_f2py_oracle, test_monotonicity, test_differentiable,
              test_linear_interp_factor, test_zlinterp, test_zlinterp_differentiable):
        t()
    print("All interpolation checks PASSED")


if __name__ == "__main__":
    main()
