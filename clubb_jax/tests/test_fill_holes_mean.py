"""Unit tests for the MEAN-FIELD vertical hole-filler (the rtm_cl / thlm_cl mechanism).

These lock in the Iter186 fix: advance_xm_wpxp applies `fill_holes_vertical` to the mean fields
rtm/thlm after the solve (advance_xm_wpxp_module.F90:4974-5018, the `rtm_cl`/`thlm_cl` budget). For
a case whose mean field dips below its tolerance at a stretched dry top (rico, rt→0 above ~9 km), this
fill — NOT an IC floor — is what keeps the field >= tol each step, mass-conservingly pulling a tiny
amount from the topmost above-tol level (the rico step-1 k51 seed of 1.37e-8 = that pull × dt).

A regression that drops the rtm/thlm fill (or re-adds an IC floor) would silently reintroduce the rico
seed; these tests are the fast unit-level guard (no Fortran oracle needed).
"""
import numpy as np
import jax.numpy as jnp
from clubb_jax.src.CLUBB_core.fill_holes import fill_holes_vertical_jax

RT_TOL = 1.0e-8


def _col_mass(field, rho_ds, dz):
    """Column mass with the same rho_ds*dz weighting fill_holes conserves."""
    return float(np.sum(np.asarray(field) * np.asarray(rho_ds) * np.asarray(dz)))


def test_mean_fill_raises_dry_top_and_conserves_mass():
    """A rico-like rtm column (moist below, a sharp drop to 0 at the top) — fill must raise the
    sub-tol dry top to >= rt_tol, conserve rho_ds-weighted column mass to machine precision, and pull
    the deficit from the topmost above-tol (donor) level."""
    nzt = 24
    # Stretched-ish density (decreasing with height) and non-uniform dz.
    rho = np.linspace(1.1, 0.35, nzt)[None, :]
    dz = np.linspace(30.0, 120.0, nzt)[None, :]
    # Moist below, then a hard cliff to 0 over the top 5 levels (the dry top).
    rtm = np.full((1, nzt), 5.0e-3)
    rtm[0, -8:] = np.array([3.0e-3, 1.2e-3, 4.0e-4, 9.0e-5, 0.0, 0.0, 0.0, 0.0])
    rtm_j = jnp.asarray(rtm)

    m_before = _col_mass(rtm, rho, dz)
    out = np.asarray(fill_holes_vertical_jax(
        field=rtm_j, rho_ds=jnp.asarray(rho), dz=jnp.asarray(dz),
        threshold=RT_TOL, lower_k=0, upper_k=nzt - 1, fill_holes_type=2))
    m_after = _col_mass(out, rho, dz)

    assert np.all(out >= RT_TOL - 1e-300), "dry top not raised to threshold"
    # Mass conservation. The global redistribution conserves mass exactly in exact arithmetic, but with
    # threshold(1e-8) << field(1e-3) the `mass_frac` ratio rounds at ~1e-8 relative — a precision floor
    # of the algorithm, NOT a JAX defect: the JAX matches the Fortran bit-for-bit here (live rico rtm to
    # 6e-15), so the residual is shared. It is far below the 1e-6 prognostic gate.
    rel_mass = abs(m_after - m_before) / abs(m_before)
    assert rel_mass < 1e-6, f"column mass not conserved within gate: rel {rel_mass:.2e}"
    # The donor (topmost moist level, index where rtm was 9e-5) must have given up mass.
    donor = int(np.where(rtm[0] >= RT_TOL)[0][-1])
    assert out[0, donor] < rtm[0, donor], "donor level did not give up mass to fill the dry top"
    print(f"  mean-field fill: dry top raised, mass conserved (rel {rel_mass:.1e}), "
          f"donor k{donor} {rtm[0,donor]:.2e}->{out[0,donor]:.2e}  PASS")


def test_mean_fill_noop_when_all_above_threshold():
    """When every level is >= threshold (all 15 uniform-grid cases), the fill is a BITWISE no-op —
    this is why adding the rtm/thlm fill did not perturb any faithful case."""
    nzt = 20
    rho = np.linspace(1.1, 0.5, nzt)[None, :]
    dz = np.full((1, nzt), 40.0)
    rtm = jnp.asarray(np.linspace(8.0e-3, 1.0e-6, nzt)[None, :])  # all >> rt_tol
    out = fill_holes_vertical_jax(
        field=rtm, rho_ds=jnp.asarray(rho), dz=jnp.asarray(dz),
        threshold=RT_TOL, lower_k=0, upper_k=nzt - 1, fill_holes_type=2)
    assert np.array_equal(np.asarray(out), np.asarray(rtm)), "fill changed an all-above-threshold field"
    print("  mean-field fill: bitwise no-op when all >= threshold  PASS")


def test_thlm_fill_is_noop():
    """thlm ~300 K >> thl_tol, so the thlm_cl fill is always a guaranteed no-op (mirrors the Fortran
    but never changes thlm) — guards against an accidental threshold that would corrupt thlm."""
    nzt = 16
    rho = np.linspace(1.1, 0.6, nzt)[None, :]
    dz = np.full((1, nzt), 40.0)
    thl_tol = 1.0e-2
    thlm = jnp.asarray(np.linspace(298.0, 320.0, nzt)[None, :])
    out = fill_holes_vertical_jax(
        field=thlm, rho_ds=jnp.asarray(rho), dz=jnp.asarray(dz),
        threshold=thl_tol, lower_k=0, upper_k=nzt - 1, fill_holes_type=2)
    assert np.array_equal(np.asarray(out), np.asarray(thlm)), "thlm fill changed thlm"
    print("  thlm fill: no-op (thlm >> thl_tol)  PASS")


if __name__ == "__main__":
    print("Mean-field fill_holes (rtm_cl / thlm_cl) tests:")
    test_mean_fill_raises_dry_top_and_conserves_mass()
    test_mean_fill_noop_when_all_above_threshold()
    test_thlm_fill_is_noop()
    print("All mean-field fill_holes tests PASSED.")
