"""JAX implementations of advance_wp2_wp3 sub-functions.

Faithful port of CLUBB_core/advance_wp2_wp3_module.F90.

Functions implemented:
  advance_wp2_wp3 -- full pentadiagonal LHS/RHS assembly and solve
                         for the coupled wp2/wp3 system (ADG1 PDF, ascending grid).

All functions operate in float64.

References:
  src/CLUBB_core/advance_wp2_wp3_module.F90
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.grid_class import zm2zt_jax, zm2zt2zm_jax, zm2zt, zt2zm
from clubb_jax.src.CLUBB_core.diffusion import (
    diffusion_zm_lhs_jax,
    diffusion_zt_lhs_jax,
)
from clubb_jax.src.CLUBB_core.mean_adv import term_ma_zm_lhs_jax, term_ma_zt_lhs_jax
from clubb_jax.src.CLUBB_core.penta_lu_solver import penta_lu_solve_jax
from clubb_jax.src.CLUBB_core.fill_holes import (
    fill_holes_vertical, fill_holes_wp2_from_horz_tke,
)
from clubb_jax.src.CLUBB_core.clip_explicit import clip_variance, clip_skewness
from clubb_jax.src.CLUBB_core.advance_xp2_xpyp_module import (
    apply_lhs_band3_interior_jax, apply_lhs_band2_zt2zm_interior_jax,
    finalize_implicit_budget_interior_jax,
)
from clubb_jax.src.CLUBB_core.tracer_numpy import _asarray, _xp, _iset

import clubb_jax.src.CLUBB_core.constants_clubb as cc

_gamma = cc.gamma_over_implicit_ts   # 1.5
_grav  = cc.grav                     # 9.81
_two_thirds = cc.two_thirds          # 2/3
_one_third  = 1.0 / 3.0


# ===========================================================================
# Private helpers
# ===========================================================================

def wp2_term_ta_lhs(invrs_rho_ds_zm, rho_ds_zt, gr):
    """Turbulent advection LHS for wp2 (zm-level variable).

    LHS[0] = coeff of wp3[k_py]   (super1 in pentadiag)
    LHS[1] = coeff of wp3[k_py-1] (sub1 in pentadiag)

    Shape: (2, ngrdcol, nzm).  Boundaries are 0.
    """
    fac = invrs_rho_ds_zm[:, 1:-1] * gr.invrs_dzm[:, 1:-1]
    lhs = jnp.zeros((2,) + invrs_rho_ds_zm.shape)
    lhs = lhs.at[0, :, 1:-1].set(fac * rho_ds_zt[:, 1:])
    lhs = lhs.at[1, :, 1:-1].set(-fac * rho_ds_zt[:, :-1])
    return lhs


def wp3_term_ta_ADG1_lhs(wp2, a1_coef_zt, a3_coef_zt, wp3_on_wp2,
                           rho_ds_zm, invrs_rho_ds_zt, gr):
    """5-band ADG1 turbulent advection LHS for wp3 (zt-level variable).

    Band 0 (super2): coeff of wp3[k+1]
    Band 1 (super1): coeff of wp2[k+1]
    Band 2 (main):   coeff of wp3[k]
    Band 3 (sub1):   coeff of wp2[k]
    Band 4 (sub2):   coeff of wp3[k-1]

    Shape: (5, ngrdcol, nzt).  Boundaries are 0.

    Faithful port of wp3_term_ta_ADG1_lhs for l_standard_term_ta=False
    (non-standard TA, uses a1 for wp3 fluxes, a3 for wp2 fluxes).
    """
    lhs = jnp.zeros((5,) + invrs_rho_ds_zt.shape)

    inv  = invrs_rho_ds_zt[:, 1:-1]    # (ngrdcol, nzt-2)
    a1   = a1_coef_zt[:, 1:-1]
    a3   = a3_coef_zt[:, 1:-1]
    idzt = gr.invrs_dzt[:, 1:-1]

    # At zt level k_py (1..nzm-3), zm levels k_py and k_py+1 are adjacent.
    # gr.weights_zt2zm indexed by zm level; T_ABOVE=0, T_BELOW=1.
    rho_up  = rho_ds_zm[:, 2:-1]    # rho_ds_zm[k_py+1]
    rho_lo  = rho_ds_zm[:, 1:-2]    # rho_ds_zm[k_py]
    wp2_up  = wp2[:, 2:-1]          # wp2[k_py+1]
    wp2_lo  = wp2[:, 1:-2]          # wp2[k_py]
    w3w2_up = wp3_on_wp2[:, 2:-1]   # wp3_on_wp2[k_py+1]
    w3w2_lo = wp3_on_wp2[:, 1:-2]   # wp3_on_wp2[k_py]
    wt_up_tab = gr.weights_zt2zm[:, 2:-1, 0]  # T_ABOVE for zm[k_py+1]
    wt_up_tbe = gr.weights_zt2zm[:, 2:-1, 1]  # T_BELOW for zm[k_py+1]
    wt_lo_tab = gr.weights_zt2zm[:, 1:-2, 0]  # T_ABOVE for zm[k_py]
    wt_lo_tbe = gr.weights_zt2zm[:, 1:-2, 1]  # T_BELOW for zm[k_py]

    # Band 0: coefficient of wp3[k_py+1]
    lhs = lhs.at[0, :, 1:-1].set(inv * a1 * idzt * rho_up * w3w2_up * wt_up_tab)
    # Band 1: coefficient of wp2[k_py+1]
    lhs = lhs.at[1, :, 1:-1].set(inv * a3 * idzt * rho_up * wp2_up)
    # Band 2: coefficient of wp3[k_py]
    lhs = lhs.at[2, :, 1:-1].set(
        inv * a1 * idzt * (rho_up * w3w2_up * wt_up_tbe - rho_lo * w3w2_lo * wt_lo_tab)
    )
    # Band 3: coefficient of wp2[k_py]
    lhs = lhs.at[3, :, 1:-1].set(-inv * a3 * idzt * rho_lo * wp2_lo)
    # Band 4: coefficient of wp3[k_py-1]
    lhs = lhs.at[4, :, 1:-1].set(-inv * a1 * idzt * rho_lo * w3w2_lo * wt_lo_tbe)
    return lhs


def wp3_term_tp_lhs(coef, wp2, rho_ds_zm, invrs_rho_ds_zt, gr):
    """Turbulent production LHS for wp3.

    coef: (ngrdcol,) scalar coefficient per column.
    Shape: (2, ngrdcol, nzt).  Boundaries are 0.

    Band 0 (k_mdiag):   coeff of wp2[k_py+1]
    Band 1 (km1_mdiag): coeff of wp2[k_py]
    """
    lhs  = jnp.zeros((2,) + invrs_rho_ds_zt.shape)
    c    = coef[:, None]                         # (ngrdcol, 1)
    inv  = invrs_rho_ds_zt[:, 1:-1]
    idzt = gr.invrs_dzt[:, 1:-1]
    rho_up = rho_ds_zm[:, 2:-1]
    wp2_up = wp2[:, 2:-1]
    rho_lo = rho_ds_zm[:, 1:-2]
    wp2_lo = wp2[:, 1:-2]
    lhs = lhs.at[0, :, 1:-1].set(
        c * (-3.0 * inv * idzt * rho_up * wp2_up + 1.5 * idzt * wp2_up)
    )
    lhs = lhs.at[1, :, 1:-1].set(
        c * (3.0 * inv * idzt * rho_lo * wp2_lo - 1.5 * idzt * wp2_lo)
    )
    return lhs


def wp3_terms_ac_pr2_lhs(C11_Skw_fnc, wm_zm, gr):
    """Accumulation + pr2 LHS for wp3. (ngrdcol, nzt). Boundaries are 0."""
    lhs = jnp.zeros(C11_Skw_fnc.shape)
    d_wm = wm_zm[:, 2:-1] - wm_zm[:, 1:-2]   # wm_zm[k_py+1] - wm_zm[k_py]
    lhs = lhs.at[:, 1:-1].set(
        (1.0 - C11_Skw_fnc[:, 1:-1]) * 3.0 * gr.invrs_dzt[:, 1:-1] * d_wm
    )
    return lhs


def wp2_terms_ac_pr2_lhs(C_uu_shr, wm_zt, gr):
    """Accumulation + pr2 LHS for wp2. (ngrdcol, nzm). Boundaries are 0.

    C_uu_shr: (ngrdcol,)
    """
    lhs = jnp.zeros((C_uu_shr.shape[0], gr.invrs_dzm.shape[1]))
    d_wm = wm_zt[:, 1:] - wm_zt[:, :-1]    # wm_zt[k_py] - wm_zt[k_py-1]
    lhs = lhs.at[:, 1:-1].set(
        (1.0 - C_uu_shr[:, None]) * 2.0 * gr.invrs_dzm[:, 1:-1] * d_wm
    )
    return lhs


def wp2_term_dp1_lhs(C1_Skw_fnc, invrs_tau_C1_zm):
    """Dissipation term 1 LHS for wp2. (ngrdcol, nzm). Boundaries are 0."""
    lhs = jnp.zeros_like(C1_Skw_fnc)
    lhs = lhs.at[:, 1:-1].set(C1_Skw_fnc[:, 1:-1] * invrs_tau_C1_zm[:, 1:-1])
    return lhs


def wp2_term_pr1_lhs(C4, invrs_tau_C4_zm):
    """Pressure term 1 LHS for wp2 (l_tke_aniso path). (ngrdcol, nzm). Boundaries 0.

    C4: (ngrdcol,)
    """
    lhs = jnp.zeros_like(invrs_tau_C4_zm)
    lhs = lhs.at[:, 1:-1].set(
        (2.0 * C4[:, None] * invrs_tau_C4_zm[:, 1:-1]) / 3.0
    )
    return lhs


def wp3_term_pr1_lhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt):
    """Pressure term 1 LHS for wp3 (l_damp_wp3_Skw_squared path). (ngrdcol, nzt). Boundaries 0."""
    lhs = jnp.zeros_like(invrs_tau_wp3_zt)
    lhs = lhs.at[:, 1:-1].set(
        C8[:, None] * invrs_tau_wp3_zt[:, 1:-1] * (3.0 * C8b[:, None] * Skw_zt[:, 1:-1] ** 2 + 1.0)
    )
    return lhs


# ---------------------------------------------------------------------------
# RHS sub-functions
# ---------------------------------------------------------------------------

def wp2_term_pr_dfsn_rhs(C_wp2_pr_dfsn, rho_ds_zt, invrs_rho_ds_zm, wpup2, wpvp2, wp3, gr):
    """Pressure-diffusion RHS for wp2. (ngrdcol, nzm).

    C_wp2_pr_dfsn: (ngrdcol,)
    """
    wpuip2 = wpup2 + wpvp2 + wp3                 # (ngrdcol, nzt)
    rhs = jnp.zeros_like(invrs_rho_ds_zm)
    fac = C_wp2_pr_dfsn[:, None] * invrs_rho_ds_zm[:, 1:-1] * gr.invrs_dzm[:, 1:-1]
    interior = fac * (rho_ds_zt[:, 1:] * wpuip2[:, 1:] - rho_ds_zt[:, :-1] * wpuip2[:, :-1])
    rhs = rhs.at[:, 1:-1].set(interior)
    rhs = rhs.at[:, 0].set(rhs[:, 1])   # lower BC = first interior
    return rhs


def wp3_term_pr_turb_rhs(C_wp3_pr_turb, rho_ds_zm, invrs_rho_ds_zt, wp2_smth, em_smth, gr):
    """Pressure-turbulence RHS for wp3 (l_use_tke_in_wp3_pr_turb_term=True). (ngrdcol, nzt).

    C_wp3_pr_turb: (ngrdcol,)
    wp2_smth, em_smth: (ngrdcol, nzm) smoothed zm-level arrays.
    """
    rhs = jnp.zeros_like(invrs_rho_ds_zt)
    fac = -C_wp3_pr_turb[:, None] * invrs_rho_ds_zt[:, 1:-1] * gr.invrs_dzt[:, 1:-1]
    val = (
        rho_ds_zm[:, 2:-1] * wp2_smth[:, 2:-1] * em_smth[:, 2:-1]
        - rho_ds_zm[:, 1:-2] * wp2_smth[:, 1:-2] * em_smth[:, 1:-2]
    )
    rhs = rhs.at[:, 1:-1].set(fac * val)
    return rhs


def wp3_term_pr_dfsn_rhs(C_wp3_pr_dfsn, rho_ds_zm, invrs_rho_ds_zt,
                           wp2up2, wp2vp2, wp4, up2, vp2, wp2, gr):
    """Pressure-diffusion RHS for wp3. (ngrdcol, nzt).

    C_wp3_pr_dfsn: (ngrdcol,)
    All zm-level inputs.
    """
    wp2uip2   = wp2up2 + wp2vp2 + wp4           # (ngrdcol, nzm)
    wp2_uip2  = wp2 * (up2 + vp2 + wp2)         # (ngrdcol, nzm)
    net       = wp2uip2 - wp2_uip2
    rhs = jnp.zeros_like(invrs_rho_ds_zt)
    fac = C_wp3_pr_dfsn[:, None] * invrs_rho_ds_zt[:, 1:-1] * gr.invrs_dzt[:, 1:-1]
    val = rho_ds_zm[:, 2:-1] * net[:, 2:-1] - rho_ds_zm[:, 1:-2] * net[:, 1:-2]
    rhs = rhs.at[:, 1:-1].set(fac * val)
    return rhs


def wp2_terms_bp_pr2_rhs(C_uu_buoy, thv_ds_zm, wpthvp):
    """Buoyancy + pressure-2 RHS for wp2. (ngrdcol, nzm). Boundaries 0.

    C_uu_buoy: (ngrdcol,)
    """
    rhs = jnp.zeros_like(thv_ds_zm)
    rhs = rhs.at[:, 1:-1].set(
        (1.0 - C_uu_buoy[:, None]) * 2.0
        * (_grav / thv_ds_zm[:, 1:-1]) * wpthvp[:, 1:-1]
    )
    return rhs


def wp2_term_dp1_rhs(C1_Skw_fnc, invrs_tau_C1_zm, up2, vp2):
    """Dissipation-1 RHS for wp2 (l_damp_wp2_using_em=True). (ngrdcol, nzm). Boundaries 0."""
    rhs = jnp.zeros_like(C1_Skw_fnc)
    rhs = rhs.at[:, 1:-1].set(
        -(C1_Skw_fnc[:, 1:-1] * invrs_tau_C1_zm[:, 1:-1]) * (up2[:, 1:-1] + vp2[:, 1:-1])
    )
    return rhs


def wp2_term_pr3_rhs(C_uu_shr, C_uu_buoy, thv_ds_zm, wpthvp, upwp, um, vpwp, vm, gr):
    """Pressure-3 RHS for wp2. (ngrdcol, nzm). Boundaries 0.

    C_uu_shr, C_uu_buoy: (ngrdcol,)
    um, vm: (ngrdcol, nzt) zt-level!  Fortran advance_wp2_wp3 signature line 243.
    upwp, vpwp: (ngrdcol, nzm) zm-level.
    """
    rhs = jnp.zeros_like(thv_ds_zm)
    buoy = C_uu_buoy[:, None] * (_grav / thv_ds_zm[:, 1:-1]) * wpthvp[:, 1:-1]
    # Fortran loop k=2..nzm-1 accesses um(i,k) and um(i,k-1) where um is zt-level.
    # In Python (0-based): um[k_py] at k_py=1..nzm-2 → um[:, 1:] (nzt elements starting at 1)
    # and um[k_py-1] → um[:, :-1].  Both have shape (ngrdcol, nzm-2).
    shear = C_uu_shr[:, None] * (
        -upwp[:, 1:-1] * gr.invrs_dzm[:, 1:-1] * (um[:, 1:] - um[:, :-1])
        - vpwp[:, 1:-1] * gr.invrs_dzm[:, 1:-1] * (vm[:, 1:] - vm[:, :-1])
    )
    val = _two_thirds * (buoy + shear)
    val = jnp.maximum(val, 0.0)    # zero_threshold clip (Fortran line 4025)
    rhs = rhs.at[:, 1:-1].set(val)
    return rhs


def wp2_term_pr1_rhs(C4, up2, vp2, invrs_tau_C4_zm):
    """Pressure-1 RHS for wp2 (l_tke_aniso=True). (ngrdcol, nzm). Boundaries 0.

    C4: (ngrdcol,)
    """
    rhs = jnp.zeros_like(invrs_tau_C4_zm)
    rhs = rhs.at[:, 1:-1].set(
        (C4[:, None] * (up2[:, 1:-1] + vp2[:, 1:-1]) * invrs_tau_C4_zm[:, 1:-1]) / 3.0
    )
    return rhs


def wp3_terms_bp1_pr2_rhs(C11_Skw_fnc, thv_ds_zt, wp2thvp):
    """Buoyancy + pressure-2 RHS for wp3. (ngrdcol, nzt). Boundaries 0."""
    rhs = jnp.zeros_like(thv_ds_zt)
    rhs = rhs.at[:, 1:-1].set(
        (1.0 - C11_Skw_fnc[:, 1:-1]) * 3.0
        * (_grav / thv_ds_zt[:, 1:-1]) * wp2thvp[:, 1:-1]
    )
    return rhs


def wp3_term_pr1_rhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt, wp3):
    """Pressure-1 RHS for wp3 (l_damp_wp3_Skw_squared=True). (ngrdcol, nzt). Boundaries 0."""
    rhs = jnp.zeros_like(wp3)
    rhs = rhs.at[:, 1:-1].set(
        C8[:, None] * invrs_tau_wp3_zt[:, 1:-1]
        * (2.0 * C8b[:, None] * Skw_zt[:, 1:-1] ** 2) * wp3[:, 1:-1]
    )
    return rhs


# ===========================================================================
# Main advance function
# ===========================================================================

def wp23_rhs(*, nzm, ngrdcol, invrs_dt,
                 rhs_pr_turb_wp3, rhs_pr_dfsn_wp3, rhs_pr_dfsn_wp2,
                 rhs_pr1_wp2, rhs_bp1_pr2_wp3, rhs_pr1_wp3,
                 rhs_bp_pr2_wp2, rhs_pr3_wp2, rhs_dp1_wp2,
                 lhs_pr1_wp2, lhs_tp_wp3, lhs_pr1_wp3, lhs_dp1_wp2, lhs_ta_wp3,
                 wp2, wp3, wp2up, upwp,
                 l_ho_nontrad_coriolis, fcor_y, gr):
    """advance_wp2_wp3_module.F90:wp23_rhs — assemble the coupled wp2/wp3 explicit RHS vector.

    Interleaving: wp2[k] at global index 2k, wp3[k] at 2k+1 (ascending grid). The over-implicit
    contributions use the lhs_* terms; BCs set the four corner rows. Returns rhs (ngrdcol, 2*nzm-1).
    """
    ndim = 2 * nzm - 1
    rhs  = jnp.zeros((ngrdcol, ndim))

    # --- wp3 interior: global indices 3, 5, ..., 2*nzm-5 (slice 3:-2:2) ---
    # rhs_pr_turb_wp3 + rhs_pr_dfsn_wp3 (set, not +=)
    rhs = rhs.at[:, 3:-2:2].set(rhs_pr_turb_wp3[:, 1:-1] + rhs_pr_dfsn_wp3[:, 1:-1])

    # --- wp2 interior: global indices 2, 4, ..., 2*nzm-4 (slice 2:-1:2) ---
    # rhs_pr_dfsn_wp2 (set, not +=)
    rhs = rhs.at[:, 2:-1:2].set(rhs_pr_dfsn_wp2[:, 1:-1])

    # --- l_tke_aniso=True: add pr1 and over-implicit pr1 to wp2 rows ---
    rhs = rhs.at[:, 2:-1:2].add(rhs_pr1_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add((1.0 - _gamma) * (-lhs_pr1_wp2[:, 1:-1] * wp2[:, 1:-1]))

    # --- Time tendency and main terms ---
    # wp3:
    rhs = rhs.at[:, 3:-2:2].add(invrs_dt * wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add(
        (1.0 - _gamma) * (-lhs_tp_wp3[0, :, 1:-1] * wp2[:, 2:-1]
                          - lhs_tp_wp3[1, :, 1:-1] * wp2[:, 1:-2])
    )
    rhs = rhs.at[:, 3:-2:2].add(rhs_bp1_pr2_wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add(rhs_pr1_wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add((1.0 - _gamma) * (-lhs_pr1_wp3[:, 1:-1] * wp3[:, 1:-1]))

    # wp2:
    rhs = rhs.at[:, 2:-1:2].add(invrs_dt * wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_bp_pr2_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_pr3_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_dp1_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add((1.0 - _gamma) * (-lhs_dp1_wp2[:, 1:-1] * wp2[:, 1:-1]))

    # --- ADG1 TA for wp3 (implicit, over-implicit) ---
    # RHS over-implicit contribution: (1-gamma)*(-lhs_ta_wp3 * [wp3/wp2])
    # bands: 0→wp3[k+1], 1→wp2[k+1], 2→wp3[k], 3→wp2[k], 4→wp3[k-1]
    rhs = rhs.at[:, 3:-2:2].add(
        (1.0 - _gamma) * (
            -lhs_ta_wp3[0, :, 1:-1] * wp3[:, 2:]    # wp3[k_py+1]
            - lhs_ta_wp3[1, :, 1:-1] * wp2[:, 2:-1] # wp2[k_py+1]
            - lhs_ta_wp3[2, :, 1:-1] * wp3[:, 1:-1] # wp3[k_py]
            - lhs_ta_wp3[3, :, 1:-1] * wp2[:, 1:-2] # wp2[k_py]
            - lhs_ta_wp3[4, :, 1:-1] * wp3[:, :-2]  # wp3[k_py-1]
        )
    )

    # --- Non-traditional Coriolis (Iter103); no-op for the 15 cases + rico ---
    if l_ho_nontrad_coriolis:
        _fy = jnp.asarray(fcor_y)
        if _fy.ndim == 1:
            _fy = _fy[:, None]
        # wp2 RHS += 2*fcor_y*upwp   (upwp on zm levels, interior)
        rhs = rhs.at[:, 2:-1:2].add(2.0 * _fy * upwp[:, 1:-1])
        # wp3 RHS += 3*fcor_y*wp2up  (wp2up on zt levels, interior)
        rhs = rhs.at[:, 3:-2:2].add(3.0 * _fy * jnp.asarray(wp2up)[:, 1:-1])

    # --- Boundary conditions ---
    # Lower wp2: global 0
    k_lb_zm = gr.k_lb_zm   # Python 0-based lower boundary index for zm
    rhs = rhs.at[:, 0].set(wp2[:, k_lb_zm])
    # Lower wp3: global 1
    rhs = rhs.at[:, 1].set(0.0)
    # Upper wp3: global 2*nzm-3 (= 2*(nzt-1)-1 = 2*nzt-3)
    rhs = rhs.at[:, 2 * nzm - 3].set(0.0)
    # Upper wp2: global 2*nzm-2
    rhs = rhs.at[:, 2 * nzm - 2].set(cc.w_tol_sqd)
    return rhs


def wp23_lhs(*, nzm, ngrdcol, ndim, invrs_dt,
                 lhs_ma_zm, lhs_diff_zm, lhs_ta_wp2, lhs_ac_pr2_wp2,
                 lhs_dp1_wp2, lhs_pr1_wp2, lhs_splat_wp2,
                 lhs_ma_zt, lhs_diff_zt, lhs_tp_wp3, lhs_ac_pr2_wp3,
                 lhs_pr1_wp3, lhs_splat_wp3, lhs_ta_wp3):
    """advance_wp2_wp3_module.F90:wp23_lhs — assemble the coupled wp2/wp3 pentadiagonal LHS matrix.

    Interleaving: wp2[k] at global index 2k, wp3[k] at 2k+1 (ascending grid). The over-implicit
    contributions scale the implicit lhs_* terms by gamma_over_implicit_ts. Identity rows pin the
    four BC corners. Returns lhs (5, ngrdcol, 2*nzm-1).
    """
    lhs = jnp.zeros((5, ngrdcol, ndim))

    # Lower boundary: wp2 (global 0), wp3 (global 1) → identity
    lhs = lhs.at[2, :, 0].set(1.0)
    lhs = lhs.at[2, :, 1].set(1.0)

    # wp2 interior rows (global 2*k_py, k_py=1..nzm-2)
    # Band 0 (super2): lhs_ma_zm[0] + lhs_diff_zm[0]
    lhs = lhs.at[0, :, 2:-1:2].set(lhs_ma_zm[0, :, 1:-1] + lhs_diff_zm[0, :, 1:-1])
    # Band 1 (super1): lhs_ta_wp2[0]  (coeff of wp3[k_py])
    lhs = lhs.at[1, :, 2:-1:2].set(lhs_ta_wp2[0, :, 1:-1])
    # Band 2 (main): lhs_ma_zm[1] + lhs_diff_zm[1] + lhs_ac_pr2_wp2 + gamma*lhs_dp1_wp2 + invrs_dt
    lhs = lhs.at[2, :, 2:-1:2].set(
        lhs_ma_zm[1, :, 1:-1] + lhs_diff_zm[1, :, 1:-1]
        + lhs_ac_pr2_wp2[:, 1:-1]
        + _gamma * lhs_dp1_wp2[:, 1:-1]
        + invrs_dt
    )
    # Band 3 (sub1): lhs_ta_wp2[1]  (coeff of wp3[k_py-1])
    lhs = lhs.at[3, :, 2:-1:2].set(lhs_ta_wp2[1, :, 1:-1])
    # Band 4 (sub2): lhs_ma_zm[2] + lhs_diff_zm[2]
    lhs = lhs.at[4, :, 2:-1:2].set(lhs_ma_zm[2, :, 1:-1] + lhs_diff_zm[2, :, 1:-1])

    # l_tke_aniso=True: add gamma*lhs_pr1_wp2 to main diagonal of wp2 rows
    lhs = lhs.at[2, :, 2:-1:2].add(_gamma * lhs_pr1_wp2[:, 1:-1])

    # lhs_splat_wp2 on main diagonal of wp2 rows
    lhs = lhs.at[2, :, 2:-1:2].add(lhs_splat_wp2[:, 1:-1])

    # wp3 interior rows (global 2*k_py+1, k_py=1..nzt-2)
    # Band 0 (super2): lhs_ma_zt[0] + lhs_diff_zt[0]
    lhs = lhs.at[0, :, 3:-2:2].set(lhs_ma_zt[0, :, 1:-1] + lhs_diff_zt[0, :, 1:-1])
    # Band 1 (super1): gamma*lhs_tp_wp3[0]  (coeff of wp2[k_py+1])
    lhs = lhs.at[1, :, 3:-2:2].set(_gamma * lhs_tp_wp3[0, :, 1:-1])
    # Band 2 (main): lhs_ma_zt[1] + lhs_diff_zt[1] + lhs_ac_pr2_wp3 + gamma*lhs_pr1_wp3 + lhs_splat_wp3 + invrs_dt
    lhs = lhs.at[2, :, 3:-2:2].set(
        lhs_ma_zt[1, :, 1:-1] + lhs_diff_zt[1, :, 1:-1]
        + lhs_ac_pr2_wp3[:, 1:-1]
        + _gamma * lhs_pr1_wp3[:, 1:-1]
        + lhs_splat_wp3[:, 1:-1]
        + invrs_dt
    )
    # Band 3 (sub1): gamma*lhs_tp_wp3[1]  (coeff of wp2[k_py])
    lhs = lhs.at[3, :, 3:-2:2].set(_gamma * lhs_tp_wp3[1, :, 1:-1])
    # Band 4 (sub2): lhs_ma_zt[2] + lhs_diff_zt[2]
    lhs = lhs.at[4, :, 3:-2:2].set(lhs_ma_zt[2, :, 1:-1] + lhs_diff_zt[2, :, 1:-1])

    # ADG1 TA for wp3: add gamma * lhs_ta_wp3 to all 5 bands of wp3 rows
    for b in range(5):
        lhs = lhs.at[b, :, 3:-2:2].add(_gamma * lhs_ta_wp3[b, :, 1:-1])

    # Upper boundaries: wp3 (global 2*nzm-3), wp2 (global 2*nzm-2) → identity
    lhs = lhs.at[2, :, 2 * nzm - 3].set(1.0)
    lhs = lhs.at[2, :, 2 * nzm - 2].set(1.0)
    # Ensure off-diagonals are zero at upper boundaries (they were zero-initialized)
    return lhs


def wp23_solve(lhs, rhs):
    """Pentadiagonal solve of the coupled wp2/wp3 system, then de-interleave the solution —
    the JAX analog of advance_wp2_wp3_module.F90:wp23_solve. The solution vector packs the two
    prognostics on alternating slots; returns ``(wp2_new, wp3_new)`` (wp2 on even slots, wp3 on odd)."""
    solution = penta_lu_solve_jax(lhs, rhs)   # (ngrdcol, 2*nzm-1)
    return solution[:, 0::2], solution[:, 1::2]


def advance_wp2_wp3(
    wp2,              # (ngrdcol, nzm)
    wp3,              # (ngrdcol, nzt)
    up2, vp2,         # (ngrdcol, nzm)
    sigma_sqd_w,      # (ngrdcol, nzm) from pdf_params
    wp3_on_wp2,       # (ngrdcol, nzm)
    em,               # (ngrdcol, nzm)
    wpup2, wpvp2,     # (ngrdcol, nzt)
    wp2up2, wp2vp2, wp4,   # (ngrdcol, nzm)
    wpthvp,           # (ngrdcol, nzm)  zm-level
    wp2thvp,          # (ngrdcol, nzt)  zt-level
    um, vm,           # (ngrdcol, nzt) zt-level (Fortran advance_wp2_wp3 signature)
    upwp, vpwp,       # (ngrdcol, nzm)
    wm_zm, wm_zt,     # (ngrdcol, nzm), (ngrdcol, nzt)
    Kh_zm, Kh_zt,     # (ngrdcol, nzm), (ngrdcol, nzt)
    invrs_tau_C4_zm,  # (ngrdcol, nzm)
    invrs_tau_wp3_zt, # (ngrdcol, nzt)
    invrs_tau_C1_zm,  # (ngrdcol, nzm)
    Skw_zm, Skw_zt,   # (ngrdcol, nzm), (ngrdcol, nzt)
    rho_ds_zm, rho_ds_zt,
    invrs_rho_ds_zm, invrs_rho_ds_zt,
    thv_ds_zm, thv_ds_zt,
    lhs_splat_wp2,    # (ngrdcol, nzm)
    lhs_splat_wp3,    # (ngrdcol, nzt)
    wprtp, wpthlp, rtp2, thlp2,   # (ngrdcol, nzm)
    clubb_params,     # (ngrdcol, nparams)
    dt: float,
    nu1: float,       # background diffusivity for wp2
    nu8: float,       # background diffusivity for wp3
    gr,
    flags=None,
    sfc_elevation=None,
    stats_writer=None,
    l_sample: bool = False,
    l_ho_nontrad_coriolis: bool = False,
    fcor_y=None,
    wp2up=None,
) -> tuple:
    """Full JAX solve for wp2 (zm-level) and wp3 (zt-level).

    Implements the coupled pentadiagonal system for the ADG1 PDF,
    ascending grid, with ARM model flags:
      l_standard_term_ta=False, l_damp_wp2_using_em=True,
      l_use_tke_in_wp3_pr_turb_term=True, l_damp_wp3_Skw_squared=True,
      l_use_C11_Richardson=False, l_tke_aniso=True,
      l_crank_nich_diff=False, l_use_tke_in_wp2_wp3_K_dfsn=False,
      l_lmm_stepping=False.

    Non-traditional Coriolis (Iter103): when l_ho_nontrad_coriolis=True, add the
    explicit RHS contributions wp2 += 2·fcor_y·upwp and wp3 += 3·fcor_y·wp2up
    (fcor_y the meridional Coriolis parameter, wp2up = <w'^2 u'> on zt). It is a
    no-op for the 15 faithful cases + rico (flag False); only coriolis_test uses it.

    Returns (wp2_new, wp3_new) as JAX arrays (raw solve result,
    before fill_holes and clip_variance/skewness).
    """
    ngrdcol = wp2.shape[0]
    nzm     = wp2.shape[1]
    nzt     = wp3.shape[1]
    invrs_dt = 1.0 / dt

    # Broadcast nu to per-column arrays
    nu1_arr = jnp.full((ngrdcol,), nu1)
    nu8_arr = jnp.full((ngrdcol,), nu8)

    # ------------------------------------------------------------------
    # Extract per-column parameters (1-based constants in clubb_params)
    # ------------------------------------------------------------------
    C4       = clubb_params[:, cc.iC4      - 1]
    C8       = clubb_params[:, cc.iC8      - 1]
    C8b      = clubb_params[:, cc.iC8b     - 1]
    C11      = clubb_params[:, cc.iC11     - 1]
    C11b     = clubb_params[:, cc.iC11b    - 1]
    C11c     = clubb_params[:, cc.iC11c    - 1]
    C1       = clubb_params[:, cc.iC1      - 1]
    C1b      = clubb_params[:, cc.iC1b     - 1]
    C1c      = clubb_params[:, cc.iC1c     - 1]
    C12      = clubb_params[:, cc.iC12     - 1]   # (ngrdcol,)
    a3_min   = clubb_params[:, cc.ia3_coef_min - 1]
    C_uu_shr  = clubb_params[:, cc.iC_uu_shr  - 1]
    C_uu_buoy = clubb_params[:, cc.iC_uu_buoy - 1]
    C_wp2_pr_dfsn = clubb_params[:, cc.iC_wp2_pr_dfsn - 1]
    C_wp3_pr_tp   = clubb_params[:, cc.iC_wp3_pr_tp   - 1]
    C_wp3_pr_turb = clubb_params[:, cc.iC_wp3_pr_turb - 1]
    C_wp3_pr_dfsn = clubb_params[:, cc.iC_wp3_pr_dfsn - 1]
    c_K1     = clubb_params[:, cc.ic_K1 - 1]
    c_K8     = clubb_params[:, cc.ic_K8 - 1]

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    # a3_coef (zm): -2*(1-sigma_sqd_w)^2 + 3, clipped from below by a3_coef_min
    a3_coef = -2.0 * (1.0 - sigma_sqd_w) ** 2 + 3.0
    a3_coef = jnp.maximum(a3_coef, a3_min[:, None])

    # a1_coef (zm): 1 / (1 - sigma_sqd_w)
    a1_coef = 1.0 / (1.0 - sigma_sqd_w)

    # Interpolate a3 and a1 to zt levels
    a3_coef_zt = zm2zt_jax(a3_coef, gr)
    a1_coef_zt = zm2zt_jax(a1_coef, gr)

    # C11_Skw_fnc (zt): skewness-dependent C11
    diff_C11 = jnp.abs(C11 - C11b)
    thresh_C11 = jnp.abs(C11 + C11b) * cc.eps / 2.0
    C11_Skw_fnc = jnp.where(
        diff_C11[:, None] > thresh_C11[:, None],
        C11b[:, None] + (C11[:, None] - C11b[:, None]) * jnp.exp(-0.5 * (Skw_zt / C11c[:, None]) ** 2),
        C11b[:, None] * jnp.ones_like(Skw_zt),
    )

    # C1_Skw_fnc (zm): skewness-dependent C1 (x1/3 for l_damp_wp2_using_em=True)
    diff_C1 = jnp.abs(C1 - C1b)
    thresh_C1 = jnp.abs(C1 + C1b) * cc.eps / 2.0
    C1_Skw_fnc = jnp.where(
        diff_C1[:, None] > thresh_C1[:, None],
        C1b[:, None] + (C1[:, None] - C1b[:, None]) * jnp.exp(-0.5 * (Skw_zm / C1c[:, None]) ** 2),
        C1b[:, None] * jnp.ones_like(Skw_zm),
    )
    C1_Skw_fnc = C1_Skw_fnc * _one_third   # absorb 1/3 for l_damp_wp2_using_em=True

    # Eddy diffusivities
    Kw1 = c_K1[:, None] * Kh_zt           # (ngrdcol, nzt) zt-level for wp2
    Kw8 = c_K8[:, None] * Kh_zm           # (ngrdcol, nzm) zm-level for wp3

    # Smoothed fields for wp3_term_pr_turb
    em_smth  = zm2zt2zm_jax(em, gr)       # (ngrdcol, nzm)
    wp2_smth = zm2zt2zm_jax(wp2, gr)      # (ngrdcol, nzm)

    # ------------------------------------------------------------------
    # RHS sub-terms
    # ------------------------------------------------------------------

    rhs_pr_turb_wp3 = wp3_term_pr_turb_rhs(
        C_wp3_pr_turb, rho_ds_zm, invrs_rho_ds_zt, wp2_smth, em_smth, gr
    )
    rhs_pr_dfsn_wp3 = wp3_term_pr_dfsn_rhs(
        C_wp3_pr_dfsn, rho_ds_zm, invrs_rho_ds_zt,
        wp2up2, wp2vp2, wp4, up2, vp2, wp2, gr
    )
    rhs_pr_dfsn_wp2 = wp2_term_pr_dfsn_rhs(
        C_wp2_pr_dfsn, rho_ds_zt, invrs_rho_ds_zm, wpup2, wpvp2, wp3, gr
    )
    rhs_bp_pr2_wp2 = wp2_terms_bp_pr2_rhs(C_uu_buoy, thv_ds_zm, wpthvp)
    rhs_dp1_wp2    = wp2_term_dp1_rhs(C1_Skw_fnc, invrs_tau_C1_zm, up2, vp2)
    rhs_pr3_wp2    = wp2_term_pr3_rhs(C_uu_shr, C_uu_buoy, thv_ds_zm, wpthvp, upwp, um, vpwp, vm, gr)
    rhs_pr1_wp2    = wp2_term_pr1_rhs(C4, up2, vp2, invrs_tau_C4_zm)    # l_tke_aniso=True
    rhs_bp1_pr2_wp3 = wp3_terms_bp1_pr2_rhs(C11_Skw_fnc, thv_ds_zt, wp2thvp)
    rhs_pr1_wp3     = wp3_term_pr1_rhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt, wp3)

    # ------------------------------------------------------------------
    # LHS sub-terms (computed before wp23_rhs but used in both RHS and LHS)
    # ------------------------------------------------------------------

    # Diffusion LHSes
    lhs_diff_zm = diffusion_zm_lhs_jax(Kw1, nu1_arr, invrs_rho_ds_zm, rho_ds_zt, gr)
    lhs_diff_zt = diffusion_zt_lhs_jax(Kw8, nu8_arr, invrs_rho_ds_zt, rho_ds_zm, gr)

    # wp3 turbulent production (two calls: C=1 and C=-C_wp3_pr_tp)
    ones_col = jnp.ones((ngrdcol,))
    lhs_adv_tp_wp3 = wp3_term_tp_lhs(ones_col, wp2, rho_ds_zm, invrs_rho_ds_zt, gr)
    lhs_pr_tp_wp3  = wp3_term_tp_lhs(-C_wp3_pr_tp, wp2, rho_ds_zm, invrs_rho_ds_zt, gr)
    lhs_tp_wp3     = lhs_adv_tp_wp3 + lhs_pr_tp_wp3   # (2, ngrdcol, nzt)

    # wp3 pressure-1 and dissipation
    lhs_pr1_wp3 = wp3_term_pr1_lhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt)
    lhs_dp1_wp2 = wp2_term_dp1_lhs(C1_Skw_fnc, invrs_tau_C1_zm)
    lhs_pr1_wp2 = wp2_term_pr1_lhs(C4, invrs_tau_C4_zm)   # l_tke_aniso=True

    # ADG1 TA for wp3
    lhs_ta_wp3 = wp3_term_ta_ADG1_lhs(
        wp2, a1_coef_zt, a3_coef_zt, wp3_on_wp2,
        rho_ds_zm, invrs_rho_ds_zt, gr
    )

    # ------------------------------------------------------------------
    # wp23_rhs assembly
    # Interleaving: wp2[k_py] at global index 2*k_py, wp3[k_py] at 2*k_py+1
    # Ascending grid: global indices 0..2*nzm-2
    # ------------------------------------------------------------------

    ndim = 2 * nzm - 1
    rhs = wp23_rhs(
        nzm=nzm, ngrdcol=ngrdcol, invrs_dt=invrs_dt,
        rhs_pr_turb_wp3=rhs_pr_turb_wp3, rhs_pr_dfsn_wp3=rhs_pr_dfsn_wp3,
        rhs_pr_dfsn_wp2=rhs_pr_dfsn_wp2, rhs_pr1_wp2=rhs_pr1_wp2,
        rhs_bp1_pr2_wp3=rhs_bp1_pr2_wp3, rhs_pr1_wp3=rhs_pr1_wp3,
        rhs_bp_pr2_wp2=rhs_bp_pr2_wp2, rhs_pr3_wp2=rhs_pr3_wp2, rhs_dp1_wp2=rhs_dp1_wp2,
        lhs_pr1_wp2=lhs_pr1_wp2, lhs_tp_wp3=lhs_tp_wp3, lhs_pr1_wp3=lhs_pr1_wp3,
        lhs_dp1_wp2=lhs_dp1_wp2, lhs_ta_wp3=lhs_ta_wp3,
        wp2=wp2, wp3=wp3, wp2up=wp2up, upwp=upwp,
        l_ho_nontrad_coriolis=l_ho_nontrad_coriolis, fcor_y=fcor_y, gr=gr)

    # ------------------------------------------------------------------
    # Scale lhs_diff_zt by C12 (AFTER wp23_rhs, BEFORE wp23_lhs)
    # ------------------------------------------------------------------

    lhs_diff_zt = lhs_diff_zt * C12[None, :, None]   # (3, ngrdcol, nzt)

    # ------------------------------------------------------------------
    # Remaining LHS sub-terms (computed after C12 scaling)
    # ------------------------------------------------------------------

    lhs_ma_zm  = term_ma_zm_lhs_jax(wm_zm, gr)             # (3, ngrdcol, nzm)
    lhs_ma_zt  = term_ma_zt_lhs_jax(wm_zt, gr)             # (3, ngrdcol, nzt)
    lhs_ta_wp2 = wp2_term_ta_lhs(invrs_rho_ds_zm, rho_ds_zt, gr)   # (2, ngrdcol, nzm)
    lhs_ac_pr2_wp2 = wp2_terms_ac_pr2_lhs(C_uu_shr, wm_zt, gr)     # (ngrdcol, nzm)
    lhs_ac_pr2_wp3 = wp3_terms_ac_pr2_lhs(C11_Skw_fnc, wm_zm, gr)  # (ngrdcol, nzt)

    # ------------------------------------------------------------------
    # wp23_lhs assembly
    # ------------------------------------------------------------------

    lhs = wp23_lhs(
        nzm=nzm, ngrdcol=ngrdcol, ndim=ndim, invrs_dt=invrs_dt,
        lhs_ma_zm=lhs_ma_zm, lhs_diff_zm=lhs_diff_zm, lhs_ta_wp2=lhs_ta_wp2,
        lhs_ac_pr2_wp2=lhs_ac_pr2_wp2, lhs_dp1_wp2=lhs_dp1_wp2,
        lhs_pr1_wp2=lhs_pr1_wp2, lhs_splat_wp2=lhs_splat_wp2,
        lhs_ma_zt=lhs_ma_zt, lhs_diff_zt=lhs_diff_zt, lhs_tp_wp3=lhs_tp_wp3,
        lhs_ac_pr2_wp3=lhs_ac_pr2_wp3, lhs_pr1_wp3=lhs_pr1_wp3,
        lhs_splat_wp3=lhs_splat_wp3, lhs_ta_wp3=lhs_ta_wp3)

    # ------------------------------------------------------------------
    # Solve the pentadiagonal system
    # ------------------------------------------------------------------

    # Solve + de-interleave wp2 (even slots) and wp3 (odd slots) — wp23_solve
    wp2_new, wp3_new = wp23_solve(lhs, rhs)    # wp2 (ngrdcol, nzm), wp3 (ngrdcol, nzt)

    # ------------------------------------------------------------------
    # Budget stats dict — intermediate terms for caller
    # ------------------------------------------------------------------
    # wp2 stats-only terms (C_uu_buoy=0 gives bp, C_uu_buoy+1 gives pr2)
    rhs_bp_wp2     = wp2_terms_bp_pr2_rhs(jnp.zeros_like(C_uu_buoy), thv_ds_zm, wpthvp)
    rhs_pr2_wp2    = wp2_terms_bp_pr2_rhs(C_uu_buoy + 1.0, thv_ds_zm, wpthvp)
    lhs_wp2_pr2_term = wp2_terms_ac_pr2_lhs(C_uu_shr + 1.0, wm_zt, gr)

    # wp3 stats-only terms
    rhs_bp1_wp3    = wp3_terms_bp1_pr2_rhs(jnp.zeros_like(C11_Skw_fnc), thv_ds_zt, wp2thvp)
    rhs_pr2_wp3    = wp3_terms_bp1_pr2_rhs(C11_Skw_fnc + 1.0, thv_ds_zt, wp2thvp)
    lhs_wp3_pr2_term = wp3_terms_ac_pr2_lhs(C11_Skw_fnc + 1.0, wm_zm, gr)

    # ------------------------------------------------------------------
    # Post-solve fill_holes / clip / clip_skewness + budget stats
    # (advance_wp2_wp3 subroutine tail; mirrors the Fortran in-routine
    #  fill_holes/clip_variance/clip_skewness + stat_update calls)
    # ------------------------------------------------------------------
    _wp2_jax_w23 = _asarray(wp2_new, dtype=np.float64).copy()
    _wp3_jax_w23 = _asarray(wp3_new, dtype=np.float64).copy()

    # fill_holes_vertical on wp2
    _hf_lower_w23 = gr.k_lb_zm + gr.grid_dir_indx   # Python 0-based
    _hf_upper_w23 = gr.k_ub_zm - gr.grid_dir_indx   # Python 0-based
    _wp2_jax_w23 = _asarray(fill_holes_vertical(
        field=jnp.asarray(_wp2_jax_w23),
        rho_ds=jnp.asarray(rho_ds_zm),
        dz=jnp.asarray(gr.dzm),
        threshold=float(cc.w_tol_sqd),
        lower_k=_hf_lower_w23, upper_k=_hf_upper_w23,
        fill_holes_type=flags.fill_holes_type,
    ), dtype=np.float64)

    # fill_holes_wp2_from_horz_tke (ARM: l_wp2_fill_holes_tke=True)
    _up2_out_w23 = up2
    _vp2_out_w23 = vp2
    if flags.l_wp2_fill_holes_tke:
        # Fortran lower_hf_level=1 (1-based) → Python lower_k=0
        # Fortran upper_hf_level=nzm-2 (1-based) → Python upper_k=nzm-3
        (_wp2_jax_w23, _up2_out_w23, _vp2_out_w23) = [
            _asarray(x, dtype=np.float64)
            for x in fill_holes_wp2_from_horz_tke(
                wp2=jnp.asarray(_wp2_jax_w23),
                up2=jnp.asarray(_asarray(up2, dtype=np.float64)),
                vp2=jnp.asarray(_asarray(vp2, dtype=np.float64)),
                threshold=float(cc.w_tol_sqd),
                lower_k=0,
                upper_k=nzm - 3,
            )
        ]

    # clip_variance on wp2 (solve_type=12, wp2_min=cc.w_tol_sqd for l_min_wp2_from_corr_wx=False)
    _wp2_min_w23 = _xp.full_like(_wp2_jax_w23, float(cc.w_tol_sqd))
    if flags.l_min_wp2_from_corr_wx:
        _corr_max2 = cc.max_mag_correlation_flux ** 2   # wp2_min from corr_wx (advance_wp2_wp3_module.F90:1986-1987)
        _wprtp_np_w23 = _asarray(wprtp, dtype=np.float64)
        _wpthlp_np_w23 = _asarray(wpthlp, dtype=np.float64)
        _upwp_np_w23 = _asarray(upwp, dtype=np.float64)
        _vpwp_np_w23 = _asarray(vpwp, dtype=np.float64)
        _rtp2_np_w23 = _asarray(rtp2, dtype=np.float64)
        _thlp2_np_w23 = _asarray(thlp2, dtype=np.float64)
        _up2_np_w23 = _asarray(up2, dtype=np.float64)
        _vp2_np_w23 = _asarray(vp2, dtype=np.float64)
        _wp2_min_w23 = _xp.minimum(1.0, _xp.maximum(_wp2_min_w23,
            _wprtp_np_w23 ** 2 / (_rtp2_np_w23 * _corr_max2),
            _wpthlp_np_w23 ** 2 / (_thlp2_np_w23 * _corr_max2),
            _upwp_np_w23 ** 2 / (_up2_np_w23 * _corr_max2),
            _vpwp_np_w23 ** 2 / (_vp2_np_w23 * _corr_max2),
        ))
    # clip_variance on wp2
    _wp2_jax_w23 = _asarray(clip_variance(
        xp2=jnp.asarray(_wp2_jax_w23),
        threshold_lo=jnp.asarray(_wp2_min_w23),
    ), dtype=np.float64)

    # zm2zt to get wp2_zt
    _wp2_zt_jax_w23 = _asarray(zm2zt(_wp2_jax_w23, gr), dtype=np.float64)
    _wp2_zt_jax_w23 = _xp.maximum(_wp2_zt_jax_w23, cc.w_tol_sqd)   # positive definite

    # clip_skewness on wp3 — the budget wrapper records the wp3_cl clip tendency internally
    # (mirroring the Fortran clip_skewness → clip_skewness_core split).
    _skw_max_w23 = clubb_params[:, cc.iSkw_max_mag - 1]
    _wp3_jax_w23 = _asarray(clip_skewness(
        wp3=jnp.asarray(_wp3_jax_w23),
        wp2_zt=jnp.asarray(_wp2_zt_jax_w23),
        zt=jnp.asarray(gr.zt),
        sfc_elevation=jnp.asarray(sfc_elevation),
        Skw_max_mag=jnp.asarray(_skw_max_w23),
        dt=float(dt),
        l_use_wp3_lim_with_smth_Heaviside=flags.l_use_wp3_lim_with_smth_Heaviside,
        stats_writer=stats_writer, l_sample=l_sample,
    ), dtype=np.float64)


    # advance_wp2_wp3_module.F90 stats writes (C1/C11_Skw_fnc, budgets)
    if l_sample and stats_writer is not None:
        stats_writer.update("C1_Skw_fnc",  _asarray(C1_Skw_fnc,  dtype=np.float64))
        stats_writer.update("C11_Skw_fnc", _asarray(C11_Skw_fnc, dtype=np.float64))

        # ---- post-advance wp2_zt and wp3_zm (fix: written here, not earlier) ----
        _wp3_zm_jax_w23 = _asarray(zt2zm(jnp.asarray(_wp3_jax_w23), gr), dtype=np.float64)
        stats_writer.update("wp2_zt", _wp2_zt_jax_w23)
        stats_writer.update("wp3_zm", _wp3_zm_jax_w23)

        # ---- wp2/wp3 clipping budget stats ----
        # (wp3_cl is now recorded inside clip_skewness — the Fortran clip_skewness budget wrapper)
        _dt_w23 = float(dt)
        stats_writer.update("wp2_pd",
            (_asarray(_wp2_jax_w23, dtype=np.float64) - _asarray(wp2_new, dtype=np.float64)) / _dt_w23)
        _up2_out_w23_np = _asarray(_up2_out_w23 if flags.l_wp2_fill_holes_tke else up2, dtype=np.float64)
        _vp2_out_w23_np = _asarray(_vp2_out_w23 if flags.l_wp2_fill_holes_tke else vp2, dtype=np.float64)
        # Mirror Fortran: up2_pd/vp2_pd contribution from wp2 block uses
        # l_count_sample=.false. so nsamples is NOT incremented here.
        up2_np = _asarray(up2, dtype=np.float64)
        vp2_np = _asarray(vp2, dtype=np.float64)
        stats_writer.begin_budget("up2_pd", up2_np / _dt_w23)
        stats_writer.begin_budget("vp2_pd", vp2_np / _dt_w23)
        stats_writer.finalize_budget("up2_pd", _up2_out_w23_np / _dt_w23, l_count_sample=False)
        stats_writer.finalize_budget("vp2_pd", _vp2_out_w23_np / _dt_w23, l_count_sample=False)

        # ---- wp2/wp3 budget terms (pre/post advance values) ----
        _g = cc.gamma_over_implicit_ts
        _wp2_pre = _asarray(wp2, dtype=np.float64)
        _wp3_pre = _asarray(wp3, dtype=np.float64)
        _wp2_post = _asarray(wp2_new, dtype=np.float64)  # pre-clip post-solve
        _wp3_post = _asarray(wp3_new, dtype=np.float64)  # pre-clip post-solve
        _wp2_mix = (1.0 - _g) * _wp2_pre + _g * _wp2_post
        _wp3_mix = (1.0 - _g) * _wp3_pre + _g * _wp3_post

        # ------ wp2 budget terms ------

        # wp2_bp: C_uu_buoy=0 → 2*g/thv*wpthvp
        stats_writer.update("wp2_bp",   _asarray(rhs_bp_wp2, dtype=np.float64))
        # wp2_pr3: explicit RHS pressure term 3
        stats_writer.update("wp2_pr3",  _asarray(rhs_pr3_wp2, dtype=np.float64))
        # wp2_splat: in Fortran stats_update("wp2_splat", -lhs_splat_wp2*wp2_old) before solve
        stats_writer.update("wp2_splat", -(_asarray(lhs_splat_wp2, dtype=np.float64) * _wp2_pre))

        # wp2_dp1: rhs_dp1_wp2 - lhs_dp1_wp2 * wp2_mix — diagonal implicit-budget finalize kernel
        _lhs_dp1 = _asarray(lhs_dp1_wp2, dtype=np.float64)
        _rhs_dp1 = _asarray(rhs_dp1_wp2, dtype=np.float64)
        stats_writer.update("wp2_dp1",
            finalize_implicit_budget_interior_jax(_rhs_dp1, _lhs_dp1, _wp2_mix))

        # wp2_pr1: rhs_pr1_wp2 - lhs_pr1_wp2 * wp2_mix  (l_tke_aniso=True)
        _lhs_pr1 = _asarray(lhs_pr1_wp2, dtype=np.float64)
        _rhs_pr1 = _asarray(rhs_pr1_wp2, dtype=np.float64)
        stats_writer.update("wp2_pr1",
            finalize_implicit_budget_interior_jax(_rhs_pr1, _lhs_pr1, _wp2_mix))

        # wp2_pr2: rhs_pr2_wp2 - lhs_wp2_pr2_term * wp2_post
        _rhs_pr2 = _asarray(rhs_pr2_wp2, dtype=np.float64)
        _lhs_pr2t = _asarray(lhs_wp2_pr2_term, dtype=np.float64)
        stats_writer.update("wp2_pr2",
            finalize_implicit_budget_interior_jax(_rhs_pr2, _lhs_pr2t, _wp2_post))

        # wp2_dp2: fully implicit, -(lhs_diff_zm @ wp2_post) tri-band — the shared 3-band interior
        # apply kernel (advance_xp2_xpyp_module.apply_lhs_band3_interior_jax), same matrix-vector form
        # the Fortran budget-finalize uses for the implicit diffusion contribution.
        _lhs_dz = _asarray(lhs_diff_zm, dtype=np.float64)  # (3, ngrdcol, nzm)
        _wp2_dp2 = -apply_lhs_band3_interior_jax(_lhs_dz, _wp2_post)
        stats_writer.update("wp2_dp2", _wp2_dp2)

        # wp2_ta: fully implicit, -(lhs_ta_wp2 @ wp3_post) — shared zt→zm 2-band apply kernel
        _lhs_ta2 = _asarray(lhs_ta_wp2, dtype=np.float64)  # (2, ngrdcol, nzm)
        _wp2_ta = -apply_lhs_band2_zt2zm_interior_jax(_lhs_ta2, _wp3_post)
        stats_writer.update("wp2_ta", _wp2_ta)

        # wp2_pr_dfsn already in stats? Skip (passes). wp2_ac, wp2_ma → already pass.

        # ------ wp3 budget terms ------

        # wp3_bp1: C11_Skw_fnc=0 → 3*g/thv*wp2thvp
        stats_writer.update("wp3_bp1",     _asarray(rhs_bp1_wp3, dtype=np.float64))
        # wp3_pr_turb: explicit RHS
        stats_writer.update("wp3_pr_turb", _asarray(rhs_pr_turb_wp3, dtype=np.float64))

        # wp3_pr1: rhs_pr1_wp3 - lhs_pr1_wp3 * wp3_mix — diagonal implicit-budget finalize kernel
        _lhs_pr1_3 = _asarray(lhs_pr1_wp3, dtype=np.float64)
        _rhs_pr1_3 = _asarray(rhs_pr1_wp3, dtype=np.float64)
        stats_writer.update("wp3_pr1",
            finalize_implicit_budget_interior_jax(_rhs_pr1_3, _lhs_pr1_3, _wp3_mix))

        # wp3_pr2: rhs_pr2_wp3 - lhs_wp3_pr2_term * wp3_post — diagonal implicit-budget finalize kernel
        _rhs_pr2_3 = _asarray(rhs_pr2_wp3, dtype=np.float64)
        _lhs_pr2t_3 = _asarray(lhs_wp3_pr2_term, dtype=np.float64)
        stats_writer.update("wp3_pr2",
            finalize_implicit_budget_interior_jax(_rhs_pr2_3, _lhs_pr2t_3, _wp3_post))

        # wp3_dp1: fully implicit (ARM: l_crank_nich_diff=False), -(lhs_diff_zt @ wp3_post) — the same
        # shared 3-band interior apply kernel (apply_lhs_band3_interior_jax).
        _lhs_dt = _asarray(lhs_diff_zt, dtype=np.float64)  # (3, ngrdcol, nzt)
        _wp3_dp1 = -apply_lhs_band3_interior_jax(_lhs_dt, _wp3_post)
        stats_writer.update("wp3_dp1", _wp3_dp1)

        # wp3_ta: -(lhs_ta_wp3 @ mixed) 5-band
        _lhs_ta3 = _asarray(lhs_ta_wp3, dtype=np.float64)  # (5, ngrdcol, nzt)
        _wp3_ta = _xp.zeros_like(_wp3_pre)
        _wp3_ta = _iset(_wp3_ta, np.s_[:, 1:-1], -(
            _lhs_ta3[0, :, 1:-1] * _wp3_mix[:, 2:]     # wp3[k+1]
            + _lhs_ta3[1, :, 1:-1] * _wp2_mix[:, 2:-1]   # wp2[k+1]
            + _lhs_ta3[2, :, 1:-1] * _wp3_mix[:, 1:-1]   # wp3[k]
            + _lhs_ta3[3, :, 1:-1] * _wp2_mix[:, 1:-2]   # wp2[k]
            + _lhs_ta3[4, :, 1:-1] * _wp3_mix[:, :-2]    # wp3[k-1]
        ))
        stats_writer.update("wp3_ta", _wp3_ta)

        # wp3_tp: -(lhs_adv_tp_wp3 @ wp2_mix)
        _lhs_tp3 = _asarray(lhs_adv_tp_wp3, dtype=np.float64)  # (2, ngrdcol, nzt)
        _wp3_tp = _xp.zeros_like(_wp3_pre)
        _wp3_tp = _iset(_wp3_tp, np.s_[:, 1:-1], -(
            _lhs_tp3[0, :, 1:-1] * _wp2_mix[:, 2:-1]   # wp2[k+1]
            + _lhs_tp3[1, :, 1:-1] * _wp2_mix[:, 1:-2]   # wp2[k]
        ))
        stats_writer.update("wp3_tp", _wp3_tp)

    return _wp2_jax_w23, _wp3_jax_w23, _wp2_zt_jax_w23
