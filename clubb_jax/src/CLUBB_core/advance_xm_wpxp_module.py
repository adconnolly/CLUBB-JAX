"""JAX implementations of advance_xm_wpxp sub-functions.

Faithful ports of CLUBB_core/advance_xm_wpxp_module.F90. The mean-advection
LHS operators (term_ma_zt_lhs_jax/term_ma_zm_lhs_jax) live in mean_adv.py, mirroring
the Fortran `use mean_adv`; they are imported back in here.

Functions implemented:
  xm_term_ta_lhs       -- turbulent advection LHS for xm (zt-level)
  wpxp_term_tp_lhs     -- turbulent production LHS for w'x' (zm-level)
  wpxp_terms_ac_pr2_lhs -- accumulation + pressure-2 LHS for w'x' (zm-level)
  wpxp_term_pr1_lhs    -- pressure-1 LHS for w'x' (zm-level)
  wpxp_terms_bp_pr3_rhs -- buoyancy-production + pressure-3 RHS for w'x' (zm-level)
  xm_wpxp_lhs          -- full interleaved pentadiagonal LHS assembly
  xm_wpxp_rhs          -- full interleaved RHS assembly
  solve_xm_wpxp_with_single_lhs -- per-field solve: returns (wpxp_new, xm_new)
  advance_xm_wpxp      -- whole-driver: advances rt/thl + um/vm, returns state dict
  clip_covar            -- covariance clipping post-solve
  apply_sponge_field_jax    -- sponge-layer damping of a mean field (advance_xm_wpxp tail block)

All functions operate in float64.

References:
  src/CLUBB_core/advance_xm_wpxp_module.F90
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.tracer_numpy import _asarray, _xp, _iset
from clubb_jax.src.CLUBB_core.grid_class import zm2zt_jax, zt2zm_jax, zm2zt2zm, zt2zm2zt, ddzt
from clubb_jax.src.CLUBB_core.diffusion import diffusion_zm_lhs_jax
from clubb_jax.src.CLUBB_core.turbulent_adv_pdf import xpyp_term_ta_pdf_lhs_jax
from clubb_jax.src.CLUBB_core.mean_adv import term_ma_zt_lhs_jax, term_ma_zm_lhs_jax
from clubb_jax.src.CLUBB_core.penta_lu_solver import penta_lu_solve
# clip_covar lives in its Fortran home clip_explicit.F90 (the Fortran `use clip_explicit`)
from clubb_jax.src.CLUBB_core.clip_explicit import clip_covar, clip_rcm
# fill_holes_vertical (fill_holes.F90) + the monotonic flux limiter (mono_flux_limiter.F90) —
# called by xm_wpxp_clipping_and_stats per the Fortran `use` chain
from clubb_jax.src.CLUBB_core.fill_holes import fill_holes_vertical
from clubb_jax.src.CLUBB_core.mono_flux_limiter import (
    monotonic_turbulent_flux_limit, MFL_UM, MFL_VM, MFL_RTM, MFL_THLM,
    calc_turb_adv_range, mean_vert_vel_up_down,
)
from clubb_jax.src.CLUBB_core.constants_clubb import eps as _EPS
from clubb_jax.src.CLUBB_core.advance_xp2_xpyp_module import (
    apply_lhs_band2_zt2zm_interior_jax, apply_lhs_band3_interior_jax,
)
from clubb_jax.src.CLUBB_core.constants_clubb import (
    rt_tol, thl_tol, rt_tol as _RT_TOL, thl_tol as _THL_TOL,
    rt_tol_mfl as _RT_TOL_MFL, thl_tol_mfl as _THL_TOL_MFL,
    w_tol as _W_TOL, w_tol_sqd as _W_TOL_SQD,
    ep1, grav, iC6rt, iC6thl, iC_uu_shr, ibeta, ic_K6, zero_threshold,
    gamma_over_implicit_ts,
)
# sponge_damp_xm lives in its Fortran home sponge_layer_damping.F90 (the Fortran `use sponge_layer_damping`)
from clubb_jax.src.CLUBB_core.sponge_layer_damping import sponge_damp_xm

# gamma_over_implicit_ts = 3/2 (constants_clubb.F90); advance_xm_wpxp_module.F90:1255 `use constants_clubb`
_gamma = gamma_over_implicit_ts


def apply_sponge_field_jax(key, xm, xm_ref, gr, dt_advance, sponge_cfg):
    """Apply sponge-layer damping to a mean field toward its reference profile.

    Faithful to the sponge block at the end of advance_xm_wpxp
    (advance_xm_wpxp_module.F90:1053-1123). A no-op unless `sponge_cfg` contains
    `key` (i.e. that field's l_sponge_damping is set). tau/depth are precomputed
    once at init (sponge_layer_damping.initialize_tau_sponge_damp). The reference
    profile xm_ref is the initial sounding profile (clubb_driver.F90:5298-5316).
    """
    if not sponge_cfg or key not in sponge_cfg:
        return xm
    prof = sponge_cfg[key]
    tau, depth = prof['tau'], prof['depth']
    zt_a = _asarray(gr.zt, dtype=np.float64)
    zm_a = _asarray(gr.zm, dtype=np.float64)
    ref_a = _asarray(xm_ref, dtype=np.float64)
    dt = float(dt_advance)
    # Vectorized + tracer-transparent (REFACTOR B5): sponge_damp_xm is now pure broadcast arithmetic
    # (tau (nz,) broadcasts over the (ngrdcol, nz) field), so this is bit-identical to the per-column loop
    # while keeping the prognostic xm on the autodiff graph (the old `np.array(xm)`+in-place loop severed it).
    return sponge_damp_xm(_asarray(xm, dtype=np.float64), ref_a, zt_a, zm_a[:, -1:], tau, depth, dt)


# ---------------------------------------------------------------------------
# xm_term_ta_lhs
# ---------------------------------------------------------------------------

def xm_term_ta_lhs(
    invrs_rho_ds_zt: jnp.ndarray,
    rho_ds_zm: jnp.ndarray,
    gr,
) -> jnp.ndarray:
    """Turbulent advection LHS for xm (thermodynamic levels).

    Faithful port of advance_xm_wpxp_module.F90:xm_term_ta_lhs.

    Computes: (1/rho_ds_zt) * d(rho_ds_zm * w'x') / dz, solved implicitly.

    Output shape: (2, ngrdcol, nzt).
      out[0, :, k] = coeff of wpxp[k+1]   (momentum superdiagonal)
      out[1, :, k] = coeff of wpxp[k]     (momentum subdiagonal)

    Args:
        invrs_rho_ds_zt: (ngrdcol, nzt)
        rho_ds_zm:       (ngrdcol, nzm)
        gr:              grid object
    """
    invrs_dzt = gr.invrs_dzt   # (ngrdcol, nzt)
    # super: coeff of wpxp(k+1) = rho_ds_zm[k+1]
    sup = invrs_rho_ds_zt * invrs_dzt * rho_ds_zm[:, 1:]   # (ngrdcol, nzt)
    # sub:  coeff of wpxp(k)   = rho_ds_zm[k]
    sub = -invrs_rho_ds_zt * invrs_dzt * rho_ds_zm[:, :-1]  # (ngrdcol, nzt)
    return jnp.stack([sup, sub], axis=0)   # (2, ngrdcol, nzt)


# ---------------------------------------------------------------------------
# wpxp_term_tp_lhs
# ---------------------------------------------------------------------------

def wpxp_term_tp_lhs(
    wp2: jnp.ndarray,
    gr,
) -> jnp.ndarray:
    """Turbulent production LHS for w'x' (momentum levels).

    Faithful port of advance_xm_wpxp_module.F90:wpxp_term_tp_lhs.

    Output shape: (2, ngrdcol, nzm).
      out[0, :, k] = coeff of xm[k]     (zt level k, thermodynamic superdiagonal)
      out[1, :, k] = coeff of xm[k-1]   (zt level k-1, thermodynamic subdiagonal)
    Boundaries k=0 and k=nzm-1 are zero.

    Args:
        wp2: (ngrdcol, nzm)
        gr:  grid object
    """
    nzm = wp2.shape[1]
    invrs_dzm = gr.invrs_dzm   # (ngrdcol, nzm)

    interior = wp2[:, 1:-1] * invrs_dzm[:, 1:-1]   # (ngrdcol, nzm-2)
    zeros_col = jnp.zeros((wp2.shape[0], 1))

    sup_interior = interior     # coeff of xm[k]   = +wp2[k]*invrs_dzm[k]
    sub_interior = -interior    # coeff of xm[k-1] = -wp2[k]*invrs_dzm[k]

    # Pad boundaries with zeros
    sup = jnp.concatenate([zeros_col, sup_interior, zeros_col], axis=1)  # (ngrdcol, nzm)
    sub = jnp.concatenate([zeros_col, sub_interior, zeros_col], axis=1)
    return jnp.stack([sup, sub], axis=0)   # (2, ngrdcol, nzm)


# ---------------------------------------------------------------------------
# wpxp_terms_ac_pr2_lhs
# ---------------------------------------------------------------------------

def wpxp_terms_ac_pr2_lhs(
    C7_Skw_fnc: jnp.ndarray,
    wm_zt: jnp.ndarray,
    gr,
) -> jnp.ndarray:
    """Accumulation + pressure-2 LHS for w'x' (momentum levels).

    Faithful port of advance_xm_wpxp_module.F90:wpxp_terms_ac_pr2_lhs.

    Computes: (1 - C7) * wpxp * d(wm_zt)/dz  at each zm level.

    Output shape: (ngrdcol, nzm).  Boundaries are zero.

    Args:
        C7_Skw_fnc: (ngrdcol, nzm)
        wm_zt:      (ngrdcol, nzt)
        gr:         grid object
    """
    nzm = C7_Skw_fnc.shape[1]
    invrs_dzm = gr.invrs_dzm   # (ngrdcol, nzm)

    # Interior k=1..nzm-2 (Python 0-based).
    # At zm level k_py (k_py=1..nzm-2): d(wm_zt)/dz = (wm_zt[k_py] - wm_zt[k_py-1]) * invrs_dzm[k_py]
    # Fortran: (wm_zt(k) - wm_zt(k-1)) where k is Fortran 1-based zm index.
    # Python: wm_zt[:, k_py] - wm_zt[:, k_py-1] for k_py=1..nzm-2.
    # Vectorized: wm_zt[:, 1:nzm-1] - wm_zt[:, 0:nzm-2]
    #           = wm_zt[:, 1:-1] - wm_zt[:, :-2]  (since nzm-1 = nzt)
    # But wm_zt has shape (ngrdcol, nzt) where nzt = nzm-1, so nzm-1 <= nzt means this
    # equals wm_zt[:, 1:] - wm_zt[:, :-1] when nzm-1 = nzt exactly (always true).
    # wm_zt[:, 1:] - wm_zt[:, :-1] has shape (ngrdcol, nzt-1) = (ngrdcol, nzm-2). ✓
    d_wm = wm_zt[:, 1:] - wm_zt[:, :-1]   # (ngrdcol, nzm-2)

    interior = (1.0 - C7_Skw_fnc[:, 1:-1]) * invrs_dzm[:, 1:-1] * d_wm
    zeros_col = jnp.zeros((C7_Skw_fnc.shape[0], 1))
    return jnp.concatenate([zeros_col, interior, zeros_col], axis=1)  # (ngrdcol, nzm)


# ---------------------------------------------------------------------------
# wpxp_term_pr1_lhs
# ---------------------------------------------------------------------------

def wpxp_term_pr1_lhs(
    C6_Skw_fnc: jnp.ndarray,
    invrs_tau_C6_zm: jnp.ndarray,
) -> jnp.ndarray:
    """Pressure-1 LHS for w'x' (momentum levels).

    Faithful port of advance_xm_wpxp_module.F90:wpxp_term_pr1_lhs.

    Computes: C6 * (1/tau_m) * wpxp at each zm level.

    Output shape: (ngrdcol, nzm).  Boundaries (k=0, k=nzm-1) are zero.

    Args:
        C6_Skw_fnc:     (ngrdcol, nzm)
        invrs_tau_C6_zm:(ngrdcol, nzm)
    """
    nzm = C6_Skw_fnc.shape[1]
    result = C6_Skw_fnc * invrs_tau_C6_zm   # (ngrdcol, nzm)
    # Zero out boundaries
    result = result.at[:, 0].set(0.0)
    result = result.at[:, -1].set(0.0)
    return result


# ---------------------------------------------------------------------------
# wpxp_terms_bp_pr3_rhs
# ---------------------------------------------------------------------------

def wpxp_terms_bp_pr3_rhs(
    C7_Skw_fnc: jnp.ndarray,
    thv_ds_zm: jnp.ndarray,
    xpthvp: jnp.ndarray,
    grav: float = grav,   # constants_clubb.grav (mirrors the Fortran `use constants_clubb, only: grav`)
) -> jnp.ndarray:
    """Buoyancy-production + pressure-3 RHS for w'x' (momentum levels).

    Faithful port of advance_xm_wpxp_module.F90:wpxp_terms_bp_pr3_rhs.

    Computes: (1 - C7) * (g/thv_ds) * x'thv'.

    Output shape: (ngrdcol, nzm).  Boundaries are zero.

    Args:
        C7_Skw_fnc: (ngrdcol, nzm)
        thv_ds_zm:  (ngrdcol, nzm)
        xpthvp:     (ngrdcol, nzm)  -- r't' or thl'thv' or similar
    """
    result = (grav / thv_ds_zm) * (1.0 - C7_Skw_fnc) * xpthvp
    result = result.at[:, 0].set(0.0)
    result = result.at[:, -1].set(0.0)
    return result


# ---------------------------------------------------------------------------
# xm_wpxp_lhs
# ---------------------------------------------------------------------------

def xm_wpxp_lhs(
    lhs_diff_zm: jnp.ndarray,
    lhs_ma_zm: jnp.ndarray,
    lhs_ma_zt: jnp.ndarray,
    lhs_ta_wpxp: jnp.ndarray,
    lhs_ta_xm: jnp.ndarray,
    lhs_tp: jnp.ndarray,
    lhs_ac_pr2: jnp.ndarray,
    lhs_pr1: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Assemble the interleaved pentadiagonal LHS for the xm/w'x' system.

    Faithful port of advance_xm_wpxp_module.F90:xm_wpxp_lhs.
    Handles: l_implemented=False (standalone), l_diffuse_rtm_and_thlm=False,
    l_iter=True (always), ascending grid.

    Interleaving scheme (Python 0-based, nzm zm-levels, nzt=nzm-1 zt-levels):
      j=0, 2, ..., 2*(nzm-1)  →  wpxp at zm levels 0..nzm-1
      j=1, 3, ..., 2*nzm-3    →  xm at zt levels 0..nzt-1
    Total size = 2*nzm-1.

    Output shape: (5, ngrdcol, 2*nzm-1).
      out[0] = super2, out[1] = super1, out[2] = diag, out[3] = sub1, out[4] = sub2

    Args:
        lhs_diff_zm:  (3, ngrdcol, nzm)
        lhs_ma_zm:    (3, ngrdcol, nzm)
        lhs_ma_zt:    (3, ngrdcol, nzt)  -- for l_implemented=False
        lhs_ta_wpxp:  (3, ngrdcol, nzm)  -- gamma-weighted implicitly
        lhs_ta_xm:    (2, ngrdcol, nzt)
        lhs_tp:       (2, ngrdcol, nzm)
        lhs_ac_pr2:   (ngrdcol, nzm)
        lhs_pr1:      (ngrdcol, nzm)
        dt:           scalar timestep
    """
    ngrdcol = lhs_diff_zm.shape[1]
    nzm = lhs_diff_zm.shape[2]
    nzt = nzm - 1
    ndim = 2 * nzm - 1
    invrs_dt = 1.0 / dt
    gamma = _gamma

    lhs = jnp.zeros((5, ngrdcol, ndim))

    # ------------------------------------------------------------------ #
    # xm rows: j = 2*k_zt + 1, k_zt = 0..nzt-1                          #
    # Python lhs penta band: [super2, super1, diag, sub1, sub2]           #
    # super2 (j+2=next xm): coeff of xm[k+1] = lhs_ma_zt[0]             #
    # super1 (j+1=next wpxp): coeff of wpxp[k+1] = lhs_ta_xm[0]         #
    # diag   (j=this xm): invrs_dt + lhs_ma_zt[1]                        #
    # sub1   (j-1=this wpxp): coeff of wpxp[k] = lhs_ta_xm[1]           #
    # sub2   (j-2=prev xm): coeff of xm[k-1] = lhs_ma_zt[2]             #
    # ------------------------------------------------------------------ #
    # j=1,3,...,2*nzt-1 are all xm rows (slice 1::2 of lhs)
    lhs = lhs.at[0, :, 1::2].set(lhs_ma_zt[0])   # super2: xm[k+1]
    lhs = lhs.at[1, :, 1::2].set(lhs_ta_xm[0])   # super1: wpxp[k+1]
    lhs = lhs.at[2, :, 1::2].set(invrs_dt + lhs_ma_zt[1])  # diag
    lhs = lhs.at[3, :, 1::2].set(lhs_ta_xm[1])   # sub1: wpxp[k]
    lhs = lhs.at[4, :, 1::2].set(lhs_ma_zt[2])   # sub2: xm[k-1]

    # ------------------------------------------------------------------ #
    # wpxp rows: j = 2*k_zm, k_zm = 0..nzm-1                            #
    # Interior (k_zm=1..nzm-2):                                          #
    # super2 (j+2=next wpxp): lhs_ma_zm[0]+lhs_diff_zm[0]+gamma*ta[0]   #
    # super1 (j+1=this xm ): lhs_tp[0]   (coeff of xm[k_zm])            #
    # diag   (this wpxp):    lhs_ma_zm[1]+lhs_diff_zm[1]+lhs_ac+gamma*(ta[1]+pr1)+invrs_dt
    # sub1   (j-1=prev xm ): lhs_tp[1]   (coeff of xm[k_zm-1])          #
    # sub2   (j-2=prev wpxp): lhs_ma_zm[2]+lhs_diff_zm[2]+gamma*ta[2]   #
    # ------------------------------------------------------------------ #
    # Interior: k_zm=1..nzm-2, j=2..2*(nzm-2), step 2
    interior_sup2 = (lhs_ma_zm[0, :, 1:-1] + lhs_diff_zm[0, :, 1:-1]
                     + gamma * lhs_ta_wpxp[0, :, 1:-1])
    interior_sup1 = lhs_tp[0, :, 1:-1]
    interior_diag = (lhs_ma_zm[1, :, 1:-1] + lhs_diff_zm[1, :, 1:-1]
                     + lhs_ac_pr2[:, 1:-1]
                     + gamma * (lhs_ta_wpxp[1, :, 1:-1] + lhs_pr1[:, 1:-1])
                     + invrs_dt)
    interior_sub1 = lhs_tp[1, :, 1:-1]
    interior_sub2 = (lhs_ma_zm[2, :, 1:-1] + lhs_diff_zm[2, :, 1:-1]
                     + gamma * lhs_ta_wpxp[2, :, 1:-1])

    lhs = lhs.at[0, :, 2:2*nzm-2:2].set(interior_sup2)
    lhs = lhs.at[1, :, 2:2*nzm-2:2].set(interior_sup1)
    lhs = lhs.at[2, :, 2:2*nzm-2:2].set(interior_diag)
    lhs = lhs.at[3, :, 2:2*nzm-2:2].set(interior_sub1)
    lhs = lhs.at[4, :, 2:2*nzm-2:2].set(interior_sub2)

    # ------------------------------------------------------------------ #
    # Lower boundary wpxp BC: j=0 → diag=1, others=0                    #
    # ------------------------------------------------------------------ #
    lhs = lhs.at[0, :, 0].set(0.0)
    lhs = lhs.at[1, :, 0].set(0.0)
    lhs = lhs.at[2, :, 0].set(1.0)
    lhs = lhs.at[3, :, 0].set(0.0)
    lhs = lhs.at[4, :, 0].set(0.0)

    # ------------------------------------------------------------------ #
    # Upper boundary wpxp BC: j=2*(nzm-1) → diag=1, others=0            #
    # ------------------------------------------------------------------ #
    lhs = lhs.at[0, :, -1].set(0.0)
    lhs = lhs.at[1, :, -1].set(0.0)
    lhs = lhs.at[2, :, -1].set(1.0)
    lhs = lhs.at[3, :, -1].set(0.0)
    lhs = lhs.at[4, :, -1].set(0.0)

    return lhs   # (5, ngrdcol, 2*nzm-1)


# ---------------------------------------------------------------------------
# xm_wpxp_rhs
# ---------------------------------------------------------------------------

def xm_wpxp_rhs(
    wpxp: jnp.ndarray,
    xm: jnp.ndarray,
    wpxp_forcing: jnp.ndarray,
    xm_forcing: jnp.ndarray,
    rhs_bp_pr3: jnp.ndarray,
    rhs_ta: jnp.ndarray,
    lhs_ta_wpxp: jnp.ndarray,
    lhs_pr1: jnp.ndarray,
    dt: float,
    k_lb_zm: int,
) -> jnp.ndarray:
    """Assemble the interleaved RHS for the xm/w'x' system.

    Faithful port of advance_xm_wpxp_module.F90:xm_wpxp_rhs.
    l_iter=True always (adds wpxp*invrs_dt to RHS), ascending grid.

    Output shape: (ngrdcol, 2*nzm-1).

    Args:
        wpxp:         (ngrdcol, nzm)
        xm:           (ngrdcol, nzt)
        wpxp_forcing: (ngrdcol, nzm)
        xm_forcing:   (ngrdcol, nzt)
        rhs_bp_pr3:   (ngrdcol, nzm)
        rhs_ta:       (ngrdcol, nzm)  -- 0 for ADG1
        lhs_ta_wpxp:  (3, ngrdcol, nzm)
        lhs_pr1:      (ngrdcol, nzm)
        dt:           scalar
        k_lb_zm:      lower boundary zm index (0 for ascending grid)
    """
    ngrdcol, nzm = wpxp.shape
    nzt = nzm - 1
    ndim = 2 * nzm - 1
    invrs_dt = 1.0 / dt
    gamma = _gamma

    rhs = jnp.zeros((ngrdcol, ndim))

    # ---- Lower boundary wpxp BC: j=0 = wpxp at lower boundary ----
    rhs = rhs.at[:, 0].set(wpxp[:, k_lb_zm])

    # ---- xm rows: j=2k+1 for k=0..nzt-1 ----
    rhs = rhs.at[:, 1::2].set(xm * invrs_dt + xm_forcing)

    # ---- Interior wpxp rows: j=2k for k=1..nzm-2 ----
    # Ascending grid: grid_dir_indx=1
    # lhs_ta_wpxp[0] = coeff of wpxp[k+1], [1] = coeff of wpxp[k], [2] = coeff of wpxp[k-1]
    wpxp_kp1 = wpxp[:, 2:]      # k+1, shape (ngrdcol, nzm-2)
    wpxp_k   = wpxp[:, 1:-1]    # k,   shape (ngrdcol, nzm-2)
    wpxp_km1 = wpxp[:, :-2]     # k-1, shape (ngrdcol, nzm-2)

    ta = lhs_ta_wpxp[:, :, 1:-1]   # (3, ngrdcol, nzm-2)
    pr1 = lhs_pr1[:, 1:-1]          # (ngrdcol, nzm-2)

    rhs_int = (
        rhs_bp_pr3[:, 1:-1]
        + wpxp_forcing[:, 1:-1]
        + rhs_ta[:, 1:-1]
        + (1.0 - gamma) * (
            -ta[0] * wpxp_kp1
            - ta[1] * wpxp_k
            - ta[2] * wpxp_km1
            - pr1 * wpxp_k
        )
        + wpxp_k * invrs_dt     # l_iter = True always
    )
    rhs = rhs.at[:, 2:-1:2].set(rhs_int)

    # ---- Upper boundary wpxp BC: j=ndim-1=2*(nzm-1) ----
    rhs = rhs.at[:, -1].set(0.0)

    return rhs   # (ngrdcol, 2*nzm-1)


# ---------------------------------------------------------------------------
# xm_wpxp_solve
# ---------------------------------------------------------------------------

def xm_wpxp_solve(lhs, rhs):
    """Pentadiagonal solve of the coupled xm/wpxp system, then de-interleave the solution —
    the JAX analog of advance_xm_wpxp_module.F90:xm_wpxp_solve. The solution vector packs the two
    prognostics on alternating slots; returns ``(wpxp_new, xm_new)`` (wpxp on even slots, xm on odd)."""
    soln = penta_lu_solve(lhs, rhs)   # (ngrdcol, 2*nzm-1)
    return soln[:, 0::2], soln[:, 1::2]


# ---------------------------------------------------------------------------
# advance_xm_wpxp
# ---------------------------------------------------------------------------

def solve_xm_wpxp_with_single_lhs(
    wpxp: jnp.ndarray,
    xm: jnp.ndarray,
    wpxp_forcing: jnp.ndarray,
    xm_forcing: jnp.ndarray,
    C6_Skw_fnc: jnp.ndarray,
    C7_Skw_fnc: jnp.ndarray,
    invrs_tau_C6_zm: jnp.ndarray,
    lhs_ta_wpxp: jnp.ndarray,
    lhs_diff_zm: jnp.ndarray,
    lhs_ma_zm: jnp.ndarray,
    lhs_ma_zt: jnp.ndarray,
    lhs_ta_xm: jnp.ndarray,
    lhs_tp: jnp.ndarray,
    lhs_ac_pr2: jnp.ndarray,
    thv_ds_zm: jnp.ndarray,
    xpthvp: jnp.ndarray,
    wm_zt: jnp.ndarray,
    dt: float,
    gr,
    wp2: jnp.ndarray | None = None,
    xp2_relaxed: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Full JAX solve for one xm/w'x' variable pair.

    Returns (wpxp_new, xm_new), each (ngrdcol, nzm) and (ngrdcol, nzt).

    Faithful port of solve_xm_wpxp_with_single_lhs for ADG1, ascending grid,
    l_not_diffuse, l_not_implemented cases.

    If wp2 and xp2_relaxed are provided, covariance clipping is applied after
    the solve (matching xm_wpxp_clipping_and_stats in clip_explicit.F90).

    Args:
        wpxp:            (ngrdcol, nzm)  current value
        xm:              (ngrdcol, nzt)  current value
        wpxp_forcing:    (ngrdcol, nzm)
        xm_forcing:      (ngrdcol, nzt)
        C6_Skw_fnc:      (ngrdcol, nzm)  C6rt or C6thl
        C7_Skw_fnc:      (ngrdcol, nzm)
        invrs_tau_C6_zm: (ngrdcol, nzm)
        lhs_ta_wpxp:     (3, ngrdcol, nzm)  variable-specific TA LHS
        lhs_diff_zm:     (3, ngrdcol, nzm)  shared diffusion LHS
        lhs_ma_zm:       (3, ngrdcol, nzm)  shared MA LHS (zm)
        lhs_ma_zt:       (3, ngrdcol, nzt)  shared MA LHS (zt)
        lhs_ta_xm:       (2, ngrdcol, nzt)  shared TA-xm LHS
        lhs_tp:          (2, ngrdcol, nzm)  shared TP LHS
        lhs_ac_pr2:      (ngrdcol, nzm)     shared AC+PR2 LHS
        thv_ds_zm:       (ngrdcol, nzm)
        xpthvp:          (ngrdcol, nzm)  rt'thv' or thl'thv'
        wm_zt:           (ngrdcol, nzt)
        dt:              scalar
        gr:              grid object
        wp2:             (ngrdcol, nzm) optional; if given, clip wpxp after solve
        xp2_relaxed:     (ngrdcol, nzm) optional; xp2 (possibly floored) for clipping
    """
    k_lb_zm = gr.k_lb_zm

    # Pressure-1 LHS (variable-specific)
    lhs_pr1 = wpxp_term_pr1_lhs(C6_Skw_fnc, invrs_tau_C6_zm)

    # Buoyancy-production + pressure-3 RHS
    rhs_bp_pr3 = wpxp_terms_bp_pr3_rhs(C7_Skw_fnc, thv_ds_zm, xpthvp)

    # rhs_ta = 0 for ADG1
    rhs_ta = jnp.zeros_like(wpxp)

    # Assemble LHS and RHS
    lhs = xm_wpxp_lhs(
        lhs_diff_zm, lhs_ma_zm, lhs_ma_zt,
        lhs_ta_wpxp, lhs_ta_xm, lhs_tp,
        lhs_ac_pr2, lhs_pr1, dt,
    )
    rhs = xm_wpxp_rhs(
        wpxp, xm, wpxp_forcing, xm_forcing,
        rhs_bp_pr3, rhs_ta, lhs_ta_wpxp, lhs_pr1, dt, k_lb_zm,
    )

    # Solve the pentadiagonal system + de-interleave (xm_wpxp_solve)
    wpxp_new, xm_new = xm_wpxp_solve(lhs, rhs)   # wpxp (ngrdcol, nzm), xm (ngrdcol, nzt)

    # Apply covariance clipping if wp2/xp2 provided
    if wp2 is not None and xp2_relaxed is not None:
        wpxp_new = clip_covar(wpxp_new, wp2, xp2_relaxed)

    return wpxp_new, xm_new


# ---------------------------------------------------------------------------
# calc_xm_wpxp_ta_terms
# ---------------------------------------------------------------------------

def calc_xm_wpxp_ta_terms(
    sigma_sqd_w: jnp.ndarray,
    wp3_on_wp2_zt: jnp.ndarray,
    rho_ds_zt: jnp.ndarray,
    invrs_rho_ds_zm: jnp.ndarray,
    gr,
) -> jnp.ndarray:
    """The ADG1 turbulent-advection LHS operator for w'x' — the JAX analog of the Fortran
    `calc_xm_wpxp_ta_terms` (advance_xm_wpxp_module.F90:1996), which the Fortran computes as a
    SEPARATE call (its out-arg `lhs_ta_wprtp`) and passes into the xm/wpxp LHS assembly.

    For ADG1 the implicit turbulent-advection coefficient is
        coef_wp2rtp = a1_coef_zt * wp3_on_wp2_zt,   a1_coef = 1/(1 - sigma_sqd_w) regridded zm->zt,
    and lhs_ta_wprtp = xpyp_term_ta_pdf_lhs(coef_wp2rtp). The SAME operator serves wpthlp and (when
    predicted) wpup/wpvp — the moment pairs share one ADG1 TA LHS.

    sigma_sqd_w: (ngrdcol, nzm); wp3_on_wp2_zt: (ngrdcol, nzt); rho_ds_zt: (ngrdcol, nzt);
    invrs_rho_ds_zm: (ngrdcol, nzm). Returns lhs_ta_wprtp: (3, ngrdcol, nzm)."""
    a1_coef = 1.0 / (1.0 - sigma_sqd_w)                         # (ngrdcol, nzm)
    a1_coef_zt = zm2zt_jax(a1_coef, gr)                          # (ngrdcol, nzt)
    coef_wp2rtp = a1_coef_zt * wp3_on_wp2_zt                     # (ngrdcol, nzt)
    return xpyp_term_ta_pdf_lhs_jax(coef_wp2rtp, rho_ds_zt, invrs_rho_ds_zm, gr)


# ---------------------------------------------------------------------------
# calc_xm_wpxp_lhs_terms
# ---------------------------------------------------------------------------

def calc_xm_wpxp_lhs_terms(
    wm_zm: jnp.ndarray,
    wm_zt: jnp.ndarray,
    wp2: jnp.ndarray,
    Kw6: jnp.ndarray,
    nu6: float,
    C7_Skw_fnc: jnp.ndarray,
    invrs_rho_ds_zm: jnp.ndarray,
    rho_ds_zt: jnp.ndarray,
    rho_ds_zm: jnp.ndarray,
    invrs_rho_ds_zt: jnp.ndarray,
    gr,
) -> dict:
    """Compute the shared LHS terms for the xm/w'x' system — the JAX analog of
    advance_xm_wpxp_module.F90:calc_xm_wpxp_lhs_terms (the Fortran out-args lhs_diff_zm/lhs_ma_zt/
    lhs_ma_zm/lhs_tp/lhs_ta_xm/lhs_ac_pr2), computed once and shared across the moment pairs. As in the
    Fortran, the ADG1 turbulent-advection operator `lhs_ta_wprtp` is computed SEPARATELY by
    `calc_xm_wpxp_ta_terms` (a sibling call) and is NOT produced here.

    Returns a dict with keys: lhs_diff_zm, lhs_ma_zm, lhs_ma_zt, lhs_ta_xm, lhs_tp, lhs_ac_pr2.

    Args:
        wm_zm:           (ngrdcol, nzm)
        wm_zt:           (ngrdcol, nzt)
        wp2:             (ngrdcol, nzm)
        Kw6:             (ngrdcol, nzt)  = c_K6 * Kh_zt
        nu6:             scalar (background diffusivity for w'x')
        C7_Skw_fnc:      (ngrdcol, nzm)
        invrs_rho_ds_zm: (ngrdcol, nzm)
        rho_ds_zt:       (ngrdcol, nzt)
        rho_ds_zm:       (ngrdcol, nzm)
        invrs_rho_ds_zt: (ngrdcol, nzt)
        gr:              grid object
    """
    # Diffusion LHS for w'x' (zm-level variable)
    nu6_arr = jnp.broadcast_to(jnp.array(nu6), (invrs_rho_ds_zm.shape[0],))
    lhs_diff_zm = diffusion_zm_lhs_jax(Kw6, nu6_arr, invrs_rho_ds_zm, rho_ds_zt, gr)

    # Mean advection LHS for w'x' (zm-level)
    lhs_ma_zm = term_ma_zm_lhs_jax(wm_zm, gr)

    # Mean advection LHS for xm (zt-level, upwind)
    lhs_ma_zt = term_ma_zt_lhs_jax(wm_zt, gr)

    # Turbulent advection LHS for xm
    lhs_ta_xm = xm_term_ta_lhs(invrs_rho_ds_zt, rho_ds_zm, gr)

    # Turbulent production LHS for w'x'
    lhs_tp = wpxp_term_tp_lhs(wp2, gr)

    # Accumulation + pressure-2 LHS for w'x'
    lhs_ac_pr2 = wpxp_terms_ac_pr2_lhs(C7_Skw_fnc, wm_zt, gr)

    return dict(
        lhs_diff_zm=lhs_diff_zm,
        lhs_ma_zm=lhs_ma_zm,
        lhs_ma_zt=lhs_ma_zt,
        lhs_ta_xm=lhs_ta_xm,
        lhs_tp=lhs_tp,
        lhs_ac_pr2=lhs_ac_pr2,
    )




def diagnose_upxp(ypwp, xm, wpxp, ym, C6x_Skw_fnc, tau_C6_zm, C7_Skw_fnc, gr):
    """advance_xm_wpxp_module.F90:diagnose_upxp (line 6052).

    Diagnose the turbulent horizontal flux of a conserved scalar (upthlp/uprtp/vpthlp/vprtp)
    — Andre et al. (1978) eqn. 7 / Bougeault et al. (1981) eqn. 4:
        ypxp[k] = (tau_C6/C6x) * ( -ypwp*d(xm)/dz - (1-C7)*wpxp*d(ym)/dz )   for k=2..nzm-1
    with the top/bottom boundary levels left at 0. d/dz of xm and ym are formed here (the
    caller passes the smoothed velocity as `ym`, mirroring the Fortran `um_smth`/`vm_smth` args).
    Pure-jnp → differentiable.
    """
    ypwp = jnp.asarray(ypwp); xm = jnp.asarray(xm); wpxp = jnp.asarray(wpxp); ym = jnp.asarray(ym)
    C6x = jnp.asarray(C6x_Skw_fnc); tau = jnp.asarray(tau_C6_zm); C7 = jnp.asarray(C7_Skw_fnc)
    ddzt_xm = ddzt(xm, gr)
    ddzt_ym = ddzt(ym, gr)
    interior = (tau[:, 1:-1] / C6x[:, 1:-1]) * (
        -ypwp[:, 1:-1] * ddzt_xm[:, 1:-1]
        - (1.0 - C7[:, 1:-1]) * wpxp[:, 1:-1] * ddzt_ym[:, 1:-1]
    )
    return jnp.zeros_like(ypwp).at[:, 1:-1].set(interior)


def xm_wpxp_clipping_and_stats(
    solve_type, xm, wpxp_preclip, xm_old, xp2, xp2_clip, wp2,
    wm_zt, xm_forcing, rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
    xp2_threshold, xm_tol, low_lev_effect, high_lev_effect,
    field_tol, fill_holes_type, l_mono_flux_lim, dt, gr,
):
    """Per-field post-solve clipping for advance_xm_wpxp — mirrors the Fortran
    `xm_wpxp_clipping_and_stats` (advance_xm_wpxp_module.F90:4410), applied once per scalar
    after its xm/wpxp solve:

      1. the monotonic turbulent-flux limiter (`monotonic_turbulent_flux_limit`, no-op unless
         `l_mono_flux_lim`; adjusts BOTH the mean field `xm` and the flux `wpxp` — fixes atex),
      2. `fill_holes_vertical` on the mean field `xm` (gated `fill_holes_type/=0` AND
         `solve_type/=um,vm` — the Fortran skips the mean-field fill for the wind components;
         zt-level, full zt range, threshold=`field_tol`; a bitwise no-op where xm>=field_tol
         everywhere, fires only at a stretched dry top — e.g. rico's moist/dry interface),
      3. `clip_covar` on the flux `wpxp` (bounded by `wp2` and `xp2_clip`).

    Called once per scalar (rt/thl) and once per wind component (um/vm); `solve_type` is the field's
    MFL id (MFL_RTM/MFL_THLM/MFL_UM/MFL_VM). `xp2` (raw) feeds the limiter; `xp2_clip` (the
    relaxed-clipping-floored variance) bounds the covariance clip. Returns `(xm, wpxp)`. The Fortran
    subroutine's clip *budget* stat_updates are not reproduced here (the JAX path omits them)."""
    if l_mono_flux_lim:
        xm, wpxp_preclip = monotonic_turbulent_flux_limit(
            solve_type, xm, wpxp_preclip, xm_old, xp2, wm_zt, xm_forcing,
            rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
            xp2_threshold, xm_tol, low_lev_effect, high_lev_effect, gr, dt)
    if fill_holes_type != 0 and solve_type not in (MFL_UM, MFL_VM):
        xm = jnp.asarray(fill_holes_vertical(
            field=jnp.asarray(xm), rho_ds=jnp.asarray(rho_ds_zt),
            dz=jnp.asarray(gr.dzt), threshold=float(field_tol),
            lower_k=gr.k_lb_zt, upper_k=gr.k_ub_zt,
            fill_holes_type=fill_holes_type))
    wpxp = clip_covar(wpxp_preclip, wp2, xp2_clip)
    # NOTE: the Fortran here optionally calls `xm_correction_wpxp_cl` to adjust xm for the
    # amount w'x' was clipped — but ONLY under `l_clip_turb_adv`, which is OFF in the validated
    # config (the covariance clip DOES fire here, so applying the correction would change the
    # prognostics — verified: wiring it in fails the bit gate with ProgFail 16). The correction
    # is therefore intentionally NOT applied; `xm_correction_wpxp_cl` (below) is the faithful
    # mirror of that Fortran routine, kept + unit-tested for the `l_clip_turb_adv=.true.` config.
    return xm, wpxp


def xm_correction_wpxp_cl(xm, wpxp_chnge, invrs_dzt, dt):
    """Correct xm when w'x' was clipped — the Fortran `xm_correction_wpxp_cl`
    (advance_xm_wpxp_module.F90:5766). Because xm's time-tendency carries the implicit
    turbulent-advection term -d(w'x')/dz, clipping w'x' by `wpxp_chnge`
    (= clipped - unclipped, zm-level) needs the explicit adjuster

        xm_tndcy_wpxp_cl(k) = -invrs_dzt(k) * ( wpxp_chnge(k+1) - wpxp_chnge(k) )
        xm(k) += xm_tndcy_wpxp_cl(k) * dt                       (k over the zt levels)

    applied per column only where any |wpxp_chnge| > eps (the Fortran `l_clipping_needed(i)`
    gate; columns with no clipping are left untouched). `xm`/`invrs_dzt`: (ngrdcol, nzt);
    `wpxp_chnge`: (ngrdcol, nzm), nzm = nzt+1. Pure-jnp (jnp.where gate) → differentiable.

    NOT called from the live `xm_wpxp_clipping_and_stats` path: the Fortran gates this
    correction on `l_clip_turb_adv`, which is OFF in the validated config (the covariance clip
    DOES fire there, so applying the correction changes the prognostics — verified to fail the
    bit gate with ProgFail 16). This is the faithful, unit-tested mirror kept for the
    `l_clip_turb_adv=.true.` config. (The Fortran's `xm_tacl` budget stat_update is not done.)"""
    xm = jnp.asarray(xm)
    wpxp_chnge = jnp.asarray(wpxp_chnge)
    invrs_dzt = jnp.asarray(invrs_dzt)
    l_clip = jnp.any(jnp.abs(wpxp_chnge) > _EPS, axis=1, keepdims=True)   # (ngrdcol, 1)
    xm_tndcy = -invrs_dzt * (wpxp_chnge[:, 1:] - wpxp_chnge[:, :-1])      # (ngrdcol, nzt)
    return jnp.where(l_clip, xm + xm_tndcy * dt, xm)


def advance_xm_wpxp(*, Cx_fnc_Richardson, Kh_zt, clubb_params, dt_advance, fcor, fcor_y, flags, gr,
    invrs_rho_ds_zm, invrs_rho_ds_zt, invrs_tau_C6_zm, l_sample, mixt_frac_zm, ngrdcol,
    nu_vert_res_dep, nzm, nzt, rc_coef_zm, rho_ds_zm, rho_ds_zt, rtm_forcing, rtm_ref, rtp2,
    rtpthvp, sigma_sqd_w, sponge_cfg, stats_writer, thlm_forcing, thlm_ref, thlp2, thlpthvp,
    thv_ds_zm, ts_nudge, ug, um_forcing, um_ref, up2, uprcp, varnce_w_1_zm, varnce_w_2_zm,
    vg, vm_forcing, vm_ref, vp2, vprcp, w_1_zm, w_2_zm, wm_zm, wm_zt, wp2, wp3_on_wp2_zt,
    wprtp_forcing, wpthlp_forcing,
    rcm, rtm, thlm, um, upwp, vm, vpwp, wprtp, wpthlp):
    """Whole-driver advance of the xm/w'x' system — mirrors advance_xm_wpxp_module.F90:advance_xm_wpxp.
    Advances the rt/thl scalar pairs and (l_predict_upwp_vpwp) the um/vm wind pairs: builds the shared
    TA + LHS terms (calc_xm_wpxp_ta_terms / calc_xm_wpxp_lhs_terms), solves each field via
    solve_xm_wpxp_with_single_lhs, applies the per-field clipping (xm_wpxp_clipping_and_stats), the
    wind forcing/Coriolis/diagnose_upxp setup, the sponge/nudge, clip_rcm, and the budget stat_updates.
    Relocated verbatim from advance_clubb_core's inline Block V (iter 160). Returns the advanced state as a
    dict (wprtp/rtm/wpthlp/thlm/upwp/um/vpwp/vm/rcm)."""
    # ---- Save pre-call state for advance_xm_wpxp ----
    _wprtp_pre_xw = wprtp.copy()
    _rtm_pre_xw   = rtm.copy()
    _wpthlp_pre_xw = wpthlp.copy()
    _thlm_pre_xw   = thlm.copy()
    # ---- Save pre-call state for the upwp/vpwp wind prediction ----
    _um_pre_uv   = _asarray(um,   dtype=np.float64).copy()
    _vm_pre_uv   = _asarray(vm,   dtype=np.float64).copy()
    _upwp_pre_uv = _asarray(upwp, dtype=np.float64).copy()
    _vpwp_pre_uv = _asarray(vpwp, dtype=np.float64).copy()


    # ============================================================ #
    # Block V: advance_xm_wpxp (wprtp/rtm, wpthlp/thlm)            #
    # ARM: ADG1, l_diag_Lscale_from_tau=True, l_use_C7_Richardson   #
    # ============================================================ #
    _c_K6_xw = float(clubb_params[0, ic_K6 - 1])
    _Kw6_xw = _c_K6_xw * Kh_zt   # (ngrdcol, nzt)
    _nu6_xw = float(_asarray(nu_vert_res_dep.nu6, dtype=np.float64).flat[0])
    # For ARM l_diag_Lscale_from_tau=True: C6 is constant per column
    _C6rt_xw = jnp.broadcast_to(
        jnp.array(clubb_params[:, iC6rt - 1])[:, None], (ngrdcol, nzm))
    _C6thl_xw = jnp.broadcast_to(
        jnp.array(clubb_params[:, iC6thl - 1])[:, None], (ngrdcol, nzm))
    # For ARM l_use_C7_Richardson=True: C7 = Cx_fnc_Richardson
    _C7_xw = jnp.array(Cx_fnc_Richardson)   # (ngrdcol, nzm)

    # advance_xm_wpxp_module.F90 stats (C6/C7 Skw_fnc, C6_term)
    if l_sample and stats_writer is not None:
        stats_writer.update("C7_Skw_fnc",   _asarray(_C7_xw,   dtype=np.float64))
        stats_writer.update("C6rt_Skw_fnc", _asarray(_C6rt_xw, dtype=np.float64))
        stats_writer.update("C6thl_Skw_fnc",_asarray(_C6thl_xw,dtype=np.float64))
        _C6_term_xw = _C6rt_xw * jnp.asarray(invrs_tau_C6_zm)
        stats_writer.update("C6_term",      _asarray(_C6_term_xw, dtype=np.float64))
        # coef_wp2rtp/thlp_implicit (ADG1): a1_zt * wp3_on_wp2_zt
        _a1_zm_xw = 1.0 / (1.0 - jnp.asarray(sigma_sqd_w))
        _a1_zt_xw = jnp.maximum(zm2zt_jax(_a1_zm_xw, gr), zero_threshold)
        _coef_wp2rtp_xw = _a1_zt_xw * jnp.asarray(wp3_on_wp2_zt)
        stats_writer.update("coef_wp2rtp_implicit",  _asarray(_coef_wp2rtp_xw, dtype=np.float64))
        stats_writer.update("coef_wp2thlp_implicit", _asarray(_coef_wp2rtp_xw, dtype=np.float64))
        # coef_wprtp2/thlp2/rtpthlp_implicit (ADG1): (1/3)*beta*a1_zt*wp3_on_wp2_zt
        _beta_xw = jnp.asarray(clubb_params[:, ibeta - 1])[:, None]
        _coef_wprtp2_xw = (1.0 / 3.0) * _beta_xw * _a1_zt_xw * jnp.asarray(wp3_on_wp2_zt)
        stats_writer.update("coef_wprtp2_implicit",    _asarray(_coef_wprtp2_xw, dtype=np.float64))
        stats_writer.update("coef_wpthlp2_implicit",   _asarray(_coef_wprtp2_xw, dtype=np.float64))
        stats_writer.update("coef_wprtpthlp_implicit", _asarray(_coef_wprtp2_xw, dtype=np.float64))

    # ADG1 turbulent-advection LHS operator (mirrors the Fortran calc_xm_wpxp_ta_terms routine),
    # shared across the rt/thl/um/vm pairs.
    _lhs_ta_wprtp_xw = calc_xm_wpxp_ta_terms(
        sigma_sqd_w=jnp.array(sigma_sqd_w),
        wp3_on_wp2_zt=jnp.array(wp3_on_wp2_zt),
        rho_ds_zt=jnp.array(rho_ds_zt),
        invrs_rho_ds_zm=jnp.array(invrs_rho_ds_zm),
        gr=gr,
    )
    # The remaining shared LHS terms (diffusion, MA, TP, AC+PR2, ta_xm)
    _sh_xw = calc_xm_wpxp_lhs_terms(
        wm_zm=jnp.array(wm_zm), wm_zt=jnp.array(wm_zt),
        wp2=jnp.array(wp2), Kw6=jnp.array(_Kw6_xw), nu6=_nu6_xw,
        C7_Skw_fnc=_C7_xw,
        invrs_rho_ds_zm=jnp.array(invrs_rho_ds_zm),
        rho_ds_zt=jnp.array(rho_ds_zt),
        rho_ds_zm=jnp.array(rho_ds_zm),
        invrs_rho_ds_zt=jnp.array(invrs_rho_ds_zt),
        gr=gr,
    )
    _sh_xw['lhs_ta_wprtp'] = _lhs_ta_wprtp_xw

    # Clipping parameters (matches xm_wpxp_clipping_and_stats in Fortran)
    # l_enable_relaxed_clipping: xp2_floor = 1e-7 for rtp2, 0.01 for thlp2
    _wp2_jax = jnp.array(wp2)
    _rtp2_jax = jnp.array(rtp2)
    _thlp2_jax = jnp.array(thlp2)
    if flags.l_enable_relaxed_clipping:
        _rtp2_clip = jnp.maximum(_rtp2_jax, 1e-7)
        _thlp2_clip = jnp.maximum(_thlp2_jax, 0.01)
    else:
        _rtp2_clip = _rtp2_jax
        _thlp2_clip = _thlp2_jax

    # monotonic flux limiter setup. The turbulent-advection range is
    # field-independent — compute once, reuse for rtm/thlm/um/vm. The limiter is
    # applied after each solve and before clip_covar (matching the Fortran
    # xm_wpxp_clipping_and_stats order). It is a no-op unless w'x' exceeds the
    # monotonic bounds (fixes atex; no-op for the bit-faithful set).
    _lle_mfl, _hle_mfl = calc_turb_adv_range(
        w_1_zm, w_2_zm, varnce_w_1_zm, varnce_w_2_zm, mixt_frac_zm,
        gr, float(dt_advance))
    _rho_ds_zm_mfl = _asarray(rho_ds_zm, np.float64)
    _rho_ds_zt_mfl = _asarray(rho_ds_zt, np.float64)
    _irho_zm_mfl = _asarray(invrs_rho_ds_zm, np.float64)
    _irho_zt_mfl = _asarray(invrs_rho_ds_zt, np.float64)
    _wm_zt_mfl = _asarray(wm_zt, np.float64)
    # The field-path limiter (monotonic_turbulent_flux_limit, now invoked inside
    # xm_wpxp_clipping_and_stats for each of rtm/thlm/um/vm — mirroring the Fortran which calls
    # monotonic_turbulent_flux_limit per field) is JAX (lax.scan), bit-exact to the NumPy reference
    # (tests/test_mono_flux_limiter.py) and differentiable w.r.t. the fields. calc_turb_adv_range
    # (the integer ranges _lle_mfl/_hle_mfl) stays NumPy — they derive from the w-PDF, not the
    # limited fields, so they are structural constants for the grad.

    # Solve wprtp/rtm pair — no clipping in solve; apply separately to get pre-clip value
    _wprtp_preclip_xw, _rtm_jax_xw = solve_xm_wpxp_with_single_lhs(
        wpxp=jnp.array(_wprtp_pre_xw),
        xm=jnp.array(_rtm_pre_xw),
        wpxp_forcing=jnp.array(wprtp_forcing),
        xm_forcing=jnp.array(rtm_forcing),
        C6_Skw_fnc=_C6rt_xw,
        C7_Skw_fnc=_C7_xw,
        invrs_tau_C6_zm=jnp.array(invrs_tau_C6_zm),
        lhs_ta_wpxp=_sh_xw['lhs_ta_wprtp'],
        lhs_diff_zm=_sh_xw['lhs_diff_zm'],
        lhs_ma_zm=_sh_xw['lhs_ma_zm'],
        lhs_ma_zt=_sh_xw['lhs_ma_zt'],
        lhs_ta_xm=_sh_xw['lhs_ta_xm'],
        lhs_tp=_sh_xw['lhs_tp'],
        lhs_ac_pr2=_sh_xw['lhs_ac_pr2'],
        thv_ds_zm=jnp.array(thv_ds_zm),
        xpthvp=jnp.array(rtpthvp),
        wm_zt=jnp.array(wm_zt),
        dt=float(dt_advance),
        gr=gr,
    )
    # Per-field post-solve clipping (MFL + fill_holes + clip_covar) — the Fortran
    # xm_wpxp_clipping_and_stats (advance_xm_wpxp_module.F90:4410), called once per scalar.
    _rtm_jax_xw, _wprtp_jax_xw = xm_wpxp_clipping_and_stats(
        MFL_RTM, _rtm_jax_xw, _wprtp_preclip_xw, _rtm_pre_xw, rtp2, _rtp2_clip, _wp2_jax,
        _wm_zt_mfl, rtm_forcing, _rho_ds_zm_mfl, _rho_ds_zt_mfl, _irho_zm_mfl, _irho_zt_mfl,
        _RT_TOL ** 2, _RT_TOL_MFL, _lle_mfl, _hle_mfl,
        rt_tol, flags.fill_holes_type, getattr(flags, 'l_mono_flux_lim_rtm', False),
        float(dt_advance), gr)

    # Solve wpthlp/thlm pair (same lhs_ta_wprtp for ADG1)
    _wpthlp_preclip_xw, _thlm_jax_xw = solve_xm_wpxp_with_single_lhs(
        wpxp=jnp.array(_wpthlp_pre_xw),
        xm=jnp.array(_thlm_pre_xw),
        wpxp_forcing=jnp.array(wpthlp_forcing),
        xm_forcing=jnp.array(thlm_forcing),
        C6_Skw_fnc=_C6thl_xw,
        C7_Skw_fnc=_C7_xw,
        invrs_tau_C6_zm=jnp.array(invrs_tau_C6_zm),
        lhs_ta_wpxp=_sh_xw['lhs_ta_wprtp'],
        lhs_diff_zm=_sh_xw['lhs_diff_zm'],
        lhs_ma_zm=_sh_xw['lhs_ma_zm'],
        lhs_ma_zt=_sh_xw['lhs_ma_zt'],
        lhs_ta_xm=_sh_xw['lhs_ta_xm'],
        lhs_tp=_sh_xw['lhs_tp'],
        lhs_ac_pr2=_sh_xw['lhs_ac_pr2'],
        thv_ds_zm=jnp.array(thv_ds_zm),
        xpthvp=jnp.array(thlpthvp),
        wm_zt=jnp.array(wm_zt),
        dt=float(dt_advance),
        gr=gr,
    )
    # Per-field post-solve clipping (xm_wpxp_clipping_and_stats, F90:4410) for the thl pair.
    _thlm_jax_xw, _wpthlp_jax_xw = xm_wpxp_clipping_and_stats(
        MFL_THLM, _thlm_jax_xw, _wpthlp_preclip_xw, _thlm_pre_xw, thlp2, _thlp2_clip, _wp2_jax,
        _wm_zt_mfl, thlm_forcing, _rho_ds_zm_mfl, _rho_ds_zt_mfl, _irho_zm_mfl, _irho_zt_mfl,
        _THL_TOL ** 2, _THL_TOL_MFL, _lle_mfl, _hle_mfl,
        thl_tol, flags.fill_holes_type, getattr(flags, 'l_mono_flux_lim_thlm', False),
        float(dt_advance), gr)


    # ============================================================ #
    # Override advance_xm_wpxp state with JAX values        #
    # wprtp/rtm/wpthlp/thlm computed in JAX.                        #
    # ============================================================ #
    wprtp  = _asarray(_wprtp_jax_xw,  dtype=np.float64).copy()
    rtm    = _asarray(_rtm_jax_xw,    dtype=np.float64).copy()
    wpthlp = _asarray(_wpthlp_jax_xw, dtype=np.float64).copy()
    thlm   = _asarray(_thlm_jax_xw,   dtype=np.float64).copy()

    # Sponge-layer damping for rtm/thlm (advance_xm_wpxp_module.F90:1053-1093).
    # No-op unless sponge_cfg enables the field (e.g. ekman).
    rtm  = apply_sponge_field_jax('rtm',  rtm,  rtm_ref, gr, dt_advance, sponge_cfg)
    thlm = apply_sponge_field_jax('thlm', thlm, thlm_ref, gr, dt_advance, sponge_cfg)

    # clip_rcm using JAX-updated rtm
    _rcm_presave = _asarray(rcm, dtype=np.float64).copy()
    _rcm_adg = _asarray(clip_rcm(
        rcm=jnp.asarray(_rcm_presave),
        rtm=jnp.asarray(_asarray(rtm, dtype=np.float64)),
    ))
    rcm = _asarray(_rcm_adg, dtype=np.float64).copy()

    # ============================================================ #
    # advance_xm_wpxp wind pair: um/upwp and vm/vpwp               #
    # ARM: l_predict_upwp_vpwp=True, l_implemented=False,           #
    # l_lmm_stepping=False, l_ho_trad/nontrad_coriolis=False        #
    # ============================================================ #
    # upwp_forcing = C_uu_shr * wp2 * ddzt(um_pre)  (zm-level)
    _C_uu_shr_uv = _asarray(
        clubb_params[:, iC_uu_shr - 1], dtype=np.float64
    )[:, np.newaxis]
    _wp2_uv     = _asarray(wp2, dtype=np.float64)
    _ddzt_um_uv  = _asarray(ddzt(jnp.asarray(_um_pre_uv), gr))
    _ddzt_vm_uv  = _asarray(ddzt(jnp.asarray(_vm_pre_uv), gr))
    _upwp_forcing_uv = _C_uu_shr_uv * _wp2_uv * _ddzt_um_uv  # (ngrdcol, nzm)
    _vpwp_forcing_uv = _C_uu_shr_uv * _wp2_uv * _ddzt_vm_uv  # (ngrdcol, nzm)
    # Nontraditional Coriolis term for upwp (advance_xm_wpxp_module.F90:3098-3106).
    if getattr(flags, 'l_ho_nontrad_coriolis', False):
        _fcy_uv = _asarray(fcor_y, dtype=np.float64)[:, np.newaxis]
        _upwp_forcing_uv = _upwp_forcing_uv + _fcy_uv * (
            _asarray(up2, dtype=np.float64) - _wp2_uv)

    # Coriolis + large-scale forcing for um/vm  (l_implemented=False)
    # um_tndcy = um_forcing - fcor * (vg - vm_pre)
    _fcor_uv = _asarray(fcor, dtype=np.float64)[:, np.newaxis]  # (ngrdcol,1)
    _um_tndcy_uv = (_asarray(um_forcing, dtype=np.float64)
                   - _fcor_uv * (_asarray(vg, dtype=np.float64) - _vm_pre_uv))
    _vm_tndcy_uv = (_asarray(vm_forcing, dtype=np.float64)
                   + _fcor_uv * (_asarray(ug, dtype=np.float64) - _um_pre_uv))

    # diagnose_upxp: upthvp via upthlp + ep1*thv_ds*uprtp + rc_coef*uprcp
    # um_smth = zt2zm2zt(um)  (smoothed zt-level)
    _um_smth_uv = _asarray(zt2zm2zt(jnp.asarray(_um_pre_uv), gr))
    _vm_smth_uv = _asarray(zt2zm2zt(jnp.asarray(_vm_pre_uv), gr))

    _invrs_tau_C6_uv  = _asarray(invrs_tau_C6_zm, dtype=np.float64)
    _tau_C6_uv        = _xp.where(_invrs_tau_C6_uv > 1e-30,
                                 1.0 / _invrs_tau_C6_uv, 0.0)
    _C6thl_uv = _xp.broadcast_to(
        _asarray(clubb_params[:, iC6thl - 1], dtype=np.float64)[:, np.newaxis],
        (ngrdcol, nzm)).copy()
    _C6rt_uv  = _xp.broadcast_to(
        _asarray(clubb_params[:, iC6rt - 1], dtype=np.float64)[:, np.newaxis],
        (ngrdcol, nzm)).copy()
    _C7_uv    = _asarray(Cx_fnc_Richardson, dtype=np.float64)  # (ngrdcol, nzm)

    _wpthlp_uv = _asarray(_wpthlp_pre_xw, dtype=np.float64)
    _wprtp_uv  = _asarray(_wprtp_pre_xw,  dtype=np.float64)

    # diagnose_upxp (advance_xm_wpxp_module.F90:6052) — horizontal scalar fluxes
    # upthlp/uprtp/vpthlp/vprtp. ym = um_smth/vm_smth (the Fortran smoothed-velocity
    # arg); d/dz of xm and ym are formed inside the routine.
    _upthlp_uv = _asarray(diagnose_upxp(
        _upwp_pre_uv, _thlm_pre_xw, _wpthlp_uv, _um_smth_uv, _C6thl_uv, _tau_C6_uv, _C7_uv, gr))
    _uprtp_uv  = _asarray(diagnose_upxp(
        _upwp_pre_uv, _rtm_pre_xw,  _wprtp_uv,  _um_smth_uv, _C6rt_uv,  _tau_C6_uv, _C7_uv, gr))
    _vpthlp_uv = _asarray(diagnose_upxp(
        _vpwp_pre_uv, _thlm_pre_xw, _wpthlp_uv, _vm_smth_uv, _C6thl_uv, _tau_C6_uv, _C7_uv, gr))
    _vprtp_uv  = _asarray(diagnose_upxp(
        _vpwp_pre_uv, _rtm_pre_xw,  _wprtp_uv,  _vm_smth_uv, _C6rt_uv,  _tau_C6_uv, _C7_uv, gr))

    _ep1_uv   = float(ep1)
    _thv_ds_uv = _asarray(thv_ds_zm, dtype=np.float64)
    _rc_cf_uv  = _asarray(rc_coef_zm, dtype=np.float64)
    _uprcp_uv  = _asarray(uprcp, dtype=np.float64)
    _vprcp_uv  = _asarray(vprcp, dtype=np.float64)

    _upthvp_tmp_uv = _upthlp_uv + _ep1_uv * _thv_ds_uv * _uprtp_uv + _rc_cf_uv * _uprcp_uv
    _vpthvp_tmp_uv = _vpthlp_uv + _ep1_uv * _thv_ds_uv * _vprtp_uv + _rc_cf_uv * _vprcp_uv
    # smooth via zm2zt2zm
    _upthvp_uv = _asarray(zm2zt2zm(jnp.asarray(_upthvp_tmp_uv), gr))
    _vpthvp_uv = _asarray(zm2zt2zm(jnp.asarray(_vpthvp_tmp_uv), gr))

    # Clipping floor: up2/vp2 (l_tke_aniso=True → use up2 directly, no relaxed floor)
    _up2_uv = jnp.asarray(_asarray(up2, dtype=np.float64))
    _vp2_uv = jnp.asarray(_asarray(vp2, dtype=np.float64))

    # Solve upwp/um pair — no clipping in solve; apply separately
    _upwp_preclip_uv, _um_jax_uv = solve_xm_wpxp_with_single_lhs(
        wpxp=jnp.asarray(_upwp_pre_uv),
        xm=jnp.asarray(_um_pre_uv),
        wpxp_forcing=jnp.asarray(_upwp_forcing_uv),
        xm_forcing=jnp.asarray(_um_tndcy_uv),
        C6_Skw_fnc=_C6rt_xw,
        C7_Skw_fnc=_C7_xw,
        invrs_tau_C6_zm=jnp.asarray(_invrs_tau_C6_uv),
        lhs_ta_wpxp=_sh_xw['lhs_ta_wprtp'],
        lhs_diff_zm=_sh_xw['lhs_diff_zm'],
        lhs_ma_zm=_sh_xw['lhs_ma_zm'],
        lhs_ma_zt=_sh_xw['lhs_ma_zt'],
        lhs_ta_xm=_sh_xw['lhs_ta_xm'],
        lhs_tp=_sh_xw['lhs_tp'],
        lhs_ac_pr2=_sh_xw['lhs_ac_pr2'],
        thv_ds_zm=jnp.asarray(_thv_ds_uv),
        xpthvp=jnp.asarray(_upthvp_uv),
        wm_zt=jnp.asarray(_asarray(wm_zt, dtype=np.float64)),
        dt=float(dt_advance),
        gr=gr,
    )
    # Per-component post-solve clipping (xm_wpxp_clipping_and_stats, F90:4410); the wind
    # components skip the mean-field fill_holes (gated solve_type/=um,vm inside the routine).
    _um_jax_uv, _upwp_jax_uv = xm_wpxp_clipping_and_stats(
        MFL_UM, _um_jax_uv, _upwp_preclip_uv, _um_pre_uv, up2, _up2_uv, _wp2_jax,
        _wm_zt_mfl, _um_tndcy_uv, _rho_ds_zm_mfl, _rho_ds_zt_mfl, _irho_zm_mfl, _irho_zt_mfl,
        _W_TOL_SQD, _W_TOL, _lle_mfl, _hle_mfl,
        rt_tol, flags.fill_holes_type, getattr(flags, 'l_mono_flux_lim_um', False),
        float(dt_advance), gr)

    # Solve vpwp/vm pair — no clipping in solve; apply separately
    _vpwp_preclip_uv, _vm_jax_uv = solve_xm_wpxp_with_single_lhs(
        wpxp=jnp.asarray(_vpwp_pre_uv),
        xm=jnp.asarray(_vm_pre_uv),
        wpxp_forcing=jnp.asarray(_vpwp_forcing_uv),
        xm_forcing=jnp.asarray(_vm_tndcy_uv),
        C6_Skw_fnc=_C6rt_xw,
        C7_Skw_fnc=_C7_xw,
        invrs_tau_C6_zm=jnp.asarray(_invrs_tau_C6_uv),
        lhs_ta_wpxp=_sh_xw['lhs_ta_wprtp'],
        lhs_diff_zm=_sh_xw['lhs_diff_zm'],
        lhs_ma_zm=_sh_xw['lhs_ma_zm'],
        lhs_ma_zt=_sh_xw['lhs_ma_zt'],
        lhs_ta_xm=_sh_xw['lhs_ta_xm'],
        lhs_tp=_sh_xw['lhs_tp'],
        lhs_ac_pr2=_sh_xw['lhs_ac_pr2'],
        thv_ds_zm=jnp.asarray(_thv_ds_uv),
        xpthvp=jnp.asarray(_vpthvp_uv),
        wm_zt=jnp.asarray(_asarray(wm_zt, dtype=np.float64)),
        dt=float(dt_advance),
        gr=gr,
    )
    # Per-component post-solve clipping (xm_wpxp_clipping_and_stats, F90:4410) for the vm pair.
    _vm_jax_uv, _vpwp_jax_uv = xm_wpxp_clipping_and_stats(
        MFL_VM, _vm_jax_uv, _vpwp_preclip_uv, _vm_pre_uv, vp2, _vp2_uv, _wp2_jax,
        _wm_zt_mfl, _vm_tndcy_uv, _rho_ds_zm_mfl, _rho_ds_zt_mfl, _irho_zm_mfl, _irho_zt_mfl,
        _W_TOL_SQD, _W_TOL, _lle_mfl, _hle_mfl,
        rt_tol, flags.fill_holes_type, getattr(flags, 'l_mono_flux_lim_vm', False),
        float(dt_advance), gr)


    # ============================================================ #
    # Store advance_xm_wpxp wind state (um/upwp, vm/vpwp)           #
    # computed in JAX.                                              #
    # ============================================================ #
    upwp = _asarray(_upwp_jax_uv, dtype=np.float64).copy()
    um   = _asarray(_um_jax_uv,   dtype=np.float64).copy()
    vpwp = _asarray(_vpwp_jax_uv, dtype=np.float64).copy()
    vm   = _asarray(_vm_jax_uv,   dtype=np.float64).copy()

    # Sponge-layer damping for um/vm (advance_xm_wpxp_module.F90:1095-1123,
    # under l_predict_upwp_vpwp + uv_sponge). No-op unless sponge_cfg enables 'uv'.
    um = apply_sponge_field_jax('uv', um, um_ref, gr, dt_advance, sponge_cfg)
    vm = apply_sponge_field_jax('uv', vm, vm_ref, gr, dt_advance, sponge_cfg)

    # uv nudging toward the initial reference profiles (advance_xm_wpxp_module.F90:
    # 1126-1151, under l_predict_upwp_vpwp + l_uv_nudge). No-op unless l_uv_nudge
    # (none of the cloud/dry cases use it; coriolis_test does, ts_nudge=dt → full reset).
    if getattr(flags, 'l_uv_nudge', False):
        _nf = float(dt_advance) / float(ts_nudge)
        um = um - (um - _asarray(um_ref, dtype=np.float64)) * _nf
        vm = vm - (vm - _asarray(vm_ref, dtype=np.float64)) * _nf

    # ============================================================ #
    # advance_xm_wpxp budget term stats writes               #
    # ============================================================ #
    if l_sample and stats_writer is not None:
        _grav_dg = float(grav)
        _gamma_dg = gamma_over_implicit_ts

        # --- Pre-advance snapshots ---
        stats_writer.update("rtm_old",  _asarray(_rtm_pre_xw,   dtype=np.float64))
        stats_writer.update("thlm_old", _asarray(_thlm_pre_xw,  dtype=np.float64))

        # --- Forcings ---
        stats_writer.update("rtm_forcing",  _asarray(rtm_forcing,  dtype=np.float64))
        stats_writer.update("thlm_forcing", _asarray(thlm_forcing, dtype=np.float64))

        # --- Geostrophic and Coriolis terms for um/vm ---
        _fcor_dg = _asarray(fcor, dtype=np.float64)[:, np.newaxis]  # (ngrdcol,1)
        stats_writer.update("um_gf", -_fcor_dg * _asarray(vg, dtype=np.float64))
        stats_writer.update("vm_gf",  _fcor_dg * _asarray(ug, dtype=np.float64))
        stats_writer.update("um_cf",  _fcor_dg * _vm_pre_uv)
        stats_writer.update("vm_cf", -_fcor_dg * _um_pre_uv)

        # --- diagnose_upxp results ---
        stats_writer.update("upthlp", _upthlp_uv)
        stats_writer.update("uprtp",  _uprtp_uv)
        stats_writer.update("upthvp", _upthvp_uv)
        stats_writer.update("vpthlp", _vpthlp_uv)
        stats_writer.update("vprtp",  _vprtp_uv)
        stats_writer.update("vpthvp", _vpthvp_uv)

        # --- Shared LHS terms (already computed in _sh_xw) ---
        _lhs_ta_wpxp_dg = _asarray(_sh_xw['lhs_ta_wprtp'], dtype=np.float64)
        _lhs_diff_dg    = _asarray(_sh_xw['lhs_diff_zm'],  dtype=np.float64)
        _lhs_tp_dg      = _asarray(_sh_xw['lhs_tp'],       dtype=np.float64)
        _lhs_ta_xm_dg   = _asarray(_sh_xw['lhs_ta_xm'],    dtype=np.float64)
        _thv_ds_dg      = _asarray(thv_ds_zm,             dtype=np.float64)

        # lhs_pr1 for each pair (variable-specific)
        _invrs_tau_C6_dg  = _asarray(invrs_tau_C6_zm, dtype=np.float64)
        _lhs_pr1_rtp_dg   = _asarray(wpxp_term_pr1_lhs(
            jnp.asarray(_asarray(_C6rt_xw)),
            jnp.asarray(_invrs_tau_C6_dg)), dtype=np.float64)
        _lhs_pr1_thl_dg   = _asarray(wpxp_term_pr1_lhs(
            jnp.asarray(_asarray(_C6thl_xw)),
            jnp.asarray(_invrs_tau_C6_dg)), dtype=np.float64)

        def _wpxp_budgets_dg(name_bp, name_pr3, name_ta, name_pr1,
                             name_tp, name_dp1, name_xm_ta,
                             wpxp_pre, wpxp_new, xm_new, xpthvp_np,
                             C7_np, lhs_pr1_np):
            """Write wpxp and xm budget term stats for one variable pair."""
            # bp / pr3: exactly as the Fortran (advance_xm_wpxp_module.F90:1894-1913), the two explicit
            # contributions are obtained by calling wpxp_terms_bp_pr3_rhs with C7_Skw_fnc=0 (→ bp) and
            # C7_Skw_fnc+1 (→ pr3); the routine itself zeros the boundaries and returns the full profile.
            _C7_j = jnp.asarray(C7_np)
            _thv_j = jnp.asarray(_thv_ds_dg)
            _xpthvp_j = jnp.asarray(xpthvp_np)
            _bp = _asarray(wpxp_terms_bp_pr3_rhs(
                jnp.zeros_like(_C7_j), _thv_j, _xpthvp_j, _grav_dg), dtype=np.float64)
            stats_writer.update(name_bp, _bp)
            _pr3 = _asarray(wpxp_terms_bp_pr3_rhs(
                _C7_j + 1.0, _thv_j, _xpthvp_j, _grav_dg), dtype=np.float64)
            stats_writer.update(name_pr3, _pr3)
            # ta: explicit over-implicit part (1-gamma, pre-solve) + implicit finalize (gamma,
            # post-solve), both the same 3-band turb-adv LHS applied via the shared band3 kernel.
            _ta_over = (1.0 - _gamma_dg) * (-apply_lhs_band3_interior_jax(_lhs_ta_wpxp_dg, wpxp_pre))
            _ta_impl = -_gamma_dg * apply_lhs_band3_interior_jax(_lhs_ta_wpxp_dg, wpxp_new)
            stats_writer.update(name_ta, _ta_over + _ta_impl)
            # pr1: begin -(1-gamma)*lhs_pr1*wpxp_pre, finalize -gamma*lhs_pr1*wpxp_new
            _pr1 = _xp.zeros_like(wpxp_pre)
            _pr1 = _iset(_pr1, np.s_[:, 1:-1], (
                -(1.0 - _gamma_dg) * lhs_pr1_np[:, 1:-1] * wpxp_pre[:, 1:-1]
                - _gamma_dg * lhs_pr1_np[:, 1:-1] * wpxp_new[:, 1:-1]
            ))
            stats_writer.update(name_pr1, _pr1)
            # tp: implicit, uses post-solve xm — shared zt→zm 2-band apply kernel
            _tp = -apply_lhs_band2_zt2zm_interior_jax(_lhs_tp_dg, xm_new)
            stats_writer.update(name_tp, _tp)
            # dp1: diffusion, implicit, uses post-solve wpxp — shared band3 kernel
            _dp1 = -apply_lhs_band3_interior_jax(_lhs_diff_dg, wpxp_new)
            stats_writer.update(name_dp1, _dp1)
            # xm_ta: implicit, uses post-solve wpxp
            _xm_ta = (
                -_lhs_ta_xm_dg[1] * wpxp_new[:, :-1]
                - _lhs_ta_xm_dg[0] * wpxp_new[:, 1:]
            )
            stats_writer.update(name_xm_ta, _xm_ta)

        _C7_np_dg = _asarray(_C7_xw, dtype=np.float64)
        _dt_dg    = float(dt_advance)

        # wprtp/rtm budget terms — use pre-clip wprtp for dp1/ta/pr1
        _wprtp_pc_xw = _asarray(_wprtp_preclip_xw, dtype=np.float64)
        _wpxp_budgets_dg(
            "wprtp_bp", "wprtp_pr3", "wprtp_ta", "wprtp_pr1",
            "wprtp_tp", "wprtp_dp1", "rtm_ta",
            _asarray(_wprtp_pre_xw, dtype=np.float64),
            _wprtp_pc_xw,
            _asarray(_rtm_jax_xw,   dtype=np.float64),
            _asarray(rtpthvp,      dtype=np.float64),
            _C7_np_dg, _lhs_pr1_rtp_dg,
        )
        # wprtp_cl: clipping rate = (post_clip - pre_clip) / dt
        _wprtp_jax_xw_np = _asarray(_wprtp_jax_xw, dtype=np.float64)
        stats_writer.update("wprtp_cl", (_wprtp_jax_xw_np - _wprtp_pc_xw) / _dt_dg)

        # wpthlp/thlm budget terms — use pre-clip wpthlp
        _wpthlp_pc_xw = _asarray(_wpthlp_preclip_xw, dtype=np.float64)
        _wpxp_budgets_dg(
            "wpthlp_bp", "wpthlp_pr3", "wpthlp_ta", "wpthlp_pr1",
            "wpthlp_tp", "wpthlp_dp1", "thlm_ta",
            _asarray(_wpthlp_pre_xw, dtype=np.float64),
            _wpthlp_pc_xw,
            _asarray(_thlm_jax_xw,   dtype=np.float64),
            _asarray(thlpthvp,      dtype=np.float64),
            _C7_np_dg, _lhs_pr1_thl_dg,
        )
        _wpthlp_jax_xw_np = _asarray(_wpthlp_jax_xw, dtype=np.float64)
        stats_writer.update("wpthlp_cl", (_wpthlp_jax_xw_np - _wpthlp_pc_xw) / _dt_dg)

        # upwp/um budget terms — use pre-clip upwp
        _upwp_pc_uv = _asarray(_upwp_preclip_uv, dtype=np.float64)
        _wpxp_budgets_dg(
            "upwp_bp", "upwp_pr3", "upwp_ta", "upwp_pr1",
            "upwp_tp", "upwp_dp1", "um_ta",
            _asarray(_upwp_pre_uv, dtype=np.float64),
            _upwp_pc_uv,
            _asarray(_um_jax_uv,   dtype=np.float64),
            _upthvp_uv,
            _C7_np_dg, _lhs_pr1_rtp_dg,
        )
        _upwp_jax_uv_np = _asarray(_upwp_jax_uv, dtype=np.float64)
        stats_writer.update("upwp_cl", (_upwp_jax_uv_np - _upwp_pc_uv) / _dt_dg)

        # vpwp/vm budget terms — use pre-clip vpwp
        _vpwp_pc_uv = _asarray(_vpwp_preclip_uv, dtype=np.float64)
        _wpxp_budgets_dg(
            "vpwp_bp", "vpwp_pr3", "vpwp_ta", "vpwp_pr1",
            "vpwp_tp", "vpwp_dp1", "vm_ta",
            _asarray(_vpwp_pre_uv, dtype=np.float64),
            _vpwp_pc_uv,
            _asarray(_vm_jax_uv,   dtype=np.float64),
            _vpthvp_uv,
            _C7_np_dg, _lhs_pr1_rtp_dg,
        )
        _vpwp_jax_uv_np = _asarray(_vpwp_jax_uv, dtype=np.float64)
        stats_writer.update("vpwp_cl", (_vpwp_jax_uv_np - _vpwp_pc_uv) / _dt_dg)

        # upwp_pr4/vpwp_pr4: C_uu_shr * wp2 * ddzt_um/vm (zt-level gradient)
        # Mirrors Fortran advance_xm_wpxp_module: tmp_zm=C_uu_shr*wp2*ddzt_um
        stats_writer.update("upwp_pr4", _C_uu_shr_uv * _wp2_uv * _ddzt_um_uv)
        stats_writer.update("vpwp_pr4", _C_uu_shr_uv * _wp2_uv * _ddzt_vm_uv)

        # ---- MFL stats: mean_w_up/down + rtm/thlm MFL bounds ----
        # Uses previous-timestep PDF params (w_1_zm etc.) per ARM post-advance path.
        # Fortran: mono_flux_limiter.F90 / calc_turb_adv_range / mean_vert_vel_up_down
        _l_mfl_rt  = getattr(flags, 'l_mono_flux_lim_rtm',  False)
        _l_mfl_thl = getattr(flags, 'l_mono_flux_lim_thlm', False)
        _l_mfl_spk = getattr(flags, 'l_mono_flux_lim_spikefix', True)

        if _l_mfl_rt or _l_mfl_thl:
            _gd_mfl  = float(gr.grid_dir)
            _idt_mfl = 1.0 / float(dt_advance)
            _dt_mfl  = float(dt_advance)
            _dzm_mfl = _asarray(gr.dzm, dtype=np.float64)
            _dzt_mfl = (_asarray(gr.zm)[:, 1:] - _asarray(gr.zm)[:, :-1]) * _gd_mfl

            # mean_w_up/down (Fortran mean_vert_vel_up_down) + the turb-adv ranges, both
            # via mono_flux_limiter. The per-component means are mf-weighted (as the Fortran
            # caller / calc_turb_adv_range does); the integer ranges coincide with the
            # field-MFL call above (same w-PDF inputs + dt), so reuse _lle_mfl/_hle_mfl.
            _wm_mfl = _gd_mfl * _dzm_mfl * _idt_mfl
            _vvd_mfl, _vvu_mfl = mean_vert_vel_up_down(
                w_1_zm, w_2_zm, varnce_w_1_zm, varnce_w_2_zm, mixt_frac_zm, _wm_mfl)
            _vvd_mfl = _asarray(_vvd_mfl, dtype=np.float64)
            _vvu_mfl = _asarray(_vvu_mfl, dtype=np.float64)
            stats_writer.update("mean_w_down", _vvd_mfl)
            stats_writer.update("mean_w_up",   _vvu_mfl)
            _ll_mfl, _hl_mfl = _lle_mfl, _hle_mfl

            _rdszt = _asarray(rho_ds_zt,       dtype=np.float64)
            _rdszm = _asarray(rho_ds_zm,       dtype=np.float64)
            _irdzm = _asarray(invrs_rho_ds_zm, dtype=np.float64)

            def _mfl_scalar(xm_old_np, xm_new, xm_frc_np,
                            xp2_zm, xm_tol_v, max_xp2_v, xp2_thr,
                            wpxp_in,
                            nm_xe, nm_xx, nm_wta, nm_xmn, nm_xmx,
                            nm_we, nm_wx, nm_wmn, nm_wmx,
                            l_spk):
                _xm_n = _asarray(xm_new,   dtype=np.float64)
                _wp   = _asarray(wpxp_in,  dtype=np.float64)
                # Entry stats
                stats_writer.update(nm_xe, _xm_n)
                stats_writer.update(nm_we, _wp)
                # xm_without_ta = xm_old + dt * xm_forcing  (m_adv_term = 0)
                _wta = (_asarray(xm_old_np, dtype=np.float64)
                        + _dt_mfl * _asarray(xm_frc_np, dtype=np.float64))
                stats_writer.update(nm_wta, _wta)
                # Clip xp2 zm → zt
                _xp2z = _xp.clip(
                    _asarray(zm2zt_jax(jnp.asarray(xp2_zm), gr), dtype=np.float64),
                    xp2_thr, max_xp2_v)
                _mxd = _xp.maximum(2.0 * _xp.sqrt(_xp2z), xm_tol_v)
                _mnl = _xp.maximum(_wta - _mxd, 0.0)   # positive-definite (rt, thl)
                _mxl = _wta + _mxd
                # Multi-level min/max over advection range
                _mna = _xp.empty_like(_wta)
                _mxa = _xp.empty_like(_wta)
                for _ii in range(ngrdcol):
                    for _kk in range(nzt):
                        _lo = int(_ll_mfl[_ii, _kk])
                        _hi = int(_hl_mfl[_ii, _kk])
                        _mna = _iset(_mna, np.s_[_ii, _kk], _xp.min(_mnl[_ii, _lo:_hi+1]))
                        _mxa = _iset(_mxa, np.s_[_ii, _kk], _xp.max(_mxl[_ii, _lo:_hi+1]))
                stats_writer.update(nm_xmn, _mna)
                stats_writer.update(nm_xmx, _mxa)
                # wpxp MFL bounds (Fortran lines 672-756)
                _thr_zt = _idt_mfl * _gd_mfl * _dzt_mfl * (_wta - _mna)
                _mxt_zt = _rdszt * _thr_zt
                _mnt_zt = _rdszt * _idt_mfl * _gd_mfl * _dzt_mfl * (_wta - _mxa)
                _thr_zm = _asarray(zt2zm_jax(jnp.asarray(_thr_zt), gr), dtype=np.float64)
                _wpmx   = np.zeros((ngrdcol, nzm))
                _wpmn   = np.zeros((ngrdcol, nzm))
                for _ii in range(ngrdcol):
                    for _kk in range(1, nzm - 1):
                        _km1 = _kk - 1   # k - grid_dir_indx (ascending)
                        _kzt = _kk - 1   # k_zt (ascending)
                        if (l_spk
                                and abs(_wp[_ii, _km1]) > _thr_zm[_ii, _km1]
                                and _wp[_ii, _km1] < 0.0):
                            _wpmx = _iset(_wpmx, np.s_[_ii, _kk], 0.0)
                        else:
                            _wpmx = _iset(_wpmx, np.s_[_ii, _kk], _irdzm[_ii, _kk] * (
                                _mxt_zt[_ii, _kzt] + _rdszm[_ii, _km1] * _wp[_ii, _km1]))
                        _wpmn = _iset(_wpmn, np.s_[_ii, _kk], _irdzm[_ii, _kk] * (
                            _mnt_zt[_ii, _kzt] + _rdszm[_ii, _km1] * _wp[_ii, _km1]))
                stats_writer.update(nm_wmx, _wpmx)
                stats_writer.update(nm_wmn, _wpmn)
                # Exit stats (= entry for ARM; MFL rarely adjusts)
                stats_writer.update(nm_xx, _xm_n)
                stats_writer.update(nm_wx, _wp)

            if _l_mfl_rt:
                _mfl_scalar(
                    _rtm_pre_xw, _rtm_jax_xw, rtm_forcing,
                    rtp2, 1e-4, 5e-6, float(rt_tol**2),
                    _wprtp_preclip_xw,
                    "rtm_enter_mfl", "rtm_exit_mfl", "rtm_without_ta",
                    "rtm_mfl_min", "rtm_mfl_max",
                    "wprtp_enter_mfl", "wprtp_exit_mfl",
                    "wprtp_mfl_min", "wprtp_mfl_max",
                    _l_mfl_spk,
                )
            if _l_mfl_thl:
                _mfl_scalar(
                    _thlm_pre_xw, _thlm_jax_xw, thlm_forcing,
                    thlp2, 0.2, 5.0, float(thl_tol**2),
                    _wpthlp_preclip_xw,
                    "thlm_enter_mfl", "thlm_exit_mfl", "thlm_without_ta",
                    "thlm_mfl_min", "thlm_mfl_max",
                    "wpthlp_enter_mfl", "wpthlp_exit_mfl",
                    "wpthlp_mfl_min", "wpthlp_mfl_max",
                    False,
                )
    return dict(wprtp=wprtp, rtm=rtm, wpthlp=wpthlp, thlm=thlm,
                upwp=upwp, um=um, vpwp=vpwp, vm=vm, rcm=rcm)
