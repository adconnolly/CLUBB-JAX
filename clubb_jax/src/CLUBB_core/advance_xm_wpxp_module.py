"""JAX implementations of advance_xm_wpxp sub-functions.

Faithful ports of CLUBB_core/advance_xm_wpxp_module.F90 and mean_adv.F90.

Functions implemented:
  xm_term_ta_lhs_jax       -- turbulent advection LHS for xm (zt-level)
  wpxp_term_tp_lhs_jax     -- turbulent production LHS for w'x' (zm-level)
  wpxp_terms_ac_pr2_lhs_jax -- accumulation + pressure-2 LHS for w'x' (zm-level)
  wpxp_term_pr1_lhs_jax    -- pressure-1 LHS for w'x' (zm-level)
  wpxp_terms_bp_pr3_rhs_jax -- buoyancy-production + pressure-3 RHS for w'x' (zm-level)
  term_ma_zt_lhs_jax       -- mean-advection LHS for xm (zt-level), upwind scheme
  xm_wpxp_lhs_jax          -- full interleaved pentadiagonal LHS assembly
  xm_wpxp_rhs_jax          -- full interleaved RHS assembly
  advance_xm_wpxp_jax      -- full solve: returns (wpxp_new, xm_new)
  clip_covar_jax            -- covariance clipping post-solve

All functions operate in float64.

References:
  src/CLUBB_core/advance_xm_wpxp_module.F90
  src/CLUBB_core/mean_adv.F90  (term_ma_zt_lhs)
"""

from __future__ import annotations

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.grid_class import zm2zt_jax, zt2zm_jax
from clubb_jax.src.CLUBB_core.diffusion import (
    diffusion_zm_lhs_jax,
    term_ma_zm_lhs_jax,
    xpyp_term_ta_pdf_lhs_jax,
)
from clubb_jax.src.CLUBB_core.matrix_solver_wrapper import penta_lu_solve_jax, penta_lu_solve

# gamma_over_implicit_ts = 3/2 (from constants_clubb.F90)
_gamma = 1.5


# ---------------------------------------------------------------------------
# xm_term_ta_lhs_jax
# ---------------------------------------------------------------------------

def xm_term_ta_lhs_jax(
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
# wpxp_term_tp_lhs_jax
# ---------------------------------------------------------------------------

def wpxp_term_tp_lhs_jax(
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
# wpxp_terms_ac_pr2_lhs_jax
# ---------------------------------------------------------------------------

def wpxp_terms_ac_pr2_lhs_jax(
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
# wpxp_term_pr1_lhs_jax
# ---------------------------------------------------------------------------

def wpxp_term_pr1_lhs_jax(
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
# wpxp_terms_bp_pr3_rhs_jax
# ---------------------------------------------------------------------------

def wpxp_terms_bp_pr3_rhs_jax(
    C7_Skw_fnc: jnp.ndarray,
    thv_ds_zm: jnp.ndarray,
    xpthvp: jnp.ndarray,
    grav: float = 9.81,
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
# term_ma_zt_lhs_jax  (upwind scheme, ascending grid)
# ---------------------------------------------------------------------------

def term_ma_zt_lhs_jax(
    wm_zt: jnp.ndarray,
    gr,
) -> jnp.ndarray:
    """Mean-advection LHS for a zt-level variable, upwind scheme.

    Faithful port of mean_adv.F90:term_ma_zt_lhs with l_upwind_xm_ma=True
    and ascending grid (grid_dir=+1).

    For ascending grid and wm_zt >= 0 (upward wind):
      super = 0,  main = +wm_zt*invrs_dzm[k],  sub = -wm_zt*invrs_dzm[k]
    For ascending grid and wm_zt < 0 (downward wind):
      super = +wm_zt*invrs_dzm[k+1],  main = -wm_zt*invrs_dzm[k+1],  sub = 0

    Output shape: (3, ngrdcol, nzt).
      out[0] = superdiagonal (coeff of xm[k+1])
      out[1] = main diagonal (coeff of xm[k])
      out[2] = subdiagonal   (coeff of xm[k-1])

    Args:
        wm_zt: (ngrdcol, nzt)
        gr:    grid object with invrs_dzm (ngrdcol, nzm)
    """
    ngrdcol, nzt = wm_zt.shape
    nzm = nzt + 1
    invrs_dzm = gr.invrs_dzm   # (ngrdcol, nzm)

    # ---- Interior k=1..nzt-2 (Fortran k=2..nzt-1) ----
    wm_int = wm_zt[:, 1:-1]                  # (ngrdcol, nzt-2)
    idzm_k  = invrs_dzm[:, 1:-2]             # invrs_dzm[k]   for k=1..nzt-2 (Python zm)
    idzm_kp1 = invrs_dzm[:, 2:-1]            # invrs_dzm[k+1] for k=1..nzt-2

    mask = wm_int >= 0.0                     # True → upward wind

    sup_int = jnp.where(mask, 0.0,            wm_int * idzm_kp1)
    mid_int = jnp.where(mask,  wm_int * idzm_k, -wm_int * idzm_kp1)
    sub_int = jnp.where(mask, -wm_int * idzm_k, 0.0)

    # ---- Upper boundary k=nzt-1 (Fortran k=nzt) ----
    wm_top = wm_zt[:, -1]                    # (ngrdcol,)
    idzm_top = invrs_dzm[:, nzm-2]           # invrs_dzm[nzm-2] (Fortran invrs_dzm(nzm-1))
    mask_top = wm_top >= 0.0

    sup_top = jnp.zeros((ngrdcol,))
    mid_top = jnp.where(mask_top,  wm_top * idzm_top, 0.0)
    sub_top = jnp.where(mask_top, -wm_top * idzm_top, 0.0)

    # ---- Lower boundary k=0 (Fortran k=1) ----
    wm_bot = wm_zt[:, 0]                     # (ngrdcol,)
    idzm_2 = invrs_dzm[:, 1]                 # invrs_dzm[1] (Fortran invrs_dzm(2))
    mask_bot = wm_bot >= 0.0

    sup_bot = jnp.where(mask_bot, 0.0,  wm_bot * idzm_2)
    mid_bot = jnp.where(mask_bot, 0.0, -wm_bot * idzm_2)
    sub_bot = jnp.zeros((ngrdcol,))

    # ---- Assemble (ngrdcol, nzt) ----
    sup = jnp.concatenate([sup_bot[:, None], sup_int, sup_top[:, None]], axis=1)
    mid = jnp.concatenate([mid_bot[:, None], mid_int, mid_top[:, None]], axis=1)
    sub = jnp.concatenate([sub_bot[:, None], sub_int, sub_top[:, None]], axis=1)

    return jnp.stack([sup, mid, sub], axis=0)   # (3, ngrdcol, nzt)


# ---------------------------------------------------------------------------
# xm_wpxp_lhs_jax
# ---------------------------------------------------------------------------

def xm_wpxp_lhs_jax(
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
# xm_wpxp_rhs_jax
# ---------------------------------------------------------------------------

def xm_wpxp_rhs_jax(
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
# advance_xm_wpxp_jax
# ---------------------------------------------------------------------------

def advance_xm_wpxp_jax(
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
    lhs_pr1 = wpxp_term_pr1_lhs_jax(C6_Skw_fnc, invrs_tau_C6_zm)

    # Buoyancy-production + pressure-3 RHS
    rhs_bp_pr3 = wpxp_terms_bp_pr3_rhs_jax(C7_Skw_fnc, thv_ds_zm, xpthvp)

    # rhs_ta = 0 for ADG1
    rhs_ta = jnp.zeros_like(wpxp)

    # Assemble LHS and RHS
    lhs = xm_wpxp_lhs_jax(
        lhs_diff_zm, lhs_ma_zm, lhs_ma_zt,
        lhs_ta_wpxp, lhs_ta_xm, lhs_tp,
        lhs_ac_pr2, lhs_pr1, dt,
    )
    rhs = xm_wpxp_rhs_jax(
        wpxp, xm, wpxp_forcing, xm_forcing,
        rhs_bp_pr3, rhs_ta, lhs_ta_wpxp, lhs_pr1, dt, k_lb_zm,
    )

    # Solve the pentadiagonal system
    soln = penta_lu_solve(lhs, rhs)   # (ngrdcol, 2*nzm-1)

    # Extract wpxp (even slots) and xm (odd slots)
    wpxp_new = soln[:, 0::2]   # (ngrdcol, nzm)
    xm_new   = soln[:, 1::2]   # (ngrdcol, nzt)

    # Apply covariance clipping if wp2/xp2 provided
    if wp2 is not None and xp2_relaxed is not None:
        wpxp_new = clip_covar_jax(wpxp_new, wp2, xp2_relaxed)

    return wpxp_new, xm_new


# ---------------------------------------------------------------------------
# compute_shared_xm_wpxp_lhs_terms
# ---------------------------------------------------------------------------

def compute_shared_xm_wpxp_lhs_terms(
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
    sigma_sqd_w: jnp.ndarray,
    wp3_on_wp2_zt: jnp.ndarray,
    gr,
) -> dict:
    """Compute all shared LHS terms for the xm/w'x' system.

    Returns a dict with keys: lhs_diff_zm, lhs_ma_zm, lhs_ma_zt, lhs_ta_xm,
    lhs_tp, lhs_ac_pr2, lhs_ta_wprtp.

    lhs_ta_wpthlp = lhs_ta_wprtp for ADG1 (and upwp/vpwp if l_predict_upwp_vpwp).

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
        sigma_sqd_w:     (ngrdcol, nzm)
        wp3_on_wp2_zt:   (ngrdcol, nzt)
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
    lhs_ta_xm = xm_term_ta_lhs_jax(invrs_rho_ds_zt, rho_ds_zm, gr)

    # Turbulent production LHS for w'x'
    lhs_tp = wpxp_term_tp_lhs_jax(wp2, gr)

    # Accumulation + pressure-2 LHS for w'x'
    lhs_ac_pr2 = wpxp_terms_ac_pr2_lhs_jax(C7_Skw_fnc, wm_zt, gr)

    # ADG1 TA coefficient for w'x'
    a1_coef = 1.0 / (1.0 - sigma_sqd_w)                         # (ngrdcol, nzm)
    a1_coef_zt = zm2zt_jax(a1_coef, gr)                          # (ngrdcol, nzt)
    coef_wp2rtp = a1_coef_zt * wp3_on_wp2_zt                     # (ngrdcol, nzt)
    lhs_ta_wprtp = xpyp_term_ta_pdf_lhs_jax(
        coef_wp2rtp, rho_ds_zt, invrs_rho_ds_zm, gr
    )   # (3, ngrdcol, nzm)  — lhs_ta_wpthlp = lhs_ta_wprtp for ADG1

    return dict(
        lhs_diff_zm=lhs_diff_zm,
        lhs_ma_zm=lhs_ma_zm,
        lhs_ma_zt=lhs_ma_zt,
        lhs_ta_xm=lhs_ta_xm,
        lhs_tp=lhs_tp,
        lhs_ac_pr2=lhs_ac_pr2,
        lhs_ta_wprtp=lhs_ta_wprtp,
    )


# ---------------------------------------------------------------------------
# clip_covar_jax
# ---------------------------------------------------------------------------

def clip_covar_jax(
    wpxp: jnp.ndarray,
    wp2: jnp.ndarray,
    xp2: jnp.ndarray,
    max_mag_corr: float = 0.99,
) -> jnp.ndarray:
    """Covariance clipping applied after the pentadiagonal solve.

    Faithful port of clip_explicit.F90:clip_covar (for wprtp/wpthlp case).
    Clips wpxp at interior levels k=1..nzm-2 (Python 0-based).

    xpyp_bound = max_mag_corr * sqrt(wp2 * xp2)
    wpxp_clipped = clip(wpxp, -xpyp_bound, xpyp_bound)

    Boundaries (k=0, k=nzm-1) are left unchanged.

    Args:
        wpxp: (ngrdcol, nzm)
        wp2:  (ngrdcol, nzm)
        xp2:  (ngrdcol, nzm)
        max_mag_corr: max magnitude of correlation (0.99 for wprtp/wpthlp)
    """
    xpyp_bound = max_mag_corr * jnp.sqrt(wp2 * xp2)   # (ngrdcol, nzm)
    wpxp_clipped = jnp.clip(wpxp, -xpyp_bound, xpyp_bound)
    # Preserve boundaries unchanged
    wpxp_clipped = wpxp_clipped.at[:, 0].set(wpxp[:, 0])
    wpxp_clipped = wpxp_clipped.at[:, -1].set(wpxp[:, -1])
    return wpxp_clipped
