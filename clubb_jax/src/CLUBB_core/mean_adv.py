"""JAX port of mean_adv.F90 — mean-advection LHS operators.

Mirrors clubb_release/src/CLUBB_core/mean_adv.F90:
  term_ma_zt_lhs   — tridiagonal mean-advection LHS for a zt-level variable.
                     As in the single Fortran subroutine, the `l_upwind_xm_ma`
                     flag selects the centered scheme (False) or the upwind
                     scheme (True, default) via an internal runtime branch.
  term_ma_zm_lhs   — tridiagonal mean-advection LHS for a zm-level variable

Output layout lhs[3, ngrdcol, nz]:
  lhs[0] = superdiagonal  (coefficient of var[k+1])
  lhs[1] = main diagonal  (coefficient of var[k])
  lhs[2] = subdiagonal    (coefficient of var[k-1])

Array layout: (ngrdcol, nz), ascending grid (index 0 = lowest level, grid_dir=+1).
Pure-jnp → differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit


# ---------------------------------------------------------------------------
# term_ma_zt_lhs  (zt-level variable) — centered + upwind branches via l_upwind_xm_ma
# ---------------------------------------------------------------------------

def term_ma_zt_lhs_jax(
    wm_zt: jnp.ndarray,
    gr,
    l_upwind_xm_ma: bool = True,
) -> jnp.ndarray:
    """Mean-advection LHS for a zt-level variable.

    Faithful port of the single mean_adv.F90:term_ma_zt_lhs subroutine for an
    ascending grid (grid_dir=+1). As in the Fortran, the `l_upwind_xm_ma` flag
    selects the scheme at runtime via an internal branch:

    Centered scheme (l_upwind_xm_ma=False), interior k=1..nzt-2 (Fortran k=2..nzt-1):
      super[k] = wm_zt[k] * invrs_dzt[k] * weights_zt2zm[k+1, 0]
      main[k]  = wm_zt[k] * invrs_dzt[k] * (weights_zt2zm[k+1, 1] - weights_zt2zm[k, 0])
      sub[k]   = -wm_zt[k] * invrs_dzt[k] * weights_zt2zm[k, 1]

    Upwind scheme (l_upwind_xm_ma=True):
      wm_zt >= 0 (upward wind):   super = 0,  main = +wm_zt*invrs_dzm[k],  sub = -wm_zt*invrs_dzm[k]
      wm_zt < 0  (downward wind): super = +wm_zt*invrs_dzm[k+1],  main = -wm_zt*invrs_dzm[k+1],  sub = 0

    Output shape: (3, ngrdcol, nzt).
      out[0] = superdiagonal (coeff of xm[k+1])
      out[1] = main diagonal (coeff of xm[k])
      out[2] = subdiagonal   (coeff of xm[k-1])

    Args:
        wm_zt:          (ngrdcol, nzt)
        gr:             grid object with invrs_dzm (ngrdcol, nzm), invrs_dzt, weights_zt2zm
        l_upwind_xm_ma: True → upwind branch (default), False → centered branch
    """
    ngrdcol, nzt = wm_zt.shape

    if not l_upwind_xm_ma:
        # ===== Centered-differencing branch (Fortran `.not. l_upwind_xm_ma` block) =====
        invrs_dzt = gr.invrs_dzt       # (ngrdcol, nzt)
        w = gr.weights_zt2zm           # (ngrdcol, nzm, 2); index 0=t_above, 1=t_below

        # Interior k=1..nzt-2 (Python)
        fac = wm_zt[:, 1:-1] * invrs_dzt[:, 1:-1]      # (ngrdcol, nzt-2)
        super_int = fac * w[:, 2:-1, 0]                  # w[k_py+1, t_above]
        main_int  = fac * (w[:, 2:-1, 1] - w[:, 1:-2, 0])
        sub_int   = -fac * w[:, 1:-2, 1]

        # Lower boundary k=0
        fac0 = wm_zt[:, 0] * invrs_dzt[:, 0]            # (ngrdcol,)
        sup_bot = fac0 * w[:, 1, 0]
        mid_bot = -fac0 * (1.0 - w[:, 1, 1])
        sub_bot = jnp.zeros((ngrdcol,), dtype=wm_zt.dtype)

        # Upper boundary k=nzt-1
        fac_top = wm_zt[:, -1] * invrs_dzt[:, -1]       # (ngrdcol,)
        sup_top = jnp.zeros((ngrdcol,), dtype=wm_zt.dtype)
        mid_top = fac_top * (1.0 - w[:, -2, 0])
        sub_top = -fac_top * w[:, -2, 1]

        sup = jnp.concatenate([sup_bot[:, None], super_int, sup_top[:, None]], axis=1)
        mid = jnp.concatenate([mid_bot[:, None], main_int,  mid_top[:, None]], axis=1)
        sub = jnp.concatenate([sub_bot[:, None], sub_int,   sub_top[:, None]], axis=1)

        return jnp.stack([sup, mid, sub], axis=0)   # (3, ngrdcol, nzt)

    # ===== Upwind-differencing branch (Fortran `else` block) =====
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
# term_ma_zm_lhs  (zm-level variable)
# ---------------------------------------------------------------------------

def term_ma_zm_lhs_jax(
    wm_zm: jnp.ndarray,
    gr,
) -> jnp.ndarray:
    """Tridiagonal LHS for implicit mean advection of a zm-level variable.

    Faithful port of Fortran term_ma_zm_lhs (mean_adv.F90).

    Discretizes w * d(var_zm)/dz implicitly at zm levels.  Boundary rows
    (k=0 and k=nzm-1) are set to zero (fixed-value BCs applied by caller).

    Fortran index conventions (1-based → 0-based Python):
      k=1       → k=0   (lower boundary, zero)
      k=2..nzm-1→ k=1..nzm-2 (interior)
      k=nzm     → k=nzm-1   (upper boundary, zero)

    Args:
        wm_zm:  Mean vertical velocity on momentum levels (ngrdcol, nzm).
        gr:     Grid object with:
                  .invrs_dzm  (ngrdcol, nzm): 1 / (zt[k] - zt[k-1])
                  .weights_zm2zt (ngrdcol, nzt=nzm-1, 2):
                    [:,k,0] = M_ABOVE weight for zm[k] contributing to zt[k]
                    [:,k,1] = M_BELOW weight for zm[k+1] contributing to zt[k]

    Returns:
        lhs: shape (3, ngrdcol, nzm).
    """
    invrs_dzm = gr.invrs_dzm            # (ngrdcol, nzm)
    w2zt = gr.weights_zm2zt             # (ngrdcol, nzt=nzm-1, 2)

    ngrdcol = wm_zm.shape[0]

    # Interior k_p=1..nzm-2 (Fortran k=2..nzm-1)
    fac = wm_zm[:, 1:-1] * invrs_dzm[:, 1:-1]  # (ngrdcol, nzm-2)

    # For momentum level k_p, the zt level "above" is zt[k_p] and "below" is zt[k_p-1].
    # weights_zm2zt[:, k_p, 0] = M_ABOVE: weight for zm[k_p] at zt[k_p]
    # weights_zm2zt[:, k_p, 1] = M_BELOW: weight for zm[k_p+1] at zt[k_p]
    # Fortran lhs_ma(kp1_mdiag) uses weights_zm2zt[i, k, m_above]   → w2zt[:, 1:, 0]
    # Fortran lhs_ma(km1_mdiag) uses weights_zm2zt[i, k-1, m_below] → w2zt[:, :-1, 1]
    super_int = fac * w2zt[:, 1:, 0]
    main_int  = fac * (w2zt[:, 1:, 1] - w2zt[:, :-1, 0])
    sub_int   = -fac * w2zt[:, :-1, 1]

    zeros_bnd = jnp.zeros((ngrdcol, 1), dtype=wm_zm.dtype)

    superdiag = jnp.concatenate([zeros_bnd, super_int, zeros_bnd], axis=1)
    maindiag  = jnp.concatenate([zeros_bnd, main_int,  zeros_bnd], axis=1)
    subdiag   = jnp.concatenate([zeros_bnd, sub_int,   zeros_bnd], axis=1)

    return jnp.stack([superdiag, maindiag, subdiag], axis=0)   # (3, ngrdcol, nzm)


# JIT-compiled production versions
term_ma_zt_lhs = jit(term_ma_zt_lhs_jax)
term_ma_zm_lhs = jit(term_ma_zm_lhs_jax)


__all__ = [
    "term_ma_zt_lhs_jax",
    "term_ma_zm_lhs_jax",
    "term_ma_zt_lhs",
    "term_ma_zm_lhs",
]
