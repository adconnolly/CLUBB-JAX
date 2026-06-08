"""JAX port of turbulent_adv_pdf.F90 — turbulent-advection LHS/RHS for xp2/xpyp.

Mirrors clubb_release/src/CLUBB_core/turbulent_adv_pdf.F90, which holds the
turbulent-advection discretization of the <w'x'y'> term:
  xpyp_term_ta_pdf_lhs          — implicit (LHS) tridiagonal term (xpyp_term_ta_pdf_lhs_jax).
                                  As in the single Fortran subroutine, the
                                  l_upwind_xpyp_turbulent_adv flag selects the centered
                                  (.false., default) or upwind (.true.) scheme via an
                                  internal runtime branch.
  xpyp_term_ta_pdf_rhs          — explicit (RHS) flux divergence (xpyp_term_ta_pdf_rhs_jax;
                                  same l_upwind_xpyp_turbulent_adv internal branch).
  xpyp_term_ta_pdf_lhs_godunov  — Godunov-like upwind LHS (flux-split stencil)
                                    xpyp_term_ta_pdf_lhs_godunov
  xpyp_term_ta_pdf_rhs_godunov  — Godunov-like upwind RHS
                                    xpyp_term_ta_pdf_rhs_godunov

Output layout lhs[3, ngrdcol, nzm]: [0]=superdiagonal, [1]=main, [2]=subdiagonal.
Array layout: (ngrdcol, nz), ascending grid (index 0 = lowest level, grid_dir=+1).
Pure-jnp → differentiable.

References:
  src/CLUBB_core/turbulent_adv_pdf.F90, xpyp_term_ta_pdf_{lhs,rhs}[_godunov].
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit


def xpyp_term_ta_pdf_lhs_jax(
    coef_wpxpyp_implicit: jnp.ndarray,
    rho_ds_zt: jnp.ndarray,
    invrs_rho_ds_zm: jnp.ndarray,
    gr,
    l_upwind_xpyp_turbulent_adv: bool = False,
    rho_ds_zm: jnp.ndarray = None,
    sgn_turbulent_vel: jnp.ndarray = None,
    coef_wpxpyp_implicit_zm: jnp.ndarray = None,
    grid_dir: float = 1.0,
) -> jnp.ndarray:
    """Tridiagonal LHS for turbulent advection of xp2/xpyp.

    Faithful port of the single Fortran xpyp_term_ta_pdf_lhs subroutine. As in the
    Fortran, the `l_upwind_xpyp_turbulent_adv` flag selects the centered (.false.,
    default) or upwind (.true.) discretization via an internal branch.

    Centered branch discretizes, at zm levels using zm→zt interpolation weights:
      (1/rho_ds_zm) * d( rho_ds_zt * coef_wpxpyp_implicit * var_zm ) / dz
    Fortran index conventions (1-based → 0-based Python):
      k=1 → k=0 (lower boundary, zero); k=2..nzm-1 → k=1..nzm-2 (interior);
      k=nzm → k=nzm-1 (upper boundary, zero).

    Upwind branch (per interior k_py=1..nzm-2) uses a one-sided stencil keyed on
    grid_dir*sgn_turbulent_vel, with zm-level rho_ds_zm / coef_wpxpyp_implicit_zm:
      grid_dir*sgn > 0 (upward): super=0;  main=irho*idzt[k-1]*rho[k]*coef[k];
                                 sub=-irho*idzt[k-1]*rho[k-1]*coef[k-1]
      else (downward):           super=irho*idzt[k]*rho[k+1]*coef[k+1];
                                 main=-irho*idzt[k]*rho[k]*coef[k];  sub=0

    Args (centered): coef_wpxpyp_implicit (zt, ngrdcol×nzt), rho_ds_zt (zt),
        invrs_rho_ds_zm (zm), gr with .invrs_dzm and .weights_zm2zt.
    Args (upwind):   rho_ds_zm (zm), invrs_rho_ds_zm (zm), sgn_turbulent_vel (zm),
        coef_wpxpyp_implicit_zm (zm), gr with .invrs_dzt, grid_dir.

    Returns:
        lhs: shape (3, ngrdcol, nzm).  [0=super, 1=main, 2=sub]
    """
    if l_upwind_xpyp_turbulent_adv:
        # ===== Upwind discretization (Fortran l_upwind_xpyp_turbulent_adv branch) =====
        invrs_dzt = gr.invrs_dzt    # (ngrdcol, nzt = nzm-1)
        ngrdcol = rho_ds_zm.shape[0]

        # Interior slice indices: k_py=1..nzm-2
        irho = invrs_rho_ds_zm[:, 1:-1]        # (ngrdcol, nzm-2)
        sgn = sgn_turbulent_vel[:, 1:-1]

        rho_k   = rho_ds_zm[:, 1:-1]
        coef_k  = coef_wpxpyp_implicit_zm[:, 1:-1]
        rho_km1 = rho_ds_zm[:, :-2]
        coef_km1 = coef_wpxpyp_implicit_zm[:, :-2]
        rho_kp1 = rho_ds_zm[:, 2:]
        coef_kp1 = coef_wpxpyp_implicit_zm[:, 2:]

        idzt_km1 = invrs_dzt[:, :-1]   # invrs_dzt[k_py-1] for k_py=1..nzm-2
        idzt_k   = invrs_dzt[:, 1:]    # invrs_dzt[k_py]   for k_py=1..nzm-2

        # Upward wind branch
        zeros_int = jnp.zeros_like(rho_k)
        sup_up   = zeros_int
        main_up  = irho * idzt_km1 * rho_k * coef_k
        sub_up   = -irho * idzt_km1 * rho_km1 * coef_km1

        # Downward wind branch
        sup_dn  = irho * idzt_k * rho_kp1 * coef_kp1
        main_dn = -irho * idzt_k * rho_k * coef_k
        sub_dn  = zeros_int

        is_up = (grid_dir * sgn) > 0.0
        super_int = jnp.where(is_up, sup_up, sup_dn)
        main_int  = jnp.where(is_up, main_up, main_dn)
        sub_int   = jnp.where(is_up, sub_up, sub_dn)

        zeros_bnd = jnp.zeros((ngrdcol, 1), dtype=rho_ds_zm.dtype)
        superdiag = jnp.concatenate([zeros_bnd, super_int, zeros_bnd], axis=1)
        maindiag  = jnp.concatenate([zeros_bnd, main_int,  zeros_bnd], axis=1)
        subdiag   = jnp.concatenate([zeros_bnd, sub_int,   zeros_bnd], axis=1)
        return jnp.stack([superdiag, maindiag, subdiag], axis=0)

    # ===== Centered discretization (Fortran .not. l_upwind_xpyp_turbulent_adv branch) =====
    invrs_dzm = gr.invrs_dzm            # (ngrdcol, nzm)
    w2zt = gr.weights_zm2zt             # (ngrdcol, nzt=nzm-1, 2)
    ngrdcol = coef_wpxpyp_implicit.shape[0]

    # Interior k_py=1..nzm-2 (Fortran k=2..nzm-1)
    # coef[k] / rho_ds_zt[k] (Fortran zt index k=2..nzm-1) → Python [:,1:]
    # coef[k-1] / rho_ds_zt[k-1]                            → Python [:,:-1]
    fac = invrs_rho_ds_zm[:, 1:-1] * invrs_dzm[:, 1:-1]   # (ngrdcol, nzm-2)

    rho_coef_k   = rho_ds_zt[:, 1:] * coef_wpxpyp_implicit[:, 1:]    # k
    rho_coef_km1 = rho_ds_zt[:, :-1] * coef_wpxpyp_implicit[:, :-1]  # k-1

    super_int = fac * rho_coef_k   * w2zt[:, 1:, 0]   # m_above=0
    main_int  = fac * (rho_coef_k  * w2zt[:, 1:, 1]   # m_below=1
                     - rho_coef_km1 * w2zt[:, :-1, 0])  # m_above=0
    sub_int   = -fac * rho_coef_km1 * w2zt[:, :-1, 1]  # m_below=1

    zeros_bnd = jnp.zeros((ngrdcol, 1), dtype=coef_wpxpyp_implicit.dtype)
    superdiag = jnp.concatenate([zeros_bnd, super_int, zeros_bnd], axis=1)
    maindiag  = jnp.concatenate([zeros_bnd, main_int,  zeros_bnd], axis=1)
    subdiag   = jnp.concatenate([zeros_bnd, sub_int,   zeros_bnd], axis=1)

    return jnp.stack([superdiag, maindiag, subdiag], axis=0)  # (3, ngrdcol, nzm)


# JIT-compiled production version
xpyp_term_ta_pdf_lhs = jit(xpyp_term_ta_pdf_lhs_jax)


def xpyp_term_ta_pdf_rhs_jax(
    term_wpxpyp_explicit: jnp.ndarray,
    rho_ds_zt: jnp.ndarray,
    invrs_rho_ds_zm: jnp.ndarray,
    gr,
    l_upwind_xpyp_turbulent_adv: bool = False,
    rho_ds_zm: jnp.ndarray = None,
    sgn_turbulent_vel: jnp.ndarray = None,
    term_wpxpyp_explicit_zm: jnp.ndarray = None,
    grid_dir: float = 1.0,
) -> jnp.ndarray:
    """Explicit RHS turbulent-advection flux divergence for xp2/xpyp.

    Faithful port of the single Fortran xpyp_term_ta_pdf_rhs subroutine. As in the
    Fortran, the `l_upwind_xpyp_turbulent_adv` flag selects the centered (.false.,
    default) or upwind (.true.) path via an internal branch.

    Centered branch, interior k=1..nzm-2 (Python 0-indexed); boundaries = 0:
      rhs[k] = -(1/rho_ds_zm[k]) * invrs_dzm[k] *
                (rho_ds_zt[k] * term[k] - rho_ds_zt[k-1] * term[k-1])

    Upwind branch (per interior k_py=1..nzm-2), keyed on grid_dir*sgn_turbulent_vel,
    using zm-level rho_ds_zm / term_wpxpyp_explicit_zm:
      grid_dir*sgn > 0: rhs = -irho*idzt[k-1]*(rho[k]*term[k] - rho[k-1]*term[k-1])
      else:             rhs = -irho*idzt[k]  *(rho[k+1]*term[k+1] - rho[k]*term[k])

    Args (centered): term_wpxpyp_explicit (zt), rho_ds_zt (zt), invrs_rho_ds_zm (zm),
        gr with .invrs_dzm.
    Args (upwind):   rho_ds_zm (zm), invrs_rho_ds_zm (zm), sgn_turbulent_vel (zm),
        term_wpxpyp_explicit_zm (zm), gr with .invrs_dzt, grid_dir.

    Returns:
        rhs: shape (ngrdcol, nzm).  Boundaries (k=0 and k=nzm-1) are zero.
    """
    if l_upwind_xpyp_turbulent_adv:
        # ===== Upwind discretization (Fortran l_upwind_xpyp_turbulent_adv branch) =====
        invrs_dzt = gr.invrs_dzt    # (ngrdcol, nzt = nzm-1)
        ngrdcol = rho_ds_zm.shape[0]

        irho   = invrs_rho_ds_zm[:, 1:-1]
        sgn    = sgn_turbulent_vel[:, 1:-1]

        rho_k   = rho_ds_zm[:, 1:-1]
        term_k  = term_wpxpyp_explicit_zm[:, 1:-1]
        rho_km1 = rho_ds_zm[:, :-2]
        term_km1 = term_wpxpyp_explicit_zm[:, :-2]
        rho_kp1 = rho_ds_zm[:, 2:]
        term_kp1 = term_wpxpyp_explicit_zm[:, 2:]

        idzt_km1 = invrs_dzt[:, :-1]
        idzt_k   = invrs_dzt[:, 1:]

        rhs_up = -irho * idzt_km1 * (rho_k * term_k - rho_km1 * term_km1)
        rhs_dn = -irho * idzt_k   * (rho_kp1 * term_kp1 - rho_k * term_k)

        is_up   = (grid_dir * sgn) > 0.0
        rhs_int = jnp.where(is_up, rhs_up, rhs_dn)

        zeros_bnd = jnp.zeros((ngrdcol, 1), dtype=rho_ds_zm.dtype)
        return jnp.concatenate([zeros_bnd, rhs_int, zeros_bnd], axis=1)

    # ===== Centered discretization (Fortran .not. l_upwind_xpyp_turbulent_adv branch) =====
    invrs_dzm = gr.invrs_dzm  # (ngrdcol, nzm)
    # Interior k=1..nzm-2; zt index k maps to Python [:,1:] and [:,:-1]
    interior = (
        -invrs_rho_ds_zm[:, 1:-1] * invrs_dzm[:, 1:-1] * (
            rho_ds_zt[:, 1:] * term_wpxpyp_explicit[:, 1:]
            - rho_ds_zt[:, :-1] * term_wpxpyp_explicit[:, :-1]
        )
    )
    ngrdcol = invrs_rho_ds_zm.shape[0]
    zeros_bnd = jnp.zeros((ngrdcol, 1), dtype=invrs_rho_ds_zm.dtype)
    return jnp.concatenate([zeros_bnd, interior, zeros_bnd], axis=1)


# JIT-compiled production version
xpyp_term_ta_pdf_rhs = jit(xpyp_term_ta_pdf_rhs_jax)


def xpyp_term_ta_pdf_lhs_godunov(
    coef_wpxpyp_implicit: jnp.ndarray,
    invrs_rho_ds_zm: jnp.ndarray,
    rho_ds_zm: jnp.ndarray,
    gr,
    grid_dir: float = 1.0,
) -> jnp.ndarray:
    """Godunov-like upwind LHS for turbulent advection of xp2/xpyp.

    Faithful port of Fortran xpyp_term_ta_pdf_lhs_godunov (turbulent_adv_pdf.F90).
    Unlike the centered/upwind variants this uses 1/dzm directly (gr%invrs_dzm) and a
    flux-split upwind stencil keyed on the *sign of the implicit coef* itself.

    For each interior k_py=1..nzm-2 (Fortran k=2..nzm-1), with gd = grid_dir:
      super[k_py] = irho[k] * idzm[k] * rho[k+1] * gd * min(0, gd*coef_zt[k])
      main[k_py]  = irho[k] * idzm[k] * rho[k]   * gd * ( max(0, gd*coef_zt[k])
                                                          - min(0, gd*coef_zt[k-1]) )
      sub[k_py]   = -irho[k] * idzm[k] * rho[k-1] * gd * max(0, gd*coef_zt[k-1])
    where coef_zt[k]→coef[:,1:], coef_zt[k-1]→coef[:,:-1] (coef on zt levels).
    Boundaries (k=0, k=nzm-1) are zero.

    Args:
        coef_wpxpyp_implicit: Coef. of <x'y'> in <w'x'y'> on zt levels, (ngrdcol, nzt).
        invrs_rho_ds_zm:      1/rho_ds on zm levels, (ngrdcol, nzm).
        rho_ds_zm:            Dry, static density on zm levels, (ngrdcol, nzm).
        gr:                   Grid with .invrs_dzm (ngrdcol, nzm).
        grid_dir:             +1 for ascending grid (default).

    Returns:
        lhs: shape (3, ngrdcol, nzm).  [0=super(kp1), 1=main(k), 2=sub(km1)].
    """
    gd = grid_dir
    invrs_dzm = gr.invrs_dzm                 # (ngrdcol, nzm)
    ngrdcol = rho_ds_zm.shape[0]

    irho   = invrs_rho_ds_zm[:, 1:-1]        # (ngrdcol, nzm-2)
    idzm   = invrs_dzm[:, 1:-1]
    rho_k   = rho_ds_zm[:, 1:-1]
    rho_kp1 = rho_ds_zm[:, 2:]
    rho_km1 = rho_ds_zm[:, :-2]
    coef_k   = coef_wpxpyp_implicit[:, 1:]   # Fortran coef(i,k),   k=2..nzm-1
    coef_km1 = coef_wpxpyp_implicit[:, :-1]  # Fortran coef(i,k-1), k-1=1..nzm-2

    zero = jnp.zeros_like(rho_k)
    super_int = irho * idzm * rho_kp1 * gd * jnp.minimum(zero, gd * coef_k)
    main_int  = irho * idzm * rho_k * gd * (
        jnp.maximum(zero, gd * coef_k) - jnp.minimum(zero, gd * coef_km1))
    sub_int   = -irho * idzm * rho_km1 * gd * jnp.maximum(zero, gd * coef_km1)

    zeros_bnd = jnp.zeros((ngrdcol, 1), dtype=rho_ds_zm.dtype)
    superdiag = jnp.concatenate([zeros_bnd, super_int, zeros_bnd], axis=1)
    maindiag  = jnp.concatenate([zeros_bnd, main_int,  zeros_bnd], axis=1)
    subdiag   = jnp.concatenate([zeros_bnd, sub_int,   zeros_bnd], axis=1)
    return jnp.stack([superdiag, maindiag, subdiag], axis=0)


def xpyp_term_ta_pdf_rhs_godunov(
    term_wpxpyp_explicit_zm: jnp.ndarray,
    invrs_rho_ds_zm: jnp.ndarray,
    sgn_turbulent_vel: jnp.ndarray,
    rho_ds_zm: jnp.ndarray,
    gr,
    grid_dir: float = 1.0,
) -> jnp.ndarray:
    """Godunov-like upwind RHS for turbulent advection of xp2/xpyp.

    Faithful port of Fortran xpyp_term_ta_pdf_rhs_godunov (turbulent_adv_pdf.F90).
    The explicit term lives on zm levels here (unlike the centered RHS, which is on zt).

    For each interior k_py=1..nzm-2 (Fortran k=2..nzm-1), with gd = grid_dir:
      rhs[k] = -irho[k] * idzm[k] * gd * (
                 min(0, gd*sgn[k])   * rho[k+1]*term[k+1]
               + max(0, gd*sgn[k])   * rho[k]  *term[k]
               - min(0, gd*sgn[k-1]) * rho[k]  *term[k]
               - max(0, gd*sgn[k-1]) * rho[k-1]*term[k-1] )
    where sgn[k]→sgn[:,1:], sgn[k-1]→sgn[:,:-1] (sgn on zt levels).
    Boundaries (k=0, k=nzm-1) are zero.

    Args:
        term_wpxpyp_explicit_zm: Explicit <w'x'y'> term on zm levels, (ngrdcol, nzm).
        invrs_rho_ds_zm:         1/rho_ds on zm levels, (ngrdcol, nzm).
        sgn_turbulent_vel:       Sign of turbulent velocity on zt levels, (ngrdcol, nzt).
        rho_ds_zm:               Dry, static density on zm levels, (ngrdcol, nzm).
        gr:                      Grid with .invrs_dzm (ngrdcol, nzm).
        grid_dir:                +1 for ascending grid (default).

    Returns:
        rhs: shape (ngrdcol, nzm).  Boundaries (k=0, k=nzm-1) are zero.
    """
    gd = grid_dir
    invrs_dzm = gr.invrs_dzm
    ngrdcol = rho_ds_zm.shape[0]

    irho   = invrs_rho_ds_zm[:, 1:-1]
    idzm   = invrs_dzm[:, 1:-1]
    rho_k   = rho_ds_zm[:, 1:-1]
    rho_kp1 = rho_ds_zm[:, 2:]
    rho_km1 = rho_ds_zm[:, :-2]
    term_k   = term_wpxpyp_explicit_zm[:, 1:-1]
    term_kp1 = term_wpxpyp_explicit_zm[:, 2:]
    term_km1 = term_wpxpyp_explicit_zm[:, :-2]
    sgn_k   = sgn_turbulent_vel[:, 1:]       # Fortran sgn(i,k),   k=2..nzm-1
    sgn_km1 = sgn_turbulent_vel[:, :-1]      # Fortran sgn(i,k-1), k-1=1..nzm-2

    zero = jnp.zeros_like(rho_k)
    rhs_int = -irho * idzm * gd * (
        jnp.minimum(zero, gd * sgn_k)   * rho_kp1 * term_kp1
        + jnp.maximum(zero, gd * sgn_k)   * rho_k * term_k
        - jnp.minimum(zero, gd * sgn_km1) * rho_k * term_k
        - jnp.maximum(zero, gd * sgn_km1) * rho_km1 * term_km1)

    zeros_bnd = jnp.zeros((ngrdcol, 1), dtype=rho_ds_zm.dtype)
    return jnp.concatenate([zeros_bnd, rhs_int, zeros_bnd], axis=1)


__all__ = [
    "xpyp_term_ta_pdf_lhs_jax",
    "xpyp_term_ta_pdf_lhs",
    "xpyp_term_ta_pdf_rhs_jax",
    "xpyp_term_ta_pdf_rhs",
    "xpyp_term_ta_pdf_lhs_godunov",
    "xpyp_term_ta_pdf_rhs_godunov",
]
