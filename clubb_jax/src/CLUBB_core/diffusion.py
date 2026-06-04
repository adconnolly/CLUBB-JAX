"""Pure JAX implementations of eddy diffusion and mean-advection LHS operators.

Faithful ports of CLUBB Fortran sources:
  - diffusion_zt_lhs_jax:          tridiagonal diffusion LHS for zt-level variables
  - diffusion_zm_lhs_jax:          tridiagonal diffusion LHS for zm-level variables
  - term_ma_zm_lhs_jax:            tridiagonal mean-advection LHS for zm-level variables
  - xpyp_term_ta_pdf_lhs_jax:      turbulent-advection LHS for xp2/xpyp equations (centered)
  - xpyp_term_ta_pdf_rhs_jax:      turbulent-advection RHS flux divergence (centered)

Output layout lhs[3, ngrdcol, nz]:
  lhs[0] = superdiagonal  (coefficient of var[k+1])
  lhs[1] = main diagonal  (coefficient of var[k])
  lhs[2] = subdiagonal    (coefficient of var[k-1])

Array layout: (ngrdcol, nz), ascending grid (index 0 = lowest level).

References:
  src/CLUBB_core/diffusion.F90, diffusion_zt_lhs, diffusion_zm_lhs.
  src/CLUBB_core/mean_adv.F90, term_ma_zm_lhs.
  src/CLUBB_core/turbulent_adv_pdf.F90, xpyp_term_ta_pdf_lhs.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit


def diffusion_zt_lhs_jax(
    K_zm: jnp.ndarray,
    nu: jnp.ndarray,
    invrs_rho_ds_zt: jnp.ndarray,
    rho_ds_zm: jnp.ndarray,
    gr,
) -> jnp.ndarray:
    """Tridiagonal LHS for implicit eddy diffusion of a zt-level variable.

    Faithful port of Fortran diffusion_zt_lhs (non-upwind path).

    Discretizes: d/dz[ (K_zm + nu) * d(var_zt)/dz ] evaluated at zt levels,
    using zero-flux boundary conditions.

    Fortran index conventions (1-based → 0-based Python):
      k=1       → k=0   (lower boundary)
      k=2..nzt-1→ k=1..nzt-2 (interior)
      k=nzt     → k=nzt-1   (upper boundary)

    Args:
        K_zm: Eddy diffusivity on momentum levels, shape (ngrdcol, nzm).
        nu:   Background diffusivity, shape (ngrdcol,).
        invrs_rho_ds_zt: 1/rho_ds on thermo levels, shape (ngrdcol, nzt).
        rho_ds_zm: rho_ds on momentum levels, shape (ngrdcol, nzm).
        gr:   Grid with .invrs_dzt (ngrdcol, nzt) and .invrs_dzm (ngrdcol, nzm).

    Returns:
        lhs: shape (3, ngrdcol, nzt).
             lhs[0]=superdiag, lhs[1]=maindiag, lhs[2]=subdiag.
    """
    K_zm_nu = K_zm + nu[:, None]          # (ngrdcol, nzm)
    invrs_dzt = gr.invrs_dzt              # (ngrdcol, nzt)
    invrs_dzm = gr.invrs_dzm              # (ngrdcol, nzm)

    # Lower boundary k=0  [Fortran k=1]
    # super = -invrs_dzt(1)*invrs_rho(1) * K_zm_nu(2)*rho(2)*invrs_dzm(2)
    # main  = +same;  sub = 0
    common_bot = (
        invrs_dzt[:, :1] * invrs_rho_ds_zt[:, :1]
        * K_zm_nu[:, 1:2] * rho_ds_zm[:, 1:2] * invrs_dzm[:, 1:2]
    )
    super_bot = -common_bot
    main_bot  =  common_bot
    sub_bot   = jnp.zeros_like(common_bot)

    # Interior k=1..nzt-2  [Fortran k=2..nzt-1]
    # super = -invrs_dzt(k)*invrs_rho(k) * K_zm_nu(k+1)*rho(k+1)*invrs_dzm(k+1)
    # sub   = -invrs_dzt(k)*invrs_rho(k) * K_zm_nu(k)  *rho(k)  *invrs_dzm(k)
    # main  = -(super + sub)   [conservation identity]
    scale_int = invrs_dzt[:, 1:-1] * invrs_rho_ds_zt[:, 1:-1]   # (ngrdcol, nzt-2)
    super_int = -scale_int * K_zm_nu[:, 2:-1] * rho_ds_zm[:, 2:-1] * invrs_dzm[:, 2:-1]
    sub_int   = -scale_int * K_zm_nu[:, 1:-2] * rho_ds_zm[:, 1:-2] * invrs_dzm[:, 1:-2]
    main_int  = -(super_int + sub_int)

    # Upper boundary k=nzt-1  [Fortran k=nzt]
    # super = 0;  sub = -invrs_dzt(nzt)*invrs_rho(nzt)*K_zm_nu(nzm-1)*rho(nzm-1)*invrs_dzm(nzm-1)
    # main  = -sub
    common_top = (
        invrs_dzt[:, -1:] * invrs_rho_ds_zt[:, -1:]
        * K_zm_nu[:, -2:-1] * rho_ds_zm[:, -2:-1] * invrs_dzm[:, -2:-1]
    )
    super_top = jnp.zeros_like(common_top)
    sub_top   = -common_top
    main_top  =  common_top

    superdiag = jnp.concatenate([super_bot, super_int, super_top], axis=1)
    maindiag  = jnp.concatenate([main_bot,  main_int,  main_top],  axis=1)
    subdiag   = jnp.concatenate([sub_bot,   sub_int,   sub_top],   axis=1)

    return jnp.stack([superdiag, maindiag, subdiag], axis=0)   # (3, ngrdcol, nzt)


def diffusion_zm_lhs_jax(
    K_zt: jnp.ndarray,
    nu: jnp.ndarray,
    invrs_rho_ds_zm: jnp.ndarray,
    rho_ds_zt: jnp.ndarray,
    gr,
) -> jnp.ndarray:
    """Tridiagonal LHS for implicit eddy diffusion of a zm-level variable.

    Faithful port of Fortran diffusion_zm_lhs (non-upwind path).

    Discretizes: d/dz[ (K_zt + nu) * d(var_zm)/dz ] evaluated at zm levels,
    using zero-flux boundary conditions.

    Note: The k=0 (lower boundary) row is computed but is NOT fed into the
    solver by the parent subroutine (per Fortran comment), so its exact value
    does not affect model results.

    Fortran index conventions (1-based → 0-based Python):
      k=1       → k=0   (lower boundary, unused in solver)
      k=2..nzm-1→ k=1..nzm-2 (interior)
      k=nzm     → k=nzm-1   (upper boundary)

    Args:
        K_zt: Eddy diffusivity on thermo levels, shape (ngrdcol, nzt).
        nu:   Background diffusivity, shape (ngrdcol,).
        invrs_rho_ds_zm: 1/rho_ds on momentum levels, shape (ngrdcol, nzm).
        rho_ds_zt: rho_ds on thermo levels, shape (ngrdcol, nzt).
        gr:   Grid with .invrs_dzm (ngrdcol, nzm) and .invrs_dzt (ngrdcol, nzt).

    Returns:
        lhs: shape (3, ngrdcol, nzm).
             lhs[0]=superdiag, lhs[1]=maindiag, lhs[2]=subdiag.
    """
    K_zt_nu = K_zt + nu[:, None]          # (ngrdcol, nzt)
    invrs_dzm = gr.invrs_dzm              # (ngrdcol, nzm)
    invrs_dzt = gr.invrs_dzt              # (ngrdcol, nzt)

    # Lower boundary k=0  [Fortran k=1; not used in final solver]
    # super = -invrs_dzm(1)*invrs_rho(1)*K_zt_nu(1)*rho_zt(1)*invrs_dzt(1)
    # main  = +same;  sub = 0
    common_bot = (
        invrs_dzm[:, :1] * invrs_rho_ds_zm[:, :1]
        * K_zt_nu[:, :1] * rho_ds_zt[:, :1] * invrs_dzt[:, :1]
    )
    super_bot = -common_bot
    main_bot  =  common_bot
    sub_bot   = jnp.zeros_like(common_bot)

    # Interior k=1..nzm-2  [Fortran k=2..nzm-1]
    # super = -invrs_dzm(k)*invrs_rho(k) * K_zt_nu(k)  *rho_zt(k)  *invrs_dzt(k)
    # sub   = -invrs_dzm(k)*invrs_rho(k) * K_zt_nu(k-1)*rho_zt(k-1)*invrs_dzt(k-1)
    # main  = -(super + sub)
    scale_int = invrs_dzm[:, 1:-1] * invrs_rho_ds_zm[:, 1:-1]   # (ngrdcol, nzm-2)
    super_int = -scale_int * K_zt_nu[:, 1:] * rho_ds_zt[:, 1:] * invrs_dzt[:, 1:]
    sub_int   = -scale_int * K_zt_nu[:, :-1] * rho_ds_zt[:, :-1] * invrs_dzt[:, :-1]
    main_int  = -(super_int + sub_int)

    # Upper boundary k=nzm-1  [Fortran k=nzm]
    # super = 0;  sub = -invrs_dzm(nzm)*invrs_rho(nzm)*K_zt_nu(nzt)*rho_zt(nzt)*invrs_dzt(nzt)
    # main  = -sub
    common_top = (
        invrs_dzm[:, -1:] * invrs_rho_ds_zm[:, -1:]
        * K_zt_nu[:, -1:] * rho_ds_zt[:, -1:] * invrs_dzt[:, -1:]
    )
    super_top = jnp.zeros_like(common_top)
    sub_top   = -common_top
    main_top  =  common_top

    superdiag = jnp.concatenate([super_bot, super_int, super_top], axis=1)
    maindiag  = jnp.concatenate([main_bot,  main_int,  main_top],  axis=1)
    subdiag   = jnp.concatenate([sub_bot,   sub_int,   sub_top],   axis=1)

    return jnp.stack([superdiag, maindiag, subdiag], axis=0)   # (3, ngrdcol, nzm)


# JIT-compiled production versions
diffusion_zt_lhs = jit(diffusion_zt_lhs_jax)
diffusion_zm_lhs = jit(diffusion_zm_lhs_jax)


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


# JIT-compiled production version
term_ma_zm_lhs = jit(term_ma_zm_lhs_jax)


def xpyp_term_ta_pdf_lhs_jax(
    coef_wpxpyp_implicit: jnp.ndarray,
    rho_ds_zt: jnp.ndarray,
    invrs_rho_ds_zm: jnp.ndarray,
    gr,
) -> jnp.ndarray:
    """Tridiagonal LHS for turbulent advection of xp2/xpyp (centered discretization).

    Faithful port of Fortran xpyp_term_ta_pdf_lhs, centered branch
    (l_upwind_xpyp_turbulent_adv = .false.).

    Discretizes:
      (1/rho_ds_zm) * d( rho_ds_zt * coef_wpxpyp_implicit * var_zm ) / dz
    at zm levels, using zm→zt interpolation weights.

    Fortran index conventions (1-based → 0-based Python):
      k=1       → k=0   (lower boundary, zero)
      k=2..nzm-1→ k=1..nzm-2 (interior)
      k=nzm     → k=nzm-1   (upper boundary, zero)

    Args:
        coef_wpxpyp_implicit: PDF coef of <x'y'> in <w'x'y'>, shape (ngrdcol, nzt).
        rho_ds_zt:            Dry, static density on zt levels, shape (ngrdcol, nzt).
        invrs_rho_ds_zm:      1/rho_ds on zm levels, shape (ngrdcol, nzm).
        gr:                   Grid with .invrs_dzm (ngrdcol, nzm)
                                and .weights_zm2zt (ngrdcol, nzt, 2).

    Returns:
        lhs: shape (3, ngrdcol, nzm).
    """
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


def term_dp1_lhs_jax(
    Cn: jnp.ndarray,
    invrs_tau_zm: jnp.ndarray,
) -> jnp.ndarray:
    """Main-diagonal coefficient for dissipation term 1 of x_a'x_b' equations.

    Faithful port of Fortran term_dp1_lhs (advance_xp2_xpyp_module.F90).

    The d(x_a'x_b')/dt equation contains the implicit dissipation:
      + (C_n / tau_zm) * x_a'x_b'(t+1)
    which contributes only to the main diagonal of the tridiagonal system.

    Fortran index conventions (1-based → 0-based Python):
      k=1     → k=0        (lower boundary, zero)
      k=2..nzm-1 → k=1..nzm-2 (interior)
      k=nzm   → k=nzm-1   (upper boundary, zero)

    Args:
        Cn:           Dissipation coefficient, shape (ngrdcol, nzm).
        invrs_tau_zm: Inverse timescale on zm levels, shape (ngrdcol, nzm).

    Returns:
        lhs: shape (ngrdcol, nzm). Main diagonal only — no super/sub-diagonal.
    """
    interior = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1]
    ngrdcol = Cn.shape[0]
    zeros_bnd = jnp.zeros((ngrdcol, 1), dtype=Cn.dtype)
    return jnp.concatenate([zeros_bnd, interior, zeros_bnd], axis=1)


# JIT-compiled production version
term_dp1_lhs = jit(term_dp1_lhs_jax)


_GAMMA_OVER_IMPLICIT_TS = 1.5   # constants_clubb.F90: gamma_over_implicit_ts


def xp2_xpyp_lhs_jax(
    lhs_ta: jnp.ndarray,
    lhs_ma: jnp.ndarray,
    lhs_diff: jnp.ndarray,
    lhs_dp1: jnp.ndarray,
    dt: float,
    gamma: float = _GAMMA_OVER_IMPLICIT_TS,
) -> jnp.ndarray:
    """Assemble full tridiagonal LHS for xp2/xpyp equations.

    Faithful port of Fortran xp2_xpyp_lhs (advance_xp2_xpyp_module.F90).

    For interior k=1..nzm-2 (Python):
      lhs[0,k] = lhs_diff[0,k] + lhs_ma[0,k] + lhs_ta[0,k] * gamma
      lhs[1,k] = lhs_diff[1,k] + lhs_ma[1,k] + lhs_ta[1,k] * gamma
                 + lhs_dp1[k]  (pre-scaled by caller)  + 1/dt
      lhs[2,k] = lhs_diff[2,k] + lhs_ma[2,k] + lhs_ta[2,k] * gamma

    Boundaries k=0 and k=nzm-1: lhs = [0, 1, 0] (fixed-value BCs).

    Args:
        lhs_ta:   Turbulent-advection LHS,   shape (3, ngrdcol, nzm).
        lhs_ma:   Mean-advection LHS,         shape (3, ngrdcol, nzm).
        lhs_diff: Eddy-diffusion LHS,         shape (3, ngrdcol, nzm).
        lhs_dp1:  Dissipation-term-1 diagonal, shape (ngrdcol, nzm),
                  already multiplied by gamma_over_implicit_ts by the caller.
        dt:       Timestep [s].
        gamma:    Over-implicit weight for turbulent-advection terms (default 1.5).

    Returns:
        lhs: shape (3, ngrdcol, nzm).
    """
    ngrdcol = lhs_ta.shape[1]

    # Interior k=1..nzm-2
    super_int = lhs_diff[0, :, 1:-1] + lhs_ma[0, :, 1:-1] + lhs_ta[0, :, 1:-1] * gamma
    main_int  = (lhs_diff[1, :, 1:-1] + lhs_ma[1, :, 1:-1] + lhs_ta[1, :, 1:-1] * gamma
                 + lhs_dp1[:, 1:-1] + 1.0 / dt)
    sub_int   = lhs_diff[2, :, 1:-1] + lhs_ma[2, :, 1:-1] + lhs_ta[2, :, 1:-1] * gamma

    # Boundaries: fixed-value BC → [super=0, main=1, sub=0]
    zeros_bnd = jnp.zeros((ngrdcol, 1), dtype=lhs_ta.dtype)
    ones_bnd  = jnp.ones((ngrdcol, 1),  dtype=lhs_ta.dtype)

    superdiag = jnp.concatenate([zeros_bnd, super_int, zeros_bnd], axis=1)
    maindiag  = jnp.concatenate([ones_bnd,  main_int,  ones_bnd],  axis=1)
    subdiag   = jnp.concatenate([zeros_bnd, sub_int,   zeros_bnd], axis=1)

    return jnp.stack([superdiag, maindiag, subdiag], axis=0)  # (3, ngrdcol, nzm)


# JIT-compiled production version
xp2_xpyp_lhs = jit(xp2_xpyp_lhs_jax)


def term_dp1_rhs_jax(
    Cn: jnp.ndarray,
    invrs_tau_zm: jnp.ndarray,
    threshold: float,
) -> jnp.ndarray:
    """Explicit portion of dissipation term 1 for x'y' equations (all levels).

    Faithful port of Fortran term_dp1_rhs (advance_xp2_xpyp_module.F90:5727).

    The explicit part of -(C_n/tau_zm) * (x'y' - threshold) is
    +(C_n/tau_zm) * threshold, applied at every level including boundaries.

    Args:
        Cn:           Dissipation coefficient, shape (ngrdcol, nzm).
        invrs_tau_zm: Inverse timescale on zm, shape (ngrdcol, nzm).
        threshold:    Scalar minimum value for x'y'.

    Returns:
        rhs: shape (ngrdcol, nzm).  All levels, no boundary zeroing.
    """
    return Cn * invrs_tau_zm * threshold


# JIT-compiled production version
term_dp1_rhs = jit(term_dp1_rhs_jax)


def xp2_xpyp_rhs_jax(
    lhs_ta: jnp.ndarray,
    rhs_ta: jnp.ndarray,
    Cn: jnp.ndarray,
    invrs_tau_zm: jnp.ndarray,
    threshold: float,
    xapxbp: jnp.ndarray,
    xam: jnp.ndarray,
    xbm: jnp.ndarray,
    wpxap: jnp.ndarray,
    wpxbp: jnp.ndarray,
    invrs_dzm: jnp.ndarray,
    xpyp_forcing: jnp.ndarray,
    dt: float,
    gamma: float = _GAMMA_OVER_IMPLICIT_TS,
) -> jnp.ndarray:
    """Explicit RHS of x'^2 and x'y' equations.

    Faithful port of Fortran xp2_xpyp_rhs (advance_xp2_xpyp_module.F90:3453).

    For interior levels k=1..nzm-2 (Python 0-based, ascending grid):
      rhs[k] = rhs_ta[k]
             + (1-gamma)*(-lhs_ta[0,k]*xapxbp[k+1]
                          -lhs_ta[1,k]*xapxbp[k]
                          -lhs_ta[2,k]*xapxbp[k-1])     [TA over-implicit]
             - wpxbp[k]*invrs_dzm[k]*(xam[k]-xam[k-1])  [TP term, part 1]
             - wpxap[k]*invrs_dzm[k]*(xbm[k]-xbm[k-1])  [TP term, part 2]
             + Cn[k]*invrs_tau_zm[k]*threshold            [DP1 explicit]
             + (1-gamma)*(-Cn[k]*invrs_tau_zm[k]*xapxbp[k]) [DP1 over-implicit]
             + xpyp_forcing[k]                            [forcing]
             + (1/dt)*xapxbp[k]                          [time tendency]

    Boundary conditions (ascending grid, fixed-value BCs):
      rhs[:, 0]  = xapxbp[:, 0]   (lower boundary: carry current value)
      rhs[:, -1] = threshold       (upper boundary: set to threshold)

    Band ordering: lhs_ta[0=super, 1=main, 2=sub], same as LHS convention.
    For ascending grid: super couples k to k+1 (above), sub couples k to k-1 (below).

    xam/xbm have shape (ngrdcol, nzt) where nzt=nzm-1.
    At interior k (Python 1..nzm-2): uses xam[:,1:]-xam[:,:-1] differences.

    Args:
        lhs_ta:       TA LHS, shape (3, ngrdcol, nzm). [0=super,1=main,2=sub].
        rhs_ta:       TA RHS, shape (ngrdcol, nzm).
        Cn:           Dissipation coefficient, shape (ngrdcol, nzm).
        invrs_tau_zm: Inverse timescale on zm, shape (ngrdcol, nzm).
        threshold:    Scalar minimum allowable value for x'y'.
        xapxbp:       x'y' on zm, shape (ngrdcol, nzm).
        xam:          x_am on zt, shape (ngrdcol, nzt).
        xbm:          x_bm on zt, shape (ngrdcol, nzt).
        wpxap:        w'x_a' on zm, shape (ngrdcol, nzm).
        wpxbp:        w'x_b' on zm, shape (ngrdcol, nzm).
        invrs_dzm:    1/dz on zm levels, shape (ngrdcol, nzm).
        xpyp_forcing: Forcing on zm, shape (ngrdcol, nzm).
        dt:           Timestep [s].
        gamma:        Over-implicit weight (default 1.5).

    Returns:
        rhs: shape (ngrdcol, nzm).
    """
    one_minus_gamma = 1.0 - gamma

    # term_tp_rhs: turbulent production at interior k=1..nzm-2
    # xam[:,1:]-xam[:,:-1] gives differences spanning k=1..nzm-2 on zm
    rhs_tp_int = (
        -wpxbp[:, 1:-1] * invrs_dzm[:, 1:-1] * (xam[:, 1:] - xam[:, :-1])
        - wpxap[:, 1:-1] * invrs_dzm[:, 1:-1] * (xbm[:, 1:] - xbm[:, :-1])
    )

    # term_dp1_rhs: Cn * invrs_tau_zm * threshold (all levels, used at interior)
    rhs_dp1_int = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1] * threshold

    # term_dp1_lhs (interior, unscaled): Cn * invrs_tau_zm
    lhs_dp1_int = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1]

    # Assemble interior RHS
    rhs_int = (
        rhs_ta[:, 1:-1]
        + one_minus_gamma * (
            - lhs_ta[0, :, 1:-1] * xapxbp[:, 2:]    # super * xapxbp[k+1]
            - lhs_ta[1, :, 1:-1] * xapxbp[:, 1:-1]  # main  * xapxbp[k]
            - lhs_ta[2, :, 1:-1] * xapxbp[:, :-2]   # sub   * xapxbp[k-1]
        )
        + rhs_tp_int
        + rhs_dp1_int
        + one_minus_gamma * (-lhs_dp1_int * xapxbp[:, 1:-1])
        + xpyp_forcing[:, 1:-1]
        + (1.0 / dt) * xapxbp[:, 1:-1]
    )

    # Boundary conditions: lower=current value, upper=threshold
    ngrdcol = Cn.shape[0]
    rhs_lb = xapxbp[:, 0:1]
    rhs_ub = jnp.full((ngrdcol, 1), threshold, dtype=Cn.dtype)

    return jnp.concatenate([rhs_lb, rhs_int, rhs_ub], axis=1)


# JIT-compiled production version
xp2_xpyp_rhs = jit(xp2_xpyp_rhs_jax)


def xpyp_term_ta_pdf_rhs_jax(
    term_wpxpyp_explicit: jnp.ndarray,
    rho_ds_zt: jnp.ndarray,
    invrs_rho_ds_zm: jnp.ndarray,
    gr,
) -> jnp.ndarray:
    """Explicit RHS turbulent-advection flux divergence for xp2/xpyp (centered).

    Faithful port of Fortran xpyp_term_ta_pdf_rhs (turbulent_adv_pdf.F90),
    centered path (l_upwind_xpyp_turbulent_adv = .false.).

    rhs[k] = -(1/rho_ds_zm[k]) * invrs_dzm[k] *
              (rho_ds_zt[k] * term[k] - rho_ds_zt[k-1] * term[k-1])
    for interior k=1..nzm-2 (Python 0-indexed); boundaries = 0.

    Args:
        term_wpxpyp_explicit: Explicit turbulent flux at zt levels, shape (ngrdcol, nzt).
        rho_ds_zt:            Dry static density at zt levels, shape (ngrdcol, nzt).
        invrs_rho_ds_zm:      1/rho_ds at zm levels, shape (ngrdcol, nzm).
        gr:                   Grid with .invrs_dzm (ngrdcol, nzm).

    Returns:
        rhs: shape (ngrdcol, nzm).  Boundaries (k=0 and k=nzm-1) are zero.
    """
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


def xpyp_term_ta_pdf_lhs_upwind_jax(
    rho_ds_zm: jnp.ndarray,
    invrs_rho_ds_zm: jnp.ndarray,
    sgn_turbulent_vel: jnp.ndarray,
    coef_wpxpyp_implicit_zm: jnp.ndarray,
    gr,
    grid_dir: float = 1.0,
) -> jnp.ndarray:
    """Upwind LHS for turbulent advection of xp2/xpyp.

    Faithful port of Fortran xpyp_term_ta_pdf_lhs (turbulent_adv_pdf.F90),
    l_upwind_xpyp_turbulent_adv = .true.

    For each interior k_py=1..nzm-2 (Fortran k=2..nzm-1):
      if grid_dir * sgn[k_py] > 0 (wind blowing up for ascending grid):
        super[k_py] = 0
        main[k_py]  = invrs_rho[k_py] * invrs_dzt[k_py-1] * rho[k_py] * coef[k_py]
        sub[k_py]   = -invrs_rho[k_py] * invrs_dzt[k_py-1] * rho[k_py-1] * coef[k_py-1]
      else:
        super[k_py] = invrs_rho[k_py] * invrs_dzt[k_py] * rho[k_py+1] * coef[k_py+1]
        main[k_py]  = -invrs_rho[k_py] * invrs_dzt[k_py] * rho[k_py] * coef[k_py]
        sub[k_py]   = 0

    Args:
        rho_ds_zm:               Dry static density at zm levels, (ngrdcol, nzm).
        invrs_rho_ds_zm:         1/rho_ds at zm levels, (ngrdcol, nzm).
        sgn_turbulent_vel:       Sign of turbulent velocity at zm levels, (ngrdcol, nzm).
        coef_wpxpyp_implicit_zm: Implicit coefficient at zm levels, (ngrdcol, nzm).
        gr:                      Grid with .invrs_dzt (ngrdcol, nzt=nzm-1).
        grid_dir:                +1 for ascending grid (default).

    Returns:
        lhs: shape (3, ngrdcol, nzm).  [0=super, 1=main, 2=sub]
    """
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


def xpyp_term_ta_pdf_rhs_upwind_jax(
    rho_ds_zm: jnp.ndarray,
    invrs_rho_ds_zm: jnp.ndarray,
    sgn_turbulent_vel: jnp.ndarray,
    term_wpxpyp_explicit_zm: jnp.ndarray,
    gr,
    grid_dir: float = 1.0,
) -> jnp.ndarray:
    """Upwind RHS for turbulent advection of xp2/xpyp.

    Faithful port of Fortran xpyp_term_ta_pdf_rhs (turbulent_adv_pdf.F90),
    l_upwind_xpyp_turbulent_adv = .true.

    For each interior k_py=1..nzm-2 (Fortran k=2..nzm-1):
      if grid_dir * sgn[k_py] > 0:
        rhs[k_py] = -invrs_rho[k_py] * invrs_dzt[k_py-1] *
                     (rho[k_py]*term[k_py] - rho[k_py-1]*term[k_py-1])
      else:
        rhs[k_py] = -invrs_rho[k_py] * invrs_dzt[k_py] *
                     (rho[k_py+1]*term[k_py+1] - rho[k_py]*term[k_py])

    Args:
        rho_ds_zm:               Dry static density at zm levels, (ngrdcol, nzm).
        invrs_rho_ds_zm:         1/rho_ds at zm levels, (ngrdcol, nzm).
        sgn_turbulent_vel:       Sign of turbulent velocity, (ngrdcol, nzm).
        term_wpxpyp_explicit_zm: Explicit flux at zm levels, (ngrdcol, nzm).
        gr:                      Grid with .invrs_dzt (ngrdcol, nzt=nzm-1).
        grid_dir:                +1 for ascending grid (default).

    Returns:
        rhs: shape (ngrdcol, nzm).  Boundaries (k=0, k=nzm-1) are zero.
    """
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


def xpyp_term_ta_pdf_lhs_godunov_jax(
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


def xpyp_term_ta_pdf_rhs_godunov_jax(
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
    "diffusion_zt_lhs_jax",
    "diffusion_zm_lhs_jax",
    "diffusion_zt_lhs",
    "diffusion_zm_lhs",
    "term_ma_zm_lhs_jax",
    "term_ma_zm_lhs",
    "xpyp_term_ta_pdf_lhs_jax",
    "xpyp_term_ta_pdf_lhs",
    "xpyp_term_ta_pdf_rhs_jax",
    "xpyp_term_ta_pdf_rhs",
    "term_dp1_lhs_jax",
    "term_dp1_lhs",
    "xp2_xpyp_lhs_jax",
    "xp2_xpyp_lhs",
    "term_dp1_rhs_jax",
    "term_dp1_rhs",
    "xp2_xpyp_rhs_jax",
    "xp2_xpyp_rhs",
    "xpyp_term_ta_pdf_lhs_upwind_jax",
    "xpyp_term_ta_pdf_rhs_upwind_jax",
    "xpyp_term_ta_pdf_lhs_godunov_jax",
    "xpyp_term_ta_pdf_rhs_godunov_jax",
]
