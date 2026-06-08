"""Pure JAX implementations of the eddy-diffusion LHS operators (mirror of diffusion.F90).

Faithful ports of CLUBB Fortran source diffusion.F90:
  - diffusion_zt_lhs_jax:          tridiagonal diffusion LHS for zt-level variables
  - diffusion_zm_lhs_jax:          tridiagonal diffusion LHS for zm-level variables

Output layout lhs[3, ngrdcol, nz]:
  lhs[0] = superdiagonal  (coefficient of var[k+1])
  lhs[1] = main diagonal  (coefficient of var[k])
  lhs[2] = subdiagonal    (coefficient of var[k-1])

Array layout: (ngrdcol, nz), ascending grid (index 0 = lowest level).

References:
  src/CLUBB_core/diffusion.F90, diffusion_zt_lhs, diffusion_zm_lhs.

(The mean-advection `term_ma_zm_lhs` and turbulent-advection `xpyp_term_ta_pdf_*` operators that
formerly also lived here were dead duplicates; they live in their Fortran homes mean_adv.py /
turbulent_adv_pdf.py and the diffusion.py copies were deleted — mirror-refactor iters 228-229.)
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


__all__ = [
    "diffusion_zt_lhs_jax",
    "diffusion_zm_lhs_jax",
    "diffusion_zt_lhs",
    "diffusion_zm_lhs",
]
