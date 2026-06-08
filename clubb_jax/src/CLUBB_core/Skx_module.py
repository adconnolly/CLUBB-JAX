"""JAX port of Skx_module.F90 — skewness-of-x diagnostics.

Mirrors clubb_release/src/CLUBB_core/Skx_module.F90:
  Skx_func             — skewness of x with the sensitivity-reduction denominator
  compute_gamma_Skw    — the tunable gamma(Skw) Gaussian function
  LG_2005_ansatz       — Larson & Golaz (2005) skewness ansatz (eqs. 11, 16, 33)
  xp3_LG_2005_ansatz   — <x'^3> from the LG05 skewness ansatz (inverse of Skx_func)

advance_xp3_module.py imports `Skx_func`/`xp3_LG_2005_ansatz` from here, exactly as the Fortran
advance_xp3_module `use`s Skx_module. Pure-jnp → differentiable.
"""
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import iSkw_denom_coef, igamma_coef, igamma_coefb, igamma_coefc, eps, w_tol_sqd


def Skx_func(xp2, xp3, x_tol, clubb_params):
    """Compute skewness of x with sensitivity-reduction formula (Skx_module.F90:Skx_func).

    Skx = xp3 * (xp2 + denom_tol)^(-3/2)
    where denom_tol = iSkw_denom_coef * x_tol^2
    """
    denom_tol = clubb_params[:, iSkw_denom_coef - 1:iSkw_denom_coef] * x_tol ** 2
    return xp3 * (xp2 + denom_tol) ** (-1.5)


def compute_gamma_Skw(Skw, clubb_params, l_gamma_Skw):
    """Gamma coefficient as a Gaussian function of w skewness (Skx_module.F90:compute_gamma_Skw).

    When l_gamma_Skw and the two coefficients differ meaningfully
    (|γ_coef − γ_coefb| > |γ_coef + γ_coefb|·eps/2):
        gamma = γ_coefb + (γ_coef − γ_coefb)·exp(−½ (Skw/γ_coefc)²),
    otherwise (degenerate coefficients, or l_gamma_Skw off) gamma = γ_coef (constant). The branch depends only
    on the per-column tunable parameters, not on Skw. Skw is (ngrdcol, nz) — pass Skw_zm or Skw_zt. Pure-jnp →
    differentiable. Returns gamma_Skw_fnc with Skw's shape."""
    Skw = jnp.asarray(Skw, dtype=jnp.float64)
    cp = jnp.asarray(clubb_params, dtype=jnp.float64)
    gc = cp[:, igamma_coef - 1:igamma_coef]      # (ngrdcol, 1)
    gb = cp[:, igamma_coefb - 1:igamma_coefb]
    gcf = cp[:, igamma_coefc - 1:igamma_coefc]
    if not l_gamma_Skw:
        return gc + jnp.zeros_like(Skw)               # broadcast (ngrdcol,1) over (ngrdcol,nz)
    cond = jnp.abs(gc - gb) > jnp.abs(gc + gb) * eps / 2.0
    varying = gb + (gc - gb) * jnp.exp(-0.5 * (Skw / gcf) ** 2)
    return jnp.where(cond, varying, gc + jnp.zeros_like(Skw))


def xp3_LG_2005_ansatz(Skw_zt, wpxp_zt, wp2_zt, xp2_zt, sigma_sqd_w_zt,
                        beta, clubb_params, x_tol):
    """Compute <x'^3> via LG05 ansatz (Skx_module.F90:xp3_LG_2005_ansatz)."""
    Skx_denom_tol = clubb_params[:, iSkw_denom_coef - 1:iSkw_denom_coef] * x_tol ** 2
    Skx_zt = LG_2005_ansatz(Skw_zt, wpxp_zt, wp2_zt, xp2_zt, sigma_sqd_w_zt, beta, x_tol)
    # Reverse of Skx_func: xp3 = Skx * (xp2 + denom_tol)^(3/2)
    xp3 = Skx_zt * (xp2_zt + Skx_denom_tol) * jnp.sqrt(xp2_zt + Skx_denom_tol)
    return xp3


def LG_2005_ansatz(Skw, wpxp, wp2, xp2, sigma_sqd_w, beta, x_tol):
    """LG 2005 eqs. 11, 16, 33 (Skx_module.F90:LG_2005_ansatz) — skewness of x from skewness of w."""
    one_minus_ssw = 1.0 - sigma_sqd_w
    nrmlzd_corr_wx = wpxp / jnp.sqrt(
        jnp.maximum(wp2, w_tol_sqd) * jnp.maximum(xp2, x_tol ** 2) * one_minus_ssw
    )
    nrmlzd_Skw = Skw / (one_minus_ssw * jnp.sqrt(one_minus_ssw))
    Skx = nrmlzd_Skw * nrmlzd_corr_wx * (beta + (1.0 - beta) * nrmlzd_corr_wx ** 2)
    return Skx
