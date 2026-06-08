"""JAX implementation of compute_xp3/advance_xp3 from advance_xp3_module.F90.

The skewness diagnostics (Skx_func, compute_gamma_Skw, LG_2005_ansatz, xp3_LG_2005_ansatz) live in
Skx_module.py, mirroring the Fortran advance_xp3_module's `use Skx_module`."""

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.grid_class import zm2zt_jax, zt2zm_jax, ddzm_jax
from clubb_jax.src.CLUBB_core.constants_clubb import (
    grav,
    ibeta,
    iSkw_denom_coef,
    ixp3_coef_base,
    ixp3_coef_slope,
    rt_tol,
    thl_tol,
    w_tol,
    w_tol_sqd,
    zero_threshold,
)
from clubb_jax.src.CLUBB_core.Skx_module import xp3_LG_2005_ansatz


def term_tp_rhs(xp2_zt, wpxpp1, wpxp, rho_ds_zmp1, rho_ds_zm, invrs_rho_ds_zt, invrs_dzt):
    """Turbulent production of <x'^3>, explicit portion (advance_xp3_module.F90:term_tp_rhs).

    + 3 * ( <x'^2>|_zt / rho_ds ) * d( rho_ds * <w'x'> )/dz, on the central thermo level: xp2 interp. to zt,
    rho_ds_zm*wpxp differenced over zt, scaled by invrs_rho_ds_zt and 3*xp2_zt. Element-wise (vectorized over the
    column); mirrors the Fortran scalar function applied per level."""
    return 3.0 * xp2_zt * invrs_rho_ds_zt * invrs_dzt * (rho_ds_zmp1 * wpxpp1 - rho_ds_zm * wpxp)


def term_ac_rhs(xm_zmp1, xm_zm, wpxp2, invrs_dzt):
    """Accumulation of <x'^3>, explicit portion (advance_xp3_module.F90:term_ac_rhs).

    - 3 * <w'x'^2> * d<x>/dz, with <x> interp. to momentum levels and differenced over the central thermo level.
    Element-wise (vectorized over the column); mirrors the Fortran scalar function applied per level."""
    return -3.0 * wpxp2 * invrs_dzt * (xm_zmp1 - xm_zm)


def compute_xp3(
    wp2,          # (ngrdcol, nzm) zm-level
    wp3,          # (ngrdcol, nzt)
    wprtp,        # (ngrdcol, nzm)
    wpthlp,       # (ngrdcol, nzm)
    rtp2,         # (ngrdcol, nzm)
    thlp2,        # (ngrdcol, nzm)
    upwp,         # (ngrdcol, nzm)
    vpwp,         # (ngrdcol, nzm)
    up2,          # (ngrdcol, nzm)
    vp2,          # (ngrdcol, nzm)
    sigma_sqd_w,  # (ngrdcol, nzm)
    clubb_params, # (ngrdcol, nparams)
    gr,
):
    """Diagnose third-order moments (ADG1 PDF, sclr_dim=0).

    Matches Fortran advance_clubb_core behavior: wp2 is interpolated zm→zt
    (with w_tol_sqd floor) before use in Skw and LG05 ansatz — mirrors
    compute_xp3 in advance_xp3_module.F90.

    Returns: (rtp3, thlp3, up3, vp3), all shape (ngrdcol, nzt).
    """
    # Interpolate wp2 from zm to zt levels with positive-definite floor
    # (matches Fortran advance_clubb_core which passes zm2zt(wp2) to compute_xp3)
    wp2_zt = jnp.maximum(zm2zt_jax(wp2, gr), w_tol_sqd)

    # Skewness of w: Skw = wp3 * (wp2_zt + denom_tol)^(-3/2)
    denom_tol_w = clubb_params[:, iSkw_denom_coef - 1:iSkw_denom_coef] * w_tol**2
    Skw_zt = wp3 * (wp2_zt + denom_tol_w) ** (-1.5)

    # Interpolate zm → zt with floors for positive-definite quantities
    wpthlp_zt = zm2zt_jax(wpthlp, gr)
    wprtp_zt  = zm2zt_jax(wprtp, gr)
    thlp2_zt  = jnp.maximum(zm2zt_jax(thlp2, gr), thl_tol ** 2)
    rtp2_zt   = jnp.maximum(zm2zt_jax(rtp2, gr), rt_tol ** 2)
    upwp_zt   = zm2zt_jax(upwp, gr)
    vpwp_zt   = zm2zt_jax(vpwp, gr)
    up2_zt    = jnp.maximum(zm2zt_jax(up2, gr), w_tol_sqd)
    vp2_zt    = jnp.maximum(zm2zt_jax(vp2, gr), w_tol_sqd)
    sigma_sqd_w_zt = jnp.maximum(zm2zt_jax(sigma_sqd_w, gr), zero_threshold)

    beta = clubb_params[:, ibeta - 1:ibeta]  # (ngrdcol, 1)

    thlp3 = xp3_LG_2005_ansatz(
        Skw_zt, wpthlp_zt, wp2_zt, thlp2_zt, sigma_sqd_w_zt, beta, clubb_params, thl_tol)
    rtp3  = xp3_LG_2005_ansatz(
        Skw_zt, wprtp_zt,  wp2_zt, rtp2_zt,  sigma_sqd_w_zt, beta, clubb_params, rt_tol)
    up3   = xp3_LG_2005_ansatz(
        Skw_zt, upwp_zt,   wp2_zt, up2_zt,   sigma_sqd_w_zt, beta, clubb_params, w_tol)
    vp3   = xp3_LG_2005_ansatz(
        Skw_zt, vpwp_zt,   wp2_zt, vp2_zt,   sigma_sqd_w_zt, beta, clubb_params, w_tol)

    return rtp3, thlp3, up3, vp3


def advance_xp3_simplified(xm, xp2, wpxp, wpxp2, rho_ds_zm, invrs_rho_ds_zt,
                                  invrs_tau_zt, tau_max_zt, x_tol, gr):
    """Steady-state xp3 (advance_xp3_module.F90:advance_xp3_simplified, l_predict_xp3=False).

    C_xp3_dissipation=1.0, so:
      xp3[k] = min(tau_zt[k], tau_max_zt[k]) * (term_tp[k] + term_ac[k])
      term_tp[k] = 3*xp2_zt[k]*irho[k]*idzt[k]*(rho[kp1]*wpxp[kp1] - rho[k]*wpxp[k])
      term_ac[k] = -3*wpxp2[k]*idzt[k]*(xm_zm[kp1] - xm_zm[k])
    Loop k=0..nzt-2; kp1=min(k+1,nzt-1); top (k=nzt-1) = 0.
    """
    nzt = invrs_tau_zt.shape[1]

    xm_zm  = zt2zm_jax(xm, gr, zm_min=zero_threshold)  # (ngrdcol, nzm)
    xp2_zt = zm2zt_jax(xp2, gr)                        # (ngrdcol, nzt)
    xp2_zt = jnp.maximum(xp2_zt, x_tol ** 2)

    # Interior k_py=0..nzt-2; kp1_py=min(k_py+1, nzt-1) — the level above, clamped to the top.
    # NB (iter 491): the Fortran advance_xp3_simplified writes this clamp as `kp1 = max( k+1, nzt )`
    # (advance_xp3_module.F90:812), which evaluates to the CONSTANT nzt for every k in the 1..nzt-1 loop — an
    # apparent typo (the analogous clamps elsewhere, pdf_closure_module.F90:5556 / interpolation.F90:480, use
    # `min(k+1, nzt)`). This JAX uses the correct `min` (the level above). This path is GATED OFF (non-ADG1; the
    # iiPDF_type init-guard rejects it, l_advance_xp3=False by default), so neither the Fortran typo nor this
    # deviation is ever exercised — the JAX is intentionally NOT bit-faithful to the Fortran's `max` here. Do not
    # "correct" the JAX to match the Fortran bug.
    k_idx   = jnp.arange(nzt - 1)
    kp1_idx = jnp.minimum(k_idx + 1, nzt - 1)

    # Slice interior zt-level arrays
    xp2_zt_int = xp2_zt[:, :nzt - 1]              # (ngrdcol, nzt-1)
    irho_int   = invrs_rho_ds_zt[:, :nzt - 1]
    idzt_int   = gr.invrs_dzt[:, :nzt - 1]
    wpxp2_int  = wpxp2[:, :nzt - 1]

    # Neighboring zm-level values (indexed by k and kp1)
    wpxp_k   = wpxp[:, k_idx]
    wpxp_kp1 = wpxp[:, kp1_idx]
    rho_k    = rho_ds_zm[:, k_idx]
    rho_kp1  = rho_ds_zm[:, kp1_idx]
    xm_k     = xm_zm[:, k_idx]
    xm_kp1   = xm_zm[:, kp1_idx]

    term_tp = term_tp_rhs(xp2_zt_int, wpxp_kp1, wpxp_k, rho_kp1, rho_k, irho_int, idzt_int)
    term_ac = term_ac_rhs(xm_kp1, xm_k, wpxp2_int, idzt_int)

    tau_int = jnp.minimum(1.0 / invrs_tau_zt[:, :nzt - 1], tau_max_zt[:, :nzt - 1])
    xp3_int = tau_int * (term_tp + term_ac)

    top_zeros = jnp.zeros((xm.shape[0], 1), dtype=xp3_int.dtype)
    return jnp.concatenate([xp3_int, top_zeros], axis=1)  # (ngrdcol, nzt)


def advance_xp3(
    rtm, thlm, rtp2, thlp2, wprtp, wpthlp, wprtp2, wpthlp2, rho_ds_zm,
    invrs_rho_ds_zt, invrs_tau_zt, tau_max_zt,
    wp2, wp3, upwp, vpwp, up2, vp2,
    thvm, clubb_params, gr,
):
    """Advance third-order moments for non-ADG1 PDF (advance_xp3_module.F90:advance_xp3).

    Uses steady-state approximation (l_predict_xp3=False, C_xp3_dissipation=1.0).
    sclr_dim=0 variant — passive scalars not handled.

    Returns: (rtp3, thlp3, up3, vp3), all shape (ngrdcol, nzt).
    """
    # rtp3 and thlp3 via steady-state advance_xp3_simplified
    rtp3 = advance_xp3_simplified(
        rtm, rtp2, wprtp, wprtp2, rho_ds_zm, invrs_rho_ds_zt,
        invrs_tau_zt, tau_max_zt, rt_tol, gr,
    )
    thlp3 = advance_xp3_simplified(
        thlm, thlp2, wpthlp, wpthlp2, rho_ds_zm, invrs_rho_ds_zt,
        invrs_tau_zt, tau_max_zt, thl_tol, gr,
    )

    # wp2_zt: zm→zt with floor w_tol_sqd
    wp2_zt = zm2zt_jax(wp2, gr)
    wp2_zt = jnp.maximum(wp2_zt, w_tol_sqd)

    # Skw_zt from Skx_func (same formula, applied to w)
    denom_tol_w = clubb_params[:, iSkw_denom_coef - 1:iSkw_denom_coef] * w_tol ** 2
    Skw_zt = wp3 * (wp2_zt + denom_tol_w) ** (-1.5)

    # Interpolate momentum fluxes and variances to zt
    upwp_zt = zm2zt_jax(upwp, gr)
    vpwp_zt = zm2zt_jax(vpwp, gr)
    up2_zt  = jnp.maximum(zm2zt_jax(up2, gr), w_tol_sqd)
    vp2_zt  = jnp.maximum(zm2zt_jax(vp2, gr), w_tol_sqd)

    # Buoyancy frequency squared at zt levels (for xp3_coef_fnc)
    thvm_zm      = zt2zm_jax(thvm, gr, zm_min=zero_threshold)  # (ngrdcol, nzm)
    ddzm_thvm_zm = ddzm_jax(thvm_zm, gr)                       # (ngrdcol, nzt)
    bv_sqd_zt    = jnp.maximum((grav / thvm) * ddzm_thvm_zm, 0.0)

    # xp3_coef_fnc: non-ADG1 coefficient in place of sigma_sqd_w_zt
    coef_base  = clubb_params[:, ixp3_coef_base - 1:ixp3_coef_base]   # (ngrdcol, 1)
    coef_slope = clubb_params[:, ixp3_coef_slope - 1:ixp3_coef_slope] # (ngrdcol, 1)
    xp3_coef_fnc = coef_base + (1.0 - coef_slope) * (1.0 - jnp.exp(bv_sqd_zt / coef_slope))

    # up3, vp3 via xp3_LG_2005_ansatz with xp3_coef_fnc
    beta = clubb_params[:, ibeta - 1:ibeta]
    up3 = xp3_LG_2005_ansatz(Skw_zt, upwp_zt, wp2_zt, up2_zt, xp3_coef_fnc,
                               beta, clubb_params, w_tol)
    vp3 = xp3_LG_2005_ansatz(Skw_zt, vpwp_zt, wp2_zt, vp2_zt, xp3_coef_fnc,
                               beta, clubb_params, w_tol)

    return rtp3, thlp3, up3, vp3
