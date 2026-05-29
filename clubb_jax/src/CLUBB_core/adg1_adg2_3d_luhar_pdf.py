"""JAX implementation of ADG1 PDF driver (adg1_adg2_3d_luhar_pdf.F90).

Implements:
  ADG1_w_closure_jax             - mixture fraction and w PDF component params
  ADG1_ADG2_responder_params_jax - PDF component params for rt/thl/u/v/sclr
  ADG1_pdf_driver_jax            - top-level ADG1 PDF parameter driver
  calc_comp_corrs_binormal_jax   - PDF component correlations (pdf_utilities.F90)
"""

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import (
    rt_tol, thl_tol, w_tol_sqd, zero_threshold,
    ibeta,
    eps, min_max_smth_mag, max_mag_correlation,
)


def ADG1_w_closure_jax(wm, wp2, Skw, sigma_sqd_w, sqrt_wp2, mixt_frac_max_mag):
    """adg1_adg2_3d_luhar_pdf.F90:ADG1_w_closure.

    Computes mixture fraction and w PDF component parameters for ADG1 closure.

    Args:
        wm:               (ngrdcol, nzt) — mean w [m/s]
        wp2:              (ngrdcol, nzt) — w variance [m^2/s^2]
        Skw:              (ngrdcol, nzt) — w skewness [-]
        sigma_sqd_w:      (ngrdcol, nzt) — PDF width parameter [-]
        sqrt_wp2:         (ngrdcol, nzt) — sqrt(wp2) [m/s]
        mixt_frac_max_mag: scalar — max |mixt_frac - 0.5| + 0.5

    Returns:
        w_1, w_2:           (ngrdcol, nzt) — w means of each PDF component
        w_1_n, w_2_n:       (ngrdcol, nzt) — normalized w means
        varnce_w_1, varnce_w_2: (ngrdcol, nzt) — w variances per component
        mixt_frac:          (ngrdcol, nzt) — mixture fraction (weight of 1st)
    """
    _SKW_TOL = 1.0e-5

    # Mixture fraction
    # If |Skw| <= 1e-5: mixt_frac = 0.5
    # Otherwise: mixt_frac = 0.5 * (1 - Skw / sqrt(4*(1-sigma_sqd_w)^3 + Skw^2))
    denom_sq = 4.0 * (1.0 - sigma_sqd_w) ** 3 + Skw ** 2
    mf_formula = 0.5 * (1.0 - Skw / jnp.sqrt(denom_sq))
    mixt_frac = jnp.where(jnp.abs(Skw) <= _SKW_TOL, 0.5, mf_formula)

    # Clip to [1 - mixt_frac_max_mag, mixt_frac_max_mag]
    mixt_frac = jnp.clip(mixt_frac, 1.0 - mixt_frac_max_mag, mixt_frac_max_mag)

    # Normalized means: w_1_n = sqrt((1-mf)/mf * (1-sigma_sqd_w))  > 0
    #                   w_2_n = -sqrt(mf/(1-mf) * (1-sigma_sqd_w)) < 0
    one_minus_mf = 1.0 - mixt_frac
    sigma_factor = 1.0 - sigma_sqd_w
    w_1_n = jnp.sqrt(one_minus_mf / mixt_frac * sigma_factor)
    w_2_n = -jnp.sqrt(mixt_frac / one_minus_mf * sigma_factor)

    # Actual means
    w_1 = wm + sqrt_wp2 * w_1_n
    w_2 = wm + sqrt_wp2 * w_2_n

    # Variances (both equal to sigma_sqd_w * wp2)
    varnce_w_1 = sigma_sqd_w * wp2
    varnce_w_2 = sigma_sqd_w * wp2

    return w_1, w_2, w_1_n, w_2_n, varnce_w_1, varnce_w_2, mixt_frac


def ADG1_ADG2_responder_params_jax(xm, xp2, wp2, sqrt_wp2, wpxp,
                                   w_1_n, w_2_n, mixt_frac,
                                   sigma_sqd_w, beta):
    """adg1_adg2_3d_luhar_pdf.F90:ADG1_ADG2_responder_params.

    Computes PDF component means and variances for a "responder" variable x
    (rt, thl, u, v, or passive scalar) given the ADG1 w-closure parameters.

    Args:
        xm:         (ngrdcol, nzt) — mean of x
        xp2:        (ngrdcol, nzt) — variance of x
        wp2:        (ngrdcol, nzt) — variance of w
        sqrt_wp2:   (ngrdcol, nzt) — sqrt(wp2)
        wpxp:       (ngrdcol, nzt) — covariance w'x'
        w_1_n, w_2_n: (ngrdcol, nzt) — normalized w component means
        mixt_frac:  (ngrdcol, nzt) — mixture fraction
        sigma_sqd_w: (ngrdcol, nzt) — PDF width parameter
        beta:       (ngrdcol,) or scalar — CLUBB tunable parameter beta

    Returns:
        x_1, x_2:           (ngrdcol, nzt) — component means
        varnce_x_1, varnce_x_2: (ngrdcol, nzt) — component variances
        alpha_x:            (ngrdcol, nzt) — normalized variance factor
    """
    # Component means
    # x_1 = xm - wpxp / (sqrt_wp2 * w_2_n)
    # x_2 = xm - wpxp / (sqrt_wp2 * w_1_n)
    x_1 = xm - wpxp / (sqrt_wp2 * w_2_n)
    x_2 = xm - wpxp / (sqrt_wp2 * w_1_n)

    # alpha_x = 0.5 * (1 - wpxp^2 / ((1-sigma_sqd_w)*wp2*xp2))
    alpha_x = 0.5 * (1.0 - wpxp ** 2 / ((1.0 - sigma_sqd_w) * wp2 * xp2))
    # Clip to [zero_threshold, 1]
    alpha_x = jnp.clip(alpha_x, zero_threshold, 1.0)

    # Width factor: width_factor_1 = (2/3)*beta + 2*mixt_frac*(1 - (2/3)*beta)
    if hasattr(beta, 'shape') and len(beta.shape) == 1:
        beta_bc = beta[:, None]  # (ngrdcol, 1)
    else:
        beta_bc = beta
    two_thirds_beta = (2.0 / 3.0) * beta_bc
    width_factor_1 = two_thirds_beta + 2.0 * mixt_frac * (1.0 - two_thirds_beta)

    # Component variances
    varnce_x_1 = width_factor_1 * xp2 * alpha_x / mixt_frac
    varnce_x_2 = (2.0 - width_factor_1) * xp2 * alpha_x / (1.0 - mixt_frac)

    return x_1, x_2, varnce_x_1, varnce_x_2, alpha_x


def ADG1_pdf_driver_jax(wm, rtm, thlm, um, vm,
                        wp2, rtp2, thlp2, up2, vp2,
                        Skw, wprtp, wpthlp, upwp, vpwp,
                        sqrt_wp2, sigma_sqd_w,
                        beta, mixt_frac_max_mag):
    """adg1_adg2_3d_luhar_pdf.F90:ADG1_pdf_driver — top-level ADG1 driver.

    All inputs on zt grid (ngrdcol, nzt).

    Args:
        wm:     (ngrdcol, nzt) — mean w
        rtm:    (ngrdcol, nzt) — mean total water
        thlm:   (ngrdcol, nzt) — mean theta-l
        um, vm: (ngrdcol, nzt) — mean winds
        wp2:    (ngrdcol, nzt) — w variance
        rtp2:   (ngrdcol, nzt) — rt variance
        thlp2:  (ngrdcol, nzt) — thl variance
        up2, vp2: (ngrdcol, nzt) — u,v variances
        Skw:    (ngrdcol, nzt) — w skewness
        wprtp, wpthlp, upwp, vpwp: (ngrdcol, nzt) — covariances
        sqrt_wp2: (ngrdcol, nzt) — sqrt(wp2)
        sigma_sqd_w: (ngrdcol, nzt) — PDF width parameter
        beta:   (ngrdcol,) — tunable parameter
        mixt_frac_max_mag: scalar

    Returns:
        dict with keys: w_1, w_2, varnce_w_1, varnce_w_2, mixt_frac,
                        rt_1, rt_2, varnce_rt_1, varnce_rt_2, alpha_rt,
                        thl_1, thl_2, varnce_thl_1, varnce_thl_2, alpha_thl,
                        u_1, u_2, varnce_u_1, varnce_u_2, alpha_u,
                        v_1, v_2, varnce_v_1, varnce_v_2, alpha_v
    """
    # w closure
    (w_1, w_2, w_1_n, w_2_n,
     varnce_w_1, varnce_w_2, mixt_frac) = ADG1_w_closure_jax(
        wm, wp2, Skw, sigma_sqd_w, sqrt_wp2, mixt_frac_max_mag)

    # Responder parameters for rt
    (rt_1, rt_2, varnce_rt_1, varnce_rt_2, alpha_rt) = ADG1_ADG2_responder_params_jax(
        rtm, rtp2, wp2, sqrt_wp2, wprtp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)

    # Responder parameters for thl
    (thl_1, thl_2, varnce_thl_1, varnce_thl_2, alpha_thl) = ADG1_ADG2_responder_params_jax(
        thlm, thlp2, wp2, sqrt_wp2, wpthlp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)

    # Responder parameters for u
    (u_1, u_2, varnce_u_1, varnce_u_2, alpha_u) = ADG1_ADG2_responder_params_jax(
        um, up2, wp2, sqrt_wp2, upwp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)

    # Responder parameters for v
    (v_1, v_2, varnce_v_1, varnce_v_2, alpha_v) = ADG1_ADG2_responder_params_jax(
        vm, vp2, wp2, sqrt_wp2, vpwp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)

    return {
        'w_1': w_1, 'w_2': w_2,
        'varnce_w_1': varnce_w_1, 'varnce_w_2': varnce_w_2,
        'mixt_frac': mixt_frac,
        'rt_1': rt_1, 'rt_2': rt_2,
        'varnce_rt_1': varnce_rt_1, 'varnce_rt_2': varnce_rt_2, 'alpha_rt': alpha_rt,
        'thl_1': thl_1, 'thl_2': thl_2,
        'varnce_thl_1': varnce_thl_1, 'varnce_thl_2': varnce_thl_2, 'alpha_thl': alpha_thl,
        'u_1': u_1, 'u_2': u_2,
        'varnce_u_1': varnce_u_1, 'varnce_u_2': varnce_u_2, 'alpha_u': alpha_u,
        'v_1': v_1, 'v_2': v_2,
        'varnce_v_1': varnce_v_1, 'varnce_v_2': varnce_v_2, 'alpha_v': alpha_v,
    }


def calc_comp_corrs_binormal_jax(xpyp, xm, ym,
                                 mu_x_1, mu_x_2, mu_y_1, mu_y_2,
                                 sigma_x_1_sqd, sigma_x_2_sqd,
                                 sigma_y_1_sqd, sigma_y_2_sqd,
                                 mixt_frac):
    """pdf_utilities.F90:calc_comp_corrs_binormal + smooth_corr_quotient.

    Computes the PDF component correlations of x and y in a two-component
    normal PDF, given the overall covariance and component parameters.

    The formula is:
      corr = (xpyp - mf*(mu_x_1-xm)*(mu_y_1-ym) - (1-mf)*(mu_x_2-xm)*(mu_y_2-ym))
             / smooth_denom
    where smooth_denom is the result of two smooth_max calls in smooth_corr_quotient.
    corr_x_y_1 = corr_x_y_2 = corr.

    Args (all (ngrdcol, nzt) arrays):
        xpyp:            overall covariance of x and y
        xm, ym:          overall means of x and y
        mu_x_1, mu_x_2: component means of x
        mu_y_1, mu_y_2: component means of y
        sigma_x_1_sqd, sigma_x_2_sqd: component variances of x
        sigma_y_1_sqd, sigma_y_2_sqd: component variances of y
        mixt_frac:       mixture fraction (weight of 1st component)

    Returns:
        corr_x_y_1, corr_x_y_2: (ngrdcol, nzt) — equal component correlations
    """
    one_minus_mf = 1.0 - mixt_frac

    # Numerator: residual covariance after subtracting cross-component terms
    numer = (xpyp
             - mixt_frac * (mu_x_1 - xm) * (mu_y_1 - ym)
             - one_minus_mf * (mu_x_2 - xm) * (mu_y_2 - ym))

    # Denominator: weighted sum of component geometric means of std devs
    denom = (mixt_frac * jnp.sqrt(sigma_x_1_sqd * sigma_y_1_sqd)
             + one_minus_mf * jnp.sqrt(sigma_x_2_sqd * sigma_y_2_sqd))

    # smooth_corr_quotient — two smooth_max calls with denom_thresh=eps
    _denom_thresh = eps  # = max(1e-10, machine_eps) = 1e-10
    _smth = min(min_max_smth_mag, _denom_thresh)  # = min(1e-9, 1e-10) = 1e-10

    # First smooth_max: smooth_max(|numer|/max_mag_correlation, denom, smth)
    _a1 = jnp.abs(numer) / max_mag_correlation
    _tmp = 0.5 * ((_a1 + denom) + jnp.sqrt((_a1 - denom) ** 2 + _smth ** 2))

    # Second smooth_max: smooth_max(tmp, denom_thresh, smth)
    _tmp = 0.5 * ((_tmp + _denom_thresh) + jnp.sqrt((_tmp - _denom_thresh) ** 2 + _smth ** 2))

    corr = numer / _tmp
    return corr, corr  # corr_x_y_1 = corr_x_y_2


def calc_wp2xp_pdf_jax(wm, xm, w_1, w_2, x_1, x_2,
                        varnce_w_1, varnce_w_2,
                        varnce_x_1, varnce_x_2,
                        corr_w_x_1, corr_w_x_2,
                        mixt_frac):
    """pdf_utilities.F90:calc_wp2xp_pdf — <w'^2 x'> from bivariate PDF integral.

    Formula:
      wp2xp = mf * (((w_1-wm)^2 + varnce_w_1)*(x_1-xm)
                    + 2*corr_w_x_1*sqrt(varnce_w_1*varnce_x_1)*(w_1-wm))
            + (1-mf) * (((w_2-wm)^2 + varnce_w_2)*(x_2-xm)
                        + 2*corr_w_x_2*sqrt(varnce_w_2*varnce_x_2)*(w_2-wm))

    Args (all (ngrdcol, nzt) unless noted):
        wm, xm:           overall means of w and x
        w_1, w_2:         component means of w
        x_1, x_2:         component means of x
        varnce_w_1/2:     component variances of w
        varnce_x_1/2:     component variances of x
        corr_w_x_1/2:     component correlations w-x (= 0 for ADG1)
        mixt_frac:        mixture fraction

    Returns:
        wp2xp: (ngrdcol, nzt) — <w'^2 x'>
    """
    one_minus_mf = 1.0 - mixt_frac
    dw_1 = w_1 - wm
    dw_2 = w_2 - wm
    dx_1 = x_1 - xm
    dx_2 = x_2 - xm

    term1 = ((dw_1 ** 2 + varnce_w_1) * dx_1
             + 2.0 * corr_w_x_1 * jnp.sqrt(varnce_w_1 * varnce_x_1) * dw_1)
    term2 = ((dw_2 ** 2 + varnce_w_2) * dx_2
             + 2.0 * corr_w_x_2 * jnp.sqrt(varnce_w_2 * varnce_x_2) * dw_2)

    return mixt_frac * term1 + one_minus_mf * term2


def calc_wpxp2_pdf_jax(wm, xm, w_1, w_2, x_1, x_2,
                        varnce_w_1, varnce_w_2,
                        varnce_x_1, varnce_x_2,
                        corr_w_x_1, corr_w_x_2,
                        mixt_frac):
    """pdf_utilities.F90:calc_wpxp2_pdf — <w'x'^2> from bivariate PDF integral.

    Formula:
      wpxp2 = mf * ((w_1-wm)*((x_1-xm)^2 + varnce_x_1)
                    + 2*corr_w_x_1*sqrt(varnce_w_1*varnce_x_1)*(x_1-xm))
            + (1-mf) * ((w_2-wm)*((x_2-xm)^2 + varnce_x_2)
                        + 2*corr_w_x_2*sqrt(varnce_w_2*varnce_x_2)*(x_2-xm))
    """
    one_minus_mf = 1.0 - mixt_frac
    dw_1 = w_1 - wm
    dw_2 = w_2 - wm
    dx_1 = x_1 - xm
    dx_2 = x_2 - xm

    term1 = (dw_1 * (dx_1 ** 2 + varnce_x_1)
             + 2.0 * corr_w_x_1 * jnp.sqrt(varnce_w_1 * varnce_x_1) * dx_1)
    term2 = (dw_2 * (dx_2 ** 2 + varnce_x_2)
             + 2.0 * corr_w_x_2 * jnp.sqrt(varnce_w_2 * varnce_x_2) * dx_2)

    return mixt_frac * term1 + one_minus_mf * term2


def calc_wp2xp2_pdf_jax(wm, xm, w_1, w_2, x_1, x_2,
                         varnce_w_1, varnce_w_2,
                         varnce_x_1, varnce_x_2,
                         corr_w_x_1, corr_w_x_2,
                         mixt_frac):
    """pdf_utilities.F90:calc_wp2xp2_pdf — <w'^2 x'^2> from bivariate PDF integral.

    Formula:
      wp2xp2 = mf * ((dw_1^2 * (dx_1^2 + varnce_x_1)
                      + 4*corr_w_x_1*sqrt(varnce_w_1*varnce_x_1)*dx_1*dw_1
                      + (dx_1^2 + (1+2*corr_w_x_1^2)*varnce_x_1)*varnce_w_1))
             + (1-mf) * (same for component 2)
    """
    one_minus_mf = 1.0 - mixt_frac
    dw_1 = w_1 - wm
    dw_2 = w_2 - wm
    dx_1 = x_1 - xm
    dx_2 = x_2 - xm

    term1 = (dw_1 ** 2 * (dx_1 ** 2 + varnce_x_1)
             + 4.0 * corr_w_x_1 * jnp.sqrt(varnce_w_1 * varnce_x_1) * dx_1 * dw_1
             + (dx_1 ** 2 + (1.0 + 2.0 * corr_w_x_1 ** 2) * varnce_x_1) * varnce_w_1)
    term2 = (dw_2 ** 2 * (dx_2 ** 2 + varnce_x_2)
             + 4.0 * corr_w_x_2 * jnp.sqrt(varnce_w_2 * varnce_x_2) * dx_2 * dw_2
             + (dx_2 ** 2 + (1.0 + 2.0 * corr_w_x_2 ** 2) * varnce_x_2) * varnce_w_2)

    return mixt_frac * term1 + one_minus_mf * term2


def calc_wp4_pdf_jax(wm, w_1, w_2, varnce_w_1, varnce_w_2, mixt_frac):
    """pdf_utilities.F90:calc_wp4_pdf — <w'^4> from marginal PDF integral.

    Formula:
      wp4 = mf * (3*varnce_w_1^2 + 6*(w_1-wm)^2*varnce_w_1 + (w_1-wm)^4)
           + (1-mf) * (3*varnce_w_2^2 + 6*(w_2-wm)^2*varnce_w_2 + (w_2-wm)^4)
    """
    one_minus_mf = 1.0 - mixt_frac
    dw_1 = w_1 - wm
    dw_2 = w_2 - wm

    term1 = (3.0 * varnce_w_1 ** 2
             + 6.0 * dw_1 ** 2 * varnce_w_1
             + dw_1 ** 4)
    term2 = (3.0 * varnce_w_2 ** 2
             + 6.0 * dw_2 ** 2 * varnce_w_2
             + dw_2 ** 4)

    return mixt_frac * term1 + one_minus_mf * term2


def calc_wpxpyp_pdf_jax(wm, xm, ym, w_1, w_2, x_1, x_2, y_1, y_2,
                         varnce_w_1, varnce_w_2,
                         varnce_x_1, varnce_x_2,
                         varnce_y_1, varnce_y_2,
                         corr_w_x_1, corr_w_x_2,
                         corr_w_y_1, corr_w_y_2,
                         corr_x_y_1, corr_x_y_2,
                         mixt_frac):
    """pdf_closure_module.F90:calc_wpxpyp_pdf — <w'x'y'> from trivariate PDF integral."""
    one_minus_mf = 1.0 - mixt_frac
    dw_1 = w_1 - wm; dw_2 = w_2 - wm
    dx_1 = x_1 - xm; dx_2 = x_2 - xm
    dy_1 = y_1 - ym; dy_2 = y_2 - ym

    term1 = (dw_1 * dx_1 * dy_1
             + corr_x_y_1 * jnp.sqrt(varnce_x_1 * varnce_y_1) * dw_1
             + corr_w_y_1 * jnp.sqrt(varnce_w_1 * varnce_y_1) * dx_1
             + corr_w_x_1 * jnp.sqrt(varnce_w_1 * varnce_x_1) * dy_1)
    term2 = (dw_2 * dx_2 * dy_2
             + corr_x_y_2 * jnp.sqrt(varnce_x_2 * varnce_y_2) * dw_2
             + corr_w_y_2 * jnp.sqrt(varnce_w_2 * varnce_y_2) * dx_2
             + corr_w_x_2 * jnp.sqrt(varnce_w_2 * varnce_x_2) * dy_2)

    return mixt_frac * term1 + one_minus_mf * term2
