"""JAX implementation of ADG1 PDF driver (adg1_adg2_3d_luhar_pdf.F90).

Implements:
  ADG1_w_closure_jax             - mixture fraction and w PDF component params
  ADG1_ADG2_responder_params_jax - PDF component params for rt/thl/u/v/sclr
  ADG1_pdf_driver_jax            - top-level ADG1 PDF parameter driver
  calc_comp_corrs_binormal_jax   - PDF component correlations (pdf_utilities.F90)
"""

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.tracer_numpy import _safe_sqrt  # REFACTOR B5: finite grad for sqrt(max(0,·))
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
    # _safe_sqrt (REFACTOR B5): sigma_factor = 1-sigma_sqd_w -> 0 in well-mixed/surface layers, so the bare
    # sqrt has an inf reverse-mode gradient there (the bomex thlm surface nan). Forward-identical (arg >= 0).
    w_1_n = _safe_sqrt(one_minus_mf / mixt_frac * sigma_factor)
    w_2_n = -_safe_sqrt(mixt_frac / one_minus_mf * sigma_factor)

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
    denom = (mixt_frac * _safe_sqrt(sigma_x_1_sqd * sigma_y_1_sqd)
             + one_minus_mf * _safe_sqrt(sigma_x_2_sqd * sigma_y_2_sqd))

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
             + 2.0 * corr_w_x_1 * _safe_sqrt(varnce_w_1 * varnce_x_1) * dw_1)
    term2 = ((dw_2 ** 2 + varnce_w_2) * dx_2
             + 2.0 * corr_w_x_2 * _safe_sqrt(varnce_w_2 * varnce_x_2) * dw_2)

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
             + 2.0 * corr_w_x_1 * _safe_sqrt(varnce_w_1 * varnce_x_1) * dx_1)
    term2 = (dw_2 * (dx_2 ** 2 + varnce_x_2)
             + 2.0 * corr_w_x_2 * _safe_sqrt(varnce_w_2 * varnce_x_2) * dx_2)

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
             + 4.0 * corr_w_x_1 * _safe_sqrt(varnce_w_1 * varnce_x_1) * dx_1 * dw_1
             + (dx_1 ** 2 + (1.0 + 2.0 * corr_w_x_1 ** 2) * varnce_x_1) * varnce_w_1)
    term2 = (dw_2 ** 2 * (dx_2 ** 2 + varnce_x_2)
             + 4.0 * corr_w_x_2 * _safe_sqrt(varnce_w_2 * varnce_x_2) * dx_2 * dw_2
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
             + corr_x_y_1 * _safe_sqrt(varnce_x_1 * varnce_y_1) * dw_1
             + corr_w_y_1 * _safe_sqrt(varnce_w_1 * varnce_y_1) * dx_1
             + corr_w_x_1 * _safe_sqrt(varnce_w_1 * varnce_x_1) * dy_1)
    term2 = (dw_2 * dx_2 * dy_2
             + corr_x_y_2 * _safe_sqrt(varnce_x_2 * varnce_y_2) * dw_2
             + corr_w_y_2 * _safe_sqrt(varnce_w_2 * varnce_y_2) * dx_2
             + corr_w_x_2 * _safe_sqrt(varnce_w_2 * varnce_x_2) * dy_2)

    return mixt_frac * term1 + one_minus_mf * term2


def calc_w_up_in_cloud(mixt_frac, cloud_frac_1, cloud_frac_2,
                       w_1, w_2, varnce_w_1, varnce_w_2):
    """Mean cloudy updraft / downdraft vertical velocity from the binormal w-PDF
    (pdf_closure_module.F90:calc_w_up_in_cloud). For aerosol activation, this gives a w representative of
    cloudy updrafts (an alternative to sqrt(wp2)). Per PDF component, the truncated-Gaussian updraft integral is
      w_up = 1/2 w (1+erf(r)) + sigma/sqrt(2pi) exp(-r^2),  r = w/(sqrt(2) sigma),  updraft_frac = 1/2(1+erf(r)),
    with all-updraft / all-downdraft shortcuts when |w| > max_num_stdevs*sigma. The cloudy means weight the two
    components by mixt_frac*cloud_frac. Returns
    (w_up_in_cloud, w_down_in_cloud, cloudy_updraft_frac, cloudy_downdraft_frac). Pure-jnp → differentiable.

    All inputs are (ngrdcol, nz). varnce_w_* are variances (sigma^2)."""
    import jax.scipy.special as jsp
    from clubb_jax.src.CLUBB_core.constants_clubb import sqrt_2, sqrt_2pi, max_num_stdevs

    def _component(w, varnce):
        w = jnp.asarray(w, dtype=jnp.float64)
        stdev = jnp.sqrt(jnp.asarray(varnce, dtype=jnp.float64))
        all_up = w > max_num_stdevs * stdev
        all_down = w < -max_num_stdevs * stdev
        ratio = w / (sqrt_2 * jnp.maximum(eps, stdev))
        erf_r = jsp.erf(ratio)
        exp_neg = jnp.exp(-ratio ** 2)
        w_up_mid = 0.5 * w * (1.0 + erf_r) + (stdev / sqrt_2pi) * exp_neg
        uf_mid = 0.5 * (1.0 + erf_r)
        w_down_mid = 0.5 * w * (1.0 - erf_r) - (stdev / sqrt_2pi) * exp_neg
        w_up = jnp.where(all_up, w, jnp.where(all_down, 0.0, w_up_mid))
        uf = jnp.where(all_up, 1.0, jnp.where(all_down, 0.0, uf_mid))
        w_down = jnp.where(all_up, 0.0, jnp.where(all_down, w, w_down_mid))
        df = 1.0 - uf   # holds in all three branches (Fortran: 1, 0, 1-uf_mid)
        return w_up, uf, w_down, df

    a = jnp.asarray(mixt_frac, dtype=jnp.float64)
    cf1 = jnp.asarray(cloud_frac_1, dtype=jnp.float64); cf2 = jnp.asarray(cloud_frac_2, dtype=jnp.float64)
    w_up_1, uf_1, w_down_1, df_1 = _component(w_1, varnce_w_1)
    w_up_2, uf_2, w_down_2, df_2 = _component(w_2, varnce_w_2)

    cloudy_updraft_frac = a * cf1 * uf_1 + (1.0 - a) * cf2 * uf_2
    cloudy_downdraft_frac = a * cf1 * df_1 + (1.0 - a) * cf2 * df_2
    w_up_in_cloud = ((a * cf1 * w_up_1 + (1.0 - a) * cf2 * w_up_2)
                     / jnp.maximum(eps, cloudy_updraft_frac))
    w_down_in_cloud = ((a * cf1 * w_down_1 + (1.0 - a) * cf2 * w_down_2)
                       / jnp.maximum(eps, cloudy_downdraft_frac))
    return w_up_in_cloud, w_down_in_cloud, cloudy_updraft_frac, cloudy_downdraft_frac


def calc_wp4_pdf(wm, w_1, w_2, varnce_w_1, varnce_w_2, mixt_frac):
    """<w'^4> integrated over the two-component-normal PDF of w (pdf_closure_module.F90:calc_wp4_pdf):
      <w'^4> = Σ_i weight_i (3 σ_w_i^4 + 6 (μ_w_i-<w>)^2 σ_w_i^2 + (μ_w_i-<w>)^4). Pure-jnp → differentiable."""
    wm = jnp.asarray(wm); a = jnp.asarray(mixt_frac)
    d1 = jnp.asarray(w_1) - wm; d2 = jnp.asarray(w_2) - wm
    v1 = jnp.asarray(varnce_w_1); v2 = jnp.asarray(varnce_w_2)
    return (a * (3.0 * v1 ** 2 + 6.0 * d1 ** 2 * v1 + d1 ** 4)
            + (1.0 - a) * (3.0 * v2 ** 2 + 6.0 * d2 ** 2 * v2 + d2 ** 4))


def calc_wp2xp_pdf(wm, xm, w_1, w_2, x_1, x_2, varnce_w_1, varnce_w_2,
                   varnce_x_1, varnce_x_2, corr_w_x_1, corr_w_x_2, mixt_frac):
    """<w'^2 x'> integrated over the binormal PDF of (w, x) (pdf_closure_module.F90:calc_wp2xp_pdf):
      Σ_i weight_i [ ((μ_w_i-<w>)^2 + σ_w_i^2)(μ_x_i-<x>) + 2 corr_i σ_w_i σ_x_i (μ_w_i-<w>) ]."""
    wm = jnp.asarray(wm); xm = jnp.asarray(xm); a = jnp.asarray(mixt_frac)
    dw1 = jnp.asarray(w_1) - wm; dw2 = jnp.asarray(w_2) - wm
    dx1 = jnp.asarray(x_1) - xm; dx2 = jnp.asarray(x_2) - xm
    vw1 = jnp.asarray(varnce_w_1); vw2 = jnp.asarray(varnce_w_2)
    vx1 = jnp.asarray(varnce_x_1); vx2 = jnp.asarray(varnce_x_2)
    c1 = jnp.asarray(corr_w_x_1); c2 = jnp.asarray(corr_w_x_2)
    return (a * ((dw1 ** 2 + vw1) * dx1 + 2.0 * c1 * _safe_sqrt(vw1 * vx1) * dw1)
            + (1.0 - a) * ((dw2 ** 2 + vw2) * dx2 + 2.0 * c2 * _safe_sqrt(vw2 * vx2) * dw2))


def calc_wpxp2_pdf(wm, xm, w_1, w_2, x_1, x_2, varnce_w_1, varnce_w_2,
                   varnce_x_1, varnce_x_2, corr_w_x_1, corr_w_x_2, mixt_frac):
    """<w'x'^2> integrated over the binormal PDF of (w, x) (pdf_closure_module.F90:calc_wpxp2_pdf):
      Σ_i weight_i [ (μ_w_i-<w>)((μ_x_i-<x>)^2 + σ_x_i^2) + 2 corr_i σ_w_i σ_x_i (μ_x_i-<x>) ]."""
    wm = jnp.asarray(wm); xm = jnp.asarray(xm); a = jnp.asarray(mixt_frac)
    dw1 = jnp.asarray(w_1) - wm; dw2 = jnp.asarray(w_2) - wm
    dx1 = jnp.asarray(x_1) - xm; dx2 = jnp.asarray(x_2) - xm
    vw1 = jnp.asarray(varnce_w_1); vw2 = jnp.asarray(varnce_w_2)
    vx1 = jnp.asarray(varnce_x_1); vx2 = jnp.asarray(varnce_x_2)
    c1 = jnp.asarray(corr_w_x_1); c2 = jnp.asarray(corr_w_x_2)
    return (a * (dw1 * (dx1 ** 2 + vx1) + 2.0 * c1 * _safe_sqrt(vw1 * vx1) * dx1)
            + (1.0 - a) * (dw2 * (dx2 ** 2 + vx2) + 2.0 * c2 * _safe_sqrt(vw2 * vx2) * dx2))


def calc_wpxpyp_pdf(wm, xm, ym, w_1, w_2, x_1, x_2, y_1, y_2,
                    varnce_w_1, varnce_w_2, varnce_x_1, varnce_x_2, varnce_y_1, varnce_y_2,
                    corr_w_x_1, corr_w_x_2, corr_w_y_1, corr_w_y_2, corr_x_y_1, corr_x_y_2, mixt_frac):
    """<w'x'y'> integrated over the trinormal PDF of (w, x, y) (pdf_closure_module.F90:calc_wpxpyp_pdf):
      Σ_i weight_i [ (μ_w-<w>)(μ_x-<x>)(μ_y-<y>) + corr_xy σ_x σ_y (μ_w-<w>)
                     + corr_wy σ_w σ_y (μ_x-<x>) + corr_wx σ_w σ_x (μ_y-<y>) ]_i."""
    wm = jnp.asarray(wm); xm = jnp.asarray(xm); ym = jnp.asarray(ym); a = jnp.asarray(mixt_frac)
    dw1 = jnp.asarray(w_1) - wm; dw2 = jnp.asarray(w_2) - wm
    dx1 = jnp.asarray(x_1) - xm; dx2 = jnp.asarray(x_2) - xm
    dy1 = jnp.asarray(y_1) - ym; dy2 = jnp.asarray(y_2) - ym
    vw1 = jnp.asarray(varnce_w_1); vw2 = jnp.asarray(varnce_w_2)
    vx1 = jnp.asarray(varnce_x_1); vx2 = jnp.asarray(varnce_x_2)
    vy1 = jnp.asarray(varnce_y_1); vy2 = jnp.asarray(varnce_y_2)
    cwx1 = jnp.asarray(corr_w_x_1); cwx2 = jnp.asarray(corr_w_x_2)
    cwy1 = jnp.asarray(corr_w_y_1); cwy2 = jnp.asarray(corr_w_y_2)
    cxy1 = jnp.asarray(corr_x_y_1); cxy2 = jnp.asarray(corr_x_y_2)
    comp1 = (dw1 * dx1 * dy1 + cxy1 * _safe_sqrt(vx1 * vy1) * dw1
             + cwy1 * _safe_sqrt(vw1 * vy1) * dx1 + cwx1 * _safe_sqrt(vw1 * vx1) * dy1)
    comp2 = (dw2 * dx2 * dy2 + cxy2 * _safe_sqrt(vx2 * vy2) * dw2
             + cwy2 * _safe_sqrt(vw2 * vy2) * dx2 + cwx2 * _safe_sqrt(vw2 * vx2) * dy2)
    return a * comp1 + (1.0 - a) * comp2


def calc_Luhar_params(Skx, wpxp, xp2, x_tol_sqd):
    """Luhar PDF parameters mixt_frac, big_m (M), small_m (m) from skewness and the w-x covariance
    (adg1_adg2_3d_luhar_pdf.F90:calc_Luhar_params; Larson, Golaz & Cotton 2002). Where xp2 > x_tol_sqd:
      m = max(2/3 |Skx|^(1/3), 0.05);  M = (1+m²)³ / ((3+m²)² m²);
      mixt_frac = 1/2 (1 − sgn(wpxp)·Skx·√(1/((4/M) + Skx²))).
    Otherwise (x effectively constant) mixt_frac = 1/2, m = M = 0. Pure-jnp (finite-grad cube root) →
    differentiable. Returns (mixt_frac, big_m, small_m)."""
    Skx = jnp.asarray(Skx, dtype=jnp.float64)
    wpxp = jnp.asarray(wpxp, dtype=jnp.float64)
    xp2 = jnp.asarray(xp2, dtype=jnp.float64)

    # Guarded cube root (cbrt'(0)=inf otherwise; masked by the max floor but would poison grad as 0*inf).
    absS = jnp.abs(Skx)
    cbrt = jnp.where(absS > 0.0, jnp.cbrt(jnp.where(absS > 0.0, absS, 1.0)), 0.0)
    small_m = jnp.maximum((2.0 / 3.0) * cbrt, 0.05)
    m_sqd = small_m ** 2
    big_m = (1.0 + m_sqd) ** 3 / ((3.0 + m_sqd) ** 2 * m_sqd)

    sgn = jnp.where(wpxp >= 0.0, 1.0, -1.0)
    mixt_frac_vary = 0.5 * (1.0 - sgn * Skx * jnp.sqrt(1.0 / ((4.0 / big_m) + Skx ** 2)))

    vary = xp2 > x_tol_sqd
    mixt_frac = jnp.where(vary, mixt_frac_vary, 0.5)
    big_m = jnp.where(vary, big_m, 0.0)
    small_m = jnp.where(vary, small_m, 0.0)
    return mixt_frac, big_m, small_m


def close_Luhar_pdf(xm, xp2, mixt_frac, small_m, wpxp, x_tol_sqd):
    """PDF component normalized widths/means/variances for the Luhar closure (adg1_adg2_3d_luhar_pdf.F90:
    close_Luhar_pdf). Where xp2 > x_tol_sqd:
      sigma_sqd_x_1 = (1−mf)/(mf(1+m²)),  sigma_sqd_x_2 = mf/((1−mf)(1+m²)),  varnce_x_i = sigma_sqd_x_i·xp2,
      x_i_n = ±sgn(wpxp)·m·√(sigma_sqd_x_i),  x_i = xm + √xp2·x_i_n.
    Where x is effectively constant, sigma_sqd_x_i = 1/(1+m²), varnce_x_i = 0, x_i = xm (sgn computed normally
    here — the Fortran leaves it uninitialized in that degenerate branch). Pure-jnp → differentiable. Returns
    (sigma_sqd_x_1, sigma_sqd_x_2, varnce_x_1, varnce_x_2, x_1_n, x_2_n, x_1, x_2)."""
    xm = jnp.asarray(xm, dtype=jnp.float64); xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64); m = jnp.asarray(small_m, dtype=jnp.float64)
    wpxp = jnp.asarray(wpxp, dtype=jnp.float64)
    omf = 1.0 - mf
    msq = 1.0 + m ** 2
    sgn = jnp.where(wpxp >= 0.0, 1.0, -1.0)

    vary = xp2 > x_tol_sqd
    sigma_sqd_x_1 = jnp.where(vary, omf / (mf * msq), 1.0 / msq)
    sigma_sqd_x_2 = jnp.where(vary, mf / (omf * msq), 1.0 / msq)
    varnce_x_1 = jnp.where(vary, sigma_sqd_x_1 * xp2, 0.0)
    varnce_x_2 = jnp.where(vary, sigma_sqd_x_2 * xp2, 0.0)
    x_1_n = sgn * m * _safe_sqrt(sigma_sqd_x_1)
    x_2_n = -sgn * m * _safe_sqrt(sigma_sqd_x_2)
    sqrt_xp2 = _safe_sqrt(xp2)
    x_1 = jnp.where(vary, xm + sqrt_xp2 * x_1_n, xm)
    x_2 = jnp.where(vary, xm + sqrt_xp2 * x_2_n, xm)
    return sigma_sqd_x_1, sigma_sqd_x_2, varnce_x_1, varnce_x_2, x_1_n, x_2_n, x_1, x_2


def max_cubic_root(a_coef, b_coef, c_coef, d_coef):
    """Largest real root of a·x³ + b·x² + c·x + d = 0 (adg1_adg2_3d_luhar_pdf.F90:max_cubic_root). Falls back to
    the quadratic solver when |a| is negligible vs the other coefficients, and to the linear solution when |a|
    and |b| are both negligible. For a true cubic: if the 2nd/3rd roots are real (negligible imaginary part) the
    max over all three real parts, else the always-real first root. Pure-jnp (uses the iter-71-corrected
    cubic_solve) → differentiable away from the root-multiplicity branch points. Inputs broadcast elementwise."""
    from clubb_jax.src.CLUBB_core.calc_roots import cubic_solve, quadratic_solve
    from clubb_jax.src.CLUBB_core.constants_clubb import eps
    a = jnp.asarray(a_coef, dtype=jnp.float64); b = jnp.asarray(b_coef, dtype=jnp.float64)
    c = jnp.asarray(c_coef, dtype=jnp.float64); d = jnp.asarray(d_coef, dtype=jnp.float64)

    a_thresh = 0.001 * jnp.maximum(jnp.maximum(jnp.abs(b), jnp.abs(c)), jnp.abs(d))
    b_thresh = 0.001 * jnp.maximum(jnp.abs(c), jnp.abs(d))

    cr = cubic_solve(a, b, c, d)                      # (..., 3) complex
    r0, r1, r2 = jnp.real(cr[..., 0]), jnp.real(cr[..., 1]), jnp.real(cr[..., 2])
    is_3real = (jnp.abs(jnp.imag(cr[..., 1])) < eps) & (jnp.abs(jnp.imag(cr[..., 2])) < eps)
    cubic_max = jnp.where(is_3real, jnp.maximum(jnp.maximum(r0, r1), r2), r0)

    qr = quadratic_solve(b, c, d)                     # (..., 2) complex
    quad_max = jnp.maximum(jnp.real(qr[..., 0]), jnp.real(qr[..., 1]))

    c_safe = jnp.where(c != 0.0, c, 1.0)
    linear = -d / c_safe

    return jnp.where(jnp.abs(a) > a_thresh, cubic_max,
                     jnp.where(jnp.abs(b) > b_thresh, quad_max, linear))


def backsolve_Luhar_params(Sk_max, Skx, big_m_max, mixt_frac):
    """Backsolve Luhar's m (and M) for a responding variable given the setter's mixt_frac and the variable's
    skewness (adg1_adg2_3d_luhar_pdf.F90:backsolve_Luhar_params, cubic branch — l_use_cubic_backsolve is
    hardcoded true; Sk_max/big_m_max are then unused). When mixt_frac≈0.5 → m=0.05; when |Skx|<eps → m=0;
    otherwise alpha = mf(1−mf)/(1−2mf)²·Skx², and m² is the largest real root of
    (alpha−1)(m²)³ + (3alpha−6)(m²)² + (3alpha−9)(m²) + alpha = 0 (floored at 0.05²). Pure-jnp → differentiable.
    Returns (big_m_x, small_m_x)."""
    from clubb_jax.src.CLUBB_core.constants_clubb import eps
    Skx = jnp.asarray(Skx, dtype=jnp.float64)
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64)

    near_half = jnp.abs(mf - 0.5) < 0.001
    skx_zero = jnp.abs(Skx) < eps

    m_nh = 0.05
    bigm_nh = (1.0 + m_nh ** 2) ** 3 / ((3.0 + m_nh ** 2) ** 2 * m_nh ** 2)

    denom = jnp.where(near_half, 1.0, (1.0 - 2.0 * mf) ** 2)        # guard the 1/(1-2mf)^2 at mf=0.5
    alpha = mf * (1.0 - mf) / denom * Skx ** 2
    alpha_safe = jnp.where(alpha > 0.0, alpha, 1.0)
    bigm_else = 1.0 / alpha_safe
    m2 = max_cubic_root(alpha - 1.0, 3.0 * alpha - 6.0, 3.0 * alpha - 9.0, alpha)
    m_else = jnp.sqrt(jnp.maximum(m2, 0.05 ** 2))

    small_m_x = jnp.where(near_half, m_nh, jnp.where(skx_zero, 0.0, m_else))
    big_m_x = jnp.where(near_half, bigm_nh, jnp.where(skx_zero, jnp.inf, bigm_else))
    return big_m_x, small_m_x


def Luhar_3D_pdf_driver(wm, rtm, thlm, wp2, rtp2, thlp2, Skw, Skrt, Skthl, wprtp, wpthlp):
    """3-D Luhar PDF driver (adg1_adg2_3d_luhar_pdf.F90:Luhar_3D_pdf_driver). The variable with the greatest
    |skewness| sets the PDF (calc_Luhar_params → mixt_frac); the other two backsolve their m from the setter's
    mixt_frac (backsolve_Luhar_params); each variable's PDF component means/variances follow from close_Luhar_pdf.
    Pure-jnp (vectorized 3-way setter select) → differentiable. Returns (w_1, w_2, rt_1, rt_2, thl_1, thl_2,
    varnce_w_1, varnce_w_2, varnce_rt_1, varnce_rt_2, varnce_thl_1, varnce_thl_2, mixt_frac)."""
    Skw = jnp.asarray(Skw, dtype=jnp.float64); Skrt = jnp.asarray(Skrt, dtype=jnp.float64)
    Skthl = jnp.asarray(Skthl, dtype=jnp.float64)
    wp2 = jnp.asarray(wp2, dtype=jnp.float64); rtp2 = jnp.asarray(rtp2, dtype=jnp.float64); thlp2 = jnp.asarray(thlp2, dtype=jnp.float64)
    wprtp = jnp.asarray(wprtp, dtype=jnp.float64); wpthlp = jnp.asarray(wpthlp, dtype=jnp.float64)
    w_tsq, rt_tsq, thl_tsq = w_tol_sqd, rt_tol ** 2, thl_tol ** 2

    # calc_Luhar_params for each variable as a potential setter (w uses wp2 as the covariance -> sgn = +1).
    mf_w, bigm_w, m_w = calc_Luhar_params(Skw, wp2, wp2, w_tsq)
    mf_rt, bigm_rt, m_rt = calc_Luhar_params(Skrt, wprtp, rtp2, rt_tsq)
    mf_thl, bigm_thl, m_thl = calc_Luhar_params(Skthl, wpthlp, thlp2, thl_tsq)

    aw, art, athl = jnp.abs(Skw), jnp.abs(Skrt), jnp.abs(Skthl)
    w_set = (aw >= athl) & (aw >= art)
    thl_set = (~w_set) & (athl > aw) & (athl >= art)
    rt_set = (~w_set) & (~thl_set)

    mixt_frac = jnp.where(w_set, mf_w, jnp.where(thl_set, mf_thl, mf_rt))
    setter_Sk = jnp.where(w_set, Skw, jnp.where(thl_set, Skthl, Skrt))
    setter_bigm = jnp.where(w_set, bigm_w, jnp.where(thl_set, bigm_thl, bigm_rt))

    # Each variable: its own m if it is the setter, else backsolve from the setter.
    _, bm_w_resp = backsolve_Luhar_params(setter_Sk, Skw, setter_bigm, mixt_frac)
    _, bm_rt_resp = backsolve_Luhar_params(setter_Sk, Skrt, setter_bigm, mixt_frac)
    _, bm_thl_resp = backsolve_Luhar_params(setter_Sk, Skthl, setter_bigm, mixt_frac)
    m_w_f = jnp.where(w_set, m_w, bm_w_resp)
    m_rt_f = jnp.where(rt_set, m_rt, bm_rt_resp)
    m_thl_f = jnp.where(thl_set, m_thl, bm_thl_resp)

    _, _, vw1, vw2, _, _, w1, w2 = close_Luhar_pdf(wm, wp2, mixt_frac, m_w_f, wp2, w_tsq)
    _, _, vrt1, vrt2, _, _, rt1, rt2 = close_Luhar_pdf(rtm, rtp2, mixt_frac, m_rt_f, wprtp, rt_tsq)
    _, _, vthl1, vthl2, _, _, thl1, thl2 = close_Luhar_pdf(thlm, thlp2, mixt_frac, m_thl_f, wpthlp, thl_tsq)

    return (w1, w2, rt1, rt2, thl1, thl2, vw1, vw2, vrt1, vrt2, vthl1, vthl2, mixt_frac)


def ADG2_pdf_driver(wm, rtm, thlm, wp2, rtp2, thlp2, Skw, wprtp, wpthlp, sqrt_wp2, beta):
    """ADG2 PDF driver (adg1_adg2_3d_luhar_pdf.F90:ADG2_pdf_driver), PDF-parameter outputs for w/rt/thl
    (the passive-scalar branch is omitted; the gated config uses sclr_dim=0). w's PDF comes from the Luhar
    closure (calc_Luhar_params + close_Luhar_pdf, with sigma_sqd_w overwritten to min(1/(1+m²), 0.99) for ADG1
    consistency); rt and thl are responders via ADG1_ADG2_responder_params_jax. Pure-jnp → differentiable.
    Returns (w_1, w_2, rt_1, rt_2, thl_1, thl_2, varnce_w_1, varnce_w_2, varnce_rt_1, varnce_rt_2,
    varnce_thl_1, varnce_thl_2, mixt_frac, alpha_rt, alpha_thl, sigma_sqd_w)."""
    wm = jnp.asarray(wm, dtype=jnp.float64); wp2 = jnp.asarray(wp2, dtype=jnp.float64)
    Skw = jnp.asarray(Skw, dtype=jnp.float64); sqrt_wp2 = jnp.asarray(sqrt_wp2, dtype=jnp.float64)
    w_tsq = w_tol_sqd

    mixt_frac, _, small_m_w = calc_Luhar_params(Skw, wp2, wp2, w_tsq)
    _, _, vw1, vw2, w_1_n, w_2_n, w1, w2 = close_Luhar_pdf(wm, wp2, mixt_frac, small_m_w, wp2, w_tsq)
    sigma_sqd_w = jnp.minimum(1.0 / (1.0 + small_m_w ** 2), 0.99)

    rt1, rt2, vrt1, vrt2, alpha_rt = ADG1_ADG2_responder_params_jax(
        rtm, rtp2, wp2, sqrt_wp2, wprtp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)
    thl1, thl2, vthl1, vthl2, alpha_thl = ADG1_ADG2_responder_params_jax(
        thlm, thlp2, wp2, sqrt_wp2, wpthlp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)

    return (w1, w2, rt1, rt2, thl1, thl2, vw1, vw2, vrt1, vrt2, vthl1, vthl2,
            mixt_frac, alpha_rt, alpha_thl, sigma_sqd_w)
