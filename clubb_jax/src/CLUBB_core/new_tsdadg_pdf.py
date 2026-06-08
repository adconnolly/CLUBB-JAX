"""JAX port of new_tsdadg_pdf.F90 — the "new TSDADG" PDF closure helpers.

iiPDF_new_tsdadg is an alternative PDF closure (the gated CLUBB config uses ADG1), so this is a completeness
port. calc_L_x_Skx_fnc maps the tunable l_x_1/l_x_2 spread coefficients to the skewness-dependent L_x_1/L_x_2,
swapping the two when Skx·sgn(<w'x'>) < 0. Pure-jnp → differentiable.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def calc_L_x_Skx_fnc(Skx, sgn_wpxp, small_l_x_1, small_l_x_2):
    """Skewness-dependent spread parameters L_x_1, L_x_2 (new_tsdadg_pdf.F90:calc_L_x_Skx_fnc):
      factor = |Skx| / sqrt(4 + Skx²);
      Skx·sgn(<w'x'>) >= 0:  L_x_1 = l_x_1·factor,  L_x_2 = l_x_2·factor;
      otherwise (swap):       L_x_1 = l_x_2·factor,  L_x_2 = l_x_1·factor.
    Pure-jnp → differentiable. Returns (big_L_x_1, big_L_x_2)."""
    Skx = jnp.asarray(Skx, dtype=jnp.float64)
    sgn = jnp.asarray(sgn_wpxp, dtype=jnp.float64)
    l1 = jnp.asarray(small_l_x_1, dtype=jnp.float64)
    l2 = jnp.asarray(small_l_x_2, dtype=jnp.float64)

    factor = jnp.abs(Skx) / jnp.sqrt(4.0 + Skx ** 2)
    cond = Skx * sgn >= 0.0
    big_L_x_1 = jnp.where(cond, l1, l2) * factor
    big_L_x_2 = jnp.where(cond, l2, l1) * factor
    return big_L_x_1, big_L_x_2


from clubb_jax.src.CLUBB_core.constants_clubb import eps as _EPS  # noqa: E402


# grad-safe sqrt(max(x,0)) — the canonical tracer-toolkit helper.
from clubb_jax.src.CLUBB_core.tracer_numpy import _safe_sqrt as _ssqrt


def calc_setter_parameters(xm, xp2, Skx, sgn_wpxp, big_L_x_1, big_L_x_2):
    """PDF component means/variances and mixture fraction for the setting variable in the new-TSDADG closure
    (new_tsdadg_pdf.F90:calc_setter_parameters). factor± = 1 ± Skx·sgn/√(4+Skx²); the normalized component means
    are big_L_x_1·√(f₊/f₋)·sgn and −big_L_x_2·√(f₋/f₊)·sgn; mixt_frac is a 4-branch ratio of their magnitudes;
    the component variance coefficients follow from the skewness constraint (with a thresholded mu_x_1_nrmlized
    denominator). Pure-jnp (guarded) → differentiable. Returns
    (mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd, mixt_frac, coef_sigma_x_1_sqd, coef_sigma_x_2_sqd)."""
    xm = jnp.asarray(xm, dtype=jnp.float64); xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    Skx = jnp.asarray(Skx, dtype=jnp.float64); sgn = jnp.asarray(sgn_wpxp, dtype=jnp.float64)
    L1 = jnp.asarray(big_L_x_1, dtype=jnp.float64); L2 = jnp.asarray(big_L_x_2, dtype=jnp.float64)

    t = Skx * sgn / jnp.sqrt(4.0 + Skx ** 2)
    factor_plus = 1.0 + t
    factor_minus = 1.0 - t
    mu1n = L1 * jnp.sqrt(factor_plus / factor_minus) * sgn
    mu2n = -L2 * jnp.sqrt(factor_minus / factor_plus) * sgn
    sqrt_xp2 = _ssqrt(xp2)
    mu_x_1 = xm + mu1n * sqrt_xp2
    mu_x_2 = xm + mu2n * sqrt_xp2

    a1 = jnp.abs(mu1n) >= _EPS
    a2 = jnp.abs(mu2n) >= _EPS
    mu2n_safe = jnp.where(a2, mu2n, 1.0)
    mf_both = 1.0 / (1.0 + jnp.abs(mu1n / mu2n_safe))
    mf_1 = 1.0 / (1.0 + jnp.abs(mu1n / _EPS))
    mf_2 = 1.0 / (1.0 + jnp.abs(_EPS / mu2n_safe))
    mixt_frac = jnp.where(a1 & a2, mf_both,
                          jnp.where(a1 & (~a2), mf_1,
                                    jnp.where((~a1) & a2, mf_2, 0.5)))

    mu1n_thresh = jnp.where(mu1n >= 0.0, jnp.maximum(mu1n, _EPS), jnp.minimum(mu1n, -_EPS))
    common = Skx / (3.0 * mixt_frac * mu1n_thresh) - mu1n ** 2 / 3.0 + mu2n ** 2 / 3.0
    base = 1.0 - mixt_frac * mu1n ** 2 - (1.0 - mixt_frac) * mu2n ** 2
    coef1 = base + (1.0 - mixt_frac) * common
    coef2 = base - mixt_frac * common
    return mu_x_1, mu_x_2, coef1 * xp2, coef2 * xp2, mixt_frac, coef1, coef2


def calc_respnder_parameters(xm, xp2, Skx, sgn_wpxp, mixt_frac, big_L_x_1):
    """new_tsdadg_pdf.F90:calc_respnder_parameters (the misspelling "respnder" is deliberate — it mirrors the
    Fortran subroutine name exactly, like the preserved "derrived" typo in parameters_tunable).
    PDF component means/variances for a variable responding to the new-TSDADG PDF set by another variable
    (new_tsdadg_pdf.F90:calc_respnder_parameters [sic]; takes the setter's mixt_frac). Same factor±/normalized-
    mean-1 and variance-coefficient formulas as `calc_setter_parameters`, but the 2nd component normalized mean
    follows the overall-mean constraint mu_x_2_nrmlized = −(mf/(1−mf))·mu_x_1_nrmlized. Pure-jnp → differentiable.
    Returns (mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd, coef_sigma_x_1_sqd, coef_sigma_x_2_sqd)."""
    xm = jnp.asarray(xm, dtype=jnp.float64); xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    Skx = jnp.asarray(Skx, dtype=jnp.float64); sgn = jnp.asarray(sgn_wpxp, dtype=jnp.float64)
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64); L1 = jnp.asarray(big_L_x_1, dtype=jnp.float64)

    t = Skx * sgn / jnp.sqrt(4.0 + Skx ** 2)
    mu1n = L1 * jnp.sqrt((1.0 + t) / (1.0 - t)) * sgn
    mu2n = -(mf / (1.0 - mf)) * mu1n
    sqrt_xp2 = _ssqrt(xp2)
    mu_x_1 = xm + mu1n * sqrt_xp2
    mu_x_2 = xm + mu2n * sqrt_xp2

    mu1n_thresh = jnp.where(mu1n >= 0.0, jnp.maximum(mu1n, _EPS), jnp.minimum(mu1n, -_EPS))
    common = Skx / (3.0 * mf * mu1n_thresh) - mu1n ** 2 / 3.0 + mu2n ** 2 / 3.0
    base = 1.0 - mf * mu1n ** 2 - (1.0 - mf) * mu2n ** 2
    coef1 = base + (1.0 - mf) * common
    coef2 = base - mf * common
    return mu_x_1, mu_x_2, coef1 * xp2, coef2 * xp2, coef1, coef2


def tsdadg_pdf_driver(wm, rtm, thlm, wp2, rtp2, thlp2, Skw, Skrt, Skthl, wprtp, wpthlp):
    """New-TSDADG PDF driver (new_tsdadg_pdf.F90:tsdadg_pdf_driver). The variable with the greatest |skewness|
    among w/rt/thl SETS the mixture fraction (calc_setter_parameters); the other two are RESPONDERS
    (calc_respnder_parameters) using that mixt_frac. Tunable spread params l_x_1=0.75, l_x_2=0.5 for all three.
    Negative PDF-component variances are clipped to 0. Pure-jnp (vectorized 3-way select) → differentiable.
    Returns (mu_w_1, mu_w_2, mu_rt_1, mu_rt_2, mu_thl_1, mu_thl_2, sigma_w_1_sqd, sigma_w_2_sqd,
             sigma_rt_1_sqd, sigma_rt_2_sqd, sigma_thl_1_sqd, sigma_thl_2_sqd, mixt_frac)."""
    Skw = jnp.asarray(Skw, dtype=jnp.float64); Skrt = jnp.asarray(Skrt, dtype=jnp.float64)
    Skthl = jnp.asarray(Skthl, dtype=jnp.float64)
    wprtp = jnp.asarray(wprtp, dtype=jnp.float64); wpthlp = jnp.asarray(wpthlp, dtype=jnp.float64)

    sgn_wprtp = jnp.where(wprtp >= 0.0, 1.0, -1.0)
    sgn_wpthlp = jnp.where(wpthlp >= 0.0, 1.0, -1.0)
    sgn_wp2 = jnp.ones_like(Skw)
    l1, l2 = 0.75, 0.5

    bLw1, bLw2 = calc_L_x_Skx_fnc(Skw, sgn_wp2, l1, l2)
    bLrt1, bLrt2 = calc_L_x_Skx_fnc(Skrt, sgn_wprtp, l1, l2)
    bLthl1, bLthl2 = calc_L_x_Skx_fnc(Skthl, sgn_wpthlp, l1, l2)

    aw, art, athl = jnp.abs(Skw), jnp.abs(Skrt), jnp.abs(Skthl)
    w_set = (aw >= art) & (aw >= athl)
    rt_set = (art > aw) & (art >= athl)
    thl_set = (~w_set) & (~rt_set)

    sw = calc_setter_parameters(wm, wp2, Skw, sgn_wp2, bLw1, bLw2)
    srt = calc_setter_parameters(rtm, rtp2, Skrt, sgn_wprtp, bLrt1, bLrt2)
    sthl = calc_setter_parameters(thlm, thlp2, Skthl, sgn_wpthlp, bLthl1, bLthl2)
    mixt_frac = jnp.where(w_set, sw[4], jnp.where(rt_set, srt[4], sthl[4]))

    rw = calc_respnder_parameters(wm, wp2, Skw, sgn_wp2, mixt_frac, bLw1)
    rrt = calc_respnder_parameters(rtm, rtp2, Skrt, sgn_wprtp, mixt_frac, bLrt1)
    rthl = calc_respnder_parameters(thlm, thlp2, Skthl, sgn_wpthlp, mixt_frac, bLthl1)

    def _sel(mask, setter, responder, i):
        return jnp.where(mask, setter[i], responder[i])

    mu_w_1 = _sel(w_set, sw, rw, 0); mu_w_2 = _sel(w_set, sw, rw, 1)
    sig_w_1 = _sel(w_set, sw, rw, 2); sig_w_2 = _sel(w_set, sw, rw, 3)
    mu_rt_1 = _sel(rt_set, srt, rrt, 0); mu_rt_2 = _sel(rt_set, srt, rrt, 1)
    sig_rt_1 = _sel(rt_set, srt, rrt, 2); sig_rt_2 = _sel(rt_set, srt, rrt, 3)
    mu_thl_1 = _sel(thl_set, sthl, rthl, 0); mu_thl_2 = _sel(thl_set, sthl, rthl, 1)
    sig_thl_1 = _sel(thl_set, sthl, rthl, 2); sig_thl_2 = _sel(thl_set, sthl, rthl, 3)

    clip = lambda s: jnp.where(s < 0.0, 0.0, s)
    return (mu_w_1, mu_w_2, mu_rt_1, mu_rt_2, mu_thl_1, mu_thl_2,
            clip(sig_w_1), clip(sig_w_2), clip(sig_rt_1), clip(sig_rt_2),
            clip(sig_thl_1), clip(sig_thl_2), mixt_frac)
