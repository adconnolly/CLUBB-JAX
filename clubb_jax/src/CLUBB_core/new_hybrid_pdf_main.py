"""New-hybrid PDF driver — port of new_hybrid_pdf_main.F90 (Griffin & Larson 2020).

The new-hybrid (iiPDF_new_hybrid) two-component closure. w is always the *setter* variable (it sets the
mixture fraction); rt, thl, u, v, and passive scalars are *responders*. This module ports:

  * calc_F_w_zeta_w        — PDF spread/variance parameters F_w, zeta_w for w, from Skw, the flux-based
                             correlation ceiling, and the tunable gamma / stdev-factor.
  * calc_responder_driver  — clips a responder's skewness Skx to the representable [min_Skx, max_Skx] range,
                             then builds its component params via calculate_responder_params.
  * new_hybrid_pdf_driver  — the full orchestration returning all w/rt/thl/u/v/sclr PDF component params,
                             the clipped responder skewnesses, the mixture fraction, and sigma_sqd_w = 1 - F_w.

The leaf component-parameter routines (calculate_w_params / calculate_responder_params / calculate_mixture_
fraction) live in new_pdf.py. This module is an alternative closure — not used by the gated ADG1 config — so it
is a completeness port, validated end-to-end against f2py_new_hybrid_pdf_driver (the 31 exposed PDF-param /
skewness / mixt_frac / sigma_sqd_w outputs; the implicit_coefs_terms derived type is not f2py-exposed and is
not needed by those outputs, so it is omitted). Pure-jnp, nan-safe sqrt and guarded denominators → differentiable.

Reference: clubb_release/src/CLUBB_core/new_hybrid_pdf_main.F90
"""
from __future__ import annotations

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.new_pdf import (
    _ssqrt, calculate_w_params, calculate_responder_params)

# constants_clubb.F90
_MAX_MAG_CORRELATION_FLUX = 0.99    # max_mag_correlation_flux
_LAMBDA = 0.5                       # calc_F_w_zeta_w lambda (unused: only the commented-out F_w form needs it)


def calc_F_w_zeta_w(Skw, wprtp, wpthlp, upwp, vpwp, wp2, rtp2, thlp2, up2, vp2,
                    gamma_Skw_fnc, slope_coef_spread_DG_means_w,
                    pdf_component_stdev_factor_w, max_corr_w_sclr_sqd):
    """new_hybrid_pdf_main.F90:calc_F_w_zeta_w — F_w and zeta_w for the w-setter.

    F_w is bounded below by the largest squared flux-correlation max(<w'x'>²/(<w'^2><x'^2>)) over the
    responders (capped at max_mag_correlation_flux²), floored at 1e-3 when |Skw|>0, and above by 1:
      min_F_w = max(max_corr_w_x_sqd, 1e-3 if |Skw|>0 else 0);  max_F_w = 1;
      F_w = max_F_w − gamma·(max_F_w − min_F_w)   (the ADG1-like gamma form actually used);
      zeta_w* = pdf_component_stdev_factor_w − 1;
      zeta_w = zeta_w*               if Skw ≥ 0,
             = −zeta_w*/(zeta_w*+1)  if Skw < 0   (mirror-image PDF for negative skewness).
    Pure-jnp (guarded correlation denominators) → differentiable. Returns (F_w, zeta_w, min_F_w, max_F_w)."""
    Skw = jnp.asarray(Skw, dtype=jnp.float64)
    wprtp = jnp.asarray(wprtp, dtype=jnp.float64); wpthlp = jnp.asarray(wpthlp, dtype=jnp.float64)
    upwp = jnp.asarray(upwp, dtype=jnp.float64); vpwp = jnp.asarray(vpwp, dtype=jnp.float64)
    wp2 = jnp.asarray(wp2, dtype=jnp.float64); rtp2 = jnp.asarray(rtp2, dtype=jnp.float64)
    thlp2 = jnp.asarray(thlp2, dtype=jnp.float64); up2 = jnp.asarray(up2, dtype=jnp.float64)
    vp2 = jnp.asarray(vp2, dtype=jnp.float64)
    gamma = jnp.asarray(gamma_Skw_fnc, dtype=jnp.float64)
    pstdev = jnp.asarray(pdf_component_stdev_factor_w, dtype=jnp.float64)
    mcs = jnp.asarray(max_corr_w_sclr_sqd, dtype=jnp.float64)

    def _corr_sqd(cov, va, vb):
        d = va * vb
        return jnp.where(d > 0.0, cov ** 2 / jnp.where(d > 0.0, d, 1.0), 0.0)

    corr_w_rt_sqd = _corr_sqd(wprtp, wp2, rtp2)
    corr_w_thl_sqd = _corr_sqd(wpthlp, wp2, thlp2)
    corr_u_w_sqd = _corr_sqd(upwp, up2, wp2)
    corr_v_w_sqd = _corr_sqd(vpwp, vp2, wp2)

    max_corr_w_x_sqd = jnp.maximum(
        jnp.maximum(jnp.maximum(corr_w_rt_sqd, corr_w_thl_sqd),
                    jnp.maximum(corr_u_w_sqd, corr_v_w_sqd)), mcs)
    max_corr_w_x_sqd = jnp.minimum(max_corr_w_x_sqd, _MAX_MAG_CORRELATION_FLUX ** 2)

    min_F_w = jnp.where(jnp.abs(Skw) > 0.0,
                        jnp.maximum(max_corr_w_x_sqd, 1.0e-3), max_corr_w_x_sqd)
    max_F_w = jnp.ones_like(min_F_w)

    F_w = max_F_w - gamma * (max_F_w - min_F_w)

    zeta_w_star = pstdev - 1.0
    zeta_w = jnp.where(Skw >= 0.0, zeta_w_star, -zeta_w_star / (zeta_w_star + 1.0))
    return F_w, zeta_w, min_F_w, max_F_w


def calc_responder_driver(xm, xp2, wpxp, wp2, mixt_frac, F_w, Skx):
    """new_hybrid_pdf_main.F90:calc_responder_driver — clip a responder's Skx to the representable range,
    then build its PDF component params. The PDF can only represent skewnesses in [min_Skx, max_Skx], with
    (for F_w>0, corr_w_x = <w'x'>/√(<w'^2><x'^2>)):
      A = (1+mf)/√(mf(1−mf))·corr³/F_w^{3/2} − √(mf/(1−mf))·3·corr/√F_w
      B = (mf−2)/√(mf(1−mf))·corr³/F_w^{3/2} + √((1−mf)/mf)·3·corr/√F_w
      (min_Skx, max_Skx) = (A, B) if <w'x'> ≥ 0 else (B, A);  both 0 when F_w = 0.
    Skx is clamped (Fortran order: clip-high then clip-low) and passed to calculate_responder_params.
    Pure-jnp (nan-safe sqrt, guarded F_w/denoms) → differentiable. Returns
    (Skx_clipped, mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd, coef_sigma_x_1_sqd, coef_sigma_x_2_sqd)."""
    xm = jnp.asarray(xm, dtype=jnp.float64); xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    wpxp = jnp.asarray(wpxp, dtype=jnp.float64); wp2 = jnp.asarray(wp2, dtype=jnp.float64)
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64); F_w = jnp.asarray(F_w, dtype=jnp.float64)
    Skx = jnp.asarray(Skx, dtype=jnp.float64)
    omf = 1.0 - mf

    F_pos = F_w > 0.0
    F_safe = jnp.where(F_pos, F_w, 1.0)
    wx = wp2 * xp2
    corr_w_x = jnp.where(wx > 0.0, wpxp / _ssqrt(jnp.where(wx > 0.0, wx, 1.0)), 0.0)

    sqrt_F = _ssqrt(F_safe)
    F_3half = F_safe ** 1.5                                     # F_w**three_halves (gfortran pow)
    base_mfomf = _ssqrt(mf * omf)
    corr3 = corr_w_x * corr_w_x * corr_w_x                      # gfortran expands x**3 to x*x*x
    A = ((1.0 + mf) / base_mfomf * corr3 / F_3half
         - _ssqrt(mf / omf) * 3.0 * corr_w_x / sqrt_F)
    B = ((mf - 2.0) / base_mfomf * corr3 / F_3half
         + _ssqrt(omf / mf) * 3.0 * corr_w_x / sqrt_F)

    wpxp_nonneg = wpxp >= 0.0
    min_Skx = jnp.where(F_pos, jnp.where(wpxp_nonneg, A, B), 0.0)
    max_Skx = jnp.where(F_pos, jnp.where(wpxp_nonneg, B, A), 0.0)

    # Fortran clamp order: if Skx > max -> max; elseif Skx < min -> min.
    Skx_c = jnp.where(Skx > max_Skx, max_Skx, jnp.where(Skx < min_Skx, min_Skx, Skx))

    mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd, c1, c2 = calculate_responder_params(
        xm, xp2, Skx_c, wpxp, wp2, F_w, mf)
    return Skx_c, mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd, c1, c2


def new_hybrid_pdf_driver(wm, rtm, thlm, um, vm, wp2, rtp2, thlp2, up2, vp2,
                          Skw, wprtp, wpthlp, upwp, vpwp,
                          gamma_Skw_fnc, slope_coef_spread_DG_means_w, pdf_component_stdev_factor_w,
                          Skrt, Skthl, Sku, Skv,
                          sclrm=None, sclrp2=None, wpsclrp=None, Sksclr=None):
    """new_hybrid_pdf_main.F90:new_hybrid_pdf_driver — full new-hybrid PDF-parameter driver.

    w is the setter; rt/thl/u/v/sclr are responders. Returns the dict of all f2py-exposed outputs:
    clipped responder skewnesses (Skrt/Skthl/Sku/Skv[/Sksclr]), component means mu_{w,rt,thl,u,v}_{1,2}
    [+sclr], component variances sigma_{...}_{1,2}_sqd, mixt_frac, and sigma_sqd_w = 1 − F_w. The implicit-
    coefs terms (turbulent-advection closure) are not f2py-exposed and are omitted. Scalar args are optional
    (ngrdcol,nz,sclr_dim) arrays; when omitted sclr handling is skipped (max_corr_w_sclr_sqd = 0).
    Pure-jnp → differentiable. All inputs (ngrdcol,nz) except slope/stdev-factor which are (ngrdcol,) or scalar."""
    wp2 = jnp.asarray(wp2, dtype=jnp.float64)
    slope = jnp.asarray(slope_coef_spread_DG_means_w, dtype=jnp.float64)
    pstdev = jnp.asarray(pdf_component_stdev_factor_w, dtype=jnp.float64)
    if slope.ndim == 1:
        slope = slope[:, None]
    if pstdev.ndim == 1:
        pstdev = pstdev[:, None]

    # max_corr_w_sclr_sqd: largest <w'sclr'>^2/(<w'^2><sclr'^2>) over scalars, else 0.
    if sclrm is not None and jnp.asarray(sclrm).shape[-1] > 0:
        sclrp2_a = jnp.asarray(sclrp2, dtype=jnp.float64)
        wpsclrp_a = jnp.asarray(wpsclrp, dtype=jnp.float64)
        d = wp2[..., None] * sclrp2_a
        ratio = jnp.where(d > 0.0, wpsclrp_a ** 2 / jnp.where(d > 0.0, d, 1.0), 0.0)
        max_corr_w_sclr_sqd = jnp.max(ratio, axis=-1)
    else:
        max_corr_w_sclr_sqd = jnp.zeros_like(wp2)

    F_w, zeta_w, _min_F_w, _max_F_w = calc_F_w_zeta_w(
        Skw, wprtp, wpthlp, upwp, vpwp, wp2, rtp2, thlp2, up2, vp2,
        gamma_Skw_fnc, slope, pstdev, max_corr_w_sclr_sqd)

    mu_w_1, mu_w_2, sigma_w_1, sigma_w_2, mixt_frac, _cw1, _cw2 = calculate_w_params(
        wm, wp2, Skw, F_w, zeta_w)

    out = dict(mu_w_1=mu_w_1, mu_w_2=mu_w_2,
               sigma_w_1_sqd=sigma_w_1 ** 2, sigma_w_2_sqd=sigma_w_2 ** 2,
               mixt_frac=mixt_frac, sigma_sqd_w=1.0 - F_w)

    responders = (('rt', rtm, rtp2, wprtp, Skrt), ('thl', thlm, thlp2, wpthlp, Skthl),
                  ('u', um, up2, upwp, Sku), ('v', vm, vp2, vpwp, Skv))
    for name, xm, xp2, wpxp, Skx in responders:
        Skx_c, mu1, mu2, s1, s2, _c1, _c2 = calc_responder_driver(xm, xp2, wpxp, wp2, mixt_frac, F_w, Skx)
        out[f'Sk{name}'] = Skx_c
        out[f'mu_{name}_1'] = mu1; out[f'mu_{name}_2'] = mu2
        out[f'sigma_{name}_1_sqd'] = s1; out[f'sigma_{name}_2_sqd'] = s2

    if sclrm is not None and jnp.asarray(sclrm).shape[-1] > 0:
        sclrm_a = jnp.asarray(sclrm, dtype=jnp.float64)
        Sksclr_a = jnp.asarray(Sksclr, dtype=jnp.float64)
        nd = sclrm_a.shape[-1]
        sk_l, mu1_l, mu2_l, s1_l, s2_l = [], [], [], [], []
        for j in range(nd):
            Skx_c, mu1, mu2, s1, s2, _c1, _c2 = calc_responder_driver(
                sclrm_a[..., j], sclrp2_a[..., j], wpsclrp_a[..., j], wp2, mixt_frac, F_w, Sksclr_a[..., j])
            sk_l.append(Skx_c); mu1_l.append(mu1); mu2_l.append(mu2); s1_l.append(s1); s2_l.append(s2)
        out['Sksclr'] = jnp.stack(sk_l, axis=-1)
        out['mu_sclr_1'] = jnp.stack(mu1_l, axis=-1); out['mu_sclr_2'] = jnp.stack(mu2_l, axis=-1)
        out['sigma_sclr_1_sqd'] = jnp.stack(s1_l, axis=-1); out['sigma_sclr_2_sqd'] = jnp.stack(s2_l, axis=-1)

    return out
