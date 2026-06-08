"""JAX port of new_hybrid_pdf.F90 — the new-hybrid (Griffin & Larson 2020) PDF leaf routines.

Mirrors clubb_release/src/CLUBB_core/new_hybrid_pdf.F90: the component-parameter and semi-implicit-coefficient
routines for the new-hybrid PDF closure. The top-level driver lives in new_hybrid_pdf_main.py
(new_hybrid_pdf_main.F90), which `use`s these. Several of these are sgn=+1 / capital-C specializations of the
sibling new_pdf.F90 forms; they import those shared helpers (`_ssqrt`, `calc_coef_wp4_implicit`,
`calc_mixture_fraction`) back from new_pdf.py (one-directional — new_pdf never calls these). Pure-jnp → differentiable.
"""

import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.new_pdf import _ssqrt, calc_coef_wp4_implicit, calc_mixture_fraction


def calc_coef_wp2xp_implicit(wp2, mixt_frac, F_w, coef_sigma_w_1_sqd, coef_sigma_w_2_sqd):
    """Coefficient in <w'^2 x'> = coef_wp2xp_implicit * <w'x'> for the implicit wp2xp turbulent-advection term
    (new_hybrid_pdf.F90:calc_coef_wp2xp_implicit). For F_w > 0:
      coef = sqrt(mf(1-mf)) · (F_w((1-mf)/mf − mf/(1-mf)) + c1 − c2) · sqrt(wp2/F_w);  else 0.
    Pure-jnp (nan-safe sqrt + guarded F_w) → differentiable."""
    wp2 = jnp.asarray(wp2, dtype=jnp.float64)
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64)
    F = jnp.asarray(F_w, dtype=jnp.float64)
    c1 = jnp.asarray(coef_sigma_w_1_sqd, dtype=jnp.float64)
    c2 = jnp.asarray(coef_sigma_w_2_sqd, dtype=jnp.float64)
    omf = 1.0 - mf
    F_safe = jnp.where(F > 0.0, F, 1.0)
    coef = (_ssqrt(mf * omf)
            * (F * (omf / mf - mf / omf) + c1 - c2)
            * _ssqrt(wp2 / F_safe))
    return jnp.where(F > 0.0, coef, 0.0)


# --- new_hybrid_pdf.F90 aliases (Griffin & Larson 2020) ---
# These are byte-identical / sgn=+1 specializations of the new_pdf.F90 forms; CLUBB defines them
# separately in module new_hybrid_pdf with their own (capital-C) names. Kept as thin wrappers so callers of
# either module map to one tested implementation.

def calculate_coef_wp4_implicit(mixt_frac, F_w, coef_sigma_w_1_sqd, coef_sigma_w_2_sqd):
    """new_hybrid_pdf.F90:calculate_coef_wp4_implicit — byte-identical to calc_coef_wp4_implicit."""
    return calc_coef_wp4_implicit(mixt_frac, F_w, coef_sigma_w_1_sqd, coef_sigma_w_2_sqd)


def calculate_mixture_fraction(Skw, F_w, zeta_w):
    """new_hybrid_pdf.F90:calculate_mixture_fraction (Griffin & Larson 2020) — the w-setter specialization of
    calc_mixture_fraction with sgn_wpxp = +1 (w is the setting variable)."""
    return calc_mixture_fraction(Skw, F_w, zeta_w, 1.0)


def calculate_w_params(wm, wp2, Skw, F_w, zeta_w):
    """new_hybrid_pdf.F90:calculate_w_params (Griffin & Larson 2020) — PDF component means/stdevs and the
    mixture fraction for w (the variable that sets the new-hybrid PDF):
      mixt_frac = calculate_mixture_fraction(Skw, F_w, zeta_w);
      mu_w_1 = wm + √(F_w·((1−mf)/mf)·wp2);  mu_w_2 = wm − (mf/(1−mf))·(mu_w_1 − wm);
      coef_sigma_w_1_sqd = (zeta+1)(1−F_w)/((zeta+2)mf);  coef_sigma_w_2_sqd = (1−F_w)/((zeta+2)(1−mf));
      sigma_w_i = √(coef_sigma_w_i_sqd·wp2).
    Mirrors the Fortran validity gate: when mixt_frac ∉ (0,1) (only happens for F_w=0, |Skw|>0 → mixt_frac=−1)
    all outputs except mixt_frac are 0. Pure-jnp (nan-safe sqrt, guarded denominators) → differentiable.
    Returns (mu_w_1, mu_w_2, sigma_w_1, sigma_w_2, mixt_frac, coef_sigma_w_1_sqd, coef_sigma_w_2_sqd)."""
    wm = jnp.asarray(wm, dtype=jnp.float64); wp2 = jnp.asarray(wp2, dtype=jnp.float64)
    Skw = jnp.asarray(Skw, dtype=jnp.float64); F_w = jnp.asarray(F_w, dtype=jnp.float64)
    zeta = jnp.asarray(zeta_w, dtype=jnp.float64)

    mixt_frac = calculate_mixture_fraction(Skw, F_w, zeta)
    valid = (mixt_frac > 0.0) & (mixt_frac < 1.0)
    mf = jnp.where(valid, mixt_frac, 0.5)          # avoid div-by-zero in the invalid branch
    omf = 1.0 - mf

    mu_w_1_v = wm + _ssqrt(F_w * (omf / mf) * wp2)
    mu_w_2_v = wm - (mf / omf) * (mu_w_1_v - wm)
    c1_v = ((zeta + 1.0) * (1.0 - F_w)) / ((zeta + 2.0) * mf)
    c2_v = (1.0 - F_w) / ((zeta + 2.0) * omf)
    sig1_v = _ssqrt(c1_v * wp2); sig2_v = _ssqrt(c2_v * wp2)

    z = jnp.zeros_like(mu_w_1_v)
    mu_w_1 = jnp.where(valid, mu_w_1_v, z)
    mu_w_2 = jnp.where(valid, mu_w_2_v, z)
    sigma_w_1 = jnp.where(valid, sig1_v, z)
    sigma_w_2 = jnp.where(valid, sig2_v, z)
    coef_sigma_w_1_sqd = jnp.where(valid, c1_v, z)
    coef_sigma_w_2_sqd = jnp.where(valid, c2_v, z)
    return (mu_w_1, mu_w_2, sigma_w_1, sigma_w_2, mixt_frac,
            coef_sigma_w_1_sqd, coef_sigma_w_2_sqd)


def calculate_responder_params(xm, xp2, Skx, wpxp, wp2, F_w, mixt_frac):
    """new_hybrid_pdf.F90:calculate_responder_params (Griffin & Larson 2020) — PDF component means/variances for
    a responding variable x (one that does not set the PDF), using <w'x'> explicitly. For |<w'x'>| > 0:
      mu_x_1 = xm + √((1−mf)/mf)·wpxp/√(F_w·wp2);  mu_x_2 = xm − √(mf/(1−mf))·wpxp/√(F_w·wp2);
      coef_sigma_x_1_sqd = 1 + √((1−mf)/mf)·Skx·√(F_w·wp2·xp2)/(3·wpxp) − ((1+mf)/mf)·wpxp²/(3·F_w·wp2·xp2);
      coef_sigma_x_2_sqd = 1 − √(mf/(1−mf))·Skx·√(F_w·wp2·xp2)/(3·wpxp) + ((mf−2)/(1−mf))·wpxp²/(3·F_w·wp2·xp2);
    each coef floored at 0; sigma_x_i_sqd = coef·xp2. For <w'x'> = 0 the PDF is a single Gaussian:
    mu_x_i = xm, sigma_x_i_sqd = xp2, coef = 1. Pure-jnp (nan-safe sqrt, guarded wpxp/denominators) →
    differentiable. Returns (mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd, coef_sigma_x_1_sqd, coef_sigma_x_2_sqd)."""
    xm = jnp.asarray(xm, dtype=jnp.float64); xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    Skx = jnp.asarray(Skx, dtype=jnp.float64); wpxp = jnp.asarray(wpxp, dtype=jnp.float64)
    wp2 = jnp.asarray(wp2, dtype=jnp.float64); F_w = jnp.asarray(F_w, dtype=jnp.float64)
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64)
    omf = 1.0 - mf

    gate = jnp.abs(wpxp) > 0.0
    wpxp_safe = jnp.where(gate, wpxp, 1.0)              # guard the /(3·wpxp) and /√(F_w·wp2) terms
    fw_wp2_safe = jnp.where(gate, F_w * wp2, 1.0)       # >0 whenever gate is true (Fortran note)
    den_safe = jnp.where(gate, 3.0 * F_w * wp2 * xp2, 1.0)

    mu_x_1_v = xm + _ssqrt(omf / mf) * wpxp / _ssqrt(fw_wp2_safe)
    mu_x_2_v = xm - _ssqrt(mf / omf) * wpxp / _ssqrt(fw_wp2_safe)
    c1_v = (1.0 + _ssqrt(omf / mf) * Skx * _ssqrt(fw_wp2_safe * xp2) / (3.0 * wpxp_safe)
            - ((1.0 + mf) / mf) * wpxp ** 2 / den_safe)
    c1_v = jnp.maximum(c1_v, 0.0)
    c2_v = (1.0 - _ssqrt(mf / omf) * Skx * _ssqrt(fw_wp2_safe * xp2) / (3.0 * wpxp_safe)
            + ((mf - 2.0) / omf) * wpxp ** 2 / den_safe)
    c2_v = jnp.maximum(c2_v, 0.0)

    mu_x_1 = jnp.where(gate, mu_x_1_v, xm)
    mu_x_2 = jnp.where(gate, mu_x_2_v, xm)
    coef_sigma_x_1_sqd = jnp.where(gate, c1_v, 1.0)
    coef_sigma_x_2_sqd = jnp.where(gate, c2_v, 1.0)
    sigma_x_1_sqd = coef_sigma_x_1_sqd * xp2
    sigma_x_2_sqd = coef_sigma_x_2_sqd * xp2
    return (mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd,
            coef_sigma_x_1_sqd, coef_sigma_x_2_sqd)


def calc_coefs_wpxp2_semiimpl(wp2, wpxp, mixt_frac, F_w, coef_sigma_x_1_sqd, coef_sigma_x_2_sqd):
    """Semi-implicit decomposition <w'x'²> = coef_wpxp2_implicit·<x'²> + term_wpxp2_explicit
    (new_hybrid_pdf.F90:calc_coefs_wpxp2_semiimpl). For F_w>0 and wp2>0:
      coef = √(mf(1−mf))·√(F_w·wp2)·(c_x1 − c_x2);
      term = √(mf(1−mf))·wpxp²/√(F_w·wp2)·((1−mf)/mf − mf/(1−mf));  else 0, 0.
    Pure-jnp (guarded) → differentiable. Returns (coef_wpxp2_implicit, term_wpxp2_explicit)."""
    wp2 = jnp.asarray(wp2, dtype=jnp.float64); wpxp = jnp.asarray(wpxp, dtype=jnp.float64)
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64); F_w = jnp.asarray(F_w, dtype=jnp.float64)
    cx1 = jnp.asarray(coef_sigma_x_1_sqd, dtype=jnp.float64); cx2 = jnp.asarray(coef_sigma_x_2_sqd, dtype=jnp.float64)
    omf = 1.0 - mf

    vary = (F_w > 0.0) & (wp2 > 0.0)
    sFwwp2 = _ssqrt(F_w * wp2)
    sFwwp2_safe = jnp.where(sFwwp2 > 0.0, sFwwp2, 1.0)
    base = _ssqrt(mf * omf)
    coef = base * sFwwp2 * (cx1 - cx2)
    term = base * wpxp ** 2 / sFwwp2_safe * (omf / mf - mf / omf)
    return jnp.where(vary, coef, 0.0), jnp.where(vary, term, 0.0)


def calc_coefs_wpxpyp_semiimpl(wp2, wpxp, wpyp, mixt_frac, F_w,
                               coef_sigma_x_1_sqd, coef_sigma_x_2_sqd,
                               coef_sigma_y_1_sqd, coef_sigma_y_2_sqd):
    """Semi-implicit decomposition <w'x'y'> = coef_wpxpyp_implicit·<x'y'> + term_wpxpyp_explicit
    (new_hybrid_pdf.F90:calc_coefs_wpxpyp_semiimpl). When the x·y sigma-product varies (cx1·cy1>0 or
    cx2·cy2>0) AND F_w>0 AND wp2>0:
      f_xy = (√(cx1·cy1) − √(cx2·cy2)) / (mf·√(cx1·cy1) + (1−mf)·√(cx2·cy2));
      coef = √(mf(1−mf))·√(F_w·wp2)·f_xy;
      term = √(mf(1−mf))·wpxp·wpyp/√(F_w·wp2)·((1−mf)/mf − mf/(1−mf) − f_xy).
    Else (single-Gaussian x·y): coef = √(mf(1−mf))·√(F_w·wp2)·((1−mf)/mf − mf/(1−mf)) when F_w>0 else 0; term = 0.
    The xy-companion of calc_coefs_wpxp2_semiimpl. Pure-jnp (guarded) → differentiable.
    Returns (coef_wpxpyp_implicit, term_wpxpyp_explicit)."""
    wp2 = jnp.asarray(wp2, dtype=jnp.float64); wpxp = jnp.asarray(wpxp, dtype=jnp.float64)
    wpyp = jnp.asarray(wpyp, dtype=jnp.float64)
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64); F_w = jnp.asarray(F_w, dtype=jnp.float64)
    cx1 = jnp.asarray(coef_sigma_x_1_sqd, dtype=jnp.float64); cx2 = jnp.asarray(coef_sigma_x_2_sqd, dtype=jnp.float64)
    cy1 = jnp.asarray(coef_sigma_y_1_sqd, dtype=jnp.float64); cy2 = jnp.asarray(coef_sigma_y_2_sqd, dtype=jnp.float64)
    omf = 1.0 - mf

    s1 = _ssqrt(cx1 * cy1); s2 = _ssqrt(cx2 * cy2)
    xy_vary = ((cx1 * cy1 > 0.0) | (cx2 * cy2 > 0.0)) & (F_w > 0.0) & (wp2 > 0.0)
    denom = mf * s1 + omf * s2
    denom_safe = jnp.where(denom > 0.0, denom, 1.0)
    f_xy = (s1 - s2) / denom_safe

    sFwwp2 = _ssqrt(F_w * wp2)
    sFwwp2_safe = jnp.where(sFwwp2 > 0.0, sFwwp2, 1.0)
    base = _ssqrt(mf * omf)

    coef_vary = base * sFwwp2 * f_xy
    term_vary = base * wpxp * wpyp / sFwwp2_safe * (omf / mf - mf / omf - f_xy)
    coef_else = jnp.where(F_w > 0.0, base * sFwwp2 * (omf / mf - mf / omf), 0.0)

    coef = jnp.where(xy_vary, coef_vary, coef_else)
    term = jnp.where(xy_vary, term_vary, 0.0)
    return coef, term


__all__ = [
    "calc_coef_wp2xp_implicit",
    "calculate_coef_wp4_implicit",
    "calculate_mixture_fraction",
    "calculate_w_params",
    "calculate_responder_params",
    "calc_coefs_wpxp2_semiimpl",
    "calc_coefs_wpxpyp_semiimpl",
]
