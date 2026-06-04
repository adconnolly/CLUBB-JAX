"""JAX port of new_pdf.F90 — the "new hybrid" PDF closure helpers (Griffin & Larson 2018).

This PDF type (iiPDF_new_hybrid) is an alternative to ADG1; the gated CLUBB config uses ADG1, so these are
completeness ports. Each function is the closed-form algebra from the Fortran, pure-jnp and differentiable.

calc_coef_wp4_implicit — the coefficient relating <w'^4> to <w'^2>^2 for the implicit wp4 turbulent-advection
  term (new_pdf.F90:1180), validated bit-for-bit against f2py.
calc_mixture_fraction — the PDF mixture fraction from skewness (new_pdf.F90:348), validated by literal
  transcription + the analytic symmetric limit.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def calc_coef_wp4_implicit(mixt_frac, F_w, coef_sigma_w_1_sqd, coef_sigma_w_2_sqd):
    """Coefficient in <w'^4> = coef_wp4_implicit * <w'^2>^2 (new_pdf.F90:calc_coef_wp4_implicit):
      3 mf c1² + 6 F (1−mf) c1 + F² (1−mf)²/mf + 3 (1−mf) c2² + 6 F mf c2 + F² mf²/(1−mf),
    with mf = mixt_frac, c1/c2 = coef_sigma_w_{1,2}_sqd, F = F_w. Pure-jnp → differentiable."""
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64)
    F = jnp.asarray(F_w, dtype=jnp.float64)
    c1 = jnp.asarray(coef_sigma_w_1_sqd, dtype=jnp.float64)
    c2 = jnp.asarray(coef_sigma_w_2_sqd, dtype=jnp.float64)
    omf = 1.0 - mf
    return (3.0 * mf * c1 ** 2
            + 6.0 * F * omf * c1
            + F ** 2 * omf ** 2 / mf
            + 3.0 * omf * c2 ** 2
            + 6.0 * F * mf * c2
            + F ** 2 * mf ** 2 / omf)


def calc_mixture_fraction(Skx, F_x, zeta_x, sgn_wpxp):
    """PDF mixture fraction (weight of the 1st component) from skewness (new_pdf.F90:calc_mixture_fraction,
    Griffin & Larson 2018). For F_x > 0, a closed-form in (Skx, F_x, zeta_x, sgn_wpxp); for F_x = 0 and Skx = 0
    the symmetric limit (zeta_x+1)/(zeta_x+2). (The Fortran errors for F_x = 0 with |Skx| > 0; that degenerate
    case is not produced here.) Pure-jnp (guarded denominators) → differentiable."""
    Skx = jnp.asarray(Skx, dtype=jnp.float64)
    F = jnp.asarray(F_x, dtype=jnp.float64)
    zeta = jnp.asarray(zeta_x, dtype=jnp.float64)
    sgn = jnp.asarray(sgn_wpxp, dtype=jnp.float64)
    zp2 = zeta + 2.0

    sqrt_arg = (4.0 * F ** 3 + 12.0 * F ** 2 * (1.0 - F)
                + 36.0 * F * (zeta + 1.0) * (1.0 - F) ** 2 / zp2 ** 2 + Skx ** 2)
    num = (4.0 * F ** 3
           + 18.0 * F * (zeta + 1.0) * (1.0 - F) / zp2
           + 6.0 * F ** 2 * (1.0 - F) / zp2
           + Skx ** 2
           - Skx * sgn * jnp.sqrt(jnp.maximum(sqrt_arg, 0.0)))
    den = 2.0 * F * (F - 3.0) ** 2 + 2.0 * Skx ** 2
    mf_pos = num / jnp.where(den > 0.0, den, 1.0)

    mf_symmetric = (zeta + 1.0) / zp2
    return jnp.where(F > 0.0, mf_pos, mf_symmetric)


def _ssqrt(x):
    """sqrt(max(x,0)) with a nan-free gradient at x<=0."""
    xp = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.sqrt(xp), 0.0)


def calc_coef_wpxp2_implicit(wp2, xp2, wpxp, sgn_wpxp, mixt_frac, F_w, F_x,
                             coef_sigma_w_1_sqd, coef_sigma_w_2_sqd,
                             coef_sigma_x_1_sqd, coef_sigma_x_2_sqd):
    """Coefficient in <w'x'^2> = coef_wpxp2_implicit * <x'^2> for the implicit wpxp2 turbulent-advection term
    (new_pdf.F90:calc_coef_wpxp2_implicit). Two branches: the full form when the component sigma products and
    wp2·xp2 are positive, else a reduced form. Pure-jnp (guarded sqrt/denominators) → differentiable."""
    wp2 = jnp.asarray(wp2, dtype=jnp.float64); xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    wpxp = jnp.asarray(wpxp, dtype=jnp.float64); sgn = jnp.asarray(sgn_wpxp, dtype=jnp.float64)
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64)
    F_w = jnp.asarray(F_w, dtype=jnp.float64); F_x = jnp.asarray(F_x, dtype=jnp.float64)
    cw1 = jnp.asarray(coef_sigma_w_1_sqd, dtype=jnp.float64); cw2 = jnp.asarray(coef_sigma_w_2_sqd, dtype=jnp.float64)
    cx1 = jnp.asarray(coef_sigma_x_1_sqd, dtype=jnp.float64); cx2 = jnp.asarray(coef_sigma_x_2_sqd, dtype=jnp.float64)
    omf = 1.0 - mf

    F_spread = F_x * (omf / mf - mf / omf)
    base = _ssqrt(mf * omf) * _ssqrt(wp2)

    # Reduced branch.
    branchB = base * _ssqrt(F_w) * (F_spread + (cx1 - cx2))

    # Full branch.
    cwcx1 = cw1 * cx1
    cwcx2 = cw2 * cx2
    denom = mf * _ssqrt(cwcx1) + omf * _ssqrt(cwcx2)
    coefs_factor = (_ssqrt(cwcx1) - _ssqrt(cwcx2)) / jnp.where(denom > 0.0, denom, 1.0)
    denom2 = _ssqrt(wp2) * _ssqrt(xp2)
    wpxp_ratio = wpxp / jnp.where(denom2 > 0.0, denom2, 1.0)
    branchA = base * (_ssqrt(F_w) * F_x * (omf / mf - mf / omf)
                      + _ssqrt(F_w) * (cx1 - cx2)
                      + 2.0 * _ssqrt(F_x) * coefs_factor * sgn * wpxp_ratio
                      - 2.0 * _ssqrt(F_w) * F_x * coefs_factor)

    cond = ((cwcx1 > 0.0) | (cwcx2 > 0.0)) & (wp2 * xp2 > 0.0)
    return jnp.where(cond, branchA, branchB)


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
# These are byte-identical / sgn=+1 specializations of the new_pdf.F90 forms above; CLUBB defines them
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


def calc_setter_var_params(xm, xp2, Skx, sgn_wpxp, F_x, zeta_x):
    """PDF component means/stdevs and mixture fraction for the variable that *sets* the PDF
    (new_pdf.F90:calc_setter_var_params, Griffin & Larson 2018):
      mixt_frac = calc_mixture_fraction(Skx, F_x, zeta_x, sgn_wpxp);
      mu_x_1 = xm + sgn_wpxp·√(F_x·((1−mf)/mf)·xp2);  mu_x_2 = xm − (mf/(1−mf))·(mu_x_1 − xm);
      coef_sigma_x_1_sqd = (zeta+1)(1−F_x)/((zeta+2)mf);  coef_sigma_x_2_sqd = (1−F_x)/((zeta+2)(1−mf));
      sigma_x_i = √(coef_sigma_x_i_sqd·xp2).
    Pure-jnp (nan-safe sqrt) → differentiable. Returns
    (mu_x_1, mu_x_2, sigma_x_1, sigma_x_2, mixt_frac, coef_sigma_x_1_sqd, coef_sigma_x_2_sqd)."""
    xm = jnp.asarray(xm, dtype=jnp.float64); xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    Skx = jnp.asarray(Skx, dtype=jnp.float64); sgn = jnp.asarray(sgn_wpxp, dtype=jnp.float64)
    F_x = jnp.asarray(F_x, dtype=jnp.float64); zeta = jnp.asarray(zeta_x, dtype=jnp.float64)

    mixt_frac = calc_mixture_fraction(Skx, F_x, zeta, sgn)
    omf = 1.0 - mixt_frac
    mu_x_1 = xm + sgn * _ssqrt(F_x * (omf / mixt_frac) * xp2)
    mu_x_2 = xm - (mixt_frac / omf) * (mu_x_1 - xm)
    coef_sigma_x_1_sqd = ((zeta + 1.0) * (1.0 - F_x)) / ((zeta + 2.0) * mixt_frac)
    coef_sigma_x_2_sqd = (1.0 - F_x) / ((zeta + 2.0) * omf)
    sigma_x_1 = _ssqrt(coef_sigma_x_1_sqd * xp2)
    sigma_x_2 = _ssqrt(coef_sigma_x_2_sqd * xp2)
    return mu_x_1, mu_x_2, sigma_x_1, sigma_x_2, mixt_frac, coef_sigma_x_1_sqd, coef_sigma_x_2_sqd


def calc_responder_params(xm, xp2, Skx, sgn_wpxp, F_x, mixt_frac):
    """PDF component means/variances for a variable that *responds* to the PDF set by another variable
    (new_pdf.F90:calc_responder_params, Griffin & Larson 2018). Takes the setter's mixt_frac as input. For F_x>0:
      mu_x_1 = xm + sgn·√(F_x((1−mf)/mf)xp2);  mu_x_2 = xm − (mf/(1−mf))(mu_x_1−xm);
      coef_sigma_x_1_sqd = √(mf(1−mf))·Skx·sgn/(3 mf √F_x) − (1+mf)F_x/(3 mf) + 1;
      coef_sigma_x_2_sqd = ((1−F_x) − mf·coef_sigma_x_1_sqd)/(1−mf);  sigma_x_i_sqd = coef·xp2.
    For F_x=0 the PDF is a single Gaussian: mu_x_i = xm, sigma_x_i_sqd = xp2, coef = 1. Pure-jnp (nan-safe sqrt,
    guarded F_x) → differentiable. Returns
    (mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd, coef_sigma_x_1_sqd, coef_sigma_x_2_sqd)."""
    xm = jnp.asarray(xm, dtype=jnp.float64); xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    Skx = jnp.asarray(Skx, dtype=jnp.float64); sgn = jnp.asarray(sgn_wpxp, dtype=jnp.float64)
    F_x = jnp.asarray(F_x, dtype=jnp.float64); mf = jnp.asarray(mixt_frac, dtype=jnp.float64)
    omf = 1.0 - mf
    F_safe = jnp.where(F_x > 0.0, F_x, 1.0)

    mu_x_1_v = xm + sgn * _ssqrt(F_x * (omf / mf) * xp2)
    mu_x_2_v = xm - (mf / omf) * (mu_x_1_v - xm)
    c1_v = (_ssqrt(mf * omf) * Skx * sgn / (3.0 * mf * _ssqrt(F_safe))
            - (1.0 + mf) * F_x / (3.0 * mf) + 1.0)
    c2_v = ((1.0 - F_x) - mf * c1_v) / omf

    vary = F_x > 0.0
    mu_x_1 = jnp.where(vary, mu_x_1_v, xm)
    mu_x_2 = jnp.where(vary, mu_x_2_v, xm)
    coef_sigma_x_1_sqd = jnp.where(vary, c1_v, 1.0)
    coef_sigma_x_2_sqd = jnp.where(vary, c2_v, 1.0)
    sigma_x_1_sqd = coef_sigma_x_1_sqd * xp2
    sigma_x_2_sqd = coef_sigma_x_2_sqd * xp2
    return mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd, coef_sigma_x_1_sqd, coef_sigma_x_2_sqd


def sort_roots(roots):
    """Sort the three roots of each level ascending (new_pdf.F90:sort_roots). roots is (..., 3); returns the
    sorted values — equivalent to jnp.sort along the last axis (the Fortran's sort network returns the same
    values; only ordering of equal roots could differ, which doesn't affect the values)."""
    return jnp.sort(jnp.asarray(roots, dtype=jnp.float64), axis=-1)


def calc_limits_F_x_responder(mixt_frac, Skx, sgn_wpxp,
                              max_Skx2_pos_Skx_sgn_wpxp, max_Skx2_neg_Skx_sgn_wpxp):
    """Min/max allowable F_x for a responding variable so its PDF component standard deviations stay >= 0
    (new_pdf.F90:calc_limits_F_x_responder). Solves two cubics in sqrt(F_x) (one per component-sigma>=0
    constraint) via `cubic_solve`, sorts the real parts, and selects min/max sqrt(F_x) by a 4-way branch on
    sign(Skx·<w'x'>) and Skx² vs the max-Skx² thresholds; F_x limits are the squares (clamped to [0,1]).
    All array args are the same shape. Returns (min_F_x, max_F_x). Pure-jnp → differentiable."""
    from clubb_jax.src.CLUBB_core.calc_roots import cubic_solve
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64)
    Skx = jnp.asarray(Skx, dtype=jnp.float64)
    sgn = jnp.asarray(sgn_wpxp, dtype=jnp.float64)
    max_pos = jnp.asarray(max_Skx2_pos_Skx_sgn_wpxp, dtype=jnp.float64)
    max_neg = jnp.asarray(max_Skx2_neg_Skx_sgn_wpxp, dtype=jnp.float64)
    omf = 1.0 - mf

    Dterm = _ssqrt(mf * omf) * Skx * sgn

    # Cubic for the 1st-component-sigma>=0 constraint: A1 s^3 + 0 s^2 + C1 s + D1 = 0, s = sqrt(F_x).
    r1 = cubic_solve(-(1.0 + mf), jnp.zeros_like(mf), 3.0 * mf, Dterm)
    # Cubic for the 2nd-component-sigma>=0 constraint.
    r2 = cubic_solve(-(2.0 - mf), jnp.zeros_like(mf), 3.0 * omf, -Dterm)
    r1_real = jnp.real(r1)
    r2_real = jnp.real(r2)
    r1s = sort_roots(r1_real)
    r2s = sort_roots(r2_real)

    Skx2 = Skx ** 2
    pos = Skx * sgn >= 0.0

    min_sqrt = jnp.where(pos, r2s[..., 1], r1s[..., 1])
    max_pos_branch = jnp.where(Skx2 > max_neg,
                               jnp.minimum(r1_real[..., 0], r2s[..., 2]),
                               jnp.minimum(r1s[..., 2], r2s[..., 2]))
    max_neg_branch = jnp.where(Skx2 > max_pos,
                               jnp.minimum(r2_real[..., 0], r1s[..., 2]),
                               jnp.minimum(r1s[..., 2], r2s[..., 2]))
    max_sqrt = jnp.where(pos, max_pos_branch, max_neg_branch)

    min_sqrt = jnp.maximum(min_sqrt, 0.0)
    max_sqrt = jnp.minimum(max_sqrt, 1.0)
    return min_sqrt ** 2, max_sqrt ** 2


def calc_coefs_wp2xp_semiimpl(wp2, xp2, sgn_wpxp, mixt_frac, F_w, F_x,
                              coef_sigma_w_1_sqd, coef_sigma_w_2_sqd,
                              coef_sigma_x_1_sqd, coef_sigma_x_2_sqd):
    """Semi-implicit decomposition <w'²x'> = coef_wp2xp_implicit·<w'x'> + term_wp2xp_explicit
    (new_pdf.F90:calc_coefs_wp2xp_semiimpl). Two branches: the full form when a component σ_w·σ_x product is
    positive, else the reduced (both-zero) form. Pure-jnp (nan-safe sqrt, guarded denom) → differentiable.
    Returns (coef_wp2xp_implicit, term_wp2xp_explicit)."""
    wp2 = jnp.asarray(wp2, dtype=jnp.float64); xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    sgn = jnp.asarray(sgn_wpxp, dtype=jnp.float64); mf = jnp.asarray(mixt_frac, dtype=jnp.float64)
    F_w = jnp.asarray(F_w, dtype=jnp.float64); F_x = jnp.asarray(F_x, dtype=jnp.float64)
    cw1 = jnp.asarray(coef_sigma_w_1_sqd, dtype=jnp.float64); cw2 = jnp.asarray(coef_sigma_w_2_sqd, dtype=jnp.float64)
    cx1 = jnp.asarray(coef_sigma_x_1_sqd, dtype=jnp.float64); cx2 = jnp.asarray(coef_sigma_x_2_sqd, dtype=jnp.float64)
    omf = 1.0 - mf

    base = _ssqrt(mf * omf)
    swp2 = _ssqrt(wp2)
    F_spread = omf / mf - mf / omf
    explicit_prefac = base * _ssqrt(F_x) * _ssqrt(xp2) * wp2 * sgn

    cwcx1 = cw1 * cx1
    cwcx2 = cw2 * cx2
    denom = mf * _ssqrt(cwcx1) + omf * _ssqrt(cwcx2)
    coefs_factor = (_ssqrt(cwcx1) - _ssqrt(cwcx2)) / jnp.where(denom > 0.0, denom, 1.0)

    coef_full = base * 2.0 * _ssqrt(F_w) * swp2 * coefs_factor
    term_full = explicit_prefac * (F_w * F_spread + (cw1 - cw2) - 2.0 * F_w * coefs_factor)

    coef_red = base * _ssqrt(F_w) * swp2 * F_spread
    term_red = explicit_prefac * (cw1 - cw2)

    cond = (cwcx1 > 0.0) | (cwcx2 > 0.0)
    coef_wp2xp_implicit = jnp.where(cond, coef_full, coef_red)
    term_wp2xp_explicit = jnp.where(cond, term_full, term_red)
    return coef_wp2xp_implicit, term_wp2xp_explicit


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


def calc_coefs_wpxpyp_semiimpl(wp2, xp2, yp2, wpxp, wpyp, sgn_wpxp, sgn_wpyp, mixt_frac,
                               F_w, F_x, F_y, coef_sigma_w_1_sqd, coef_sigma_w_2_sqd,
                               coef_sigma_x_1_sqd, coef_sigma_x_2_sqd,
                               coef_sigma_y_1_sqd, coef_sigma_y_2_sqd):
    """Semi-implicit decomposition <w'x'y'> = coef_wpxpyp_implicit·<x'y'> + term_wpxpyp_explicit
    (new_pdf.F90:calc_coefs_wpxpyp_semiimpl). Uses three coefs_factors (wx, wy, xy), each
    (√(c1·c1') − √(c2·c2'))/(mf√(c1·c1') + (1−mf)√(c2·c2')) when a product is positive, else 0. Branch on whether
    a σ_x·σ_y product is positive: the full form assembles coef from factor_xy and term from all three factors;
    the reduced form drops the F_x·F_y triple-product term. Pure-jnp (nan-safe sqrt, guarded denom) →
    differentiable. Returns (coef_wpxpyp_implicit, term_wpxpyp_explicit)."""
    wp2 = jnp.asarray(wp2, dtype=jnp.float64); xp2 = jnp.asarray(xp2, dtype=jnp.float64)
    yp2 = jnp.asarray(yp2, dtype=jnp.float64)
    wpxp = jnp.asarray(wpxp, dtype=jnp.float64); wpyp = jnp.asarray(wpyp, dtype=jnp.float64)
    sx = jnp.asarray(sgn_wpxp, dtype=jnp.float64); sy = jnp.asarray(sgn_wpyp, dtype=jnp.float64)
    mf = jnp.asarray(mixt_frac, dtype=jnp.float64)
    F_w = jnp.asarray(F_w, dtype=jnp.float64); F_x = jnp.asarray(F_x, dtype=jnp.float64); F_y = jnp.asarray(F_y, dtype=jnp.float64)
    cw1 = jnp.asarray(coef_sigma_w_1_sqd, dtype=jnp.float64); cw2 = jnp.asarray(coef_sigma_w_2_sqd, dtype=jnp.float64)
    cx1 = jnp.asarray(coef_sigma_x_1_sqd, dtype=jnp.float64); cx2 = jnp.asarray(coef_sigma_x_2_sqd, dtype=jnp.float64)
    cy1 = jnp.asarray(coef_sigma_y_1_sqd, dtype=jnp.float64); cy2 = jnp.asarray(coef_sigma_y_2_sqd, dtype=jnp.float64)
    omf = 1.0 - mf

    def _cfactor(p1, p2):
        s1, s2 = _ssqrt(p1), _ssqrt(p2)
        denom = mf * s1 + omf * s2
        return jnp.where((p1 > 0.0) | (p2 > 0.0), (s1 - s2) / jnp.where(denom > 0.0, denom, 1.0), 0.0)

    factor_wx = _cfactor(cw1 * cx1, cw2 * cx2)
    factor_wy = _cfactor(cw1 * cy1, cw2 * cy2)
    factor_xy = _cfactor(cx1 * cy1, cx2 * cy2)

    base = _ssqrt(mf * omf)
    sFw_wp2 = _ssqrt(F_w) * _ssqrt(wp2)
    Fx_xp2_sx = _ssqrt(F_x) * _ssqrt(xp2) * sx
    Fy_yp2_sy = _ssqrt(F_y) * _ssqrt(yp2) * sy
    spread = omf / mf - mf / omf

    coef_A = base * sFw_wp2 * factor_xy
    term_A = (base * sFw_wp2 * Fx_xp2_sx * Fy_yp2_sy * (spread - factor_xy - factor_wy - factor_wx)
              + base * Fx_xp2_sx * factor_wy * wpyp
              + base * Fy_yp2_sy * factor_wx * wpxp)
    coef_B = base * sFw_wp2 * (spread - factor_wy - factor_wx)
    term_B = base * Fx_xp2_sx * factor_wy * wpyp + base * Fy_yp2_sy * factor_wx * wpxp

    cond_A = (cx1 * cy1 > 0.0) | (cx2 * cy2 > 0.0)
    return jnp.where(cond_A, coef_A, coef_B), jnp.where(cond_A, term_A, term_B)
