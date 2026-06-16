"""JAX port of new_pdf_main.F90 — the "new" PDF top-level driver helpers (Griffin & Larson 2018).

iiPDF_new is an alternative PDF closure (the gated CLUBB config uses ADG1), so this is a completeness
port. calc_F_x_zeta_x_setter maps the tunable spread/stdev parameters to F_x (PDF-component-mean spread) and
zeta_x (PDF-component-variance ratio) for the setting variable, interpolating F_x between min and max by an
exp(-|Skx|^lambda/slope) factor. Pure-jnp → differentiable.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _spow(x, p):
    """|x|^p style positive power with a finite gradient at x=0 (x assumed >= 0 here)."""
    xp = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, xp ** p, 0.0)


def calc_F_x_zeta_x_setter(Skx, slope_coef_spread_DG_means_x, pdf_component_stdev_factor_x, lambda_):
    """F_x, zeta_x and their min/max for the setting variable (new_pdf_main.F90:calc_F_x_zeta_x_setter):
      min_F_x = 1e-3 if |Skx|>0 else 0;  max_F_x = 1;
      F_x = min_F_x·e + max_F_x·(1−e),  e = exp(−|Skx|^lambda / slope_coef);
      zeta_x = pdf_component_stdev_factor_x − 1.
    Skx is an array; the three parameters are scalars. Pure-jnp (finite-grad power) → differentiable.
    Returns (F_x, zeta_x, min_F_x, max_F_x)."""
    Skx = jnp.asarray(Skx, dtype=jnp.float64)
    slope = jnp.asarray(slope_coef_spread_DG_means_x, dtype=jnp.float64)
    stdev_factor = jnp.asarray(pdf_component_stdev_factor_x, dtype=jnp.float64)
    lam = jnp.asarray(lambda_, dtype=jnp.float64)

    absS = jnp.abs(Skx)
    min_F_x = jnp.where(absS > 0.0, 1.0e-3, 0.0)
    max_F_x = jnp.ones_like(Skx)
    e = jnp.exp(-_spow(absS, lam) / slope)
    F_x = min_F_x * e + max_F_x * (1.0 - e)
    zeta_x = (stdev_factor - 1.0) * jnp.ones_like(Skx)
    return F_x, zeta_x, min_F_x, max_F_x


from clubb_jax.src.CLUBB_core.clubb_constants import (  # noqa: E402
    rt_tol,
    thl_tol,
    max_mag_correlation,
    l_explicit_turbulent_adv_wp3,
    l_explicit_turbulent_adv_wpxp,
    l_explicit_turbulent_adv_xpyp,
)
from clubb_jax.src.CLUBB_core.new_pdf import (  # noqa: E402
    calc_setter_var_params,
    calc_responder_params,
    calc_limits_F_x_responder,
    calc_coef_wp4_implicit,
    calc_coef_wpxp2_implicit,
    calc_coefs_wp2xp_semiimpl,
    calc_coefs_wpxpyp_semiimpl,
)
from clubb_jax.src.CLUBB_core.parameter_indices import (  # noqa: E402
    islope_coef_spread_DG_means_w,
    ipdf_component_stdev_factor_w,
    icoef_spread_DG_means_rt,
    icoef_spread_DG_means_thl,
)

_LAMBDA_W = 0.5


def calc_F_x_responder(coef_spread_DG_means_x, exp_factor_x, min_F_x, max_F_x):
    """F_x for a responding variable, interpolated between min and max by coef_spread·exp_factor
    (new_pdf_main.F90:calc_F_x_responder):
      F_x = min_F_x(1 − coef_spread·exp_factor) + max_F_x·coef_spread·exp_factor."""
    cse = jnp.asarray(coef_spread_DG_means_x) * jnp.asarray(exp_factor_x)
    return jnp.asarray(min_F_x) * (1.0 - cse) + jnp.asarray(max_F_x) * cse


def calc_responder_var(xm, xp2, sgn_wpxp, mixt_frac, coef_spread_DG_means_x, exp_factor_x,
                       max_Skx2_pos_Skx_sgn_wpxp, max_Skx2_neg_Skx_sgn_wpxp, Skx):
    """PDF parameters for a responding variable (new_pdf_main.F90:calc_responder_var). Clips |Skx| to the
    realizable upper limit (√(0.99·max_Skx²)), computes the F_x limits (calc_limits_F_x_responder), interpolates
    F_x (calc_F_x_responder), then the component means/variances (calc_responder_params). Returns
    (mu_x_1, mu_x_2, sigma_x_1_sqd, sigma_x_2_sqd, coef_sigma_x_1_sqd, coef_sigma_x_2_sqd, F_x, min_F_x, max_F_x,
     Skx_clipped)."""
    Skx = jnp.asarray(Skx, dtype=jnp.float64); sgn = jnp.asarray(sgn_wpxp, dtype=jnp.float64)
    mp = jnp.asarray(max_Skx2_pos_Skx_sgn_wpxp, dtype=jnp.float64)
    mn = jnp.asarray(max_Skx2_neg_Skx_sgn_wpxp, dtype=jnp.float64)

    sk_pos = jnp.where(Skx >= 0.0, jnp.sqrt(0.99 * mp), -jnp.sqrt(0.99 * mp))
    sk_neg = jnp.where(Skx >= 0.0, jnp.sqrt(0.99 * mn), -jnp.sqrt(0.99 * mn))
    clip_pos = jnp.where(Skx ** 2 >= mp, sk_pos, Skx)
    clip_neg = jnp.where(Skx ** 2 >= mn, sk_neg, Skx)
    Skx_c = jnp.where(Skx * sgn >= 0.0, clip_pos, clip_neg)

    min_F, max_F = calc_limits_F_x_responder(mixt_frac, Skx_c, sgn, mp, mn)
    F_x = calc_F_x_responder(coef_spread_DG_means_x, exp_factor_x, min_F, max_F)
    mu1, mu2, s1sq, s2sq, c1, c2 = calc_responder_params(xm, xp2, Skx_c, sgn, F_x, mixt_frac)
    return mu1, mu2, s1sq, s2sq, c1, c2, F_x, min_F, max_F, Skx_c


def new_pdf_driver(wm, rtm, thlm, wp2, rtp2, thlp2, Skw,
                   wprtp, wpthlp, rtpthlp,
                   clubb_params, Skrt, Skthl,
                   pdf_implicit_coefs_terms):
    """new_pdf_main.F90:new_pdf_driver.

    w always sets the PDF; rt and thl respond. Returns the Fortran out/inout
    values by name, including the packed implicit_coefs_terms fields used by
    turbulent-advection solvers.
    """
    wm = jnp.asarray(wm, dtype=jnp.float64); rtm = jnp.asarray(rtm, dtype=jnp.float64); thlm = jnp.asarray(thlm, dtype=jnp.float64)
    wp2 = jnp.asarray(wp2, dtype=jnp.float64); rtp2 = jnp.asarray(rtp2, dtype=jnp.float64); thlp2 = jnp.asarray(thlp2, dtype=jnp.float64)
    Skw = jnp.asarray(Skw, dtype=jnp.float64); rtpthlp = jnp.asarray(rtpthlp, dtype=jnp.float64)
    wprtp = jnp.asarray(wprtp, dtype=jnp.float64); wpthlp = jnp.asarray(wpthlp, dtype=jnp.float64)
    cp = jnp.asarray(clubb_params, dtype=jnp.float64)

    sgn_wprtp = jnp.where(wprtp >= 0.0, 1.0, -1.0)
    sgn_wpthlp = jnp.where(wpthlp >= 0.0, 1.0, -1.0)
    sgn_wp2 = jnp.ones_like(Skw)

    # Adjusted rt-thl correlation and the exp_factor that reduces F_rt.
    has_var = (rtp2 >= rt_tol ** 2) & (thlp2 >= thl_tol ** 2)
    denom = jnp.sqrt(jnp.where(has_var, rtp2 * thlp2, 1.0))
    adj = jnp.clip(rtpthlp / denom * sgn_wprtp * sgn_wpthlp, -max_mag_correlation, max_mag_correlation)
    exp_factor_rt = jnp.where(has_var, 1.0 - jnp.exp(-0.2 * (adj + 1.0) ** 5), 1.0)
    exp_factor_thl = jnp.ones_like(Skw)

    slope_w = cp[:, islope_coef_spread_DG_means_w:islope_coef_spread_DG_means_w + 1]
    stdev_w = cp[:, ipdf_component_stdev_factor_w:ipdf_component_stdev_factor_w + 1]
    coef_rt = cp[:, icoef_spread_DG_means_rt:icoef_spread_DG_means_rt + 1]
    coef_thl = cp[:, icoef_spread_DG_means_thl:icoef_spread_DG_means_thl + 1]

    F_w, zeta_w, min_F_w, max_F_w = calc_F_x_zeta_x_setter(Skw, slope_w, stdev_w, _LAMBDA_W)
    (mu_w_1, mu_w_2, sigma_w_1, sigma_w_2, mixt_frac,
     coef_sigma_w_1_sqd, coef_sigma_w_2_sqd) = calc_setter_var_params(
        wm, wp2, Skw, sgn_wp2, F_w, zeta_w)
    sigma_w_1_sqd = sigma_w_1 ** 2
    sigma_w_2_sqd = sigma_w_2 ** 2

    mf = mixt_frac
    max_Skx2_pos = 4.0 * (1.0 - mf) ** 2 / (mf * (2.0 - mf))
    max_Skx2_neg = 4.0 * mf ** 2 / (1.0 - mf ** 2)

    (mu_rt_1, mu_rt_2, sigma_rt_1_sqd, sigma_rt_2_sqd,
     coef_sigma_rt_1_sqd, coef_sigma_rt_2_sqd,
     F_rt, min_F_rt, max_F_rt, Skrt) = calc_responder_var(
        rtm, rtp2, sgn_wprtp, mf, coef_rt, exp_factor_rt, max_Skx2_pos, max_Skx2_neg, Skrt)
    (mu_thl_1, mu_thl_2, sigma_thl_1_sqd, sigma_thl_2_sqd,
     coef_sigma_thl_1_sqd, coef_sigma_thl_2_sqd,
     F_thl, min_F_thl, max_F_thl, Skthl) = calc_responder_var(
        thlm, thlp2, sgn_wpthlp, mf, coef_thl, exp_factor_thl, max_Skx2_pos, max_Skx2_neg, Skthl)

    zero = jnp.zeros_like(wp2)

    if not l_explicit_turbulent_adv_wp3:
        coef_wp4_implicit = calc_coef_wp4_implicit(
            mixt_frac, F_w, coef_sigma_w_1_sqd, coef_sigma_w_2_sqd)
    else:
        coef_wp4_implicit = zero

    if not l_explicit_turbulent_adv_xpyp:
        coef_wprtp2_implicit = calc_coef_wpxp2_implicit(
            wp2, rtp2, wprtp, sgn_wprtp,
            mixt_frac, F_w, F_rt,
            coef_sigma_w_1_sqd, coef_sigma_w_2_sqd,
            coef_sigma_rt_1_sqd, coef_sigma_rt_2_sqd)
        coef_wpthlp2_implicit = calc_coef_wpxp2_implicit(
            wp2, thlp2, wpthlp, sgn_wpthlp,
            mixt_frac, F_w, F_thl,
            coef_sigma_w_1_sqd, coef_sigma_w_2_sqd,
            coef_sigma_thl_1_sqd, coef_sigma_thl_2_sqd)
        coef_wprtpthlp_implicit, term_wprtpthlp_explicit = calc_coefs_wpxpyp_semiimpl(
            wp2, rtp2, thlp2, wprtp, wpthlp, sgn_wprtp, sgn_wpthlp,
            mixt_frac, F_w, F_rt, F_thl,
            coef_sigma_w_1_sqd, coef_sigma_w_2_sqd,
            coef_sigma_rt_1_sqd, coef_sigma_rt_2_sqd,
            coef_sigma_thl_1_sqd, coef_sigma_thl_2_sqd)
    else:
        coef_wprtp2_implicit = zero
        coef_wpthlp2_implicit = zero
        coef_wprtpthlp_implicit = zero
        term_wprtpthlp_explicit = zero

    if not l_explicit_turbulent_adv_wpxp:
        coef_wp2rtp_implicit, term_wp2rtp_explicit = calc_coefs_wp2xp_semiimpl(
            wp2, rtp2, sgn_wprtp,
            mixt_frac, F_w, F_rt,
            coef_sigma_w_1_sqd, coef_sigma_w_2_sqd,
            coef_sigma_rt_1_sqd, coef_sigma_rt_2_sqd)
        coef_wp2thlp_implicit, term_wp2thlp_explicit = calc_coefs_wp2xp_semiimpl(
            wp2, thlp2, sgn_wpthlp,
            mixt_frac, F_w, F_thl,
            coef_sigma_w_1_sqd, coef_sigma_w_2_sqd,
            coef_sigma_thl_1_sqd, coef_sigma_thl_2_sqd)
    else:
        coef_wp2rtp_implicit = zero
        term_wp2rtp_explicit = zero
        coef_wp2thlp_implicit = zero
        term_wp2thlp_explicit = zero

    pdf_implicit_coefs_terms = pdf_implicit_coefs_terms.replace(
        coef_wp4_implicit=coef_wp4_implicit,
        coef_wprtp2_implicit=coef_wprtp2_implicit,
        coef_wpthlp2_implicit=coef_wpthlp2_implicit,
        coef_wp2rtp_implicit=coef_wp2rtp_implicit,
        term_wp2rtp_explicit=term_wp2rtp_explicit,
        coef_wp2thlp_implicit=coef_wp2thlp_implicit,
        term_wp2thlp_explicit=term_wp2thlp_explicit,
        coef_wprtpthlp_implicit=coef_wprtpthlp_implicit,
        term_wprtpthlp_explicit=term_wprtpthlp_explicit,
    )

    return {
        "Skrt": Skrt,
        "Skthl": Skthl,
        "mu_w_1": mu_w_1,
        "mu_w_2": mu_w_2,
        "mu_rt_1": mu_rt_1,
        "mu_rt_2": mu_rt_2,
        "mu_thl_1": mu_thl_1,
        "mu_thl_2": mu_thl_2,
        "sigma_w_1_sqd": sigma_w_1_sqd,
        "sigma_w_2_sqd": sigma_w_2_sqd,
        "sigma_rt_1_sqd": sigma_rt_1_sqd,
        "sigma_rt_2_sqd": sigma_rt_2_sqd,
        "sigma_thl_1_sqd": sigma_thl_1_sqd,
        "sigma_thl_2_sqd": sigma_thl_2_sqd,
        "mixt_frac": mixt_frac,
        "pdf_implicit_coefs_terms": pdf_implicit_coefs_terms,
    }
