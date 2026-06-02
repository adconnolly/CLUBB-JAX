"""KK upscaled second-moment microphysics covariances — JAX port of KK_upscaled_covariances.F90.

These give the covariances of the KK process tendencies (auto/accr/evap) with rt/thl/w that become the
SECOND-MOMENT microphysics source terms wprtp_mc/wpthlp_mc/rtp2_mc/thlp2_mc/rtpthlp_mc (the Iter158-localized
gap that keeps KK from being bit-faithful). They compose the validated `trivar_NNL_covar_eq` covariance
integral (PDF_integrals_covar.py) with the already-ported `bivar_NL_mean_eq` mean integral. Differentiable.
"""
import jax.numpy as jnp

from clubb_jax.src.Microphys.KK_microphys.PDF_integrals_covar import (
    trivar_NNL_covar_eq, quadrivar_NNLL_covar_eq)
from clubb_jax.src.Microphys.KK_microphys.KK_upscaled_means import (
    bivar_NL_mean_eq, trivar_NLL_mean_eq, KK_AUTO_RC_EXP, KK_AUTO_NC_EXP, KK_ACCR_RC_EXP, KK_ACCR_RR_EXP,
    KK_EVAP_SUPERSAT_EXP, KK_EVAP_RR_EXP, KK_EVAP_NR_EXP, NC_TOL, RR_TOL, NR_TOL)


def _covar_x_evap_comp(mu_eta, mu_chi, mu_rr, mu_Nr, mu_rr_n, mu_Nr_n, sigma_eta, sigma_chi, sigma_rr, sigma_Nr,
                       sigma_rr_n, sigma_Nr_n, corr_eta_chi, corr_eta_rr_n, corr_eta_Nr_n, corr_chi_rr_n,
                       corr_chi_Nr_n, corr_rr_Nr_n, xm, mu_x, kk_tndcy, kk_coef, eta_tol, c, precip_frac, s):
    """Per-PDF-component Cov(x, KK_evap) for x=r_t (s=+1) / thl (s=-1) — same x'=(1/(2c))(eta'∓chi') form as
    auto/accr but the 4-var subsaturated quadrivar covariance + trivar_NLL (chi^α r_r^β N_r^γ) means."""
    a, bb, gg = KK_EVAP_SUPERSAT_EXP, KK_EVAP_RR_EXP, KK_EVAP_NR_EXP
    quad = quadrivar_NNLL_covar_eq(mu_eta, mu_chi, mu_rr, mu_Nr, mu_rr_n, mu_Nr_n, sigma_eta, sigma_chi, sigma_rr,
                                   sigma_Nr, sigma_rr_n, sigma_Nr_n, corr_eta_chi, corr_eta_rr_n, corr_eta_Nr_n,
                                   corr_chi_rr_n, corr_chi_Nr_n, corr_rr_Nr_n, mu_eta, kk_tndcy, kk_coef,
                                   eta_tol, RR_TOL, NR_TOL, a, bb, gg)
    m_a1 = trivar_NLL_mean_eq(mu_chi, mu_rr, mu_Nr, mu_rr_n, mu_Nr_n, sigma_chi, sigma_rr, sigma_Nr, sigma_rr_n,
                              sigma_Nr_n, corr_chi_rr_n, corr_chi_Nr_n, corr_rr_Nr_n, a + 1.0, bb, gg)
    m_a = trivar_NLL_mean_eq(mu_chi, mu_rr, mu_Nr, mu_rr_n, mu_Nr_n, sigma_chi, sigma_rr, sigma_Nr, sigma_rr_n,
                             sigma_Nr_n, corr_chi_rr_n, corr_chi_Nr_n, corr_rr_Nr_n, a, bb, gg)
    return kk_coef * precip_frac * ((1.0 / (2.0 * c)) * (quad + s * m_a1)
                                    + (mu_x - xm - s * mu_chi / (2.0 * c)) * m_a)


def _covar_x_comp(mu_eta, mu_chi, mu_y, mu_y_n, sigma_eta, sigma_chi, sigma_y, sigma_y_n,
                  corr_eta_chi, corr_eta_y_n, corr_chi_y_n, xm, mu_x, kk_tndcy, kk_coef,
                  eta_tol, y_tol, c, alpha, beta, precip_frac, s):
    """Per-PDF-component contribution to Cov(x, KK_<process>) for x = r_t (s=+1) or thl (s=-1), via the ADG1
    transform x' = (1/(2 c))(eta' ∓ chi'). Auto (y=N_cn, precip_frac=1) and accr (y=r_r, ×precip_frac) share
    this form. KK_upscaled_covariances.F90:covar_{rt,thl}_KK_{auto,accr}."""
    tri = trivar_NNL_covar_eq(mu_eta, mu_chi, mu_y, mu_y_n, sigma_eta, sigma_chi, sigma_y,
                              sigma_y_n, corr_eta_chi, corr_eta_y_n, corr_chi_y_n,
                              mu_eta, kk_tndcy, kk_coef, eta_tol, y_tol, alpha, beta)
    biv_a1 = bivar_NL_mean_eq(mu_chi, mu_y, mu_y_n, sigma_chi, sigma_y, sigma_y_n,
                              corr_chi_y_n, y_tol, alpha + 1.0, beta)
    biv_a = bivar_NL_mean_eq(mu_chi, mu_y, mu_y_n, sigma_chi, sigma_y, sigma_y_n,
                             corr_chi_y_n, y_tol, alpha, beta)
    return kk_coef * precip_frac * ((1.0 / (2.0 * c)) * (tri + s * biv_a1)
                                    + (mu_x - xm - s * mu_chi / (2.0 * c)) * biv_a)


def covar_rt_KK_auto(mu_eta_1, mu_eta_2, mu_chi_1, mu_chi_2, mu_Ncn_1, mu_Ncn_2, mu_Ncn_1_n, mu_Ncn_2_n,
                     sigma_eta_1, sigma_eta_2, sigma_chi_1, sigma_chi_2, sigma_Ncn_1, sigma_Ncn_2,
                     sigma_Ncn_1_n, sigma_Ncn_2_n, corr_eta_chi_1, corr_eta_chi_2, corr_eta_Ncn_1_n,
                     corr_eta_Ncn_2_n, corr_chi_Ncn_1_n, corr_chi_Ncn_2_n, rtm, mu_rt_1, mu_rt_2,
                     KK_auto_tndcy, KK_auto_coef, eta_tol, crt1, crt2, mixt_frac):
    """Covariance of r_t with the KK autoconversion tendency. KK_upscaled_covariances.F90:covar_rt_KK_auto."""
    a, b = KK_AUTO_RC_EXP, KK_AUTO_NC_EXP
    c1 = _covar_x_comp(mu_eta_1, mu_chi_1, mu_Ncn_1, mu_Ncn_1_n, sigma_eta_1, sigma_chi_1, sigma_Ncn_1,
                       sigma_Ncn_1_n, corr_eta_chi_1, corr_eta_Ncn_1_n, corr_chi_Ncn_1_n, rtm, mu_rt_1,
                       KK_auto_tndcy, KK_auto_coef, eta_tol, NC_TOL, crt1, a, b, 1.0, 1.0)
    c2 = _covar_x_comp(mu_eta_2, mu_chi_2, mu_Ncn_2, mu_Ncn_2_n, sigma_eta_2, sigma_chi_2, sigma_Ncn_2,
                       sigma_Ncn_2_n, corr_eta_chi_2, corr_eta_Ncn_2_n, corr_chi_Ncn_2_n, rtm, mu_rt_2,
                       KK_auto_tndcy, KK_auto_coef, eta_tol, NC_TOL, crt2, a, b, 1.0, 1.0)
    return mixt_frac * c1 + (1.0 - mixt_frac) * c2


def covar_rt_KK_accr(mu_eta_1, mu_eta_2, mu_chi_1, mu_chi_2, mu_rr_1, mu_rr_2, mu_rr_1_n, mu_rr_2_n,
                     sigma_eta_1, sigma_eta_2, sigma_chi_1, sigma_chi_2, sigma_rr_1, sigma_rr_2,
                     sigma_rr_1_n, sigma_rr_2_n, corr_eta_chi_1, corr_eta_chi_2, corr_eta_rr_1_n,
                     corr_eta_rr_2_n, corr_chi_rr_1_n, corr_chi_rr_2_n, rtm, mu_rt_1, mu_rt_2,
                     KK_accr_tndcy, KK_accr_coef, eta_tol, crt1, crt2, mixt_frac, precip_frac_1, precip_frac_2):
    """Covariance of r_t with the KK accretion tendency (y=r_r, ×precip_frac). Same composition as auto.
    KK_upscaled_covariances.F90:covar_rt_KK_accr."""
    a, b = KK_ACCR_RC_EXP, KK_ACCR_RR_EXP
    c1 = _covar_x_comp(mu_eta_1, mu_chi_1, mu_rr_1, mu_rr_1_n, sigma_eta_1, sigma_chi_1, sigma_rr_1,
                       sigma_rr_1_n, corr_eta_chi_1, corr_eta_rr_1_n, corr_chi_rr_1_n, rtm, mu_rt_1,
                       KK_accr_tndcy, KK_accr_coef, eta_tol, RR_TOL, crt1, a, b, precip_frac_1, 1.0)
    c2 = _covar_x_comp(mu_eta_2, mu_chi_2, mu_rr_2, mu_rr_2_n, sigma_eta_2, sigma_chi_2, sigma_rr_2,
                       sigma_rr_2_n, corr_eta_chi_2, corr_eta_rr_2_n, corr_chi_rr_2_n, rtm, mu_rt_2,
                       KK_accr_tndcy, KK_accr_coef, eta_tol, RR_TOL, crt2, a, b, precip_frac_2, 1.0)
    return mixt_frac * c1 + (1.0 - mixt_frac) * c2


def covar_thl_KK_auto(mu_eta_1, mu_eta_2, mu_chi_1, mu_chi_2, mu_Ncn_1, mu_Ncn_2, mu_Ncn_1_n, mu_Ncn_2_n,
                      sigma_eta_1, sigma_eta_2, sigma_chi_1, sigma_chi_2, sigma_Ncn_1, sigma_Ncn_2,
                      sigma_Ncn_1_n, sigma_Ncn_2_n, corr_eta_chi_1, corr_eta_chi_2, corr_eta_Ncn_1_n,
                      corr_eta_Ncn_2_n, corr_chi_Ncn_1_n, corr_chi_Ncn_2_n, thlm, mu_thl_1, mu_thl_2,
                      KK_auto_tndcy, KK_auto_coef, eta_tol, cthl1, cthl2, mixt_frac):
    """Covariance of thl with the KK autoconversion tendency (thl' = (eta'-chi')/(2 c_thl), so s=-1).
    KK_upscaled_covariances.F90:covar_thl_KK_auto."""
    a, b = KK_AUTO_RC_EXP, KK_AUTO_NC_EXP
    c1 = _covar_x_comp(mu_eta_1, mu_chi_1, mu_Ncn_1, mu_Ncn_1_n, sigma_eta_1, sigma_chi_1, sigma_Ncn_1,
                       sigma_Ncn_1_n, corr_eta_chi_1, corr_eta_Ncn_1_n, corr_chi_Ncn_1_n, thlm, mu_thl_1,
                       KK_auto_tndcy, KK_auto_coef, eta_tol, NC_TOL, cthl1, a, b, 1.0, -1.0)
    c2 = _covar_x_comp(mu_eta_2, mu_chi_2, mu_Ncn_2, mu_Ncn_2_n, sigma_eta_2, sigma_chi_2, sigma_Ncn_2,
                       sigma_Ncn_2_n, corr_eta_chi_2, corr_eta_Ncn_2_n, corr_chi_Ncn_2_n, thlm, mu_thl_2,
                       KK_auto_tndcy, KK_auto_coef, eta_tol, NC_TOL, cthl2, a, b, 1.0, -1.0)
    return mixt_frac * c1 + (1.0 - mixt_frac) * c2


def covar_thl_KK_accr(mu_eta_1, mu_eta_2, mu_chi_1, mu_chi_2, mu_rr_1, mu_rr_2, mu_rr_1_n, mu_rr_2_n,
                      sigma_eta_1, sigma_eta_2, sigma_chi_1, sigma_chi_2, sigma_rr_1, sigma_rr_2,
                      sigma_rr_1_n, sigma_rr_2_n, corr_eta_chi_1, corr_eta_chi_2, corr_eta_rr_1_n,
                      corr_eta_rr_2_n, corr_chi_rr_1_n, corr_chi_rr_2_n, thlm, mu_thl_1, mu_thl_2,
                      KK_accr_tndcy, KK_accr_coef, eta_tol, cthl1, cthl2, mixt_frac, precip_frac_1, precip_frac_2):
    """Covariance of thl with the KK accretion tendency (s=-1, y=r_r, ×precip_frac).
    KK_upscaled_covariances.F90:covar_thl_KK_accr."""
    a, b = KK_ACCR_RC_EXP, KK_ACCR_RR_EXP
    c1 = _covar_x_comp(mu_eta_1, mu_chi_1, mu_rr_1, mu_rr_1_n, sigma_eta_1, sigma_chi_1, sigma_rr_1,
                       sigma_rr_1_n, corr_eta_chi_1, corr_eta_rr_1_n, corr_chi_rr_1_n, thlm, mu_thl_1,
                       KK_accr_tndcy, KK_accr_coef, eta_tol, RR_TOL, cthl1, a, b, precip_frac_1, -1.0)
    c2 = _covar_x_comp(mu_eta_2, mu_chi_2, mu_rr_2, mu_rr_2_n, sigma_eta_2, sigma_chi_2, sigma_rr_2,
                       sigma_rr_2_n, corr_eta_chi_2, corr_eta_rr_2_n, corr_chi_rr_2_n, thlm, mu_thl_2,
                       KK_accr_tndcy, KK_accr_coef, eta_tol, RR_TOL, cthl2, a, b, precip_frac_2, -1.0)
    return mixt_frac * c1 + (1.0 - mixt_frac) * c2


def covar_x_KK_auto(mu_x_1, mu_x_2, mu_chi_1, mu_chi_2, mu_Ncn_1, mu_Ncn_2, mu_Ncn_1_n, mu_Ncn_2_n,
                    sigma_x_1, sigma_x_2, sigma_chi_1, sigma_chi_2, sigma_Ncn_1, sigma_Ncn_2,
                    sigma_Ncn_1_n, sigma_Ncn_2_n, corr_x_chi_1, corr_x_chi_2, corr_x_Ncn_1_n,
                    corr_x_Ncn_2_n, corr_chi_Ncn_1_n, corr_chi_Ncn_2_n, x_mean, KK_auto_tndcy,
                    KK_auto_coef, x_tol, mixt_frac):
    """Covariance of x (=w) with the KK autoconversion tendency — x is a direct PDF variable, so this is the
    plain KK_coef·<mixt blend of trivar_NNL_covar_eq(x, chi, N_cn)>. KK_upscaled_covariances.F90:covar_x_KK_auto."""
    a, b = KK_AUTO_RC_EXP, KK_AUTO_NC_EXP
    t1 = trivar_NNL_covar_eq(mu_x_1, mu_chi_1, mu_Ncn_1, mu_Ncn_1_n, sigma_x_1, sigma_chi_1, sigma_Ncn_1,
                             sigma_Ncn_1_n, corr_x_chi_1, corr_x_Ncn_1_n, corr_chi_Ncn_1_n, x_mean,
                             KK_auto_tndcy, KK_auto_coef, x_tol, NC_TOL, a, b)
    t2 = trivar_NNL_covar_eq(mu_x_2, mu_chi_2, mu_Ncn_2, mu_Ncn_2_n, sigma_x_2, sigma_chi_2, sigma_Ncn_2,
                             sigma_Ncn_2_n, corr_x_chi_2, corr_x_Ncn_2_n, corr_chi_Ncn_2_n, x_mean,
                             KK_auto_tndcy, KK_auto_coef, x_tol, NC_TOL, a, b)
    return KK_auto_coef * (mixt_frac * t1 + (1.0 - mixt_frac) * t2)


def covar_x_KK_accr(mu_x_1, mu_x_2, mu_chi_1, mu_chi_2, mu_rr_1, mu_rr_2, mu_rr_1_n, mu_rr_2_n,
                    sigma_x_1, sigma_x_2, sigma_chi_1, sigma_chi_2, sigma_rr_1, sigma_rr_2,
                    sigma_rr_1_n, sigma_rr_2_n, corr_x_chi_1, corr_x_chi_2, corr_x_rr_1_n,
                    corr_x_rr_2_n, corr_chi_rr_1_n, corr_chi_rr_2_n, x_mean, KK_accr_tndcy,
                    KK_accr_coef, x_tol, mixt_frac, precip_frac_1, precip_frac_2):
    """Covariance of x (=w) with the KK accretion tendency. Unlike auto, accretion has the
    out-of-precipitation correction `−(1−precip_frac)·(mu_x−x_mean)·KK_accr_tndcy` per component.
    KK_upscaled_covariances.F90:covar_x_KK_accr."""
    a, b = KK_ACCR_RC_EXP, KK_ACCR_RR_EXP
    t1 = trivar_NNL_covar_eq(mu_x_1, mu_chi_1, mu_rr_1, mu_rr_1_n, sigma_x_1, sigma_chi_1, sigma_rr_1,
                             sigma_rr_1_n, corr_x_chi_1, corr_x_rr_1_n, corr_chi_rr_1_n, x_mean,
                             KK_accr_tndcy, KK_accr_coef, x_tol, RR_TOL, a, b)
    t2 = trivar_NNL_covar_eq(mu_x_2, mu_chi_2, mu_rr_2, mu_rr_2_n, sigma_x_2, sigma_chi_2, sigma_rr_2,
                             sigma_rr_2_n, corr_x_chi_2, corr_x_rr_2_n, corr_chi_rr_2_n, x_mean,
                             KK_accr_tndcy, KK_accr_coef, x_tol, RR_TOL, a, b)
    comp1 = KK_accr_coef * precip_frac_1 * t1 - (1.0 - precip_frac_1) * (mu_x_1 - x_mean) * KK_accr_tndcy
    comp2 = KK_accr_coef * precip_frac_2 * t2 - (1.0 - precip_frac_2) * (mu_x_2 - x_mean) * KK_accr_tndcy
    return mixt_frac * comp1 + (1.0 - mixt_frac) * comp2


def _covar_x_KK_evap(xm, mu_x_1, mu_x_2, c1_, c2_, s, *,
                     mu_eta_1, mu_eta_2, mu_chi_1, mu_chi_2, mu_rr_1, mu_rr_2, mu_Nr_1, mu_Nr_2,
                     mu_rr_1_n, mu_rr_2_n, mu_Nr_1_n, mu_Nr_2_n, sigma_eta_1, sigma_eta_2, sigma_chi_1,
                     sigma_chi_2, sigma_rr_1, sigma_rr_2, sigma_Nr_1, sigma_Nr_2, sigma_rr_1_n, sigma_rr_2_n,
                     sigma_Nr_1_n, sigma_Nr_2_n, corr_eta_chi_1, corr_eta_chi_2, corr_eta_rr_1_n, corr_eta_rr_2_n,
                     corr_eta_Nr_1_n, corr_eta_Nr_2_n, corr_chi_rr_1_n, corr_chi_rr_2_n, corr_chi_Nr_1_n,
                     corr_chi_Nr_2_n, corr_rr_Nr_1_n, corr_rr_Nr_2_n, KK_evap_tndcy, KK_evap_coef, eta_tol,
                     mixt_frac, precip_frac_1, precip_frac_2):
    """Shared rt/thl evaporation covariance (s=+1 for r_t, s=-1 for thl)."""
    d1 = _covar_x_evap_comp(mu_eta_1, mu_chi_1, mu_rr_1, mu_Nr_1, mu_rr_1_n, mu_Nr_1_n, sigma_eta_1, sigma_chi_1,
                            sigma_rr_1, sigma_Nr_1, sigma_rr_1_n, sigma_Nr_1_n, corr_eta_chi_1, corr_eta_rr_1_n,
                            corr_eta_Nr_1_n, corr_chi_rr_1_n, corr_chi_Nr_1_n, corr_rr_Nr_1_n, xm, mu_x_1,
                            KK_evap_tndcy, KK_evap_coef, eta_tol, c1_, precip_frac_1, s)
    d2 = _covar_x_evap_comp(mu_eta_2, mu_chi_2, mu_rr_2, mu_Nr_2, mu_rr_2_n, mu_Nr_2_n, sigma_eta_2, sigma_chi_2,
                            sigma_rr_2, sigma_Nr_2, sigma_rr_2_n, sigma_Nr_2_n, corr_eta_chi_2, corr_eta_rr_2_n,
                            corr_eta_Nr_2_n, corr_chi_rr_2_n, corr_chi_Nr_2_n, corr_rr_Nr_2_n, xm, mu_x_2,
                            KK_evap_tndcy, KK_evap_coef, eta_tol, c2_, precip_frac_2, s)
    return mixt_frac * d1 + (1.0 - mixt_frac) * d2


def covar_rt_KK_evap(rtm, mu_rt_1, mu_rt_2, crt1, crt2, **kw):
    """Cov(r_t, KK evaporation tendency). KK_upscaled_covariances.F90:covar_rt_KK_evap. (Moment kwargs match
    `_covar_x_KK_evap`.)"""
    return _covar_x_KK_evap(rtm, mu_rt_1, mu_rt_2, crt1, crt2, 1.0, **kw)


def covar_thl_KK_evap(thlm, mu_thl_1, mu_thl_2, cthl1, cthl2, **kw):
    """Cov(thl, KK evaporation tendency) (s=-1). KK_upscaled_covariances.F90:covar_thl_KK_evap."""
    return _covar_x_KK_evap(thlm, mu_thl_1, mu_thl_2, cthl1, cthl2, -1.0, **kw)


def covar_x_KK_evap(mu_x_1, mu_x_2, mu_chi_1, mu_chi_2, mu_rr_1, mu_rr_2, mu_Nr_1, mu_Nr_2, mu_rr_1_n, mu_rr_2_n,
                    mu_Nr_1_n, mu_Nr_2_n, sigma_x_1, sigma_x_2, sigma_chi_1, sigma_chi_2, sigma_rr_1, sigma_rr_2,
                    sigma_Nr_1, sigma_Nr_2, sigma_rr_1_n, sigma_rr_2_n, sigma_Nr_1_n, sigma_Nr_2_n,
                    corr_x_chi_1, corr_x_chi_2, corr_x_rr_1_n, corr_x_rr_2_n, corr_x_Nr_1_n, corr_x_Nr_2_n,
                    corr_chi_rr_1_n, corr_chi_rr_2_n, corr_chi_Nr_1_n, corr_chi_Nr_2_n, corr_rr_Nr_1_n,
                    corr_rr_Nr_2_n, x_mean, KK_evap_tndcy, KK_evap_coef, x_tol, mixt_frac, precip_frac_1,
                    precip_frac_2):
    """Cov(x=w, KK evaporation tendency) — direct quadrivar + the out-of-precip correction (like accr).
    KK_upscaled_covariances.F90:covar_x_KK_evap."""
    a, b, g = KK_EVAP_SUPERSAT_EXP, KK_EVAP_RR_EXP, KK_EVAP_NR_EXP
    q1 = quadrivar_NNLL_covar_eq(mu_x_1, mu_chi_1, mu_rr_1, mu_Nr_1, mu_rr_1_n, mu_Nr_1_n, sigma_x_1, sigma_chi_1,
                                 sigma_rr_1, sigma_Nr_1, sigma_rr_1_n, sigma_Nr_1_n, corr_x_chi_1, corr_x_rr_1_n,
                                 corr_x_Nr_1_n, corr_chi_rr_1_n, corr_chi_Nr_1_n, corr_rr_Nr_1_n, x_mean,
                                 KK_evap_tndcy, KK_evap_coef, x_tol, RR_TOL, NR_TOL, a, b, g)
    q2 = quadrivar_NNLL_covar_eq(mu_x_2, mu_chi_2, mu_rr_2, mu_Nr_2, mu_rr_2_n, mu_Nr_2_n, sigma_x_2, sigma_chi_2,
                                 sigma_rr_2, sigma_Nr_2, sigma_rr_2_n, sigma_Nr_2_n, corr_x_chi_2, corr_x_rr_2_n,
                                 corr_x_Nr_2_n, corr_chi_rr_2_n, corr_chi_Nr_2_n, corr_rr_Nr_2_n, x_mean,
                                 KK_evap_tndcy, KK_evap_coef, x_tol, RR_TOL, NR_TOL, a, b, g)
    comp1 = KK_evap_coef * precip_frac_1 * q1 - (1.0 - precip_frac_1) * (mu_x_1 - x_mean) * KK_evap_tndcy
    comp2 = KK_evap_coef * precip_frac_2 * q2 - (1.0 - precip_frac_2) * (mu_x_2 - x_mean) * KK_evap_tndcy
    return mixt_frac * comp1 + (1.0 - mixt_frac) * comp2


_LV = 2.5e6
_CP = 1004.67
_ETA_TOL = 1.0e-8   # = chi_tol
_W_TOL = 2.0e-2


def KK_upscaled_covar_driver(
        w_mean, rtm, thlm, exner, mu_w_1, mu_w_2, mu_chi_1, mu_chi_2, mu_eta_1, mu_eta_2,
        mu_rr_1, mu_rr_2, mu_Nr_1, mu_Nr_2, mu_Ncn_1, mu_Ncn_2, mu_rr_1_n, mu_rr_2_n, mu_Nr_1_n, mu_Nr_2_n,
        mu_Ncn_1_n, mu_Ncn_2_n, sigma_w_1, sigma_w_2, sigma_chi_1, sigma_chi_2, sigma_eta_1, sigma_eta_2,
        sigma_rr_1, sigma_rr_2, sigma_Nr_1, sigma_Nr_2, sigma_Ncn_1, sigma_Ncn_2, sigma_rr_1_n, sigma_rr_2_n,
        sigma_Nr_1_n, sigma_Nr_2_n, sigma_Ncn_1_n, sigma_Ncn_2_n, corr_w_chi_1, corr_w_chi_2,
        corr_w_rr_1_n, corr_w_rr_2_n, corr_w_Nr_1_n, corr_w_Nr_2_n, corr_w_Ncn_1_n, corr_w_Ncn_2_n,
        corr_chi_eta_1, corr_chi_eta_2, corr_chi_rr_1_n, corr_chi_rr_2_n, corr_chi_Nr_1_n, corr_chi_Nr_2_n,
        corr_chi_Ncn_1_n, corr_chi_Ncn_2_n, corr_eta_rr_1_n, corr_eta_rr_2_n, corr_eta_Nr_1_n, corr_eta_Nr_2_n,
        corr_eta_Ncn_1_n, corr_eta_Ncn_2_n, corr_rr_Nr_1_n, corr_rr_Nr_2_n, mixt_frac, precip_frac_1,
        precip_frac_2, KK_evap_coef, KK_auto_coef, KK_accr_coef, KK_evap_tndcy, KK_auto_tndcy, KK_accr_tndcy,
        mu_rt_1, mu_rt_2, mu_thl_1, mu_thl_2, crt1, crt2, cthl1, cthl2):
    """Assemble the 5 second-moment microphysics tendencies (wprtp_mc, wpthlp_mc, rtp2_mc, thlp2_mc,
    rtpthlp_mc) from the 9 KK covariances. Faithful port of KK_upscaled_covariances.F90:KK_upscaled_covar_driver.
    Returns (wprtp_mc, wpthlp_mc, rtp2_mc, thlp2_mc, rtpthlp_mc)."""
    # auto covariances (y = N_cn)
    a_w = covar_x_KK_auto(mu_w_1, mu_w_2, mu_chi_1, mu_chi_2, mu_Ncn_1, mu_Ncn_2, mu_Ncn_1_n, mu_Ncn_2_n,
                          sigma_w_1, sigma_w_2, sigma_chi_1, sigma_chi_2, sigma_Ncn_1, sigma_Ncn_2,
                          sigma_Ncn_1_n, sigma_Ncn_2_n, corr_w_chi_1, corr_w_chi_2, corr_w_Ncn_1_n,
                          corr_w_Ncn_2_n, corr_chi_Ncn_1_n, corr_chi_Ncn_2_n, w_mean, KK_auto_tndcy,
                          KK_auto_coef, _W_TOL, mixt_frac)
    a_rt = covar_rt_KK_auto(mu_eta_1, mu_eta_2, mu_chi_1, mu_chi_2, mu_Ncn_1, mu_Ncn_2, mu_Ncn_1_n, mu_Ncn_2_n,
                            sigma_eta_1, sigma_eta_2, sigma_chi_1, sigma_chi_2, sigma_Ncn_1, sigma_Ncn_2,
                            sigma_Ncn_1_n, sigma_Ncn_2_n, corr_chi_eta_1, corr_chi_eta_2, corr_eta_Ncn_1_n,
                            corr_eta_Ncn_2_n, corr_chi_Ncn_1_n, corr_chi_Ncn_2_n, rtm, mu_rt_1, mu_rt_2,
                            KK_auto_tndcy, KK_auto_coef, _ETA_TOL, crt1, crt2, mixt_frac)
    a_thl = covar_thl_KK_auto(mu_eta_1, mu_eta_2, mu_chi_1, mu_chi_2, mu_Ncn_1, mu_Ncn_2, mu_Ncn_1_n, mu_Ncn_2_n,
                              sigma_eta_1, sigma_eta_2, sigma_chi_1, sigma_chi_2, sigma_Ncn_1, sigma_Ncn_2,
                              sigma_Ncn_1_n, sigma_Ncn_2_n, corr_chi_eta_1, corr_chi_eta_2, corr_eta_Ncn_1_n,
                              corr_eta_Ncn_2_n, corr_chi_Ncn_1_n, corr_chi_Ncn_2_n, thlm, mu_thl_1, mu_thl_2,
                              KK_auto_tndcy, KK_auto_coef, _ETA_TOL, cthl1, cthl2, mixt_frac)
    # accretion covariances (y = r_r, ×precip_frac)
    c_w = covar_x_KK_accr(mu_w_1, mu_w_2, mu_chi_1, mu_chi_2, mu_rr_1, mu_rr_2, mu_rr_1_n, mu_rr_2_n,
                          sigma_w_1, sigma_w_2, sigma_chi_1, sigma_chi_2, sigma_rr_1, sigma_rr_2, sigma_rr_1_n,
                          sigma_rr_2_n, corr_w_chi_1, corr_w_chi_2, corr_w_rr_1_n, corr_w_rr_2_n,
                          corr_chi_rr_1_n, corr_chi_rr_2_n, w_mean, KK_accr_tndcy, KK_accr_coef, _W_TOL,
                          mixt_frac, precip_frac_1, precip_frac_2)
    c_rt = covar_rt_KK_accr(mu_eta_1, mu_eta_2, mu_chi_1, mu_chi_2, mu_rr_1, mu_rr_2, mu_rr_1_n, mu_rr_2_n,
                            sigma_eta_1, sigma_eta_2, sigma_chi_1, sigma_chi_2, sigma_rr_1, sigma_rr_2,
                            sigma_rr_1_n, sigma_rr_2_n, corr_chi_eta_1, corr_chi_eta_2, corr_eta_rr_1_n,
                            corr_eta_rr_2_n, corr_chi_rr_1_n, corr_chi_rr_2_n, rtm, mu_rt_1, mu_rt_2,
                            KK_accr_tndcy, KK_accr_coef, _ETA_TOL, crt1, crt2, mixt_frac, precip_frac_1,
                            precip_frac_2)
    c_thl = covar_thl_KK_accr(mu_eta_1, mu_eta_2, mu_chi_1, mu_chi_2, mu_rr_1, mu_rr_2, mu_rr_1_n, mu_rr_2_n,
                              sigma_eta_1, sigma_eta_2, sigma_chi_1, sigma_chi_2, sigma_rr_1, sigma_rr_2,
                              sigma_rr_1_n, sigma_rr_2_n, corr_chi_eta_1, corr_chi_eta_2, corr_eta_rr_1_n,
                              corr_eta_rr_2_n, corr_chi_rr_1_n, corr_chi_rr_2_n, thlm, mu_thl_1, mu_thl_2,
                              KK_accr_tndcy, KK_accr_coef, _ETA_TOL, cthl1, cthl2, mixt_frac, precip_frac_1,
                              precip_frac_2)
    # evaporation covariances (4-var subsaturated, ×precip_frac)
    _evk = dict(mu_eta_1=mu_eta_1, mu_eta_2=mu_eta_2, mu_chi_1=mu_chi_1, mu_chi_2=mu_chi_2, mu_rr_1=mu_rr_1,
                mu_rr_2=mu_rr_2, mu_Nr_1=mu_Nr_1, mu_Nr_2=mu_Nr_2, mu_rr_1_n=mu_rr_1_n, mu_rr_2_n=mu_rr_2_n,
                mu_Nr_1_n=mu_Nr_1_n, mu_Nr_2_n=mu_Nr_2_n, sigma_eta_1=sigma_eta_1, sigma_eta_2=sigma_eta_2,
                sigma_chi_1=sigma_chi_1, sigma_chi_2=sigma_chi_2, sigma_rr_1=sigma_rr_1, sigma_rr_2=sigma_rr_2,
                sigma_Nr_1=sigma_Nr_1, sigma_Nr_2=sigma_Nr_2, sigma_rr_1_n=sigma_rr_1_n, sigma_rr_2_n=sigma_rr_2_n,
                sigma_Nr_1_n=sigma_Nr_1_n, sigma_Nr_2_n=sigma_Nr_2_n, corr_eta_chi_1=corr_chi_eta_1,
                corr_eta_chi_2=corr_chi_eta_2, corr_eta_rr_1_n=corr_eta_rr_1_n, corr_eta_rr_2_n=corr_eta_rr_2_n,
                corr_eta_Nr_1_n=corr_eta_Nr_1_n, corr_eta_Nr_2_n=corr_eta_Nr_2_n, corr_chi_rr_1_n=corr_chi_rr_1_n,
                corr_chi_rr_2_n=corr_chi_rr_2_n, corr_chi_Nr_1_n=corr_chi_Nr_1_n, corr_chi_Nr_2_n=corr_chi_Nr_2_n,
                corr_rr_Nr_1_n=corr_rr_Nr_1_n, corr_rr_Nr_2_n=corr_rr_Nr_2_n, KK_evap_tndcy=KK_evap_tndcy,
                KK_evap_coef=KK_evap_coef, eta_tol=_ETA_TOL, mixt_frac=mixt_frac, precip_frac_1=precip_frac_1,
                precip_frac_2=precip_frac_2)
    e_rt = covar_rt_KK_evap(rtm, mu_rt_1, mu_rt_2, crt1, crt2, **_evk)
    e_thl = covar_thl_KK_evap(thlm, mu_thl_1, mu_thl_2, cthl1, cthl2, **_evk)
    _evkx = {k: v for k, v in _evk.items()
             if k not in ('mu_eta_1', 'mu_eta_2', 'sigma_eta_1', 'sigma_eta_2', 'corr_eta_chi_1',
                          'corr_eta_chi_2', 'corr_eta_rr_1_n', 'corr_eta_rr_2_n', 'corr_eta_Nr_1_n',
                          'corr_eta_Nr_2_n', 'eta_tol')}
    e_w = covar_x_KK_evap(mu_w_1, mu_w_2, sigma_x_1=sigma_w_1, sigma_x_2=sigma_w_2, corr_x_chi_1=corr_w_chi_1,
                          corr_x_chi_2=corr_w_chi_2, corr_x_rr_1_n=corr_w_rr_1_n, corr_x_rr_2_n=corr_w_rr_2_n,
                          corr_x_Nr_1_n=corr_w_Nr_1_n, corr_x_Nr_2_n=corr_w_Nr_2_n, x_mean=w_mean, x_tol=_W_TOL,
                          **_evkx)

    w_tot = a_w + c_w + e_w
    rt_tot = a_rt + c_rt + e_rt
    thl_tot = a_thl + c_thl + e_thl
    L = _LV / (_CP * exner)
    wprtp_mc = -w_tot
    wpthlp_mc = L * w_tot
    rtp2_mc = -2.0 * rt_tot
    thlp2_mc = 2.0 * L * thl_tot
    rtpthlp_mc = L * rt_tot - thl_tot
    return wprtp_mc, wpthlp_mc, rtp2_mc, thlp2_mc, rtpthlp_mc
