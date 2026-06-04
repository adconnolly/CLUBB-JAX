"""JAX port of setup_clubb_pdf_params.F90 — hydrometeor PDF component moments.

calc_comp_mu_sigma_hm — the in-precip component means/stdevs (mu_hm_1/2,
sigma_hm_1/2) of a precipitating hydrometeor, which are the rate-function inputs
(KK accr/evap consume mu_rr_i/sigma_rr_i etc.). Oracle: setup_clubb_pdf_params.F90:1653.

The two in-precip component means are solved so the overall mean <hm> and overall
variance <hm'^2> are preserved (Griffin 2015):
  <hm> = a f_p_1 mu_1 + (1-a) f_p_2 mu_2
  <hm'^2> = a f_p_1 (1+omicron Rmax (1+zeta)) mu_1^2
            + (1-a) f_p_2 (1+omicron Rmax) mu_2^2 - <hm>^2
with R = omicron*Rmax the ratio sigma_2^2/mu_2^2, and sigma_1^2/mu_1^2 = R(1+zeta).
This gives a quadratic A mu_1^2 + B mu_1 + C = 0; the root is chosen by sign(mu_thl_1 -
mu_thl_2) so the component with more cloud also has the larger in-precip mean. Minimum
bounds (mu_hm_min_coef) and an "emergency" R-recompute handle small/degenerate cases.
Branches: precip in both components / comp 1 only / comp 2 only / neither.

compute_mean_stdev / norm_transform_mean_stdev (Iter131) — the orchestration that stacks
the per-PDF-variable component means/stdevs (chi, eta, w, Ncn, then the precipitating
hydrometeors) into the (ngrdcol, nzt, pdf_dim) arrays the rate functions index, and
transforms the lognormal variables (Ncn + hydrometeors) to normal (log) space. These are
the linear/normal-space inputs assembled by setup_pdf_parameters_api before the KK rate
calls (oracle setup_clubb_pdf_params.F90:818 + :2942). The PDF-variable index layout is
chi, eta, w, Ncn, <hydrometeors in hydromet-array order>, matching the iiPDF indices
(corr_varnce_module.F90:682). For KK that is [chi, eta, w, Ncn, rr, Nr], pdf_dim = 6.
"""
import jax
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import Ncn_tol, max_mag_correlation, w_tol, rc_tol
from clubb_jax.src.CLUBB_core.pdf_utilities import mean_L2N, stdev_L2N, corr_NN2NL, corr_NN2LL
from clubb_jax.src.CLUBB_core.matrix_operations import cholesky_factor

jax.config.update("jax_enable_x64", True)

_MU_HM_MIN_COEF = 0.01   # setup_clubb_pdf_params.F90:2176

# PDF-variable index layout (0-based), matching the iiPDF indices for the KK PDF.
IIPDF_CHI = 0
IIPDF_ETA = 1
IIPDF_W = 2
IIPDF_NCN = 3
# Precipitating hydrometeors follow Ncn in hydromet-array order (rr, Nr for KK).


def _safe_sqrt(x):
    """sqrt(x) for x>0, 0 otherwise, with a clean (nan-free) gradient at x<=0.
    The inner where keeps sqrt from ever being evaluated at <=0 in the autodiff graph
    (avoids the inf*0=nan that a bare jnp.sqrt(max(x,0)) produces at x=0)."""
    xp = jnp.where(x > 0.0, x, 1.0)
    return jnp.where(x > 0.0, jnp.sqrt(xp), 0.0)


def calc_comp_mu_sigma_hm(hmm, hmp2, hmp2_ip_on_hmm2_ip, mixt_frac,
                          precip_frac, precip_frac_1, precip_frac_2,
                          hm_tol, precip_frac_tol, mu_thl_1, mu_thl_2,
                          omicron, zeta_vrnce_rat_in):
    """In-precip component means/stdevs of a precipitating hydrometeor.

    All array args are (ngrdcol, nzt) except precip_frac_tol (ngrdcol,). omicron and
    zeta_vrnce_rat_in are scalars. Returns
    (mu_hm_1, mu_hm_2, sigma_hm_1, sigma_hm_2, hm_1, hm_2,
     sigma_hm_1_sqd_on_mu_hm_1_sqd, sigma_hm_2_sqd_on_mu_hm_2_sqd)."""
    hmm = jnp.asarray(hmm, dtype=jnp.float64)
    hmp2 = jnp.asarray(hmp2, dtype=jnp.float64)
    ratio = jnp.asarray(hmp2_ip_on_hmm2_ip, dtype=jnp.float64)
    a = jnp.asarray(mixt_frac, dtype=jnp.float64)
    fp = jnp.asarray(precip_frac, dtype=jnp.float64)
    fp1 = jnp.asarray(precip_frac_1, dtype=jnp.float64)
    fp2 = jnp.asarray(precip_frac_2, dtype=jnp.float64)
    mu_thl_1 = jnp.asarray(mu_thl_1, dtype=jnp.float64)
    mu_thl_2 = jnp.asarray(mu_thl_2, dtype=jnp.float64)
    pftol = jnp.asarray(precip_frac_tol, dtype=jnp.float64)[:, None]
    oma = 1.0 - a

    # Branch masks (mirror the Fortran if/elseif ladder).
    both = (hmm >= hm_tol) & (fp1 >= pftol) & (fp2 >= pftol)
    comp1 = (~both) & (hmm >= hm_tol) & (fp1 >= pftol)
    comp2 = (~both) & (~comp1) & (hmm >= hm_tol) & (fp2 >= pftol)

    # Safe denominators so unselected branches never divide by zero / poison gradients.
    fp1_s = jnp.where(fp1 > 0.0, fp1, 1.0)
    fp2_s = jnp.where(fp2 > 0.0, fp2, 1.0)
    a_s = jnp.where(a > 0.0, a, 1.0)
    oma_s = jnp.where(oma > 0.0, oma, 1.0)

    # ----- Branch: precipitation in BOTH PDF components (the quadratic solve) -----
    # zeta sign depends on mu_thl_1 vs mu_thl_2 (so the cloudier comp has the larger mean).
    thl_le = mu_thl_1 <= mu_thl_2
    zeta_flip = (1.0 / (1.0 + zeta_vrnce_rat_in)) - 1.0
    if zeta_vrnce_rat_in >= 0.0:
        zeta = jnp.where(thl_le, zeta_vrnce_rat_in, zeta_flip)
    else:
        zeta = jnp.where(thl_le, zeta_flip, zeta_vrnce_rat_in)

    # Guard the Rmax denominator: at no-precip points fp1=fp2=0 -> 0/0 = nan (forward and grad),
    # which poisons even though the both-precip branch is unselected there.
    den_R = a * fp1 * (1.0 + zeta) + oma * fp2
    Rmax = (fp / jnp.where(den_R > 0.0, den_R, 1.0)) * ratio
    oR = omicron * Rmax
    coef_A = (a * fp1 * (1.0 + oR * (1.0 + zeta))
              + a ** 2 * fp1 ** 2 * (1.0 + oR) / (oma_s * fp2_s))
    coef_B = -2.0 * hmm * a * fp1 * (1.0 + oR) / (oma_s * fp2_s)
    coef_C = -(hmp2 + (1.0 - (1.0 + oR) / (oma_s * fp2_s)) * hmm ** 2)
    disc = jnp.maximum(coef_B ** 2 - 4.0 * coef_A * coef_C, 0.0)
    coef_A_s = jnp.where(coef_A != 0.0, coef_A, 1.0)
    mu1 = jnp.where(thl_le, (-coef_B + _safe_sqrt(disc)) / (2.0 * coef_A_s),
                    (-coef_B - _safe_sqrt(disc)) / (2.0 * coef_A_s))
    mu2 = (hmm - a * fp1 * mu1) / (oma_s * fp2_s)
    R = oR

    # Minimum allowable component means.
    hm_ip = hmm / jnp.where(fp > 0.0, fp, 1.0)
    mu1_min = jnp.where(hm_ip > hm_tol / fp1_s,
                        jnp.minimum(hm_tol / fp1_s + _MU_HM_MIN_COEF * (hm_ip - hm_tol / fp1_s),
                                    (hmm - oma * hm_tol) / (a_s * fp1_s)),
                        hm_tol / fp1_s)
    mu2_min = jnp.where(hm_ip > hm_tol / fp2_s,
                        jnp.minimum(hm_tol / fp2_s + _MU_HM_MIN_COEF * (hm_ip - hm_tol / fp2_s),
                                    (hmm - a * hm_tol) / (oma_s * fp2_s)),
                        hm_tol / fp2_s)

    def _emergency_R(m1, m2):
        num = (hmp2 + hmm ** 2 - a * fp1 * m1 ** 2 - oma * fp2 * m2 ** 2)
        den = (a * fp1 * (1.0 + zeta) * m1 ** 2 + oma * fp2 * m2 ** 2)
        return jnp.maximum(num / jnp.where(den != 0.0, den, 1.0), 0.0)

    # Emergency: mu1 < mu1_min -> set mu1=mu1_min, recompute mu2 and R.
    mu1_e1 = mu1_min
    mu2_e1 = (hmm - a * fp1 * mu1_e1) / (oma_s * fp2_s)
    R_e1 = _emergency_R(mu1_e1, mu2_e1)
    # Emergency: mu2 < mu2_min -> set mu2=mu2_min, recompute mu1 and R.
    mu2_e2 = mu2_min
    mu1_e2 = (hmm - oma * fp2 * mu2_e2) / (a_s * fp1_s)
    R_e2 = _emergency_R(mu1_e2, mu2_e2)

    use_e1 = mu1 < mu1_min
    use_e2 = (~use_e1) & (mu2 < mu2_min)
    mu1_b = jnp.where(use_e1, mu1_e1, jnp.where(use_e2, mu1_e2, mu1))
    mu2_b = jnp.where(use_e1, mu2_e1, jnp.where(use_e2, mu2_e2, mu2))
    R_b = jnp.where(use_e1, R_e1, jnp.where(use_e2, R_e2, R))

    sig1_b = _safe_sqrt(R_b * (1.0 + zeta)) * mu1_b
    sig2_b = _safe_sqrt(R_b) * mu2_b
    hm1_b = jnp.maximum(mu1_b * fp1, hm_tol)
    hm2_b = jnp.maximum(mu2_b * fp2, hm_tol)

    # ----- Branch: precip in comp 1 only (precip_frac_2 = 0) -----
    mu1_c1 = hmm / (a_s * fp1_s)
    sig1_c1 = _safe_sqrt((hmp2 + hmm ** 2 - a * fp1 * mu1_c1 ** 2) / (a_s * fp1_s))
    hm1_c1 = mu1_c1 * fp1

    # ----- Branch: precip in comp 2 only (precip_frac_1 = 0) -----
    mu2_c2 = hmm / (oma_s * fp2_s)
    sig2_c2 = _safe_sqrt((hmp2 + hmm ** 2 - oma * fp2 * mu2_c2 ** 2) / (oma_s * fp2_s))
    hm2_c2 = mu2_c2 * fp2

    # ----- Select branch -----
    z = jnp.zeros_like(hmm)
    mu_hm_1 = jnp.where(both, mu1_b, jnp.where(comp1, mu1_c1, z))
    mu_hm_2 = jnp.where(both, mu2_b, jnp.where(comp2, mu2_c2, z))
    sigma_hm_1 = jnp.where(both, sig1_b, jnp.where(comp1, sig1_c1, z))
    sigma_hm_2 = jnp.where(both, sig2_b, jnp.where(comp2, sig2_c2, z))
    hm_1 = jnp.where(both, hm1_b, jnp.where(comp1, hm1_c1, z))
    hm_2 = jnp.where(both, hm2_b, jnp.where(comp2, hm2_c2, z))

    mu1_safe = jnp.where(mu_hm_1 > 0.0, mu_hm_1, 1.0)
    mu2_safe = jnp.where(mu_hm_2 > 0.0, mu_hm_2, 1.0)
    s1r = jnp.where((both | comp1), sigma_hm_1 ** 2 / mu1_safe ** 2, 0.0)
    s2r = jnp.where(both, R_b, jnp.where(comp2, sigma_hm_2 ** 2 / mu2_safe ** 2, 0.0))

    return mu_hm_1, mu_hm_2, sigma_hm_1, sigma_hm_2, hm_1, hm_2, s1r, s2r


def compute_mean_stdev(chi_1, chi_2, stdev_chi_1, stdev_chi_2,
                       stdev_eta_1, stdev_eta_2, Ncnm, Ncnp2_on_Ncnm2,
                       l_const_Nc_in_cloud, hydromets, thl_1, thl_2, mixt_frac,
                       precip_frac, precip_frac_1, precip_frac_2, precip_frac_tol,
                       omicron, zeta_vrnce_rat,
                       w_1=None, w_2=None, stdev_w_1=None, stdev_w_2=None):
    """Means and stdevs (per PDF component) of chi, eta, w, Ncn, and the precipitating
    hydrometeors, stacked into (ngrdcol, nzt, pdf_dim) arrays. Oracle compute_mean_stdev
    (setup_clubb_pdf_params.F90:818).

    `hydromets` is a list of (hmm, hmp2_zt, ratio, hm_tol) tuples in pdf order after Ncn
    (rr, Nr for KK); each is processed by calc_comp_mu_sigma_hm. The hydrometeor in-precip
    means/stdevs depend on thl_1/thl_2 (root selection), precip fracs, omicron, zeta.

    w / eta moments are not consumed by the KK rate functions (only chi, Ncn, and the
    hydrometeors are), so w_* default to zeros here; the running-model caller passes the
    real pdf_params w values. eta component means are 0 by construction (they cancel).

    Returns (mu_x_1, mu_x_2, sigma_x_1, sigma_x_2, hm_1, hm_2, s2m2_1, s2m2_2):
      mu_x_*/sigma_x_* : (ngrdcol, nzt, pdf_dim) stacked component moments
      hm_1/hm_2        : (ngrdcol, nzt, n_hydromet) per-hydrometeor in-precip-weighted means
      s2m2_*           : (ngrdcol, nzt, pdf_dim) ratio sigma_x^2/mu_x^2 (Ncn + hydrometeors)
    """
    chi_1 = jnp.asarray(chi_1, dtype=jnp.float64)
    z = jnp.zeros_like(chi_1)
    w_1 = z if w_1 is None else jnp.asarray(w_1, dtype=jnp.float64)
    w_2 = z if w_2 is None else jnp.asarray(w_2, dtype=jnp.float64)
    stdev_w_1 = z if stdev_w_1 is None else jnp.asarray(stdev_w_1, dtype=jnp.float64)
    stdev_w_2 = z if stdev_w_2 is None else jnp.asarray(stdev_w_2, dtype=jnp.float64)
    Ncnm = jnp.asarray(Ncnm, dtype=jnp.float64)

    # Ncn standard deviation and sigma^2/mu^2 ratio (single lognormal over the box).
    if l_const_Nc_in_cloud:
        sig_Ncn = z
        s2m2_Ncn = z
    else:
        sig_Ncn = jnp.sqrt(Ncnp2_on_Ncnm2) * Ncnm
        s2m2_Ncn = jnp.broadcast_to(jnp.asarray(Ncnp2_on_Ncnm2, dtype=jnp.float64), chi_1.shape)

    # Columns in iiPDF order: chi, eta, w, Ncn, <hydrometeors>.
    mu1_cols = [chi_1, jnp.broadcast_to(z, chi_1.shape), w_1, Ncnm]
    mu2_cols = [jnp.asarray(chi_2, dtype=jnp.float64), jnp.broadcast_to(z, chi_1.shape), w_2, Ncnm]
    sig1_cols = [jnp.asarray(stdev_chi_1, dtype=jnp.float64),
                 jnp.asarray(stdev_eta_1, dtype=jnp.float64), stdev_w_1, sig_Ncn]
    sig2_cols = [jnp.asarray(stdev_chi_2, dtype=jnp.float64),
                 jnp.asarray(stdev_eta_2, dtype=jnp.float64), stdev_w_2, sig_Ncn]
    s2m2_1_cols = [z, z, z, s2m2_Ncn]
    s2m2_2_cols = [z, z, z, s2m2_Ncn]
    hm1_list, hm2_list = [], []

    for hmm, hmp2_zt, ratio, hm_tol in hydromets:
        mu1, mu2, sig1, sig2, hm1, hm2, s1r, s2r = calc_comp_mu_sigma_hm(
            hmm, hmp2_zt, jnp.broadcast_to(ratio, chi_1.shape), mixt_frac,
            precip_frac, precip_frac_1, precip_frac_2, hm_tol, precip_frac_tol,
            thl_1, thl_2, omicron, zeta_vrnce_rat)
        mu1_cols.append(mu1); mu2_cols.append(mu2)
        sig1_cols.append(sig1); sig2_cols.append(sig2)
        s2m2_1_cols.append(s1r); s2m2_2_cols.append(s2r)
        hm1_list.append(hm1); hm2_list.append(hm2)

    stack = lambda cols: jnp.stack([jnp.broadcast_to(c, chi_1.shape) for c in cols], axis=-1)
    hm_1 = jnp.stack(hm1_list, axis=-1) if hm1_list else jnp.zeros(chi_1.shape + (0,))
    hm_2 = jnp.stack(hm2_list, axis=-1) if hm2_list else jnp.zeros(chi_1.shape + (0,))
    return (stack(mu1_cols), stack(mu2_cols), stack(sig1_cols), stack(sig2_cols),
            hm_1, hm_2, stack(s2m2_1_cols), stack(s2m2_2_cols))


def norm_transform_mean_stdev(mu_x_1, mu_x_2, sigma_x_1, sigma_x_2,
                              s2m2_1, s2m2_2, Ncnm, hm_1, hm_2, hydromet_tols,
                              l_const_Nc_in_cloud):
    """Transform the lognormal PDF variables (Ncn + precipitating hydrometeors) to normal
    (log) space; chi/eta/w pass through unchanged. Oracle norm_transform_mean_stdev
    (setup_clubb_pdf_params.F90:2942).

    mu_x_*/sigma_x_*/s2m2_* are the (ngrdcol, nzt, pdf_dim) outputs of compute_mean_stdev.
    Where the (in-precip) mean is below tolerance the Fortran sets the normal-space mean to
    -huge as an "absent" sentinel; here it is a finite floor (mean_L2N of |mu| floored to
    1e-30), which keeps the consuming integrals' vanishing weight at those points while
    leaving gradients finite (the established differentiability-hardening convention).

    Returns (mu_x_1_n, mu_x_2_n, sigma_x_1_n, sigma_x_2_n)."""
    def _to_n(mu, s2m2):
        mu_safe = jnp.maximum(jnp.abs(mu), 1.0e-30)
        return mean_L2N(mu_safe, s2m2), stdev_L2N(s2m2)

    mu_x_1_n = jnp.asarray(mu_x_1, dtype=jnp.float64)
    mu_x_2_n = jnp.asarray(mu_x_2, dtype=jnp.float64)
    sigma_x_1_n = jnp.asarray(sigma_x_1, dtype=jnp.float64)
    sigma_x_2_n = jnp.asarray(sigma_x_2, dtype=jnp.float64)

    # Ncn (single lognormal: components share the moments).
    mu_Ncn_1_n, sig_Ncn_1_n = _to_n(mu_x_1_n[..., IIPDF_NCN], s2m2_1[..., IIPDF_NCN])
    mu_Ncn_2_n, sig_Ncn_2_n = _to_n(mu_x_2_n[..., IIPDF_NCN], s2m2_2[..., IIPDF_NCN])
    if l_const_Nc_in_cloud:
        sig_Ncn_1_n = jnp.zeros_like(sig_Ncn_1_n)
        sig_Ncn_2_n = jnp.zeros_like(sig_Ncn_2_n)
    mu_x_1_n = mu_x_1_n.at[..., IIPDF_NCN].set(mu_Ncn_1_n)
    mu_x_2_n = mu_x_2_n.at[..., IIPDF_NCN].set(mu_Ncn_2_n)
    sigma_x_1_n = sigma_x_1_n.at[..., IIPDF_NCN].set(sig_Ncn_1_n)
    sigma_x_2_n = sigma_x_2_n.at[..., IIPDF_NCN].set(sig_Ncn_2_n)

    # Precipitating hydrometeors (pdf indices after Ncn).
    for j, _hm_tol in enumerate(hydromet_tols):
        ivar = IIPDF_NCN + 1 + j
        m1n, s1n = _to_n(mu_x_1_n[..., ivar], s2m2_1[..., ivar])
        m2n, s2n = _to_n(mu_x_2_n[..., ivar], s2m2_2[..., ivar])
        mu_x_1_n = mu_x_1_n.at[..., ivar].set(m1n)
        mu_x_2_n = mu_x_2_n.at[..., ivar].set(m2n)
        sigma_x_1_n = sigma_x_1_n.at[..., ivar].set(s1n)
        sigma_x_2_n = sigma_x_2_n.at[..., ivar].set(s2n)

    return mu_x_1_n, mu_x_2_n, sigma_x_1_n, sigma_x_2_n


def calc_corr_w_hm_n(wm, wphydrometp, mu_w_1, mu_w_2, mu_hm_1, mu_hm_2,
                     sigma_w_1, sigma_w_2, sigma_hm_1, sigma_hm_2, sigma_hm_1_n, sigma_hm_2_n,
                     mixt_frac, precip_frac_1, precip_frac_2, hm_tol):
    """PDF-component correlation of w and ln(hm) in-precip, diagnosed from the overall w-hm flux
    (setup_clubb_pdf_params.F90:calc_corr_w_hm_n) — the inverse of the flux assembly.

    corr = (wphydrometp - Σ_i mixt_i precip_frac_i (μ_w_i - <w>) μ_hm_i) / (Σ_i mixt_i precip_frac_i σ_w_i σ_hm_i_n μ_hm_i),
    clamped to ±max_mag_correlation, with a 4-way branch on whether w and hm vary (σ > tol) in each component:
    both vary → corr_1 = corr_2; only comp i varies → that component's corr, the other 0; neither → 0,0.
    Pure jnp → differentiable (degenerate denominators guarded). Returns (corr_w_hm_1_n, corr_w_hm_2_n).
    """
    w1, w2 = mixt_frac * precip_frac_1, (1.0 - mixt_frac) * precip_frac_2
    num = wphydrometp - w1 * (mu_w_1 - wm) * mu_hm_1 - w2 * (mu_w_2 - wm) * mu_hm_2
    den_both = w1 * sigma_w_1 * sigma_hm_1_n * mu_hm_1 + w2 * sigma_w_2 * sigma_hm_2_n * mu_hm_2
    den_1 = w1 * sigma_w_1 * sigma_hm_1_n * mu_hm_1
    den_2 = w2 * sigma_w_2 * sigma_hm_2_n * mu_hm_2

    def _clamp(x):
        return jnp.clip(x, -max_mag_correlation, max_mag_correlation)

    def _safe_div(n, d):
        return n / jnp.where(d != 0.0, d, 1.0)

    c1_vary = (sigma_w_1 > w_tol) & (sigma_hm_1 > hm_tol)
    c2_vary = (sigma_w_2 > w_tol) & (sigma_hm_2 > hm_tol)
    both = c1_vary & c2_vary

    cn = _clamp(_safe_div(num, den_both))
    c1v = _clamp(_safe_div(num, den_1))
    c2v = _clamp(_safe_div(num, den_2))

    corr_1 = jnp.where(both, cn, jnp.where(c1_vary, c1v, 0.0))
    corr_2 = jnp.where(both, cn, jnp.where(c1_vary, 0.0, jnp.where(c2_vary, c2v, 0.0)))
    return corr_1, corr_2


def _corr_cloud_below(rc_1, rc_2, corr_cloud, corr_below):
    """Per-component in-precip correlation by cloud presence: rc_i > rc_tol → cloud value, else below-cloud
    value (the shared body of component_corr_{x_hm,hmx_hmy,w_hm}_n_ip). rc_1/rc_2 are (ngrdcol,nzt)."""
    rc_1 = jnp.asarray(rc_1)
    rc_2 = jnp.asarray(rc_2)
    c1 = jnp.where(rc_1 > rc_tol, corr_cloud, corr_below)
    c2 = jnp.where(rc_2 > rc_tol, corr_cloud, corr_below)
    return c1, c2


def component_corr_w_hm_n_ip(corr_w_hm_1_n_in, rc_1, corr_w_hm_2_n_in, rc_2,
                             corr_w_hm_n_NL_cloud, corr_w_hm_n_NL_below, l_calc_w_corr):
    """In-precip correlation of w and ln(hm) per PDF component
    (setup_clubb_pdf_params.F90:component_corr_w_hm_n_ip). If l_calc_w_corr, pass through the diagnosed
    corr_w_hm_i_n_in (from calc_corr_w_hm_n); otherwise select the prescribed cloud/below-cloud value by rc_i.
    Returns (corr_w_hm_1_n, corr_w_hm_2_n)."""
    if l_calc_w_corr:
        return jnp.asarray(corr_w_hm_1_n_in), jnp.asarray(corr_w_hm_2_n_in)
    return _corr_cloud_below(rc_1, rc_2, corr_w_hm_n_NL_cloud, corr_w_hm_n_NL_below)


def component_corr_x_hm_n_ip(rc_1, rc_2, corr_x_hm_n_NL_cloud, corr_x_hm_n_NL_below):
    """In-precip correlation of a normal variable x (chi or eta) and ln(hm) per PDF component
    (setup_clubb_pdf_params.F90:component_corr_x_hm_n_ip): cloud/below-cloud value selected by rc_i."""
    return _corr_cloud_below(rc_1, rc_2, corr_x_hm_n_NL_cloud, corr_x_hm_n_NL_below)


def component_corr_hmx_hmy_n_ip(rc_1, rc_2, corr_hmx_hmy_n_LL_cloud, corr_hmx_hmy_n_LL_below):
    """In-precip correlation of ln(hmx) and ln(hmy) per PDF component
    (setup_clubb_pdf_params.F90:component_corr_hmx_hmy_n_ip): cloud/below-cloud value selected by rc_i."""
    return _corr_cloud_below(rc_1, rc_2, corr_hmx_hmy_n_LL_cloud, corr_hmx_hmy_n_LL_below)


def component_corr_eta_hm_n_ip(corr_chi_eta_1, corr_chi_hm_n_1, corr_chi_eta_2, corr_chi_hm_n_2):
    """Estimate the component correlation of eta and ln(hm) as corr_chi_eta·corr_chi_hm_n
    (setup_clubb_pdf_params.F90:component_corr_eta_hm_n_ip). This product keeps the correlation array
    Cholesky-decomposable for SILHS. Returns (corr_eta_hm_n_1, corr_eta_hm_n_2)."""
    return (jnp.asarray(corr_chi_eta_1) * jnp.asarray(corr_chi_hm_n_1),
            jnp.asarray(corr_chi_eta_2) * jnp.asarray(corr_chi_hm_n_2))


# iiPDF_type enumeration (model_flags.F90:31) — the PDF types whose ADG standards fix corr(w,x)=0.
IIPDF_TYPE_ADG1 = 1
IIPDF_TYPE_ADG2 = 2
IIPDF_TYPE_NEW_HYBRID = 7


def component_corr_w_x(rc_1, rc_2, corr_w_x_NN_cloud, corr_w_x_NN_below,
                       iiPDF_type, l_follow_ADG1_PDF_standards):
    """In-precip correlation of w and a normal variable x (chi or eta) per PDF component
    (setup_clubb_pdf_params.F90:component_corr_w_x). The ADG1/ADG2/new_hybrid PDFs fix corr(w,rt)=corr(w,thl)=0
    (so corr(w,chi)=corr(w,eta)=0) when l_follow_ADG1_PDF_standards; otherwise the prescribed cloud/below-cloud
    value is selected by rc_i > rc_tol. Returns (corr_w_x_1, corr_w_x_2)."""
    if l_follow_ADG1_PDF_standards and iiPDF_type in (IIPDF_TYPE_ADG1, IIPDF_TYPE_ADG2, IIPDF_TYPE_NEW_HYBRID):
        rc_1 = jnp.asarray(rc_1)
        return jnp.zeros_like(rc_1), jnp.zeros_like(rc_1)
    return _corr_cloud_below(rc_1, rc_2, corr_w_x_NN_cloud, corr_w_x_NN_below)


def component_corr_chi_eta(rc_1, rc_2, corr_chi_eta_NN_cloud, corr_chi_eta_NN_below,
                           l_limit_corr_chi_eta):
    """Correlation of chi and eta per PDF component (setup_clubb_pdf_params.F90:component_corr_chi_eta):
    cloud/below-cloud value selected by rc_i > rc_tol, optionally clamped to ±max_mag_correlation when
    l_limit_corr_chi_eta (a perfect chi–eta correlation is unrealizable for the Cholesky decomposition).
    Returns (corr_chi_eta_1, corr_chi_eta_2)."""
    c1, c2 = _corr_cloud_below(rc_1, rc_2, corr_chi_eta_NN_cloud, corr_chi_eta_NN_below)
    if l_limit_corr_chi_eta:
        c1 = jnp.clip(c1, -max_mag_correlation, max_mag_correlation)
        c2 = jnp.clip(c2, -max_mag_correlation, max_mag_correlation)
    return c1, c2


def comp_corr_norm(mu_x_1, mu_x_2, sigma_x_1, sigma_x_2, sigma_x_1_n, sigma_x_2_n,
                   wm_zt, rc_1, rc_2, mixt_frac, precip_frac_1, precip_frac_2,
                   wpNcnp_zt, wphydrometp_zt, corr_array_n_cloud, corr_array_n_below,
                   iiPDF_chi, iiPDF_eta, iiPDF_w, iiPDF_Ncn, pdf2hydromet, hydromet_tol, Ncn_tol_val,
                   iiPDF_type, l_calc_w_corr, l_fix_w_chi_eta_correlations,
                   pdf_params_corr=None):
    """Assemble the normal-space PDF correlation arrays (setup_clubb_pdf_params.F90:comp_corr_norm).

    Builds the lower-triangular (then symmetrized) (ngrdcol, nzt, pdf_dim, pdf_dim) correlation arrays for the
    two PDF components from the component_corr_* routines plus calc_corr_w_hm_n (for the w-correlations when
    l_calc_w_corr). The prescribed-array index layout is chi, eta, w, Ncn, <hydrometeors> (iiPDF indices).

    Arrays: mu_x_*/sigma_x_* are (ngrdcol, nzt, pdf_dim); wm_zt/rc_*/mixt_frac/precip_frac_*/wpNcnp_zt are
    (ngrdcol, nzt); wphydrometp_zt is (ngrdcol, nzt, hydromet_dim); corr_array_n_cloud/below are
    (pdf_dim, pdf_dim). pdf2hydromet maps a pdf index to its hydromet index. pdf_params_corr (with keys
    corr_chi_eta_1/2, corr_w_chi_1/2) is required only for the l_fix_w_chi_eta_correlations=False path.

    Returns (corr_array_1_n, corr_array_2_n), each (ngrdcol, nzt, pdf_dim, pdf_dim), symmetric, unit diagonal.

    Faithfulness note: the l_fix_w_chi_eta_correlations=False ("preferred") branch reproduces the Fortran's
    eta–w block exactly, including its quirk (F90:1560) of writing corr_w_chi into the (w, chi) slot a second
    time rather than (w, eta) — so (w, eta) is left at 0 on that path, matching the oracle.
    """
    mu_x_1 = jnp.asarray(mu_x_1); mu_x_2 = jnp.asarray(mu_x_2)
    sigma_x_1 = jnp.asarray(sigma_x_1); sigma_x_2 = jnp.asarray(sigma_x_2)
    sigma_x_1_n = jnp.asarray(sigma_x_1_n); sigma_x_2_n = jnp.asarray(sigma_x_2_n)
    wm_zt = jnp.asarray(wm_zt); rc_1 = jnp.asarray(rc_1); rc_2 = jnp.asarray(rc_2)
    mixt_frac = jnp.asarray(mixt_frac)
    precip_frac_1 = jnp.asarray(precip_frac_1); precip_frac_2 = jnp.asarray(precip_frac_2)
    wpNcnp_zt = jnp.asarray(wpNcnp_zt); wphydrometp_zt = jnp.asarray(wphydrometp_zt)
    cc = jnp.asarray(corr_array_n_cloud); cb = jnp.asarray(corr_array_n_below)

    ng, nzt, pdf_dim = mu_x_1.shape
    ones = jnp.ones((ng, nzt))
    hm_indices = list(range(iiPDF_Ncn + 1, pdf_dim))   # hydrometeor pdf indices

    A1 = jnp.zeros((ng, nzt, pdf_dim, pdf_dim))
    A2 = jnp.zeros((ng, nzt, pdf_dim, pdf_dim))
    idx = jnp.arange(pdf_dim)
    A1 = A1.at[:, :, idx, idx].set(1.0)
    A2 = A2.at[:, :, idx, idx].set(1.0)

    def _set(A, r, c, val):
        return A.at[:, :, r, c].set(val)

    # ---- w-correlations (down-gradient diagnosis) when l_calc_w_corr ----
    corr_w_Ncn_1 = jnp.zeros((ng, nzt)); corr_w_Ncn_2 = jnp.zeros((ng, nzt))
    corr_w_hm_1 = {j: jnp.zeros((ng, nzt)) for j in hm_indices}
    corr_w_hm_2 = {j: jnp.zeros((ng, nzt)) for j in hm_indices}
    if l_calc_w_corr:
        corr_w_Ncn_1, corr_w_Ncn_2 = calc_corr_w_hm_n(
            wm_zt, wpNcnp_zt, mu_x_1[:, :, iiPDF_w], mu_x_2[:, :, iiPDF_w],
            mu_x_1[:, :, iiPDF_Ncn], mu_x_2[:, :, iiPDF_Ncn],
            sigma_x_1[:, :, iiPDF_w], sigma_x_2[:, :, iiPDF_w],
            sigma_x_1[:, :, iiPDF_Ncn], sigma_x_2[:, :, iiPDF_Ncn],
            sigma_x_1_n[:, :, iiPDF_Ncn], sigma_x_2_n[:, :, iiPDF_Ncn],
            mixt_frac, ones, ones, Ncn_tol_val)
        for j in hm_indices:
            hm_idx = int(pdf2hydromet[j])
            corr_w_hm_1[j], corr_w_hm_2[j] = calc_corr_w_hm_n(
                wm_zt, wphydrometp_zt[:, :, hm_idx], mu_x_1[:, :, iiPDF_w], mu_x_2[:, :, iiPDF_w],
                mu_x_1[:, :, j], mu_x_2[:, :, j], sigma_x_1[:, :, iiPDF_w], sigma_x_2[:, :, iiPDF_w],
                sigma_x_1[:, :, j], sigma_x_2[:, :, j], sigma_x_1_n[:, :, j], sigma_x_2_n[:, :, j],
                mixt_frac, precip_frac_1, precip_frac_2, hydromet_tol[hm_idx])

    # ---- (eta, chi) ----
    if l_fix_w_chi_eta_correlations:
        c1, c2 = component_corr_chi_eta(rc_1, rc_2, cc[iiPDF_eta, iiPDF_chi],
                                        cb[iiPDF_eta, iiPDF_chi], True)
    else:
        c1 = jnp.asarray(pdf_params_corr['corr_chi_eta_1'])
        c2 = jnp.asarray(pdf_params_corr['corr_chi_eta_2'])
    A1 = _set(A1, iiPDF_eta, iiPDF_chi, c1); A2 = _set(A2, iiPDF_eta, iiPDF_chi, c2)

    # ---- (w, chi) ----
    if l_fix_w_chi_eta_correlations:
        c1, c2 = component_corr_w_x(rc_1, rc_2, cc[iiPDF_w, iiPDF_chi], cb[iiPDF_w, iiPDF_chi],
                                    iiPDF_type, True)
    else:
        c1 = jnp.asarray(pdf_params_corr['corr_w_chi_1']); c2 = jnp.asarray(pdf_params_corr['corr_w_chi_2'])
    A1 = _set(A1, iiPDF_w, iiPDF_chi, c1); A2 = _set(A2, iiPDF_w, iiPDF_chi, c2)

    # ---- (Ncn, chi): cloud value used twice (Ncn is inherently in-cloud) ----
    c1, c2 = component_corr_x_hm_n_ip(rc_1, rc_2, cc[iiPDF_Ncn, iiPDF_chi], cc[iiPDF_Ncn, iiPDF_chi])
    A1 = _set(A1, iiPDF_Ncn, iiPDF_chi, c1); A2 = _set(A2, iiPDF_Ncn, iiPDF_chi, c2)

    # ---- (hm, chi) ----
    for j in hm_indices:
        c1, c2 = component_corr_x_hm_n_ip(rc_1, rc_2, cc[j, iiPDF_chi], cb[j, iiPDF_chi])
        A1 = _set(A1, j, iiPDF_chi, c1); A2 = _set(A2, j, iiPDF_chi, c2)

    # ---- (w, eta) ----
    if l_fix_w_chi_eta_correlations:
        c1, c2 = component_corr_w_x(rc_1, rc_2, cc[iiPDF_w, iiPDF_eta], cb[iiPDF_w, iiPDF_eta],
                                    iiPDF_type, True)
        A1 = _set(A1, iiPDF_w, iiPDF_eta, c1); A2 = _set(A2, iiPDF_w, iiPDF_eta, c2)
    else:
        # Faithful to F90:1560 quirk: re-writes (w, chi), leaving (w, eta) at 0.
        c1 = jnp.asarray(pdf_params_corr['corr_w_chi_1']); c2 = jnp.asarray(pdf_params_corr['corr_w_chi_2'])
        A1 = _set(A1, iiPDF_w, iiPDF_chi, c1); A2 = _set(A2, iiPDF_w, iiPDF_chi, c2)

    # ---- (Ncn, eta): cloud value used twice ----
    c1, c2 = component_corr_x_hm_n_ip(rc_1, rc_2, cc[iiPDF_Ncn, iiPDF_eta], cc[iiPDF_Ncn, iiPDF_eta])
    A1 = _set(A1, iiPDF_Ncn, iiPDF_eta, c1); A2 = _set(A2, iiPDF_Ncn, iiPDF_eta, c2)

    # ---- (hm, eta): estimated from (eta, chi) and (hm, chi) ----
    for j in hm_indices:
        c1, c2 = component_corr_eta_hm_n_ip(A1[:, :, iiPDF_eta, iiPDF_chi], A1[:, :, j, iiPDF_chi],
                                            A2[:, :, iiPDF_eta, iiPDF_chi], A2[:, :, j, iiPDF_chi])
        A1 = _set(A1, j, iiPDF_eta, c1); A2 = _set(A2, j, iiPDF_eta, c2)

    # ---- (Ncn, w) ----
    c1, c2 = component_corr_w_hm_n_ip(corr_w_Ncn_1, rc_1, corr_w_Ncn_2, rc_2,
                                      cc[iiPDF_Ncn, iiPDF_w], cb[iiPDF_Ncn, iiPDF_w], l_calc_w_corr)
    A1 = _set(A1, iiPDF_Ncn, iiPDF_w, c1); A2 = _set(A2, iiPDF_Ncn, iiPDF_w, c2)

    # ---- (hm, w) ----
    for j in hm_indices:
        c1, c2 = component_corr_w_hm_n_ip(corr_w_hm_1[j], rc_1, corr_w_hm_2[j], rc_2,
                                          cc[j, iiPDF_w], cb[j, iiPDF_w], l_calc_w_corr)
        A1 = _set(A1, j, iiPDF_w, c1); A2 = _set(A2, j, iiPDF_w, c2)

    # ---- (hm, Ncn) ----
    for j in hm_indices:
        c1, c2 = component_corr_hmx_hmy_n_ip(rc_1, rc_2, cc[j, iiPDF_Ncn], cb[j, iiPDF_Ncn])
        A1 = _set(A1, j, iiPDF_Ncn, c1); A2 = _set(A2, j, iiPDF_Ncn, c2)

    # ---- (hmy, hmx) for hydrometeor pairs ----
    for ii in range(iiPDF_Ncn + 1, pdf_dim - 1):
        for jj in range(ii + 1, pdf_dim):
            c1, c2 = component_corr_hmx_hmy_n_ip(rc_1, rc_2, cc[jj, ii], cb[jj, ii])
            A1 = _set(A1, jj, ii, c1); A2 = _set(A2, jj, ii, c2)

    # ---- Symmetrize (mirror lower triangle): full = L + L^T - I ----
    eye = jnp.zeros((ng, nzt, pdf_dim, pdf_dim)).at[:, :, idx, idx].set(1.0)
    A1 = A1 + jnp.swapaxes(A1, -1, -2) - eye
    A2 = A2 + jnp.swapaxes(A2, -1, -2) - eye
    return A1, A2


def denorm_transform_corr(sigma_x_1_n, sigma_x_2_n, sigma2_on_mu2_ip_1, sigma2_on_mu2_ip_2,
                          corr_array_1_n, corr_array_2_n,
                          iiPDF_chi, iiPDF_eta, iiPDF_w, iiPDF_Ncn):
    """Transform the normal-space PDF correlation arrays to real ("standard") space
    (setup_clubb_pdf_params.F90:denorm_transform_corr).

    Correlations among the normal variables (chi, eta, w) are unchanged. Correlations between a normal variable
    and a lognormal one (Ncn or a precipitating hydrometeor) use corr_NN2NL; correlations between two lognormal
    variables use corr_NN2LL. All arrays are (ngrdcol, nzt, pdf_dim[, pdf_dim]). Returns symmetric
    (corr_array_1, corr_array_2) with unit diagonal.

    Faithfulness note: matching the Fortran, the component-2 transforms involving Ncn reuse the **component-1**
    Ncn variance ratio sigma2_on_mu2_ip_1[Ncn] (F90:3332/3338/3344/3392) — Ncn is inherently in-cloud, so its
    ratio is shared across components.
    """
    s1n = jnp.asarray(sigma_x_1_n); s2n = jnp.asarray(sigma_x_2_n)
    r1 = jnp.asarray(sigma2_on_mu2_ip_1); r2 = jnp.asarray(sigma2_on_mu2_ip_2)
    Cn1 = jnp.asarray(corr_array_1_n); Cn2 = jnp.asarray(corr_array_2_n)

    ng, nzt, pdf_dim, _ = Cn1.shape
    idx = jnp.arange(pdf_dim)
    A1 = jnp.zeros((ng, nzt, pdf_dim, pdf_dim)).at[:, :, idx, idx].set(1.0)
    A2 = jnp.zeros((ng, nzt, pdf_dim, pdf_dim)).at[:, :, idx, idx].set(1.0)
    hm_indices = list(range(iiPDF_Ncn + 1, pdf_dim))

    def _s(A, r, c, val):
        return A.at[:, :, r, c].set(val)

    # Normal-normal pairs (chi/eta/w): correlations carry over unchanged.
    for (rr, cc_) in ((iiPDF_eta, iiPDF_chi), (iiPDF_w, iiPDF_chi), (iiPDF_w, iiPDF_eta)):
        A1 = _s(A1, rr, cc_, Cn1[:, :, rr, cc_]); A2 = _s(A2, rr, cc_, Cn2[:, :, rr, cc_])

    # chi/eta/w  x  Ncn (normal-lognormal). Component 2 reuses r1[Ncn].
    for x in (iiPDF_chi, iiPDF_eta, iiPDF_w):
        A1 = _s(A1, iiPDF_Ncn, x, corr_NN2NL(Cn1[:, :, iiPDF_Ncn, x], s1n[:, :, iiPDF_Ncn], r1[:, :, iiPDF_Ncn]))
        A2 = _s(A2, iiPDF_Ncn, x, corr_NN2NL(Cn2[:, :, iiPDF_Ncn, x], s2n[:, :, iiPDF_Ncn], r1[:, :, iiPDF_Ncn]))

    # chi/eta/w  x  hydrometeors (normal-lognormal).
    for x in (iiPDF_chi, iiPDF_eta, iiPDF_w):
        for j in hm_indices:
            A1 = _s(A1, j, x, corr_NN2NL(Cn1[:, :, j, x], s1n[:, :, j], r1[:, :, j]))
            A2 = _s(A2, j, x, corr_NN2NL(Cn2[:, :, j, x], s2n[:, :, j], r2[:, :, j]))

    # Ncn  x  hydrometeors (lognormal-lognormal). Component 2 reuses r1[Ncn].
    for j in hm_indices:
        A1 = _s(A1, j, iiPDF_Ncn, corr_NN2LL(Cn1[:, :, j, iiPDF_Ncn], s1n[:, :, iiPDF_Ncn], s1n[:, :, j],
                                             r1[:, :, iiPDF_Ncn], r1[:, :, j]))
        A2 = _s(A2, j, iiPDF_Ncn, corr_NN2LL(Cn2[:, :, j, iiPDF_Ncn], s2n[:, :, iiPDF_Ncn], s2n[:, :, j],
                                             r1[:, :, iiPDF_Ncn], r2[:, :, j]))

    # hydrometeor x hydrometeor pairs (lognormal-lognormal).
    for ii in range(iiPDF_Ncn + 1, pdf_dim - 1):
        for jj in range(ii + 1, pdf_dim):
            A1 = _s(A1, jj, ii, corr_NN2LL(Cn1[:, :, jj, ii], s1n[:, :, ii], s1n[:, :, jj],
                                           r1[:, :, ii], r1[:, :, jj]))
            A2 = _s(A2, jj, ii, corr_NN2LL(Cn2[:, :, jj, ii], s2n[:, :, ii], s2n[:, :, jj],
                                           r2[:, :, ii], r2[:, :, jj]))

    eye = jnp.zeros((ng, nzt, pdf_dim, pdf_dim)).at[:, :, idx, idx].set(1.0)
    A1 = A1 + jnp.swapaxes(A1, -1, -2) - eye
    A2 = A2 + jnp.swapaxes(A2, -1, -2) - eye
    return A1, A2


def calc_corr_norm_and_cholesky_factor(corr_array_n_cloud, corr_array_n_below, rc_1, rc_2,
                                       iiPDF_type, iiPDF_chi, iiPDF_eta, iiPDF_w, iiPDF_Ncn,
                                       l_follow_ADG1_PDF_standards):
    """Compute the normal-space PDF correlation arrays and their Cholesky factors for both PDF components
    (setup_clubb_pdf_params.F90:calc_corr_norm_and_cholesky_factor).

    Uses the "two unique correlation arrays" optimization: the prescribed in-cloud and below-cloud correlation
    matrices are each adjusted (ADG zeroing of corr(w,chi)/corr(w,eta); Ncn below-cloud values replaced with
    in-cloud ones since Ncn is inherently in-cloud; the eta–hm correlation estimated as the chi–eta·chi–hm
    product for Cholesky-decomposability), Cholesky-factorized once each, then assigned per grid column / level
    by whether rc_i exceeds rc_tol.

    Args: corr_array_n_cloud/below are symmetric (pdf_dim, pdf_dim); rc_1/rc_2 are (ngrdcol, nzt). Returns
    (corr_array_1_n, corr_array_2_n, corr_cholesky_mtx_1, corr_cholesky_mtx_2), each
    (ngrdcol, nzt, pdf_dim, pdf_dim) — the corr arrays symmetric, the Cholesky matrices lower-triangular.
    """
    cc = jnp.asarray(corr_array_n_cloud, dtype=jnp.float64)
    cb = jnp.asarray(corr_array_n_below, dtype=jnp.float64)
    rc_1 = jnp.asarray(rc_1, dtype=jnp.float64)
    rc_2 = jnp.asarray(rc_2, dtype=jnp.float64)
    pdf_dim = cc.shape[0]
    hm_indices = list(range(iiPDF_Ncn + 1, pdf_dim))

    # ADG standards fix corr(w, chi) = corr(w, eta) = 0.
    if l_follow_ADG1_PDF_standards and iiPDF_type in (IIPDF_TYPE_ADG1, IIPDF_TYPE_ADG2, IIPDF_TYPE_NEW_HYBRID):
        for (r, c) in ((iiPDF_w, iiPDF_chi), (iiPDF_w, iiPDF_eta)):
            cc = cc.at[r, c].set(0.0); cb = cb.at[r, c].set(0.0)

    # Ncn is inherently in-cloud: replace its below-cloud correlations with the in-cloud ones.
    cb = cb.at[iiPDF_Ncn, iiPDF_chi].set(cc[iiPDF_Ncn, iiPDF_chi])
    cb = cb.at[iiPDF_Ncn, iiPDF_eta].set(cc[iiPDF_Ncn, iiPDF_eta])

    # eta–hm correlation estimated as chi–eta * chi–hm (keeps the matrix Cholesky-decomposable).
    for j in hm_indices:
        cc = cc.at[j, iiPDF_eta].set(cc[iiPDF_eta, iiPDF_chi] * cc[j, iiPDF_chi])
        cb = cb.at[j, iiPDF_eta].set(cb[iiPDF_eta, iiPDF_chi] * cb[j, iiPDF_chi])

    # Symmetrize from the (modified) lower triangle, then factor.
    def _symm(M):
        L = jnp.tril(M)
        return L + jnp.swapaxes(L, -1, -2) - jnp.diag(jnp.diagonal(M))

    cc_s = _symm(cc); cb_s = _symm(cb)
    _, chol_cloud, _ = cholesky_factor(cc_s)
    _, chol_below, _ = cholesky_factor(cb_s)
    chol_cloud = jnp.tril(chol_cloud)        # zero the upper triangle
    chol_below = jnp.tril(chol_below)

    # Assign per column/level by rc (the "two unique arrays" selection).
    def _select(rc):
        sel = (rc > rc_tol)[:, :, None, None]
        corr = jnp.where(sel, cc_s, cb_s)
        chol = jnp.where(sel, chol_cloud, chol_below)
        return corr, chol

    corr_1, chol_1 = _select(rc_1)
    corr_2, chol_2 = _select(rc_2)
    return corr_1, corr_2, chol_1, chol_2
