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

from clubb_jax.src.CLUBB_core.constants_clubb import Ncn_tol
from clubb_jax.src.CLUBB_core.pdf_utilities import mean_L2N, stdev_L2N

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
