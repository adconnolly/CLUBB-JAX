"""JAX port of pdf_utilities.F90 — lognormal <-> normal moment/correlation conversions.

CLUBB's precipitating hydrometeors (r_r, N_r) and the simplified cloud-nuclei
concentration N_cn have an assumed LOGNORMAL marginal in each PDF component. The
upscaled-KK rate functions (KK_upscaled_means) consume the moments of ln(hm) — the
NORMAL-space mean, standard deviation, and correlations — but the PDF setup carries
the LINEAR-space mean/variance. These elemental functions convert between the two:

  mean_L2N   : linear mean   -> mean of ln x          (Garvey 2000, App. B)
  stdev_L2N  : linear var    -> std dev of ln x        (Garvey 2000, App. B)
  corr_NL2NN : corr(x, y)    -> corr(x, ln y)   (x normal, y lognormal; Garvey Eq. B-1)
  corr_LL2NN : corr(x, y)    -> corr(ln x, ln y) (x,y lognormal; Garvey Eq. C-3)
  corr_NN2NL : corr(x, ln y) -> corr(x, y)      (inverse of corr_NL2NN)
  corr_NN2LL : corr(ln x,ln y)->corr(x, y)      (inverse of corr_LL2NN)
  calc_corr_{chi,eta}_x : corr(chi/eta, x) from corr(rt, x), corr(thl, x) (chi=crt rt-cthl thl,
                          eta=crt rt+cthl thl decomposition)
  calc_corr_{rt,thl}_x  : the inverses — corr(rt/thl, x) from corr(chi, x), corr(eta, x)

Oracle: pdf_utilities.F90 (the core_rknd variants — mean_L2N keeps its max(.,tiny)
guard; the _dp variants drop it). Inputs use the ratio sigma2_on_mu2 = sigma_x^2/mu_x^2.

mean_L2N, stdev_L2N, corr_NN2NL, corr_NN2LL, calc_corr_chi_x, calc_corr_eta_x are bit-to-bit
verifiable against the f2py API; corr_NL2NN/corr_LL2NN (not exposed) are checked vs Monte-Carlo
and calc_corr_rt_x/calc_corr_thl_x via the chi/eta<->rt/thl round-trip.
See tests/test_pdf_utilities.py. All functions are jnp and differentiable.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# constants_clubb.F90
MAX_MAG_CORRELATION = 0.99
_CHI_TOL = 1.0e-8                     # max(1e-8, epsilon); eta_tol = chi_tol
_RT_TOL = 1.0e-8                      # max(1e-8, epsilon)
_THL_TOL = 1.0e-2
_TINY = jnp.finfo(jnp.float64).tiny  # Fortran tiny(mu_x) for double precision
_MIN_MAX_SMTH_MAG = 1.0e-9           # constants_clubb.F90 min_max_smth_mag
_EPS = 1.0e-10                       # constants_clubb.F90 eps = max(1e-10, epsilon)


def mean_L2N(mu_x, sigma2_on_mu2):
    """Mean of ln x for a lognormal x. pdf_utilities.F90:33 (core_rknd variant).

    mu_x_n = log( max( mu_x / sqrt(1 + sigma2_on_mu2), tiny ) ).
    The max(.,tiny) guard prevents log(0); it mirrors the core_rknd Fortran (the _dp
    variant omits it). Not meant to be called with mu_x <= 0."""
    mu_x = jnp.asarray(mu_x, dtype=jnp.float64)
    sigma2_on_mu2 = jnp.asarray(sigma2_on_mu2, dtype=jnp.float64)
    return jnp.log(jnp.maximum(mu_x / jnp.sqrt(1.0 + sigma2_on_mu2), _TINY))


def stdev_L2N(sigma2_on_mu2):
    """Std deviation of ln x for a lognormal x. pdf_utilities.F90:122.

    sigma_x_n = sqrt( log( 1 + sigma2_on_mu2 ) )."""
    sigma2_on_mu2 = jnp.asarray(sigma2_on_mu2, dtype=jnp.float64)
    return jnp.sqrt(jnp.log(1.0 + sigma2_on_mu2))


def corr_NL2NN(corr_x_y, sigma_y_n, y_sigma2_on_mu2):
    """Correlation of x (normal) and ln y (y lognormal). pdf_utilities.F90:207.

    corr_x_y_n = corr_x_y * sqrt(y_sigma2_on_mu2) / sigma_y_n   if sigma_y_n > 0,
               = corr_x_y                                        if sigma_y_n = 0,
    then clipped to +/- max_mag_correlation."""
    corr_x_y = jnp.asarray(corr_x_y, dtype=jnp.float64)
    sigma_y_n = jnp.asarray(sigma_y_n, dtype=jnp.float64)
    y_sigma2_on_mu2 = jnp.asarray(y_sigma2_on_mu2, dtype=jnp.float64)
    # guard sigma_y_n in the division so the sigma_y_n==0 branch is finite/clean
    sy_safe = jnp.where(sigma_y_n > 0.0, sigma_y_n, 1.0)
    corr_n = jnp.where(sigma_y_n > 0.0,
                       corr_x_y * jnp.sqrt(y_sigma2_on_mu2) / sy_safe,
                       corr_x_y)
    return jnp.clip(corr_n, -MAX_MAG_CORRELATION, MAX_MAG_CORRELATION)


def corr_LL2NN(corr_x_y, sigma_x_n, sigma_y_n, x_sigma2_on_mu2, y_sigma2_on_mu2):
    """Correlation of ln x and ln y (both lognormal). pdf_utilities.F90:439.

    corr_x_y_n = log(1 + corr_x_y*sqrt(x_sigma2_on_mu2*y_sigma2_on_mu2))
                 / (sigma_x_n*sigma_y_n)   if sigma_x_n>0 and sigma_y_n>0,
               = corr_x_y                  otherwise,
    then clipped to +/- max_mag_correlation."""
    corr_x_y = jnp.asarray(corr_x_y, dtype=jnp.float64)
    sigma_x_n = jnp.asarray(sigma_x_n, dtype=jnp.float64)
    sigma_y_n = jnp.asarray(sigma_y_n, dtype=jnp.float64)
    x_sigma2_on_mu2 = jnp.asarray(x_sigma2_on_mu2, dtype=jnp.float64)
    y_sigma2_on_mu2 = jnp.asarray(y_sigma2_on_mu2, dtype=jnp.float64)
    both_pos = (sigma_x_n > 0.0) & (sigma_y_n > 0.0)
    denom = jnp.where(both_pos, sigma_x_n * sigma_y_n, 1.0)
    log_arg = 1.0 + corr_x_y * jnp.sqrt(x_sigma2_on_mu2 * y_sigma2_on_mu2)
    corr_n = jnp.where(both_pos, jnp.log(log_arg) / denom, corr_x_y)
    return jnp.clip(corr_n, -MAX_MAG_CORRELATION, MAX_MAG_CORRELATION)


def corr_NN2NL(corr_x_y_n, sigma_y_n, y_sigma2_on_mu2):
    """Correlation of x (normal) and y (lognormal) from corr(x, ln y). pdf_utilities.F90:348.

    Inverse of corr_NL2NN: corr_x_y = corr_x_y_n * sigma_y_n / sqrt(max(y_sigma2_on_mu2, tiny))
    if sigma_y_n > 0, else corr_x_y_n; clipped to +/- max_mag_correlation."""
    corr_x_y_n = jnp.asarray(corr_x_y_n, dtype=jnp.float64)
    sigma_y_n = jnp.asarray(sigma_y_n, dtype=jnp.float64)
    y_sigma2_on_mu2 = jnp.asarray(y_sigma2_on_mu2, dtype=jnp.float64)
    corr = jnp.where(sigma_y_n > 0.0,
                     corr_x_y_n * sigma_y_n / jnp.sqrt(jnp.maximum(y_sigma2_on_mu2, _TINY)),
                     corr_x_y_n)
    return jnp.clip(corr, -MAX_MAG_CORRELATION, MAX_MAG_CORRELATION)


def corr_NN2LL(corr_x_y_n, sigma_x_n, sigma_y_n, x_sigma2_on_mu2, y_sigma2_on_mu2):
    """Correlation of x and y (both lognormal) from corr(ln x, ln y). pdf_utilities.F90:590.

    Inverse of corr_LL2NN: corr_x_y = (exp(sigma_x_n sigma_y_n corr_x_y_n) - 1)
    / sqrt(x_sigma2_on_mu2 y_sigma2_on_mu2) if both sigma_n>0, else corr_x_y_n; clipped."""
    corr_x_y_n = jnp.asarray(corr_x_y_n, dtype=jnp.float64)
    sigma_x_n = jnp.asarray(sigma_x_n, dtype=jnp.float64)
    sigma_y_n = jnp.asarray(sigma_y_n, dtype=jnp.float64)
    x_sigma2_on_mu2 = jnp.asarray(x_sigma2_on_mu2, dtype=jnp.float64)
    y_sigma2_on_mu2 = jnp.asarray(y_sigma2_on_mu2, dtype=jnp.float64)
    both_pos = (sigma_x_n > 0.0) & (sigma_y_n > 0.0)
    denom = jnp.where(both_pos, jnp.sqrt(x_sigma2_on_mu2 * y_sigma2_on_mu2), 1.0)
    corr = jnp.where(both_pos,
                     (jnp.exp(sigma_x_n * sigma_y_n * corr_x_y_n) - 1.0) / denom,
                     corr_x_y_n)
    return jnp.clip(corr, -MAX_MAG_CORRELATION, MAX_MAG_CORRELATION)


def calc_corr_chi_x(crt_i, cthl_i, sigma_rt_i, sigma_thl_i, sigma_chi_i,
                    corr_rt_x_i, corr_thl_x_i):
    """Correlation of chi and x in a PDF component, from corr(rt,x), corr(thl,x).
    pdf_utilities.F90:912 (Larson et al. 2001 chi linearization).

    corr_chi_x = crt (sigma_rt/sigma_chi) corr_rt_x - cthl (sigma_thl/sigma_chi) corr_thl_x
    when sigma_chi > chi_tol, else 0; clipped to +/- max_mag_correlation."""
    sigma_chi_i = jnp.asarray(sigma_chi_i, dtype=jnp.float64)
    sig_safe = jnp.where(sigma_chi_i > _CHI_TOL, sigma_chi_i, 1.0)
    corr = jnp.where(
        sigma_chi_i > _CHI_TOL,
        crt_i * (sigma_rt_i / sig_safe) * corr_rt_x_i
        - cthl_i * (sigma_thl_i / sig_safe) * corr_thl_x_i,
        0.0)
    return jnp.clip(corr, -MAX_MAG_CORRELATION, MAX_MAG_CORRELATION)


def calc_corr_eta_x(crt_i, cthl_i, sigma_rt_i, sigma_thl_i, sigma_eta_i,
                    corr_rt_x_i, corr_thl_x_i):
    """Correlation of eta (old t) and x in a PDF component. pdf_utilities.F90:1028.

    eta' = crt·rt' + cthl·thl' (note the + on cthl, vs - for chi), so
    corr_eta_x = crt (sigma_rt/sigma_eta) corr_rt_x + cthl (sigma_thl/sigma_eta) corr_thl_x
    when sigma_eta > eta_tol (= chi_tol), else 0; clipped to +/- max_mag_correlation."""
    sigma_eta_i = jnp.asarray(sigma_eta_i, dtype=jnp.float64)
    sig_safe = jnp.where(sigma_eta_i > _CHI_TOL, sigma_eta_i, 1.0)
    corr = jnp.where(
        sigma_eta_i > _CHI_TOL,
        crt_i * (sigma_rt_i / sig_safe) * corr_rt_x_i
        + cthl_i * (sigma_thl_i / sig_safe) * corr_thl_x_i,
        0.0)
    return jnp.clip(corr, -MAX_MAG_CORRELATION, MAX_MAG_CORRELATION)


def compute_mean_binormal(mu_x_1, mu_x_2, mixt_frac):
    """Overall mean of a two-component (binormal) PDF variable (pdf_utilities.F90:compute_mean_binormal).

    xm = mixt_frac * mu_x_1 + (1 - mixt_frac) * mu_x_2
    """
    return mixt_frac * mu_x_1 + (1.0 - mixt_frac) * mu_x_2


def compute_variance_binormal(xm, mu_x_1, mu_x_2, stdev_x_1, stdev_x_2, mixt_frac):
    """Overall grid-box variance of a two-component (binormal) PDF variable
    (pdf_utilities.F90:compute_variance_binormal).

    xp2 = mixt_frac*((mu_x_1 - xm)^2 + stdev_x_1^2) + (1 - mixt_frac)*((mu_x_2 - xm)^2 + stdev_x_2^2)
    """
    xm = jnp.asarray(xm, dtype=jnp.float64)
    mu_x_1 = jnp.asarray(mu_x_1, dtype=jnp.float64); mu_x_2 = jnp.asarray(mu_x_2, dtype=jnp.float64)
    stdev_x_1 = jnp.asarray(stdev_x_1, dtype=jnp.float64); stdev_x_2 = jnp.asarray(stdev_x_2, dtype=jnp.float64)
    a = jnp.asarray(mixt_frac, dtype=jnp.float64)
    return (a * ((mu_x_1 - xm) ** 2 + stdev_x_1 ** 2)
            + (1.0 - a) * ((mu_x_2 - xm) ** 2 + stdev_x_2 ** 2))


def calc_corr_rt_x(crt_i, sigma_rt_i, sigma_chi_i, sigma_eta_i,
                   corr_chi_x_i, corr_eta_x_i):
    """Correlation of rt and x from corr(chi,x), corr(eta,x). pdf_utilities.F90:1106.

    Inverts the chi/eta decomposition: corr_rt_x = (sigma_eta corr_eta_x + sigma_chi corr_chi_x)
    / (2 crt sigma_rt) when sigma_rt > rt_tol, else 0; clipped."""
    sigma_rt_i = jnp.asarray(sigma_rt_i, dtype=jnp.float64)
    sig_safe = jnp.where(sigma_rt_i > _RT_TOL, sigma_rt_i, 1.0)
    corr = jnp.where(
        sigma_rt_i > _RT_TOL,
        (sigma_eta_i * corr_eta_x_i + sigma_chi_i * corr_chi_x_i) / (2.0 * crt_i * sig_safe),
        0.0)
    return jnp.clip(corr, -MAX_MAG_CORRELATION, MAX_MAG_CORRELATION)


def calc_corr_thl_x(cthl_i, sigma_thl_i, sigma_chi_i, sigma_eta_i,
                    corr_chi_x_i, corr_eta_x_i):
    """Correlation of thl and x from corr(chi,x), corr(eta,x). pdf_utilities.F90:1175.

    corr_thl_x = (sigma_eta corr_eta_x - sigma_chi corr_chi_x) / (2 cthl sigma_thl)
    when sigma_thl > thl_tol, else 0; clipped."""
    sigma_thl_i = jnp.asarray(sigma_thl_i, dtype=jnp.float64)
    sig_safe = jnp.where(sigma_thl_i > _THL_TOL, sigma_thl_i, 1.0)
    corr = jnp.where(
        sigma_thl_i > _THL_TOL,
        (sigma_eta_i * corr_eta_x_i - sigma_chi_i * corr_chi_x_i) / (2.0 * cthl_i * sig_safe),
        0.0)
    return jnp.clip(corr, -MAX_MAG_CORRELATION, MAX_MAG_CORRELATION)


def smooth_corr_quotient(numerator, denominator, denom_thresh):
    """Smoothly bounded correlation quotient num/denom (pdf_utilities.F90:smooth_corr_quotient).

    Two smooth_max operations keep the result a valid correlation: the denominator is first raised to at least
    |num|/max_mag_correlation (so |quotient| <= max_mag_correlation) and then to at least denom_thresh (so it
    never divides by ~0). smooth_max(a,b,c)=0.5((a+b)+sqrt((a-b)^2+c^2)); smoothing coef = min(min_max_smth_mag,
    denom_thresh). Pure-jnp → differentiable."""
    num = jnp.asarray(numerator, dtype=jnp.float64)
    den = jnp.asarray(denominator, dtype=jnp.float64)
    coef = jnp.minimum(_MIN_MAX_SMTH_MAG, denom_thresh)

    def _smax(a, b):
        return 0.5 * ((a + b) + jnp.sqrt((a - b) ** 2 + coef ** 2))

    tmp = _smax(jnp.abs(num) / MAX_MAG_CORRELATION, den)
    tmp = _smax(tmp, denom_thresh)
    return num / tmp


def calc_comp_corrs_binormal(xpyp, xm, ym, mu_x_1, mu_x_2, mu_y_1, mu_y_2,
                             sigma_x_1_sqd, sigma_x_2_sqd, sigma_y_1_sqd, sigma_y_2_sqd, mixt_frac):
    """PDF-component correlation of two binormal variables x and y, diagnosed from their overall covariance
    (pdf_utilities.F90:calc_comp_corrs_binormal). The two PDF components share one correlation
    (corr_x_y_1 = corr_x_y_2), solved from <x'y'> = sum_i w_i[(mu_x_i-<x>)(mu_y_i-<y>) + corr sigma_x_i sigma_y_i]:
      corr = (<x'y'> - sum_i w_i (mu_x_i-<x>)(mu_y_i-<y>)) / (sum_i w_i sigma_x_i sigma_y_i),
    bounded by smooth_corr_quotient. All args are (ngrdcol, nz). Returns (corr_x_y_1, corr_x_y_2)."""
    xpyp = jnp.asarray(xpyp, dtype=jnp.float64); xm = jnp.asarray(xm, dtype=jnp.float64)
    ym = jnp.asarray(ym, dtype=jnp.float64)
    mu_x_1 = jnp.asarray(mu_x_1, dtype=jnp.float64); mu_x_2 = jnp.asarray(mu_x_2, dtype=jnp.float64)
    mu_y_1 = jnp.asarray(mu_y_1, dtype=jnp.float64); mu_y_2 = jnp.asarray(mu_y_2, dtype=jnp.float64)
    sx1 = jnp.asarray(sigma_x_1_sqd, dtype=jnp.float64); sx2 = jnp.asarray(sigma_x_2_sqd, dtype=jnp.float64)
    sy1 = jnp.asarray(sigma_y_1_sqd, dtype=jnp.float64); sy2 = jnp.asarray(sigma_y_2_sqd, dtype=jnp.float64)
    a = jnp.asarray(mixt_frac, dtype=jnp.float64)

    numerator = (xpyp - a * (mu_x_1 - xm) * (mu_y_1 - ym)
                 - (1.0 - a) * (mu_x_2 - xm) * (mu_y_2 - ym))
    denominator = a * jnp.sqrt(sx1 * sy1) + (1.0 - a) * jnp.sqrt(sx2 * sy2)
    corr = smooth_corr_quotient(numerator, denominator, _EPS)
    return corr, corr
