"""Normal/lognormal covariance integrals — JAX port of PDF_integrals_covar.F90.

These closed forms give the component covariance Cov_i(x1, x2^alpha x3^beta) (and the 4-variable
analogue) for the KK upscaled second-moment microphysics tendencies (KK_upscaled_covar_driver). They
reuse the same parabolic-cylinder `Dv` + gamma primitives as the already-ported mean integrals
(PDF_integrals_means.py). Ported incrementally with Monte-Carlo + oracle validation (see DESIGN.md
"KK microphysics"). Differentiable.
"""
import jax.numpy as jnp

from clubb_jax.src.Microphys.KK_microphys.PDF_integrals_means import _dvc, _signed_pow
from clubb_jax.src.Microphys.KK_microphys.parabolic_cylinder import _gamma_real

_SQRT_2PI = jnp.sqrt(2.0 * jnp.pi)


def quadrivar_NNLL_covar(mu_x1, mu_x2, mu_x3_n, mu_x4_n, sigma_x1, sigma_x2, sigma_x3_n, sigma_x4_n,
                         rho_x1x2, rho_x1x3_n, rho_x1x4_n, rho_x2x3_n, rho_x2x4_n, rho_x3x4_n,
                         x1_mean, x2_alpha_x3_beta_x4_gamma_mean, alpha_exp, beta_exp, gamma_exp):
    """Cov_i(x1, x2^α x3^β x4^γ) for x1,x2 ~ normal and x3,x4 ~ lognormal — the SUBSATURATED (chi<0)
    region form used by the KK evaporation / w covariances. Faithful port of
    PDF_integrals_covar.F90:quadrivar_NNLL_covar (base, all-varying); the `(-σ_x2)^α` is the signed
    continuation `_signed_pow(-σ_x2, α)` (same convention as the validated trivar_NLL_mean), and the
    parabolic-cylinder args use +s_cc (not -s_c)."""
    s_cc = (mu_x2 / sigma_x2
            + rho_x2x3_n * sigma_x3_n * beta_exp
            + rho_x2x4_n * sigma_x4_n * gamma_exp)
    return (1.0 / _SQRT_2PI
            * _signed_pow(-sigma_x2, alpha_exp)
            * jnp.exp(mu_x3_n * beta_exp + mu_x4_n * gamma_exp
                      + 0.5 * (1.0 - rho_x2x3_n ** 2) * sigma_x3_n ** 2 * beta_exp ** 2
                      + 0.5 * (1.0 - rho_x2x4_n ** 2) * sigma_x4_n ** 2 * gamma_exp ** 2
                      + (rho_x3x4_n - rho_x2x3_n * rho_x2x4_n)
                      * sigma_x3_n * beta_exp * sigma_x4_n * gamma_exp)
            * jnp.exp(0.25 * s_cc ** 2 - (mu_x2 / sigma_x2) * s_cc
                      + 0.5 * (mu_x2 ** 2 / sigma_x2 ** 2))
            * (-rho_x1x2 * sigma_x1 * _gamma_real(alpha_exp + 2.0) * _dvc(-(alpha_exp + 2.0), s_cc)
               + (mu_x1 - x1_mean
                  - (mu_x2 / sigma_x2) * rho_x1x2 * sigma_x1
                  + (rho_x1x3_n - rho_x1x2 * rho_x2x3_n) * sigma_x1 * sigma_x3_n * beta_exp
                  + (rho_x1x4_n - rho_x1x2 * rho_x2x4_n) * sigma_x1 * sigma_x4_n * gamma_exp)
               * _gamma_real(alpha_exp + 1.0) * _dvc(-(alpha_exp + 1.0), s_cc))
            - x2_alpha_x3_beta_x4_gamma_mean * (mu_x1 - x1_mean))


def quadrivar_NNLL_covar_const_x1(mu_x1, mu_x2, mu_x3_n, mu_x4_n, sigma_x2, sigma_x3_n, sigma_x4_n,
                                  rho_x2x3_n, rho_x2x4_n, rho_x3x4_n, x1_mean,
                                  x2_alpha_x3_beta_x4_gamma_mean, alpha_exp, beta_exp, gamma_exp):
    """x1 constant. PDF_integrals_covar.F90:quadrivar_NNLL_covar_const_x1."""
    s_cc = (mu_x2 / sigma_x2 + rho_x2x3_n * sigma_x3_n * beta_exp + rho_x2x4_n * sigma_x4_n * gamma_exp)
    return ((1.0 / _SQRT_2PI) * (mu_x1 - x1_mean)
            * _signed_pow(-sigma_x2, alpha_exp)
            * jnp.exp(mu_x3_n * beta_exp + mu_x4_n * gamma_exp
                      + 0.5 * (1.0 - rho_x2x3_n ** 2) * sigma_x3_n ** 2 * beta_exp ** 2
                      + 0.5 * (1.0 - rho_x2x4_n ** 2) * sigma_x4_n ** 2 * gamma_exp ** 2
                      + (rho_x3x4_n - rho_x2x3_n * rho_x2x4_n) * sigma_x3_n * beta_exp * sigma_x4_n * gamma_exp)
            * jnp.exp(0.25 * s_cc ** 2 - (mu_x2 / sigma_x2) * s_cc + 0.5 * (mu_x2 ** 2 / sigma_x2 ** 2))
            * _gamma_real(alpha_exp + 1.0) * _dvc(-(alpha_exp + 1.0), s_cc)
            - x2_alpha_x3_beta_x4_gamma_mean * (mu_x1 - x1_mean))


def quadrivar_NNLL_covar_const_x2(mu_x1, mu_x2, mu_x3_n, mu_x4_n, sigma_x1, sigma_x3_n, sigma_x4_n,
                                  rho_x1x3_n, rho_x1x4_n, rho_x3x4_n, x1_mean,
                                  x2_alpha_x3_beta_x4_gamma_mean, alpha_exp, beta_exp, gamma_exp):
    """x2 (chi) constant. Nonzero only for mu_x2<=0 (subsaturated). signed_pow(mu_x2,α) for the chi<0 branch.
    PDF_integrals_covar.F90:quadrivar_NNLL_covar_const_x2."""
    pos = (_signed_pow(mu_x2, alpha_exp)
           * (mu_x1 - x1_mean + rho_x1x3_n * sigma_x1 * sigma_x3_n * beta_exp
              + rho_x1x4_n * sigma_x1 * sigma_x4_n * gamma_exp)
           * jnp.exp(mu_x3_n * beta_exp + mu_x4_n * gamma_exp
                     + 0.5 * sigma_x3_n ** 2 * beta_exp ** 2 + 0.5 * sigma_x4_n ** 2 * gamma_exp ** 2
                     + rho_x3x4_n * sigma_x3_n * beta_exp * sigma_x4_n * gamma_exp)
           - x2_alpha_x3_beta_x4_gamma_mean * (mu_x1 - x1_mean))
    return jnp.where(mu_x2 <= 0.0, pos, -x2_alpha_x3_beta_x4_gamma_mean * (mu_x1 - x1_mean))


def quadrivar_NNLL_covar_const_x3(mu_x1, mu_x2, mu_x3, mu_x4_n, sigma_x1, sigma_x2, sigma_x4_n,
                                  rho_x1x2, rho_x1x4_n, rho_x2x4_n, x1_mean,
                                  x2_alpha_x3_beta_x4_gamma_mean, alpha_exp, beta_exp, gamma_exp):
    """x3 (r_r) constant. PDF_integrals_covar.F90:quadrivar_NNLL_covar_const_x3."""
    s_cc = mu_x2 / sigma_x2 + rho_x2x4_n * sigma_x4_n * gamma_exp
    return (1.0 / _SQRT_2PI
            * _signed_pow(-sigma_x2, alpha_exp) * mu_x3 ** beta_exp
            * jnp.exp(mu_x4_n * gamma_exp + 0.5 * sigma_x4_n ** 2 * gamma_exp ** 2 - 0.25 * s_cc ** 2)
            * (-rho_x1x2 * sigma_x1 * _gamma_real(alpha_exp + 2.0) * _dvc(-(alpha_exp + 2.0), s_cc)
               + (mu_x1 - x1_mean - (mu_x2 / sigma_x2) * rho_x1x2 * sigma_x1
                  + (rho_x1x4_n - rho_x1x2 * rho_x2x4_n) * sigma_x1 * sigma_x4_n * gamma_exp)
               * _gamma_real(alpha_exp + 1.0) * _dvc(-(alpha_exp + 1.0), s_cc))
            - x2_alpha_x3_beta_x4_gamma_mean * (mu_x1 - x1_mean))


def _q_const_neg(pos, mu_x1, mu_x2, x1_mean, x2a):
    """Subsaturated multi-const dispatch: the `pos` form for mu_x2<=0, else the −overall-mean term."""
    return jnp.where(mu_x2 <= 0.0, pos, -x2a * (mu_x1 - x1_mean))


def quadrivar_NNLL_covar_const_x1x2(mu_x1, mu_x2, mu_x3_n, mu_x4_n, sigma_x3_n, sigma_x4_n, rho_x3x4_n,
                                    x1_mean, x2a, alpha_exp, beta_exp, gamma_exp):
    """x1, x2 constant. PDF_integrals_covar.F90:quadrivar_NNLL_covar_const_x1x2."""
    pos = ((mu_x1 - x1_mean) * _signed_pow(mu_x2, alpha_exp)
           * jnp.exp(mu_x3_n * beta_exp + mu_x4_n * gamma_exp
                     + 0.5 * sigma_x3_n ** 2 * beta_exp ** 2 + 0.5 * sigma_x4_n ** 2 * gamma_exp ** 2
                     + rho_x3x4_n * sigma_x3_n * beta_exp * sigma_x4_n * gamma_exp)
           - x2a * (mu_x1 - x1_mean))
    return _q_const_neg(pos, mu_x1, mu_x2, x1_mean, x2a)


def quadrivar_NNLL_covar_const_x1x3(mu_x1, mu_x2, mu_x3, mu_x4_n, sigma_x2, sigma_x4_n, rho_x2x4_n,
                                    x1_mean, x2a, alpha_exp, beta_exp, gamma_exp):
    """x1, x3 constant. PDF_integrals_covar.F90:quadrivar_NNLL_covar_const_x1x3."""
    s_cc = mu_x2 / sigma_x2 + rho_x2x4_n * sigma_x4_n * gamma_exp
    return ((1.0 / _SQRT_2PI) * (mu_x1 - x1_mean) * _signed_pow(-sigma_x2, alpha_exp) * mu_x3 ** beta_exp
            * jnp.exp(mu_x4_n * gamma_exp + 0.5 * sigma_x4_n ** 2 * gamma_exp ** 2 - 0.25 * s_cc ** 2)
            * _gamma_real(alpha_exp + 1.0) * _dvc(-(alpha_exp + 1.0), s_cc)
            - x2a * (mu_x1 - x1_mean))


def quadrivar_NNLL_covar_const_x2x3(mu_x1, mu_x2, mu_x3, mu_x4_n, sigma_x1, sigma_x4_n, rho_x1x4_n,
                                    x1_mean, x2a, alpha_exp, beta_exp, gamma_exp):
    """x2, x3 constant. PDF_integrals_covar.F90:quadrivar_NNLL_covar_const_x2x3."""
    pos = (_signed_pow(mu_x2, alpha_exp) * mu_x3 ** beta_exp
           * (mu_x1 - x1_mean + rho_x1x4_n * sigma_x1 * sigma_x4_n * gamma_exp)
           * jnp.exp(mu_x4_n * gamma_exp + 0.5 * sigma_x4_n ** 2 * gamma_exp ** 2)
           - x2a * (mu_x1 - x1_mean))
    return _q_const_neg(pos, mu_x1, mu_x2, x1_mean, x2a)


def quadrivar_NNLL_covar_const_x3x4(mu_x1, mu_x2, mu_x3, mu_x4, sigma_x1, sigma_x2, rho_x1x2,
                                    x1_mean, x2a, alpha_exp, beta_exp, gamma_exp):
    """x3, x4 constant. PDF_integrals_covar.F90:quadrivar_NNLL_covar_const_x3x4."""
    r = mu_x2 / sigma_x2
    return (1.0 / _SQRT_2PI * _signed_pow(-sigma_x2, alpha_exp) * mu_x3 ** beta_exp * mu_x4 ** gamma_exp
            * jnp.exp(-0.25 * (mu_x2 ** 2 / sigma_x2 ** 2))
            * (-rho_x1x2 * sigma_x1 * _gamma_real(alpha_exp + 2.0) * _dvc(-(alpha_exp + 2.0), r)
               + (mu_x1 - x1_mean - r * rho_x1x2 * sigma_x1)
               * _gamma_real(alpha_exp + 1.0) * _dvc(-(alpha_exp + 1.0), r))
            - x2a * (mu_x1 - x1_mean))


def quadrivar_NNLL_covar_cst_x1x2x3(mu_x1, mu_x2, mu_x3, mu_x4_n, sigma_x4_n,
                                    x1_mean, x2a, alpha_exp, beta_exp, gamma_exp):
    """x1, x2, x3 constant. PDF_integrals_covar.F90:quadrivar_NNLL_covar_cst_x1x2x3."""
    pos = ((mu_x1 - x1_mean) * _signed_pow(mu_x2, alpha_exp) * mu_x3 ** beta_exp
           * jnp.exp(mu_x4_n * gamma_exp + 0.5 * sigma_x4_n ** 2 * gamma_exp ** 2)
           - x2a * (mu_x1 - x1_mean))
    return _q_const_neg(pos, mu_x1, mu_x2, x1_mean, x2a)


def quadrivar_NNLL_covar_cst_x1x3x4(mu_x1, mu_x2, mu_x3, mu_x4, sigma_x2,
                                    x1_mean, x2a, alpha_exp, beta_exp, gamma_exp):
    """x1, x3, x4 constant. PDF_integrals_covar.F90:quadrivar_NNLL_covar_cst_x1x3x4."""
    return ((1.0 / _SQRT_2PI) * (mu_x1 - x1_mean) * _signed_pow(-sigma_x2, alpha_exp)
            * mu_x3 ** beta_exp * mu_x4 ** gamma_exp
            * jnp.exp(-0.25 * (mu_x2 ** 2 / sigma_x2 ** 2))
            * _gamma_real(alpha_exp + 1.0) * _dvc(-(alpha_exp + 1.0), mu_x2 / sigma_x2)
            - x2a * (mu_x1 - x1_mean))


def quadrivar_NNLL_covar_cst_x2x3x4(mu_x1, mu_x2, mu_x3, mu_x4, x1_mean, x2a, alpha_exp, beta_exp, gamma_exp):
    """x2, x3, x4 constant (== the all-constant case). PDF_integrals_covar.F90:quadrivar_NNLL_covar_cst_x2x3x4."""
    pos = (mu_x1 - x1_mean) * (_signed_pow(mu_x2, alpha_exp) * mu_x3 ** beta_exp * mu_x4 ** gamma_exp - x2a)
    return _q_const_neg(pos, mu_x1, mu_x2, x1_mean, x2a)


# All four constant: identical formula to cst_x2x3x4.
quadrivar_NNLL_covar_const_all = quadrivar_NNLL_covar_cst_x2x3x4


def quadrivar_NNLL_covar_eq(mu_x_i, mu_chi_i, mu_rr_i, mu_Nr_i, mu_rr_i_n, mu_Nr_i_n,
                            sigma_x_i, sigma_chi_i, sigma_rr_i, sigma_Nr_i, sigma_rr_i_n, sigma_Nr_i_n,
                            corr_x_chi_i, corr_x_rr_i_n, corr_x_Nr_i_n, corr_chi_rr_i_n, corr_chi_Nr_i_n,
                            corr_rr_Nr_i_n, x_mean, mc_tndcy_mean, mc_coef, x_tol, rr_tol, Nr_tol,
                            alpha_exp, beta_exp, gamma_exp):
    """Dispatch wrapper for the quadrivariate covariance Cov_i(x, chi^α r_r^β N_r^γ) — x=w|eta, chi, r_r, N_r —
    used by the KK EVAPORATION covariances. Selects the right of the base + 11 variants by which σ≈0
    (vectorised per-level), exploiting the (x3=r_r,β)↔(x4=N_r,γ) symmetry (an x4-const branch reuses the
    x3-const variant with x3↔x4 args + β↔γ swapped). Faithful port of
    KK_upscaled_covariances.F90:quadrivar_NNLL_covar_eq."""
    mu_x1 = mu_x_i
    mu_x2 = mu_chi_i
    mu_x3 = jnp.where(beta_exp >= 0.0, mu_rr_i, jnp.maximum(mu_rr_i, rr_tol))
    mu_x4 = jnp.where(gamma_exp >= 0.0, mu_Nr_i, jnp.maximum(mu_Nr_i, Nr_tol))
    mu_x3_n, mu_x4_n = mu_rr_i_n, mu_Nr_i_n
    sg1, sg2, sg3, sg4 = sigma_x_i, sigma_chi_i, sigma_rr_i, sigma_Nr_i
    sg3n, sg4n = sigma_rr_i_n, sigma_Nr_i_n
    r12, r13n, r14n = corr_x_chi_i, corr_x_rr_i_n, corr_x_Nr_i_n
    r23n, r24n, r34n = corr_chi_rr_i_n, corr_chi_Nr_i_n, corr_rr_Nr_i_n
    x1m = x_mean
    x2a = mc_tndcy_mean / mc_coef
    x2_tol = _CHI_TOL
    a, b, g = alpha_exp, beta_exp, gamma_exp

    s2g = jnp.where(sg2 > x2_tol, sg2, 1.0)
    s_cc = jnp.where(sg2 > x2_tol, mu_x2 / s2g + r23n * sg3n * b + r24n * sg4n * g, jnp.inf)
    c1 = sg1 <= x_tol
    c2 = (sg2 <= x2_tol) | (jnp.abs(s_cc) > _PARAB_CYL_MAX)
    c3 = sg3 <= rr_tol
    c4 = sg4 <= Nr_tol

    # variants (x3-form and, for the symmetric ones, the x3<->x4 swapped form with b<->g)
    v_all = quadrivar_NNLL_covar_const_all(mu_x1, mu_x2, mu_x3, mu_x4, x1m, x2a, a, b, g)
    v123 = quadrivar_NNLL_covar_cst_x1x2x3(mu_x1, mu_x2, mu_x3, mu_x4_n, sg4n, x1m, x2a, a, b, g)
    v123s = quadrivar_NNLL_covar_cst_x1x2x3(mu_x1, mu_x2, mu_x4, mu_x3_n, sg3n, x1m, x2a, a, g, b)
    v134 = quadrivar_NNLL_covar_cst_x1x3x4(mu_x1, mu_x2, mu_x3, mu_x4, s2g, x1m, x2a, a, b, g)
    v234 = quadrivar_NNLL_covar_cst_x2x3x4(mu_x1, mu_x2, mu_x3, mu_x4, x1m, x2a, a, b, g)
    v12 = quadrivar_NNLL_covar_const_x1x2(mu_x1, mu_x2, mu_x3_n, mu_x4_n, sg3n, sg4n, r34n, x1m, x2a, a, b, g)
    v13 = quadrivar_NNLL_covar_const_x1x3(mu_x1, mu_x2, mu_x3, mu_x4_n, s2g, sg4n, r24n, x1m, x2a, a, b, g)
    v13s = quadrivar_NNLL_covar_const_x1x3(mu_x1, mu_x2, mu_x4, mu_x3_n, s2g, sg3n, r23n, x1m, x2a, a, g, b)
    v23 = quadrivar_NNLL_covar_const_x2x3(mu_x1, mu_x2, mu_x3, mu_x4_n, sg1, sg4n, r14n, x1m, x2a, a, b, g)
    v23s = quadrivar_NNLL_covar_const_x2x3(mu_x1, mu_x2, mu_x4, mu_x3_n, sg1, sg3n, r13n, x1m, x2a, a, g, b)
    v34 = quadrivar_NNLL_covar_const_x3x4(mu_x1, mu_x2, mu_x3, mu_x4, sg1, s2g, r12, x1m, x2a, a, b, g)
    v_x1 = quadrivar_NNLL_covar_const_x1(mu_x1, mu_x2, mu_x3_n, mu_x4_n, s2g, sg3n, sg4n, r23n, r24n, r34n, x1m, x2a, a, b, g)
    v_x2 = quadrivar_NNLL_covar_const_x2(mu_x1, mu_x2, mu_x3_n, mu_x4_n, sg1, sg3n, sg4n, r13n, r14n, r34n, x1m, x2a, a, b, g)
    v_x3 = quadrivar_NNLL_covar_const_x3(mu_x1, mu_x2, mu_x3, mu_x4_n, sg1, s2g, sg4n, r12, r14n, r24n, x1m, x2a, a, b, g)
    v_x3s = quadrivar_NNLL_covar_const_x3(mu_x1, mu_x2, mu_x4, mu_x3_n, sg1, s2g, sg3n, r12, r13n, r23n, x1m, x2a, a, g, b)
    v_base = quadrivar_NNLL_covar(mu_x1, mu_x2, mu_x3_n, mu_x4_n, sg1, s2g, sg3n, sg4n,
                                  r12, r13n, r14n, r23n, r24n, r34n, x1m, x2a, a, b, g)

    out = v_base
    out = jnp.where(c4, v_x3s, out)
    out = jnp.where(c3, v_x3, out)
    out = jnp.where(c2, v_x2, out)
    out = jnp.where(c1, v_x1, out)
    out = jnp.where(c3 & c4, v34, out)
    out = jnp.where(c2 & c4, v23s, out)
    out = jnp.where(c2 & c3, v23, out)
    out = jnp.where(c1 & c4, v13s, out)
    out = jnp.where(c1 & c3, v13, out)
    out = jnp.where(c1 & c2, v12, out)
    out = jnp.where(c2 & c3 & c4, v234, out)
    out = jnp.where(c1 & c3 & c4, v134, out)
    out = jnp.where(c1 & c2 & c4, v123s, out)
    out = jnp.where(c1 & c2 & c3, v123, out)
    out = jnp.where(c1 & c2 & c3 & c4, v_all, out)
    return out


def trivar_NNL_covar(mu_x1, mu_x2, mu_x3_n, sigma_x1, sigma_x2, sigma_x3_n,
                     rho_x1x2, rho_x1x3_n, rho_x2x3_n,
                     x1_mean, x2_alpha_x3_beta_mean, alpha_exp, beta_exp):
    """Cov_i(x1, x2^alpha x3^beta) for x1,x2 ~ normal and x3 ~ lognormal (ln x3 normal), with the
    component correlations. `x1_mean` / `x2_alpha_x3_beta_mean` are the OVERALL means. Faithful port
    of PDF_integrals_covar.F90:trivar_NNL_covar (the all-varying base case; sigma_x2 > 0)."""
    s_c = (mu_x2 / sigma_x2) + rho_x2x3_n * sigma_x3_n * beta_exp
    return (1.0 / _SQRT_2PI
            * sigma_x2 ** alpha_exp
            * jnp.exp(mu_x3_n * beta_exp
                      + 0.5 * sigma_x3_n ** 2 * beta_exp ** 2
                      - 0.25 * s_c ** 2)
            * (rho_x1x2 * sigma_x1 * _gamma_real(alpha_exp + 2.0)
               * _dvc(-(alpha_exp + 2.0), -s_c)
               + (mu_x1 - x1_mean
                  - (mu_x2 / sigma_x2) * rho_x1x2 * sigma_x1
                  + (rho_x1x3_n - rho_x1x2 * rho_x2x3_n)
                  * sigma_x1 * sigma_x3_n * beta_exp)
               * _gamma_real(alpha_exp + 1.0)
               * _dvc(-(alpha_exp + 1.0), -s_c))
            - x2_alpha_x3_beta_mean * (mu_x1 - x1_mean))


def trivar_NNL_covar_const_x1(mu_x1, mu_x2, mu_x3_n, sigma_x2, sigma_x3_n,
                              rho_x2x3_n, x1_mean, x2_alpha_x3_beta_mean, alpha_exp, beta_exp):
    """x1 constant within the component (sigma_x1=0). PDF_integrals_covar.F90:trivar_NNL_covar_const_x1."""
    s_c = (mu_x2 / sigma_x2) + rho_x2x3_n * sigma_x3_n * beta_exp
    return ((1.0 / _SQRT_2PI) * (mu_x1 - x1_mean)
            * sigma_x2 ** alpha_exp
            * jnp.exp(mu_x3_n * beta_exp + 0.5 * sigma_x3_n ** 2 * beta_exp ** 2 - 0.25 * s_c ** 2)
            * _gamma_real(alpha_exp + 1.0) * _dvc(-(alpha_exp + 1.0), -s_c)
            - x2_alpha_x3_beta_mean * (mu_x1 - x1_mean))


def trivar_NNL_covar_const_x2(mu_x1, mu_x2, mu_x3_n, sigma_x1, sigma_x3_n,
                              rho_x1x3_n, x1_mean, x2_alpha_x3_beta_mean, alpha_exp, beta_exp):
    """x2 (chi) constant within the component (sigma_x2=0). PDF_integrals_covar.F90:trivar_NNL_covar_const_x2.
    For mu_x2<0 the x2^alpha integrand vanishes (subsaturated), leaving only the -overall-mean term."""
    pos = (mu_x2 ** alpha_exp
           * (mu_x1 - x1_mean + rho_x1x3_n * sigma_x1 * sigma_x3_n * beta_exp)
           * jnp.exp(mu_x3_n * beta_exp + 0.5 * sigma_x3_n ** 2 * beta_exp ** 2)
           - x2_alpha_x3_beta_mean * (mu_x1 - x1_mean))
    neg = -x2_alpha_x3_beta_mean * (mu_x1 - x1_mean)
    mu_x2_safe = jnp.where(mu_x2 >= 0.0, mu_x2, 1.0)   # avoid NaN^alpha in the unused branch
    pos = (mu_x2_safe ** alpha_exp
           * (mu_x1 - x1_mean + rho_x1x3_n * sigma_x1 * sigma_x3_n * beta_exp)
           * jnp.exp(mu_x3_n * beta_exp + 0.5 * sigma_x3_n ** 2 * beta_exp ** 2)
           - x2_alpha_x3_beta_mean * (mu_x1 - x1_mean))
    return jnp.where(mu_x2 >= 0.0, pos, neg)


def trivar_NNL_covar_const_x3(mu_x1, mu_x2, mu_x3, sigma_x1, sigma_x2,
                              rho_x1x2, x1_mean, x2_alpha_x3_beta_mean, alpha_exp, beta_exp):
    """x3 (N_cn or r_r) constant within the component (sigma_x3_n=0; x3=mu_x3). The l_const_Nc path.
    PDF_integrals_covar.F90:trivar_NNL_covar_const_x3."""
    r = mu_x2 / sigma_x2
    return (1.0 / _SQRT_2PI
            * sigma_x2 ** alpha_exp * mu_x3 ** beta_exp
            * jnp.exp(-0.25 * (mu_x2 ** 2 / sigma_x2 ** 2))
            * (rho_x1x2 * sigma_x1 * _gamma_real(alpha_exp + 2.0) * _dvc(-(alpha_exp + 2.0), -r)
               + (mu_x1 - x1_mean - r * rho_x1x2 * sigma_x1)
               * _gamma_real(alpha_exp + 1.0) * _dvc(-(alpha_exp + 1.0), -r))
            - x2_alpha_x3_beta_mean * (mu_x1 - x1_mean))


def _x2pow(mu_x2, alpha_exp):
    """mu_x2^alpha, NaN-safe in the unused mu_x2<0 branch (the caller's jnp.where discards it)."""
    return jnp.where(mu_x2 >= 0.0, mu_x2, 1.0) ** alpha_exp


def trivar_NNL_covar_const_x1x2(mu_x1, mu_x2, mu_x3_n, sigma_x3_n,
                                x1_mean, x2_alpha_x3_beta_mean, alpha_exp, beta_exp):
    """x1 and x2 constant. PDF_integrals_covar.F90:trivar_NNL_covar_const_x1x2."""
    pos = (_x2pow(mu_x2, alpha_exp) * (mu_x1 - x1_mean)
           * jnp.exp(mu_x3_n * beta_exp + 0.5 * sigma_x3_n ** 2 * beta_exp ** 2)
           - x2_alpha_x3_beta_mean * (mu_x1 - x1_mean))
    return jnp.where(mu_x2 >= 0.0, pos, -x2_alpha_x3_beta_mean * (mu_x1 - x1_mean))


def trivar_NNL_covar_const_x1x3(mu_x1, mu_x2, mu_x3, sigma_x2,
                                x1_mean, x2_alpha_x3_beta_mean, alpha_exp, beta_exp):
    """x1 and x3 constant. PDF_integrals_covar.F90:trivar_NNL_covar_const_x1x3."""
    return ((1.0 / _SQRT_2PI) * (mu_x1 - x1_mean)
            * sigma_x2 ** alpha_exp * mu_x3 ** beta_exp
            * jnp.exp(-0.25 * (mu_x2 ** 2 / sigma_x2 ** 2))
            * _gamma_real(alpha_exp + 1.0) * _dvc(-(alpha_exp + 1.0), -(mu_x2 / sigma_x2))
            - x2_alpha_x3_beta_mean * (mu_x1 - x1_mean))


def trivar_NNL_covar_const_x2x3(mu_x1, mu_x2, mu_x3, x1_mean,
                                x2_alpha_x3_beta_mean, alpha_exp, beta_exp):
    """x2 and x3 constant (== the all-constant case). PDF_integrals_covar.F90:trivar_NNL_covar_const_x2x3."""
    pos = (mu_x1 - x1_mean) * (_x2pow(mu_x2, alpha_exp) * mu_x3 ** beta_exp - x2_alpha_x3_beta_mean)
    return jnp.where(mu_x2 >= 0.0, pos, -x2_alpha_x3_beta_mean * (mu_x1 - x1_mean))


# All three constant: identical formula to const_x2x3.
trivar_NNL_covar_const_all = trivar_NNL_covar_const_x2x3

_CHI_TOL = 1.0e-8           # constants_clubb chi_tol
_PARAB_CYL_MAX = 49.0       # constants_clubb parab_cyl_max_input


def trivar_NNL_covar_eq(mu_x_i, mu_chi_i, mu_y_i, mu_y_i_n, sigma_x_i, sigma_chi_i,
                        sigma_y_i, sigma_y_i_n, corr_x_chi_i, corr_x_y_i_n, corr_chi_y_i_n,
                        x_mean, mc_tndcy_mean, mc_coef, x_tol, y_tol, alpha_exp, beta_exp):
    """Dispatch wrapper for the trivariate covariance Cov_i(x, chi^α y^β) — x=w|eta, chi, y=Ncn|rr.
    Maps the component moments to the integral inputs and selects the right form by which σ≈0
    (vectorised: per-level masks via jnp.where). Faithful port of
    KK_upscaled_covariances.F90:trivar_NNL_covar_eq."""
    mu_x1 = mu_x_i
    mu_x2 = mu_chi_i
    mu_x3 = jnp.where(beta_exp >= 0.0, mu_y_i, jnp.maximum(mu_y_i, y_tol))
    mu_x3_n = mu_y_i_n
    sigma_x1, sigma_x2, sigma_x3, sigma_x3_n = sigma_x_i, sigma_chi_i, sigma_y_i, sigma_y_i_n
    rho_x1x2, rho_x1x3_n, rho_x2x3_n = corr_x_chi_i, corr_x_y_i_n, corr_chi_y_i_n
    x1m = x_mean
    x2a = mc_tndcy_mean / mc_coef
    x2_tol = _CHI_TOL

    # Guard the σ_x2 denominator for the forms that use mu_x2/σ_x2 (only SELECTED when σ_x2>tol).
    s2g = jnp.where(sigma_x2 > x2_tol, sigma_x2, 1.0)
    s_c = jnp.where(sigma_x2 > x2_tol, mu_x2 / s2g + rho_x2x3_n * sigma_x3_n * beta_exp, jnp.inf)
    c1 = sigma_x1 <= x_tol
    c2 = (sigma_x2 <= x2_tol) | (jnp.abs(s_c) > _PARAB_CYL_MAX)
    c3 = sigma_x3 <= y_tol

    v_x1x2 = trivar_NNL_covar_const_x1x2(mu_x1, mu_x2, mu_x3_n, sigma_x3_n, x1m, x2a, alpha_exp, beta_exp)
    v_x1x3 = trivar_NNL_covar_const_x1x3(mu_x1, mu_x2, mu_x3, s2g, x1m, x2a, alpha_exp, beta_exp)
    v_x2x3 = trivar_NNL_covar_const_x2x3(mu_x1, mu_x2, mu_x3, x1m, x2a, alpha_exp, beta_exp)
    v_x1 = trivar_NNL_covar_const_x1(mu_x1, mu_x2, mu_x3_n, s2g, sigma_x3_n, rho_x2x3_n, x1m, x2a, alpha_exp, beta_exp)
    v_x2 = trivar_NNL_covar_const_x2(mu_x1, mu_x2, mu_x3_n, sigma_x1, sigma_x3_n, rho_x1x3_n, x1m, x2a, alpha_exp, beta_exp)
    v_x3 = trivar_NNL_covar_const_x3(mu_x1, mu_x2, mu_x3, sigma_x1, s2g, rho_x1x2, x1m, x2a, alpha_exp, beta_exp)
    v_base = trivar_NNL_covar(mu_x1, mu_x2, mu_x3_n, sigma_x1, s2g, sigma_x3_n,
                              rho_x1x2, rho_x1x3_n, rho_x2x3_n, x1m, x2a, alpha_exp, beta_exp)

    out = v_base
    out = jnp.where(c3, v_x3, out)
    out = jnp.where(c2, v_x2, out)
    out = jnp.where(c1, v_x1, out)
    out = jnp.where(c2 & c3, v_x2x3, out)
    out = jnp.where(c1 & c3, v_x1x3, out)
    out = jnp.where(c1 & c2, v_x1x2, out)
    out = jnp.where(c1 & c2 & c3, v_x2x3, out)   # const_all == const_x2x3
    return out
