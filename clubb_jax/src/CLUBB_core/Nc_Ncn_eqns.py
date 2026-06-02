"""JAX port of Nc_Ncn_eqns.F90 — cloud-nuclei-concentration mean <Ncn>.

The simplified cloud nuclei concentration N_cn has a single lognormal marginal over the
domain; cloud droplet concentration is Nc = Ncn*H(chi). These functions invert the PDF
integral <Nc> = <Ncn> * SUM_i mixt_frac_i * <H(chi)>_i to get <Ncn> (Ncnm) from the
in-cloud mean Nc and the chi-PDF parameters — Ncnm is the autoconversion-rate input mu_Ncn.
Oracle: Nc_Ncn_eqns.F90:251 (Nc_in_cloud_to_Ncnm), :476 (Ncm_to_Ncnm), :856 (bivar_Ncnm_eqn_comp).

Verified bit-to-bit against the f2py API (f2py_nc_in_cloud_to_ncnm); see tests/test_Nc_Ncn_eqns.py.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_SQRT_2 = jnp.sqrt(2.0)
_CHI_TOL = 1.0e-8           # constants_clubb: max(1e-8, eps)
_CLOUD_FRAC_MIN = 0.005
_EPS = jnp.finfo(jnp.float64).eps


@jax.custom_jvp
def _safe_div(num, den):
    """num/den (den assumed > 0) with the FORWARD value exact but the gradient floored so
    den^2 cannot underflow to 0. At clear-air points den ~ 1e-170 and num ~ 0, where the bare
    quotient gradient -num/den^2 is 0/0 = nan; this regularizes it to a finite value."""
    return num / den


@_safe_div.defjvp
def _safe_div_jvp(primals, tangents):
    num, den = primals
    dnum, dden = tangents
    den2 = jnp.maximum(den * den, 1.0e-300)
    return num / den, (dnum * den - num * dden) / den2


def bivar_Ncnm_eqn_comp(mu_chi_i, sigma_chi_i, const_Ncnp2_on_Ncnm2, const_corr_chi_Ncn):
    """Per-component <H(chi)>_i denominator term. Nc_Ncn_eqns.F90:856.

    sigma_chi_i <= chi_tol: 1 if mu_chi_i>0 else 0; otherwise
    0.5*erfc(-(1/sqrt2)*(mu_chi_i/sigma_chi_i + const_corr*sqrt(const_Ncnp2)))
    (the single erfc form covers both const_Ncnp2>0 and =0)."""
    sig_safe = jnp.maximum(sigma_chi_i, _CHI_TOL)
    varies = 0.5 * jax.scipy.special.erfc(
        -(1.0 / _SQRT_2) * (mu_chi_i / sig_safe
                            + const_corr_chi_Ncn * jnp.sqrt(const_Ncnp2_on_Ncnm2)))
    const = jnp.where(mu_chi_i > 0.0, 1.0, 0.0)
    return jnp.where(sigma_chi_i <= _CHI_TOL, const, varies)


def Ncm_to_Ncnm(mu_chi_1, mu_chi_2, sigma_chi_1, sigma_chi_2, mixt_frac, Ncm,
                const_Ncnp2_on_Ncnm2, const_corr_chi_Ncn, Ncnm_val_denom_0):
    """<Ncn> from the overall <Nc> and the chi PDF. Nc_Ncn_eqns.F90:476."""
    denom = (mixt_frac * bivar_Ncnm_eqn_comp(mu_chi_1, sigma_chi_1,
                                             const_Ncnp2_on_Ncnm2, const_corr_chi_Ncn)
             + (1.0 - mixt_frac) * bivar_Ncnm_eqn_comp(mu_chi_2, sigma_chi_2,
                                                       const_Ncnp2_on_Ncnm2, const_corr_chi_Ncn))
    denom_safe = jnp.where(denom > 0.0, denom, 1.0)
    return jnp.where(denom > 0.0, _safe_div(Ncm, denom_safe), Ncnm_val_denom_0)


def Nc_in_cloud_to_Ncnm(mu_chi_1, mu_chi_2, sigma_chi_1, sigma_chi_2, mixt_frac,
                        Nc_in_cloud, cloud_frac_1, cloud_frac_2,
                        const_Ncnp2_on_Ncnm2, const_corr_chi_Ncn):
    """<Ncn> from the in-cloud <Nc>. Nc_Ncn_eqns.F90:251.

    cloud_frac = mixt_frac*cloud_frac_1 + (1-mixt_frac)*cloud_frac_2. When there is cloud
    AND Ncn varies, Ncm = Nc_in_cloud*cloud_frac and Ncnm = Ncm_to_Ncnm(...); otherwise
    (clear or constant Ncn) Ncnm = Nc_in_cloud."""
    cloud_frac = mixt_frac * cloud_frac_1 + (1.0 - mixt_frac) * cloud_frac_2
    Ncm = Nc_in_cloud * cloud_frac
    ncnm_vary = Ncm_to_Ncnm(mu_chi_1, mu_chi_2, sigma_chi_1, sigma_chi_2, mixt_frac, Ncm,
                            const_Ncnp2_on_Ncnm2, const_corr_chi_Ncn, Nc_in_cloud)
    cond = (cloud_frac > _CLOUD_FRAC_MIN) & \
           (jnp.abs(const_corr_chi_Ncn * const_Ncnp2_on_Ncnm2) > _EPS)
    return jnp.where(cond, ncnm_vary, Nc_in_cloud)
