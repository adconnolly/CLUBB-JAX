"""JAX port of CLUBB_core/hydromet_pdf_parameter_module.F90 — hydrometeor-PDF parameter containers.

This Fortran module is a pair of derived types plus their zero/allocate initializers (no physics):
  * `hydromet_pdf_parameter` — per-column means/variances/correlations of the hydrometeor PDF.
  * `precipitation_fractions` — the (ngrdcol, nzt) precip-fraction fields.
Ported as frozen dataclasses holding jnp arrays (differentiable-compatible, zeros carry zero gradient), with
`init_hydromet_pdf_params` / `init_precip_fracs` / `zero_precip_fracs` mirroring the Fortran initializers.
Validated in `tests/test_hydromet_pdf_parameter.py` (shapes, dims metadata, all-zero, round-trip).
"""
from dataclasses import dataclass, field

import jax.numpy as jnp

MAX_HYDROMET_DIM = 8   # hydromet_pdf_parameter_module.F90:27


@dataclass(frozen=True)
class HydrometPdfParameter:
    """Means/variances/correlations of the hydrometeor PDF (hydromet_pdf_parameter type)."""
    hm_1: jnp.ndarray
    hm_2: jnp.ndarray
    mu_hm_1: jnp.ndarray
    mu_hm_2: jnp.ndarray
    sigma_hm_1: jnp.ndarray
    sigma_hm_2: jnp.ndarray
    corr_w_hm_1: jnp.ndarray
    corr_w_hm_2: jnp.ndarray
    corr_chi_hm_1: jnp.ndarray
    corr_chi_hm_2: jnp.ndarray
    corr_eta_hm_1: jnp.ndarray
    corr_eta_hm_2: jnp.ndarray
    corr_hmx_hmy_1: jnp.ndarray
    corr_hmx_hmy_2: jnp.ndarray
    mu_Ncn_1: float
    mu_Ncn_2: float
    sigma_Ncn_1: float
    sigma_Ncn_2: float


@dataclass(frozen=True)
class PrecipitationFractions:
    """Precipitation-fraction fields, shape (ngrdcol, nzt) (precipitation_fractions type)."""
    ngrdcol: int
    nzt: int
    precip_frac: jnp.ndarray
    precip_frac_1: jnp.ndarray
    precip_frac_2: jnp.ndarray


def init_hydromet_pdf_params():
    """Zero-initialize a HydrometPdfParameter (hydromet_pdf_parameter_module.F90:init_hydromet_pdf_params)."""
    vec = lambda: jnp.zeros(MAX_HYDROMET_DIM)
    mat = lambda: jnp.zeros((MAX_HYDROMET_DIM, MAX_HYDROMET_DIM))
    return HydrometPdfParameter(
        hm_1=vec(), hm_2=vec(), mu_hm_1=vec(), mu_hm_2=vec(),
        sigma_hm_1=vec(), sigma_hm_2=vec(), corr_w_hm_1=vec(), corr_w_hm_2=vec(),
        corr_chi_hm_1=vec(), corr_chi_hm_2=vec(), corr_eta_hm_1=vec(), corr_eta_hm_2=vec(),
        corr_hmx_hmy_1=mat(), corr_hmx_hmy_2=mat(),
        mu_Ncn_1=0.0, mu_Ncn_2=0.0, sigma_Ncn_1=0.0, sigma_Ncn_2=0.0)


def init_precip_fracs(nzt, ngrdcol):
    """Allocate + zero a PrecipitationFractions (hydromet_pdf_parameter_module.F90:init_precip_fracs_api)."""
    z = jnp.zeros((ngrdcol, nzt))
    return PrecipitationFractions(ngrdcol=ngrdcol, nzt=nzt,
                                  precip_frac=z, precip_frac_1=z, precip_frac_2=z)


def zero_precip_fracs(precip_fracs):
    """Zero the precip-fraction fields, preserving dims (hydromet_pdf_parameter_module.F90:zero_precip_fracs_api)."""
    z = jnp.zeros((precip_fracs.ngrdcol, precip_fracs.nzt))
    return PrecipitationFractions(ngrdcol=precip_fracs.ngrdcol, nzt=precip_fracs.nzt,
                                  precip_frac=z, precip_frac_1=z, precip_frac_2=z)
