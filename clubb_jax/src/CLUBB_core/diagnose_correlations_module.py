"""JAX port of CLUBB_core/diagnose_correlations_module.F90.

JAX/Python adaptation: ``iiPDF_w`` is 0-based, matching ``hm_metadata_type`` in
``corr_varnce_module.py``. The underlying Fortran argument is 1-based.
"""

import numpy as np
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.clubb_constants import max_mag_correlation
from clubb_jax.src.CLUBB_core.pdf_utilities import _safe_sqrt


def diagnose_correlations(pdf_dim, iiPDF_w, corr_array_pre, l_calc_w_corr):
    """Diagnose the hydrometeor correlation matrix for SILHS microphysics."""
    if l_calc_w_corr:
        raise NotImplementedError(
            "diagnose_correlations: l_calc_w_corr=True is not ported; the "
            "underlying Fortran approx_w_corr path is commented out."
        )

    corr_array_pre_swapped = rearrange_corr_array(pdf_dim, iiPDF_w, corr_array_pre)
    corr_array_swapped = corr_array_pre_swapped
    corr_array_swapped = diagnose_corr(
        pdf_dim,
        jnp.zeros((pdf_dim,), dtype=jnp.float64),
        corr_array_pre_swapped,
        corr_array_swapped,
    )
    return rearrange_corr_array(pdf_dim, iiPDF_w, corr_array_swapped)


def diagnose_corr(n_variables, sqrt_sigma2_on_mu2_ip,
                  corr_matrix_prescribed, corr_matrix_approx):
    """Diagnose missing correlations using Larson et al. (2011), eq. 15."""
    del sqrt_sigma2_on_mu2_ip

    s_1j = _safe_sqrt(1.0 - corr_matrix_approx[:, 0] ** 2)
    f_ij = jnp.clip(
        corr_matrix_prescribed,
        -max_mag_correlation,
        max_mag_correlation,
    )
    diagnosed = (
        corr_matrix_approx[:, 0, None] * corr_matrix_approx[:, 0][None, :]
        + f_ij * s_1j[:, None] * s_1j[None, :]
    )

    rows = jnp.arange(n_variables)[:, None]
    cols = jnp.arange(n_variables)[None, :]
    mask = (rows > cols) & (cols >= 1) & (cols <= n_variables - 2)
    return jnp.where(mask, diagnosed, corr_matrix_approx)


def calc_w_corr(wpxp, stdev_w, stdev_x, w_tol, x_tol):
    """Compute the correlation of w with x, clipped to the CLUBB bound."""
    calc_w_corr_value = wpxp / (
        jnp.maximum(stdev_x, x_tol) * jnp.maximum(stdev_w, w_tol)
    )
    return jnp.clip(
        calc_w_corr_value,
        -max_mag_correlation,
        max_mag_correlation,
    )


def calc_varnce(mixt_frac, x1, x2, xm, x1p2, x2p2):
    """Calculate xp2 from two PDF components."""
    return (
        mixt_frac * ((x1 - xm) ** 2 + x1p2)
        + (1.0 - mixt_frac) * ((x2 - xm) ** 2 + x2p2)
    )


def calc_mean(mixt_frac, x1, x2):
    """Calculate xm from two PDF components."""
    return mixt_frac * x1 + (1.0 - mixt_frac) * x2


def calc_cholesky_corr_mtx_approx(n_variables, iiPDF_w, corr_matrix):
    """Calculate transposed correlation Cholesky matrix and C = L L'."""
    corr_mtx_swap = rearrange_corr_array(n_variables, iiPDF_w, corr_matrix)
    corr_cholesky_mtx_swap = setup_corr_cholesky_mtx(
        n_variables,
        corr_mtx_swap,
    )
    corr_cholesky_mtx = rearrange_corr_array(
        n_variables,
        iiPDF_w,
        corr_cholesky_mtx_swap,
    )
    corr_mtx_approx_swap = cholesky_to_corr_mtx_approx(
        n_variables,
        corr_cholesky_mtx_swap,
    )
    corr_mtx_approx = rearrange_corr_array(
        n_variables,
        iiPDF_w,
        corr_mtx_approx_swap,
    )

    # Fortran error-stops in corr_array_assertion_checks here. The JAX port
    # keeps that as an explicit concrete helper rather than adding host-side
    # side effects to this jittable routine.
    rows = jnp.arange(n_variables)[:, None]
    cols = jnp.arange(n_variables)[None, :]
    corr_mtx_approx = jnp.where(rows < cols, 0.0, corr_mtx_approx)
    return corr_cholesky_mtx, corr_mtx_approx


def setup_corr_cholesky_mtx(n_variables, corr_matrix):
    """Set up the transposed correlation Cholesky matrix."""
    s = _safe_sqrt(1.0 - corr_matrix ** 2)

    rows = jnp.arange(n_variables)[:, None]
    cols = jnp.arange(n_variables)[None, :]
    corr_cholesky_mtx_t = jnp.where(rows >= cols, 1.0, 0.0)

    for j in range(1, n_variables):
        diag_value = 1.0
        for i in range(j):
            diag_value = diag_value * s[j, i]
        corr_cholesky_mtx_t = corr_cholesky_mtx_t.at[j, j].set(diag_value)

    for j in range(1, n_variables):
        corr_cholesky_mtx_t = corr_cholesky_mtx_t.at[j, 0].set(
            corr_matrix[j, 0]
        )

    for i in range(1, n_variables - 1):
        for j in range(i + 1, n_variables):
            upper_value = 1.0
            for k in range(i):
                upper_value = upper_value * s[j, k]
            upper_value = upper_value * corr_matrix[j, i]
            corr_cholesky_mtx_t = corr_cholesky_mtx_t.at[j, i].set(upper_value)

    return corr_cholesky_mtx_t


def cholesky_to_corr_mtx_approx(n_variables, corr_cholesky_mtx_t):
    """Approximate the correlation matrix from the Cholesky matrix."""
    del n_variables
    return corr_cholesky_mtx_t @ corr_cholesky_mtx_t.T


def corr_array_assertion_checks(n_variables, corr_array):
    """Concrete correlation-matrix checks matching the Fortran debug routine."""
    corr_array = np.asarray(corr_array, dtype=np.float64)
    off_diagonal = corr_array[~np.eye(n_variables, dtype=bool)]
    off_diagonal_in_range = bool(
        np.all(np.abs(off_diagonal) <= max_mag_correlation)
    )
    diagonal_is_one = bool(
        np.all(np.abs(np.diagonal(corr_array) - 1.0) <= 1.0e-6)
    )
    return off_diagonal_in_range and diagonal_is_one


def rearrange_corr_array(pdf_dim, iiPDF_w, corr_array):
    """Swap the w correlations to/from the first row/column."""
    w_idx = int(iiPDF_w)
    swap_array = corr_array[:, 0]

    corr_array_swapped = corr_array
    corr_array_swapped = corr_array_swapped.at[:w_idx + 1, 0].set(
        corr_array[w_idx, w_idx::-1]
    )
    corr_array_swapped = corr_array_swapped.at[w_idx + 1:pdf_dim, 0].set(
        corr_array[w_idx + 1:pdf_dim, w_idx]
    )
    corr_array_swapped = corr_array_swapped.at[w_idx, :w_idx + 1].set(
        swap_array[w_idx::-1]
    )
    corr_array_swapped = corr_array_swapped.at[w_idx + 1:pdf_dim, w_idx].set(
        swap_array[w_idx + 1:pdf_dim]
    )
    return corr_array_swapped
