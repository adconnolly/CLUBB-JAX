"""JAX port of selected routines from ``src/CLUBB_core/numerical_check.F90``."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from clubb_jax.src.CLUBB_core.error_code import CLUBB_FATAL_ERROR
from clubb_jax.src.derived_types import ErrInfo


def parameterization_check(
    nzm: int,
    nzt: int,
    ngrdcol: int,
    sclr_dim: int,
    edsclr_dim: int,
    thlm_forcing,
    rtm_forcing,
    um_forcing,
    vm_forcing,
    wm_zm,
    wm_zt,
    p_in_Pa,
    rho_zm,
    rho,
    exner,
    rho_ds_zm,
    rho_ds_zt,
    invrs_rho_ds_zm,
    invrs_rho_ds_zt,
    thv_ds_zm,
    thv_ds_zt,
    wpthlp_sfc,
    wprtp_sfc,
    upwp_sfc,
    vpwp_sfc,
    p_sfc,
    um,
    upwp,
    vm,
    vpwp,
    up2,
    vp2,
    rtm,
    wprtp,
    thlm,
    wpthlp,
    wp2,
    wp3,
    rtp2,
    thlp2,
    rtpthlp,
    prefix: str,
    wpsclrp_sfc,
    wpedsclrp_sfc,
    sclrm,
    wpsclrp,
    sclrp2,
    sclrprtp,
    sclrpthlp,
    sclrm_forcing,
    edsclrm,
    edsclrm_forcing,
    err_info: ErrInfo,
):
    """Determine what input variables may have NaN or invalid negative values."""
    del nzm, nzt, ngrdcol

    proc_name = "advance_clubb_core"
    operation = prefix + proc_name

    #-------- Input Nan Check ----------------------------------------------

    err_info = check_nan(thlm_forcing, "thlm_forcing", operation, err_info)
    err_info = check_nan(rtm_forcing, "rtm_forcing", operation, err_info)
    err_info = check_nan(um_forcing, "um_forcing", operation, err_info)
    err_info = check_nan(vm_forcing, "vm_forcing", operation, err_info)

    err_info = check_nan(wm_zm, "wm_zm", operation, err_info)
    err_info = check_nan(wm_zt, "wm_zt", operation, err_info)
    err_info = check_nan(p_in_Pa, "p_in_Pa", operation, err_info)
    err_info = check_nan(rho_zm, "rho_zm", operation, err_info)
    err_info = check_nan(rho, "rho", operation, err_info)
    err_info = check_nan(exner, "exner", operation, err_info)
    err_info = check_nan(rho_ds_zm, "rho_ds_zm", operation, err_info)
    err_info = check_nan(rho_ds_zt, "rho_ds_zt", operation, err_info)
    err_info = check_nan(invrs_rho_ds_zm, "invrs_rho_ds_zm", operation, err_info)
    err_info = check_nan(invrs_rho_ds_zt, "invrs_rho_ds_zt", operation, err_info)
    err_info = check_nan(thv_ds_zm, "thv_ds_zm", operation, err_info)
    err_info = check_nan(thv_ds_zt, "thv_ds_zt", operation, err_info)

    err_info = check_nan(um, "um", operation, err_info)
    err_info = check_nan(upwp, "upwp", operation, err_info)
    err_info = check_nan(vm, "vm", operation, err_info)
    err_info = check_nan(vpwp, "vpwp", operation, err_info)
    err_info = check_nan(up2, "up2", operation, err_info)
    err_info = check_nan(vp2, "vp2", operation, err_info)
    err_info = check_nan(rtm, "rtm", operation, err_info)
    err_info = check_nan(wprtp, "wprtp", operation, err_info)
    err_info = check_nan(thlm, "thlm", operation, err_info)
    err_info = check_nan(wpthlp, "wpthlp", operation, err_info)
    err_info = check_nan(wp2, "wp2", operation, err_info)
    err_info = check_nan(wp3, "wp3", operation, err_info)
    err_info = check_nan(rtp2, "rtp2", operation, err_info)
    err_info = check_nan(thlp2, "thlp2", operation, err_info)
    err_info = check_nan(rtpthlp, "rtpthlp", operation, err_info)

    err_info = check_nan(wpthlp_sfc, "wpthlp_sfc", operation, err_info)
    err_info = check_nan(wprtp_sfc, "wprtp_sfc", operation, err_info)
    err_info = check_nan(upwp_sfc, "upwp_sfc", operation, err_info)
    err_info = check_nan(vpwp_sfc, "vpwp_sfc", operation, err_info)
    err_info = check_nan(p_sfc, "p_sfc", operation, err_info)

    for sclr in range(sclr_dim):
        err_info = check_nan(
            sclrm_forcing[:, :, sclr], "sclrm_forcing", operation, err_info,
        )
        err_info = check_nan(
            wpsclrp_sfc[:, sclr], "wpsclrp_sfc", operation, err_info,
        )
        err_info = check_nan(sclrm[:, :, sclr], "sclrm", operation, err_info)
        err_info = check_nan(wpsclrp[:, :, sclr], "wpsclrp", operation, err_info)
        err_info = check_nan(sclrp2[:, :, sclr], "sclrp2", operation, err_info)
        err_info = check_nan(
            sclrprtp[:, :, sclr], "sclrprtp", operation, err_info,
        )
        err_info = check_nan(
            sclrpthlp[:, :, sclr], "sclrpthlp", operation, err_info,
        )

    for edsclr in range(edsclr_dim):
        err_info = check_nan(
            edsclrm_forcing[:, :, edsclr], "edsclrm_forcing", operation, err_info,
        )
        err_info = check_nan(
            wpedsclrp_sfc[:, edsclr], "wpedsclrp_sfc", operation, err_info,
        )
        err_info = check_nan(edsclrm[:, :, edsclr], "edsclrm", operation, err_info)

    #---------------------------------------------------------------------

    nan_fatal = err_info.any_fatal()

    err_info = check_negative(rtm, "rtm", operation, err_info)
    err_info = check_negative(p_in_Pa, "p_in_Pa", operation, err_info)
    err_info = check_negative(rho, "rho", operation, err_info)
    err_info = check_negative(rho_zm, "rho_zm", operation, err_info)
    err_info = check_negative(exner, "exner", operation, err_info)
    err_info = check_negative(rho_ds_zm, "rho_ds_zm", operation, err_info)
    err_info = check_negative(rho_ds_zt, "rho_ds_zt", operation, err_info)
    err_info = check_negative(
        invrs_rho_ds_zm, "invrs_rho_ds_zm", operation, err_info,
    )
    err_info = check_negative(
        invrs_rho_ds_zt, "invrs_rho_ds_zt", operation, err_info,
    )
    err_info = check_negative(thv_ds_zm, "thv_ds_zm", operation, err_info)
    err_info = check_negative(thv_ds_zt, "thv_ds_zt", operation, err_info)
    err_info = check_negative(up2, "up2", operation, err_info)
    err_info = check_negative(vp2, "vp2", operation, err_info)
    err_info = check_negative(wp2, "wp2", operation, err_info)
    err_info = check_negative(thlm, "thlm", operation, err_info)
    err_info = check_negative(rtp2, "rtp2", operation, err_info)
    err_info = check_negative(thlp2, "thlp2", operation, err_info)

    if prefix == "beginning of ":
        reset = err_info.reset_code()
        reset_mask = err_info.any_fatal() & jnp.logical_not(nan_fatal)
        err_info = err_info.replace(
            err_code=jnp.where(
                reset_mask,
                reset.err_code_or_default(),
                err_info.err_code_or_default(),
            ),
            reason_code=jnp.where(
                reset_mask,
                reset.reason_code_or_default(),
                err_info.reason_code_or_default(),
            ),
        )

    # Check the first levels for temperatures greater than 200K
    # The Fortran source only writes this diagnostic and does not update
    # err_info, so there is no JAX state update here.

    return err_info


def check_negative(var, varname, operation, err_info: ErrInfo):
    """Checks for negative values in the var array."""
    del varname, operation
    if not hasattr(err_info, "set_fatal"):
        if np.any(np.asarray(var) < 0.0):
            err_info[...] = CLUBB_FATAL_ERROR
        return err_info
    return err_info.set_fatal(mask=jnp.any(jnp.asarray(var) < 0.0))


def check_nan(var, varname, operation, err_info: ErrInfo):
    """Checks for a non-finite value in var."""
    del varname, operation
    if not hasattr(err_info, "set_fatal"):
        if np.any(~np.isfinite(np.asarray(var))):
            err_info[...] = CLUBB_FATAL_ERROR
        return err_info
    return err_info.set_fatal(
        mask=jnp.any(jnp.logical_not(jnp.isfinite(jnp.asarray(var))))
    )


def calculate_spurious_source(
    integral_after,
    integral_before,
    flux_top,
    flux_sfc,
    integral_forcing,
    dt,
):
    """Return the column-conservation imbalance diagnostic."""
    return (
        (jnp.asarray(integral_after) - jnp.asarray(integral_before))
        / jnp.asarray(dt)
        + jnp.asarray(flux_top)
        - jnp.asarray(flux_sfc)
        - jnp.asarray(integral_forcing)
    )


def _all_finite(*values):
    for value in values:
        if value is None:
            continue
        if not np.all(np.isfinite(np.asarray(value))):
            return False
    return True


def sfc_varnce_check(
    sclr_dim,
    wp2_sfc,
    up2_sfc,
    vp2_sfc,
    thlp2_sfc,
    rtp2_sfc,
    rtpthlp_sfc,
    sclrp2_sfc=None,
    sclrprtp_sfc=None,
    sclrpthlp_sfc=None,
):
    """Return True when all surface variance/covariance inputs are finite."""
    values = [wp2_sfc, up2_sfc, vp2_sfc, thlp2_sfc, rtp2_sfc, rtpthlp_sfc]
    if int(sclr_dim) > 0:
        values.extend([sclrp2_sfc, sclrprtp_sfc, sclrpthlp_sfc])
    return _all_finite(*values)


def length_check(Lscale, Lscale_up, Lscale_down):
    """Return True when all mixing-length arrays are finite."""
    return _all_finite(Lscale, Lscale_up, Lscale_down)


def pdf_closure_check(closure_fields, pdf_params, sclr_dim=0, sclr_fields=None):
    """Return True when PDF closure outputs and PDF parameters are finite."""
    values = []
    if isinstance(closure_fields, dict):
        values.extend(closure_fields.values())
    else:
        values.append(closure_fields)

    if hasattr(pdf_params, "_fields"):
        values.extend(getattr(pdf_params, field) for field in pdf_params._fields)
    else:
        values.append(pdf_params)

    if int(sclr_dim) > 0 and sclr_fields:
        values.extend(sclr_fields.values())
    return _all_finite(*values)


def rad_check(thlm, rcm, rtm, rim, cloud_frac, p_in_Pa, exner, rho_zm):
    """Return True when radiation inputs are finite and non-negative."""
    values = [thlm, rcm, rtm, rim, cloud_frac, p_in_Pa, exner, rho_zm]
    if not _all_finite(*values):
        return False
    if any(np.any(np.asarray(value) < 0.0) for value in values):
        return False
    return bool(np.all(np.asarray(rtm) - np.asarray(rcm) >= 0.0))


def invalid_model_arrays(**arrays):
    """Return True if any numeric model array contains a non-finite value."""
    for value in arrays.values():
        if value is None:
            continue
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if arr.dtype.kind not in "biufc":
            continue
        if not np.all(np.isfinite(arr)):
            return True
    return False


def check_clubb_settings(*, err_info, **kwargs):
    """Compatibility wrapper for driver setup validation.

    The full Fortran routine primarily reports configuration errors and warnings.  The
    JAX driver handles unsupported options separately, so this wrapper preserves the
    runtime contract by returning the provided ErrInfo unchanged.
    """
    del kwargs
    return err_info


__all__ = [
    "CLUBB_FATAL_ERROR",
    "parameterization_check",
    "check_negative",
    "check_nan",
    "calculate_spurious_source",
    "sfc_varnce_check",
    "length_check",
    "pdf_closure_check",
    "rad_check",
    "invalid_model_arrays",
    "check_clubb_settings",
]
