"""JAX implementation of hydrostatic pressure initialization — calc_pressure.F90:init_pressure.

The `hydrostatic` wrapper + the inverse-hydrostatic / calc_ref_z sounding-altitude routines live in
Input_fields/hydrostatic_module.py (hydrostatic_module.F90); they `use` init_pressure from here.
"""

import jax
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import Cp, grav, kappa, p0, Lv, ep1, ep2
from clubb_jax.src.CLUBB_core.grid_class import zt2zm_jax

_G_OV_CP = grav / Cp


def _exner_step(exner_prev, inputs):
    """Single step of the hydrostatic Exner integration (sequential scan body)."""
    thvm_k, thvm_km1, dz_k = inputs
    delta = thvm_k - thvm_km1
    eps = jnp.finfo(jnp.float64).eps * thvm_k
    exner_k = jnp.where(
        jnp.abs(delta) > eps,
        exner_prev - _G_OV_CP * dz_k / delta * jnp.log(thvm_k / thvm_km1),
        exner_prev - _G_OV_CP * dz_k / thvm_k,
    )
    return exner_k, exner_k


def init_pressure(thvm, p_sfc, gr):
    """Compute hydrostatic pressure and Exner on zt and zm grids.

    JAX port of calc_pressure.F90:init_pressure.

    Integrates d(exner)/dz = -g/(Cp * thvm) upward from the surface.
    Uses the analytic integral with linear thvm assumption:
      exner[k] = exner[k-1] - (g/Cp)*(z[k]-z[k-1])/(thvm[k]-thvm[k-1])*log(thvm[k]/thvm[k-1])
    (or the constant-thvm limit when thvm[k] ≈ thvm[k-1]).

    Args:
        thvm:   (ngrdcol, nzt) virtual potential temperature [K]
        p_sfc:  (ngrdcol,) surface pressure [Pa]
        gr:     grid object with .zt (ngrdcol, nzt) and .zm (ngrdcol, nzm)

    Returns:
        p_in_Pa:    (ngrdcol, nzt) pressure on thermodynamic levels [Pa]
        exner:      (ngrdcol, nzt) Exner function on thermodynamic levels [-]
        p_in_Pa_zm: (ngrdcol, nzm) pressure on momentum levels [Pa]
        exner_zm:   (ngrdcol, nzm) Exner function on momentum levels [-]
    """
    zt = jnp.asarray(gr.zt)  # (ngrdcol, nzt)
    zm = jnp.asarray(gr.zm)  # (ngrdcol, nzm)

    # Interpolate thvm to momentum levels (with zero_threshold=1e-300 floor as in Fortran)
    _zero_threshold = 1.0e-300
    thvm_zm = zt2zm_jax(thvm, gr, zm_min=_zero_threshold)  # (ngrdcol, nzm)

    # Surface momentum level: exner_zm[:, 0] = (p_sfc/p0)^kappa
    exner_zm_0 = (p_sfc / p0) ** kappa  # (ngrdcol,)

    # --- First zt level (Fortran k=1, Python k=0) ---
    # Step from momentum level 0 to thermodynamic level 0
    delta_0 = thvm[:, 0] - thvm_zm[:, 0]
    dz_0 = zt[:, 0] - zm[:, 0]
    eps_0 = jnp.finfo(jnp.float64).eps * thvm[:, 0]
    exner_0 = jnp.where(
        jnp.abs(delta_0) > eps_0,
        exner_zm_0 - _G_OV_CP * dz_0 / delta_0 * jnp.log(thvm[:, 0] / thvm_zm[:, 0]),
        exner_zm_0 - _G_OV_CP * dz_0 / thvm[:, 0],
    )  # (ngrdcol,)

    # --- Scan over zt levels k=2..nzt (Fortran) = k=1..nzt-1 (Python) ---
    thvm_k   = jnp.moveaxis(thvm[:, 1:], 1, 0)           # (nzt-1, ngrdcol)
    thvm_km1 = jnp.moveaxis(thvm[:, :-1], 1, 0)           # (nzt-1, ngrdcol)
    dz_zt    = jnp.moveaxis(zt[:, 1:] - zt[:, :-1], 1, 0) # (nzt-1, ngrdcol)

    _exner_last, exner_rest = jax.lax.scan(
        _exner_step, exner_0, (thvm_k, thvm_km1, dz_zt),
    )
    # exner_rest: (nzt-1, ngrdcol) → (ngrdcol, nzt-1)
    exner_zt_rest = jnp.moveaxis(exner_rest, 0, 1)
    exner_zt = jnp.concatenate([exner_0[:, None], exner_zt_rest], axis=1)  # (ngrdcol, nzt)
    p_in_Pa = p0 * exner_zt ** (1.0 / kappa)

    # --- Momentum levels k=2..nzm (Fortran) = k=1..nzm-1 (Python) ---
    # exner_zm[k] = exner_zt[k-1] - G_OV_CP * (zm[k]-zt[k-1]) / ...
    # where k-1 is zt index 0..nzt-1 and k is zm index 1..nzm-1
    thvm_zm_k   = thvm_zm[:, 1:]    # (ngrdcol, nzt): zm levels k=1..nzm-1
    thvm_km1_zm = thvm              # (ngrdcol, nzt): zt levels k-1=0..nzt-1
    dz_zm = zm[:, 1:] - zt          # (ngrdcol, nzt): zm[k] - zt[k-1]

    delta_zm = thvm_zm_k - thvm_km1_zm
    eps_zm = jnp.finfo(jnp.float64).eps * thvm_km1_zm
    exner_zm_rest = jnp.where(
        jnp.abs(delta_zm) > eps_zm,
        exner_zt - _G_OV_CP * dz_zm / delta_zm * jnp.log(thvm_zm_k / thvm_km1_zm),
        exner_zt - _G_OV_CP * dz_zm / thvm_zm_k,
    )  # (ngrdcol, nzt)

    exner_zm = jnp.concatenate([exner_zm_0[:, None], exner_zm_rest], axis=1)  # (ngrdcol, nzm)
    p_in_Pa_zm = p0 * exner_zm ** (1.0 / kappa)

    return p_in_Pa, exner_zt, p_in_Pa_zm, exner_zm


def calculate_thvm(thlm, rtm, rcm, exner, thv_ds_zt):
    """Compute mean virtual potential temperature (calc_pressure.F90:calculate_thvm).

    thvm = thlm + ep1*thv_ds_zt*rtm + (Lv/(Cp*exner) - ep2*thv_ds_zt)*rcm
    """
    return thlm + ep1 * thv_ds_zt * rtm + (Lv / (Cp * exner) - ep2 * thv_ds_zt) * rcm
