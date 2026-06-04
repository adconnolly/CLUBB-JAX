"""JAX implementations of hydrostatic pressure initialization.

Faithful port of:
  calc_pressure.F90:init_pressure
  hydrostatic_module.F90:hydrostatic
"""

import jax
import jax.numpy as jnp
import numpy as np

from clubb_jax.src.CLUBB_core.constants_clubb import Cp, grav, kappa, p0, Rd
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


def init_pressure_jax(thvm, p_sfc, gr):
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


def hydrostatic_jax(thvm, p_sfc, gr):
    """Compute hydrostatic pressure, Exner, and density.

    JAX port of hydrostatic_module.F90:hydrostatic.

    Args:
        thvm:   (ngrdcol, nzt) virtual potential temperature [K]
        p_sfc:  (ngrdcol,) surface pressure [Pa]
        gr:     grid object

    Returns:
        p_in_Pa:    (ngrdcol, nzt) pressure [Pa]
        p_in_Pa_zm: (ngrdcol, nzm) pressure [Pa]
        exner:      (ngrdcol, nzt) Exner function [-]
        exner_zm:   (ngrdcol, nzm) Exner function [-]
        rho:        (ngrdcol, nzt) air density [kg/m^3]
        rho_zm:     (ngrdcol, nzm) air density [kg/m^3]
    """
    p_in_Pa, exner_zt, p_in_Pa_zm, exner_zm = init_pressure_jax(thvm, p_sfc, gr)

    # Interpolate thvm to momentum levels (no floor, as in Fortran hydrostatic)
    thvm_zm = zt2zm_jax(thvm, gr)

    rho    = p_in_Pa    / (Rd * thvm    * exner_zt)
    rho_zm = p_in_Pa_zm / (Rd * thvm_zm * exner_zm)

    return p_in_Pa, p_in_Pa_zm, exner_zt, exner_zm, rho, rho_zm


_CP_OV_G = Cp / grav
_THVM_TOL = 1e-12   # relative threshold for the thvm(k)=thvm(k-1) degenerate branch


def calc_ref_z_linear_thvm(thvm, exner):
    """Reference altitudes (relative to the lowest level) of a sounding from thvm + exner.

    JAX port of hydrostatic_module.F90:calc_ref_z_linear_thvm — integrates dz = -(Cp/g) thvm d(exner) assuming
    thvm varies LINEARLY IN Z between levels, which yields the LOG-MEAN of thvm:
      ref_z[k] = ref_z[k-1] - (Cp/g)(exner[k]-exner[k-1]) (thvm[k]-thvm[k-1]) / log(thvm[k]/thvm[k-1]),
    reducing to -(Cp/g)(exner[k]-exner[k-1]) thvm[k] when thvm[k]==thvm[k-1]. Differentiable (a cumulative sum).
    """
    thvm = jnp.asarray(thvm); exner = jnp.asarray(exner)
    d_exner = exner[1:] - exner[:-1]
    dthvm = thvm[1:] - thvm[:-1]
    big = jnp.abs(dthvm) > _THVM_TOL * thvm[1:]
    safe_ratio = jnp.where(big, thvm[1:] / thvm[:-1], 2.0)          # 2.0 → log(2)!=0 keeps grad finite
    logmean = jnp.where(big, dthvm / jnp.log(safe_ratio), thvm[1:])
    dz = -_CP_OV_G * d_exner * logmean
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(dz)])


def calc_ref_z_sfc_linear_thvm(thvm_km1, thvm_k, z_km1, z_k, exner_km1, exner_sfc):
    """Altitude of the surface relative to sounding level k-1 (hydrostatic_module.F90:calc_ref_z_sfc_linear_thvm)."""
    dthvm = thvm_k - thvm_km1
    big = jnp.abs(dthvm) > _THVM_TOL * jnp.abs(thvm_k)
    slope = jnp.where(big, dthvm, 1.0) / (z_k - z_km1)             # guard /dthvm via the where below
    val_big = ((z_k - z_km1) / jnp.where(big, dthvm, 1.0)) * (
        thvm_km1 * jnp.exp(-_CP_OV_G * (exner_sfc - exner_km1) * slope)
        - thvm_km1 + slope * z_km1)
    val_eq = z_km1 - _CP_OV_G * (exner_sfc - exner_km1) * thvm_k
    return jnp.where(big, val_big, val_eq)


def inverse_hydrostatic(p_sfc, zm_init, thvm, exner):
    """Heights of a pressure-coordinate sounding's levels (hydrostatic_module.F90:inverse_hydrostatic).

    Given the surface pressure/altitude and the sounding's thvm + exner profiles (exner decreasing with level),
    integrate the inverse hydrostatic equation for the altitude of each sounding level. The exact INVERSE of the
    forward `init_pressure_jax` (same log-mean scheme), so z -> exner -> z round-trips exactly.

    Args:
        p_sfc:   surface pressure [Pa] (scalar).
        zm_init: surface altitude [m] (scalar).
        thvm:    virtual potential temperature at the sounding levels (nlevels,) [K].
        exner:   Exner function at the sounding levels (nlevels,), decreasing with level.
    Returns:
        z (nlevels,) [m].
    """
    thvm = jnp.asarray(thvm); exner = jnp.asarray(exner)
    ref_z_snd = calc_ref_z_linear_thvm(thvm, exner)
    exner_sfc = (p_sfc / p0) ** kappa
    e = np.asarray(exner)
    if float(exner_sfc) < e[-1]:
        raise ValueError("inverse_hydrostatic: the entire sounding is below the model surface.")
    # Locate the sounding interval bracketing the surface (exner decreasing → last level with exner >= exner_sfc).
    if float(exner_sfc) > e[0]:
        low, high = 0, 1                       # surface below the lowest sounding level (linear extension)
    else:
        low = int(np.nonzero(e >= float(exner_sfc))[0][-1])
        high = min(low + 1, len(e) - 1)
    ref_z_sfc = calc_ref_z_sfc_linear_thvm(thvm[low], thvm[high], ref_z_snd[low], ref_z_snd[high],
                                           exner[low], exner_sfc)
    z_snd_bottom = zm_init - ref_z_sfc
    return z_snd_bottom + ref_z_snd
