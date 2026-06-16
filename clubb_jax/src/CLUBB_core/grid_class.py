"""JAX implementations of selected ``grid_class.F90`` helpers.

Description:

Definition of a grid class and associated functions

The grid specification is as follows for an ASCENDING grid:

    +                ================== zm(nzm) =================== top
    |
    |
1/dzt(nzt)   +       ------------------ zt(nzt) -------------------
    |        |
    |        |
    +  1/dzm(nzm-1)  ================== zm(nzm-1) =================
             |
             |
             +       ------------------ zt(nzt-1) -----------------

                                          .
                                          .
                                          .
                                          .

                     ================== zm(k+1) ===================


                     ------------------ zt(k+1) -------------------


    +                ================== zm(k+1) ===================
    |
    |
1/dzt(k)     +       ------------------ zt(k) ---------------------
    |        |
    |        |
    +    1/dzm(k)    ================== zm(k) =====================
             |
             |
             +       ------------------ zt(k-1) -------------------


                     ================== zm(k-1) ===================


                     ------------------ zt(k-2) -------------------

                                          .
                                          .
                                          .
                                          .

             +       ------------------ zt(2) ---------------------
             |
             |
    +    1/dzm(2)    ================== zm(2) =====================
    |        |
    |        |
1/dzt(1)     +       ------------------ zt(1) ---------------------
    |
    |
    +                ================== zm(1) =====================  zm_init
                     //////////////////////////////////////////////  surface


The variable zm(k) stands for the momentum level altitude at momentum
level k; the variable zt(k) stands for the thermodynamic level altitude at
thermodynamic level k; the variable invrs_dzt(k) is the inverse distance
between momentum levels (over a central thermodynamic level k); and the
variable invrs_dzm(k) is the inverse distance between thermodynamic levels
(over a central momentum level k).  Please note that in the above diagram,
"invrs_dzt" is denoted "dzt", and "invrs_dzm" is denoted "dzm", such that
1/dzt is the distance between successive momentum levels k and k+1 (over a
central thermodynamic level k), and 1/dzm is the distance between successive
thermodynamic levels k-1 and k (over a central momentum level k).

The grid setup is compatible with a stretched (unevely-spaced) grid.  Thus,
the distance between successive grid levels may not always be constant.

NOTE:  Any future code written for use in the CLUBB parameterization should
       use interpolation formulas consistent with a stretched grid.  The
       simplest way to do so is to call the appropriate interpolation
       function from this module.  Interpolations should *not* be handled in
       the form of:  ( var_zm(k+1) + var_zm(k) ) / 2; *nor* in the form of:
       0.5*( var_zt(k) + var_zt(k-1) ).

References:

https://arxiv.org/pdf/1711.03675v1.pdf#nameddest=url:clubb_grid

Section 3c, p. 3548 /Numerical discretization/ of:
 ``A PDF-Based Model for Boundary Layer Clouds. Part I:
   Method and Model Description'' Golaz, et al. (2002)
   JAS, Vol. 59, pp. 3540--3551.

Porting deviations:
  * The JAX ``Grid`` data type and grid constructor are implemented in
    ``clubb_jax.src.derived_types.grid_class``.  This file keeps only the
    CLUBB_core operator subset.
  * Fortran generic interfaces include scalar/1D/2D overloads and optional
    cubic interpolation through ``l_cubic_interp``.  JAX implements the active
    2D linear operator path used by the core.
  * Fortran band selector constants are 1-based; JAX uses Python 0-based
    ``T_ABOVE``, ``T_BELOW``, ``M_ABOVE``, and ``M_BELOW``.
  * Fortran mutates output arrays.  JAX returns arrays.
"""

from __future__ import annotations

import jax.numpy as jnp

T_ABOVE = 0
T_BELOW = 1
M_ABOVE = 0
M_BELOW = 1


def zt2zm(nzm: int, nzt: int, ngrdcol: int, gr, azt, zm_min=None):
    """Function to interpolate a variable located on the thermodynamic grid
    levels (azt) to the momentum grid levels (azm).  This function inputs the
    entire azt array and outputs the results as an azm array.  The
    formulation used is compatible with a stretched (unevenly-spaced) grid.
    """
    azt = jnp.asarray(azt, dtype=jnp.float64)

    # Interpolate the value of a thermodynamic-level variable to the central
    # momentum level, k, between two successive thermodynamic levels using
    # linear interpolation.
    interior = (
        gr.weights_zt2zm[:, 1:nzm - 1, T_ABOVE] * azt[:, 1:nzt]
        + gr.weights_zt2zm[:, 1:nzm - 1, T_BELOW] * azt[:, :nzt - 1]
    )

    # Set the value of the thermodynamic-level variable, azt, at momentum
    # level 1.  The name of the variable when interpolated/extended to momentum
    # levels is azm.  This is the lower boundary for an ascending grid and the
    # upper boundary for a descending grid.
    lower_ascending = azt[:, :1]
    upper_ascending = (
        gr.weights_zt2zm[:, nzm - 1:nzm, T_ABOVE] * azt[:, nzt - 1:nzt]
        + gr.weights_zt2zm[:, nzm - 1:nzm, T_BELOW] * azt[:, nzt - 2:nzt - 1]
    )
    # Use a linear extension based on the values of azt at levels 1 and 2 to
    # find the value of azm at level 1.
    lower_descending = (
        gr.weights_zt2zm[:, :1, T_ABOVE] * azt[:, 1:2]
        + gr.weights_zt2zm[:, :1, T_BELOW] * azt[:, :1]
    )
    upper_descending = azt[:, nzt - 1:nzt]

    is_ascending = gr.grid_dir_indx == 1
    lower = jnp.where(is_ascending, lower_ascending, lower_descending)
    upper = jnp.where(is_ascending, upper_ascending, upper_descending)

    azm = jnp.concatenate([lower, interior, upper], axis=1)
    if zm_min is not None:
        azm = jnp.maximum(azm, zm_min)
    return azm


def zm2zt(nzm: int, nzt: int, ngrdcol: int, gr, azm, zt_min=None):
    """Function to interpolate a variable located on the momentum grid levels
    (azm) to the thermodynamic grid levels (azt).  This function inputs the
    entire azm array and outputs the results as an azt array.  The formulation
    used is compatible with a stretched (unevenly-spaced) grid.
    """
    azm = jnp.asarray(azm, dtype=jnp.float64)
    # Interpolate the value of a momentum-level variable to the central
    # thermodynamic level, k, between two successive momentum levels using
    # linear interpolation.
    azt = (
        gr.weights_zm2zt[:, :, M_ABOVE] * azm[:, 1:nzm]
        + gr.weights_zm2zt[:, :, M_BELOW] * azm[:, :nzt]
    )
    if zt_min is not None:
        azt = jnp.maximum(azt, zt_min)
    return azt


def zt2zm2zt(nzm: int, nzt: int, ngrdcol: int, gr, azt, zt_min=None):
    """Function to interpolate a variable located on the thermodynamic grid
    levels (azt) to the momentum grid levels (azm), then interpolate back
    to thermodynamic grid levels (azt).

    Note:
      This is intended for smoothing variables.
    """
    # Interpolate azt to momentum levels
    # Interpolate back to thermodynamic levels
    return zm2zt(nzm, nzt, ngrdcol, gr, zt2zm(nzm, nzt, ngrdcol, gr, azt), zt_min)


def zm2zt2zm(nzm: int, nzt: int, ngrdcol: int, gr, azm, zm_min=None):
    """Function to interpolate a variable located on the momentum grid
    levels(azm) to thermodynamic grid levels (azt), then interpolate
    back to momentum grid levels (azm).

    Note:
      This is intended for smoothing variables.
    """
    # Interpolate azt to termodynamic levels
    # Interpolate back to momentum levels
    return zt2zm(nzm, nzt, ngrdcol, gr, zm2zt(nzm, nzt, ngrdcol, gr, azm), zm_min)


def ddzm(nzm: int, nzt: int, ngrdcol: int, gr, azm):
    """2D version of gradzm."""
    azm = jnp.asarray(azm, dtype=jnp.float64)
    # Vertical derivative of azm (thermo. levs.) [units vary / m]
    return (azm[:, 1:nzm] - azm[:, :nzt]) * gr.invrs_dzt


def ddzt(nzm: int, nzt: int, ngrdcol: int, gr, azt):
    """2D version of gradzt."""
    azt = jnp.asarray(azt, dtype=jnp.float64)
    # Vertical derivative of azt (mom.levs.) [units vary / m]
    interior = (azt[:, 1:nzt] - azt[:, :nzt - 1]) * gr.invrs_dzm[:, 1:nzm - 1]
    return jnp.concatenate([interior[:, :1], interior, interior[:, -1:]], axis=1)


__all__ = [
    "zt2zm",
    "zm2zt",
    "zt2zm2zt",
    "zm2zt2zm",
    "ddzm",
    "ddzt",
]
