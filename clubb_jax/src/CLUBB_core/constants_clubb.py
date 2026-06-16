"""JAX compatibility mirror of `src/CLUBB_core/constants_clubb.F90`.

Description:
  Contains frequently occuring model constants

References:
  None

Porting deviations:
- This module is the compatibility import surface for newer JAX code and tests.
  Older `CLUBB_core` modules still import many constants from
  `clubb_constants.py`, which is audited separately.
- Only constants needed by the JAX port are mirrored here. Constants whose
  corresponding JAX code is absent are intentionally omitted.
- `parameter_indices.F90` and `model_flags.F90` constants are re-exported from
  their JAX home modules to preserve existing `constants_clubb` imports.
- CAM preprocessor branches and Fortran file-unit I/O behavior are not
  reproduced; `fstderr` is kept only as a numeric compatibility constant.
"""

import numpy as np

# -----------------------------------------------------------------------------
# Numerical/Arbitrary Constants
# -----------------------------------------------------------------------------

# Fortran file unit I/O constants
fstderr = 0   # Fortran stderr unit (unused in Python)

# The parameter parab_cyl_max_input is the largest magnitude that the input to
# the parabolic cylinder function is allowed to have.  When the value of the
# input to the parabolic cylinder function is too large in magnitude
# (depending on the order of the parabolic cylinder function), overflow
# occurs, and the output of the parabolic cylinder function is +/-Inf.  The
# parameter parab_cyl_max_input places a limit on the absolute value of the
# input to the parabolic cylinder function.  When the value of the potential
# input exceeds this parameter (usually due to a very large ratio of ith PDF
# component mean of x to ith PDF component standard deviation of x), the
# variable x is considered to be constant and a different version of the
# equation called.
#
# The largest allowable magnitude of the input to the parabolic cylinder
# function (before overflow occurs) is dependent on the order of parabolic
# cylinder function.  However, after a lot of testing, it was determined that
# an absolute value of 49 works well for an order of 12 or less.
parab_cyl_max_input = 49.0  # Largest allowable input to parab. cyl. fnct.

# "Over-implicit" weighted time step.
#
# The weight of the implicit portion of a term is controlled by the factor
# gamma_over_implicit_ts (abbreviated "gamma" in the expression below).  A
# factor is added to the right-hand side of the equation in order to balance a
# weight that is not equal to 1, such that:
#
#      -y(t) * [ gamma * X(t+1) + ( 1 - gamma ) * X(t) ] + RHS;
#
# where X is the variable that is being solved for in a predictive equation
# (such as w'^3, w'th_l', r_t'^2, etc), y(t) is the linearized portion of the
# term that gets treated implicitly, and RHS is the portion of the term that
# is always treated explicitly.  A weight of greater than 1 can be applied to
# make the term more numerically stable.
#
#    gamma_over_implicit_ts          Effect on term
#
#            0.0               Term becomes completely explicit
#
#            1.0               Standard implicit portion of the term;
#                              as it was without the weighting factor.
#
#            1.5               Strongly weighted implicit portion of the term;
#                              increased numerical stability.
#
#            2.0               More strongly weighted implicit portion of the
#                              term; increased numerical stability.
#
# Note:  The "over-implicit" weighted time step is only applied to terms that
#        tend to significantly decrease the amount of numerical stability for
#        variable X.
#        The "over-implicit" weighted time step is applied to the turbulent
#        advection term for the following variables:
#           w'^3 (also applied to the turbulent production term), found in
#           module advance_wp2_wp3_module;
#           w'r_t', w'th_l', and w'sclr', found in
#           module advance_xm_wpxp_module; and
#           r_t'^2, th_l'^2, r_t'th_l', u'^2, v'^2, sclr'^2, sclr'r_t',
#           and sclr'th_l', found in module advance_xp2_xpyp_module.
gamma_over_implicit_ts = 1.5

# -----------------------------------------------------------------------------
# Mathematical Constants
# -----------------------------------------------------------------------------
zero = 0.0
one = 1.0
two = 2.0
three = 3.0
three_halves = 1.5
one_half = 0.5
two_thirds = 2.0 / 3.0
one_third = 1.0 / 3.0
radians_per_deg = np.pi / 180.0

# -----------------------------------------------------------------------------
# Physical constants
# -----------------------------------------------------------------------------
grav = 9.81
Cp = 1004.67
Lv = 2.5e6
Rd = 287.04
Rv = 461.5
ep = Rd / Rv
ep1 = (1.0 - ep) / ep
ep2 = 1.0 / ep
micron_per_m = 1.0e6   # Micrometers per meter        (constants_clubb.F90)
rho_lw = 1000.0        # Density of liquid water [kg/m^3] (constants_clubb.F90:241)
rho_ice = 917.0        # Density of ice          [kg/m^3] (constants_clubb.F90:246)

# Maximum allowable mean volume radii per hydrometeor species [m] (constants_clubb.F90:298-301)
mvr_rain_max    = 5.0e-3   # Max. avg. mean vol. rad. rain    [m]
mvr_ice_max     = 1.3e-4   # Max. avg. mean vol. rad. ice     [m]
mvr_snow_max    = 1.0e-2   # Max. avg. mean vol. rad. snow    [m]
mvr_graupel_max = 2.0e-2   # Max. avg. mean vol. rad. graupel [m]

cm3_per_m3 = 1.0e6     # Cubic centimeters per cubic meter (constants_clubb.F90:378)
g_per_kg = 1000.0      # Grams in a kilogram               (constants_clubb.F90:372)

sec_per_hr = 3600.0    # Seconds per hour
pascal_per_mb = 100.0  # Pascals per millibar              (constants_clubb.F90:375)
kappa = Rd / Cp
p0 = 1.0e5

vonk = 0.4       # von Karman constant
omega_planet = 7.292e-5   # Planetary rotation rate [s^-1] (constants_clubb.F90:233)

# Tolerances below which we consider moments to be zero
w_tol = 2.0e-2
thl_tol = 1.0e-2
rt_tol = 1.0e-8
thl_tol_mfl = 0.2      # [K]      monotonic-flux-limiter xm_tol (constants_clubb.F90:262)
rt_tol_mfl = 1.0e-4    # [kg/kg]  monotonic-flux-limiter xm_tol (constants_clubb.F90:263)

# The tolerance for w'^2 is the square of the tolerance for w.
w_tol_sqd = w_tol ** 2

# Set tolerances for Khairoutdinov and Kogan rain microphysics to insure
# against numerical errors.  The tolerance values for Nc, rr, and Nr insure
# against underflow errors in computing the PDF for l_kk_rain.  Basically,
# they insure that those values squared won't be less then 10^-38, which is
# the lowest number that can be numerically represented.  However, the
# tolerance value for rc doubles as the lowest mixing ratio there can be to
# still officially have a cloud at that level.  This is figured to be about
# 1.0_core_rknd x 10^-7 kg/kg.  Brian; February 10, 2007.
rc_tol = 1.0e-6        # Tolerance value for r_c  [kg/kg]
Ncn_tol = 1.0e2        # Tolerance value for N_cn [#/kg]
Nc_tol = 1.0e2         # Tolerance value for N_c  [#/kg]

# Precipitating hydrometeor tolerances for mixing ratios.
rr_tol = 1.0e-10       # Tolerance value for r_r [kg/kg]

em_min = 1.5 * w_tol_sqd   # minimum TKE

max_mag_correlation = 0.99
max_mag_correlation_flux = 0.99
wp2_max = 1000.0

eps = max(1.0e-10, np.finfo(np.float64).eps)
zero_threshold = 0.0
unused_var = -999.0
min_max_smth_mag = 1.0e-9

cloud_frac_min = 0.005

max_num_stdevs = 5.0         # Range of standard deviations for PDF truncation
chi_tol = max(1.0e-8, np.finfo(np.float64).eps)  # Tolerance for chi [kg/kg]
eta_tol = chi_tol            # Tolerance for eta [kg/kg] (constants_clubb.F90:254, = chi_tol)

stefan_boltzmann = 5.6704e-8  # Stefan-Boltzmann constant [W/(m^2 K^4)] (constants_clubb.F90:216)
T_freeze_K = 273.15          # Freezing point of water [K] (constants_clubb.F90:219)
Lf = 3.33e5                  # Latent heat of fusion      [J/kg] (constants_clubb.F90:210)
Ls = 2.834e6                 # Latent heat of sublimation [J/kg] (constants_clubb.F90:209)
cm_per_m = 100.0             # Centimeters per meter      (constants_clubb.F90:380)

import math as _math
sqrt_2   = _math.sqrt(2.0)
sqrt_2pi = _math.sqrt(2.0 * _math.pi)

# --------------------------------------------------------------------------- #
# Parameter indices  (parameter_indices.F90) — relocated to their Fortran home parameter_indices.py (iter 607) and
# re-exported here so `from constants_clubb import iC1/ic_K1/igamma_coef/…` keeps working. parameter_indices.py is a
# pure-constants leaf (no imports) → no circular dependency.
# --------------------------------------------------------------------------- #
from clubb_jax.src.CLUBB_core.parameter_indices import *  # noqa: F401,F403,E402  (re-export of the i<name> indices)

# --------------------------------------------------------------------------- #
# Model flag constants  (model_flags.F90) — re-exported from their Fortran home model_flags.py (iter 606); this keeps
# `from constants_clubb import iiPDF_ADG1/order_*/ipdf_*` working while the canonical definitions live in the JAX mirror
# of model_flags.F90. model_flags.py imports only config_flags (a NamedTuple leaf) → no circular import.
# --------------------------------------------------------------------------- #
from clubb_jax.src.CLUBB_core.model_flags import (  # noqa: F401,E402  (re-export of model_flags.F90 enum parameters)
    ipdf_pre_advance_fields, ipdf_post_advance_fields, ipdf_pre_post_advance_fields,
    iiPDF_ADG1, l_gamma_Skw, l_advance_xp3,
    order_xm_wpxp, order_xp2_xpyp, order_wp2_wp3, order_windm,
)

# Local constants used in advance_clubb_core (subroutine-local in the Fortran; centralized here for the JAX modules
# that share them — verified genuinely used, iter 608). The Fortran's `smth_type` local was dropped: the JAX hardcodes
# the smth_type=2 Lscale_width_vert_avg path (advance_helper_module._SMTH_TYPE2_HALF_WIDTH), so the constant was unused.
tau_const = 1000.0
ufmin = 0.01
l_use_invrs_tau_N2_iso = False
l_smooth_min_max = False
below_grnd_val = 0.01
