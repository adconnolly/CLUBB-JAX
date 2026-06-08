"""CLUBB constants and parameter indices for the Python port.

Constants mirror src/CLUBB_core/constants_clubb.F90.
Parameter indices mirror src/CLUBB_core/parameter_indices.F90.
Model flags mirror src/CLUBB_core/model_flags.F90.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Physical / numerical constants  (constants_clubb.F90)
# --------------------------------------------------------------------------- #
grav = 9.81
Cp = 1004.67
Lv = 2.5e6
Rd = 287.04
Rv = 461.5
ep = Rd / Rv
ep1 = (1.0 - ep) / ep
ep2 = 1.0 / ep
rc_tol = 1.0e-6        # Tolerance for r_c  [kg/kg]  (constants_clubb.F90)
Ncn_tol = 1.0e2        # Tolerance for N_cn [#/kg]   (constants_clubb.F90)
Nc_tol = 1.0e2         # Tolerance for N_c  [#/kg]   (constants_clubb.F90:279)
rr_tol = 1.0e-10       # Tolerance for r_r  [kg/kg]  (constants_clubb.F90:290)
parab_cyl_max_input = 49.0  # Largest allowable input to the parabolic-cylinder fn (constants_clubb.F90:63)
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
radians_per_deg = np.pi / 180.0
kappa = Rd / Cp
p0 = 1.0e5

vonk = 0.4       # von Karman constant
omega_planet = 7.292e-5   # Planetary rotation rate [s^-1] (constants_clubb.F90:233)

w_tol = 2.0e-2
w_tol_sqd = w_tol ** 2
thl_tol = 1.0e-2
rt_tol = 1.0e-8
thl_tol_mfl = 0.2      # [K]      monotonic-flux-limiter xm_tol (constants_clubb.F90:262)
rt_tol_mfl = 1.0e-4    # [kg/kg]  monotonic-flux-limiter xm_tol (constants_clubb.F90:263)
em_min = 1.5 * w_tol_sqd   # minimum TKE

zero = 0.0
one = 1.0
two = 2.0
three = 3.0
three_halves = 1.5
one_half = 0.5
two_thirds = 2.0 / 3.0
one_third = 1.0 / 3.0       # constants_clubb.F90:158

gamma_over_implicit_ts = 1.5
max_mag_correlation = 0.99
max_mag_correlation_flux = 0.99
wp2_max = 1000.0

eps = max(1.0e-10, np.finfo(np.float64).eps)
zero_threshold = 0.0
unused_var = -999.0
min_max_smth_mag = 1.0e-9
fstderr = 0   # Fortran stderr unit (unused in Python)

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
