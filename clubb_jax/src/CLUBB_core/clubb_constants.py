"""CLUBB constants for the Python port.

Constants mirror src/CLUBB_core/constants_clubb.F90.
Parameter indices are defined in parameter_indices.py and re-exported here for
existing imports.
Model flags mirror src/CLUBB_core/model_flags.F90.
"""

import numpy as np
import math

from clubb_jax.src.CLUBB_core.parameter_indices import *  # noqa: F403

# --------------------------------------------------------------------------- #
# Physical / numerical constants  (constants_clubb.F90)
# --------------------------------------------------------------------------- #
grav = 9.81
vonk = 0.4
rho_lw = 1000.0
rho_ice = 917.0
Cp = 1004.67
Lv = 2.5e6
Rd = 287.04
Rv = 461.5
ep = Rd / Rv
ep1 = (1.0 - ep) / ep
ep2 = 1.0 / ep
kappa = Rd / Cp
p0 = 1.0e5
T_freeze_K = 273.15

w_tol = 2.0e-2
w_tol_sqd = w_tol ** 2
thl_tol = 1.0e-2
rt_tol = 1.0e-8
thl_tol_mfl = 0.2
rt_tol_mfl = 1.0e-4
rc_tol = 1.0e-6
Nc_tol = 1.0e2
Ncn_tol = 1.0e2
em_min = 1.5 * w_tol_sqd   # minimum TKE
mvr_cloud_max = 1.6e-5
Nc_in_cloud_min = 2.0e4
rr_tol = 1.0e-10
ri_tol = 1.0e-10
rs_tol = 1.0e-10
rg_tol = 1.0e-10
mvr_rain_max = 5.0e-3
mvr_ice_max = 1.3e-4
mvr_snow_max = 1.0e-2
mvr_graupel_max = 2.0e-2

zero = 0.0
one = 1.0
two = 2.0
three = 3.0
four = 4.0
five = 5.0
one_hundred = 100.0
three_halves = 1.5
one_half = 0.5
one_fourth = 0.25
one_third = 1.0 / 3.0
two_thirds = 2.0 / 3.0
four_thirds = 4.0 / 3.0
gamma_over_implicit_ts = 1.5
Nr_tol = rr_tol / (four_thirds * math.pi * rho_lw * mvr_rain_max ** 3)
Ni_tol = ri_tol / (four_thirds * math.pi * rho_ice * mvr_ice_max ** 3)
Ns_tol = rs_tol / (four_thirds * math.pi * rho_ice * mvr_snow_max ** 3)
Ng_tol = rg_tol / (four_thirds * math.pi * rho_ice * mvr_graupel_max ** 3)

eps = max(1.0e-10, np.finfo(np.float64).eps)
zero_threshold = 0.0
unused_var = -999.0
min_max_smth_mag = 1.0e-9
num_hf_draw_points = 2
fstderr = 0   # Fortran stderr unit (unused in Python)

cloud_frac_min = 0.005
max_mag_correlation = 0.99
max_mag_correlation_flux = 0.99
wp2_max = 1000.0
max_num_stdevs = 5.0
chi_tol = max(1.0e-8, np.finfo(np.float64).eps)
eta_tol = chi_tol
sqrt_2 = math.sqrt(2.0)
sqrt_2pi = math.sqrt(2.0 * math.pi)

# --------------------------------------------------------------------------- #
# Model flag constants  (model_flags.F90)
# --------------------------------------------------------------------------- #
ipdf_pre_advance_fields = 1
ipdf_post_advance_fields = 2
ipdf_pre_post_advance_fields = 3

iiPDF_ADG1 = 1
iiPDF_ADG2 = 2
iiPDF_3D_Luhar = 3
iiPDF_new = 4
iiPDF_TSDADG = 5
iiPDF_LY93 = 6
iiPDF_new_hybrid = 7

lapack = 1
penta_lu = 2
tridiag_lu = 2
penta_bicgstab = 3

l_gamma_Skw = True       # Use Skw-dependent gamma parameter
l_advance_xp3 = False    # Use predictive xp3 equation
l_explicit_turbulent_adv_xpyp = False
l_explicit_turbulent_adv_wpxp = False
l_explicit_turbulent_adv_wp3 = False
l_pos_def = False
l_clip_turb_adv = False
l_force_descending_solves = False
l_upwind_Kh_dp_term = False

global_fill = 1
sliding_window = 2
widening_windows = 3
smart_window = 4
smart_window_smooth = 5
parallel_fill = 6

clip_wprtp = 8
clip_wpthlp = 9
clip_upwp = 10
clip_vpwp = 11
clip_wp2 = 12
clip_wpsclrp = 13

wprtp_cl_max = 3
wpthlp_cl_max = 3
upwp_cl_max = 3
vpwp_cl_max = 3

# Default advance ordering
order_xm_wpxp = 1
order_xp2_xpyp = 2
order_wp2_wp3 = 3
order_windm = 4

saturation_bolton = 1
saturation_gfdl = 2
saturation_flatau = 3
saturation_lookup = 4

# Local constants used in advance_clubb_core
tau_const = 1000.0
ufmin = 0.01
l_use_invrs_tau_N2_iso = False
l_smooth_min_max = False
smth_type = 2
below_grnd_val = 0.01
