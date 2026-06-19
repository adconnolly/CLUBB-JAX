"""Legacy aggregate constants for the JAX port.

Description:
  Contains frequently occuring model constants

References:
  None

Porting deviations:
- This module predates `constants_clubb.py` and remains as a compatibility
  aggregate for older `CLUBB_core` imports. It mirrors selected constants from
  `constants_clubb.F90`, re-exports `parameter_indices.F90`, and keeps selected
  `model_flags.F90` constants used by older modules.
- Constants whose corresponding JAX code is absent are intentionally omitted.
- Several local Fortran constants from `advance_clubb_core_module.F90`,
  `advance_helper_module.F90`, `clip_explicit.F90`, and
  `sfc_varnce_module.F90` are centralized here because multiple JAX modules use
  them.
- CAM preprocessor branches and Fortran file-unit I/O behavior are not
  reproduced; `fstderr` is kept only as a numeric compatibility constant.
"""

import numpy as np
import math

from clubb_jax.src.CLUBB_core.parameter_indices import *  # noqa: F403

# -----------------------------------------------------------------------------
# Numerical/Arbitrary Constants
# -----------------------------------------------------------------------------

# Number of neighboring points to draw from in the hole filling algorithm
num_hf_draw_points = 2

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
four = 4.0
five = 5.0
one_hundred = 100.0
three_halves = 1.5
one_half = 0.5
one_fourth = 0.25
one_third = 1.0 / 3.0
two_thirds = 2.0 / 3.0
four_thirds = 4.0 / 3.0

# -----------------------------------------------------------------------------
# Physical constants
# -----------------------------------------------------------------------------
Cp = 1004.67
Lv = 2.5e6    # Latent heat of vaporization         [J/kg]
Rd = 287.04   # Dry air gas constant                [J/kg/K]
Rv = 461.5    # Water vapor gas constant            [J/kg/K]

# Useful combinations of Rd and Rv
ep = Rd / Rv
ep1 = (1.0 - ep) / ep
ep2 = 1.0 / ep
kappa = Rd / Cp
p0 = 1.0e5
T_freeze_K = 273.15
grav = 9.81

# Von Karman's constant
# Constant of the logarithmic wind profile in the surface layer
vonk = 0.4
rho_lw = 1000.0
rho_ice = 917.0

# Tolerances below which we consider moments to be zero
w_tol = 2.0e-2
thl_tol = 1.0e-2
rt_tol = 1.0e-8

# Tolerances for use by the monatonic flux limiter.
# rt_tol_mfl is larger than rt_tol. rt_tol is extremely small
# (1e-8) to prevent spurious cloud formation aloft in LBA.
# rt_tol_mfl is larger (1e-4) to prevent the mfl from
# depositing moisture at the top of the domain.
thl_tol_mfl = 0.2
rt_tol_mfl = 1.0e-4

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
rc_tol = 1.0e-6
Nc_tol = 1.0e2
Ncn_tol = 1.0e2

mvr_cloud_max = 1.6e-5
Nc_in_cloud_min = 2.0e4

# Precipitating hydrometeor tolerances for mixing ratios.
rr_tol = 1.0e-10
ri_tol = 1.0e-10
rs_tol = 1.0e-10
rg_tol = 1.0e-10

# Maximum allowable values for the average mean volume radius of the various
# hydrometeor species.
mvr_rain_max = 5.0e-3
mvr_ice_max = 1.3e-4
mvr_snow_max = 1.0e-2
mvr_graupel_max = 2.0e-2

# Precipitating hydrometeor tolerances for concentrations.
# Tolerance value for N_r [#/kg]
Nr_tol = rr_tol / (four_thirds * math.pi * rho_lw * mvr_rain_max ** 3)

# Tolerance value for N_i [#/kg]
Ni_tol = ri_tol / (four_thirds * math.pi * rho_ice * mvr_ice_max ** 3)

# Tolerance value for N_s [#/kg]
Ns_tol = rs_tol / (four_thirds * math.pi * rho_ice * mvr_snow_max ** 3)

# Tolerance value for N_s [#/kg]
Ng_tol = rg_tol / (four_thirds * math.pi * rho_ice * mvr_graupel_max ** 3)

# Minimum value for em (turbulence kinetic energy)
# If anisotropic TKE is enabled, em = (1/2) * ( up2 + vp2 + wp2 );
# otherwise, em = (3/2) * wp2.  Since up2, vp2, and wp2 all have
# the same minimum threshold value of w_tol_sqd, em cannot be less
# than (3/2) * w_tol_sqd.  Thus, em_min = (3/2) * w_tol_sqd.
em_min = 1.5 * w_tol_sqd   # [m^2/s^2]

eps = max(1.0e-10, np.finfo(np.float64).eps)
zero_threshold = 0.0
unused_var = -999.0
min_max_smth_mag = 1.0e-9
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

# -----------------------------------------------------------------------------
# Model flag constants  (model_flags.F90)
# -----------------------------------------------------------------------------
# Options for the placement of the call to CLUBB's PDF.
ipdf_pre_advance_fields = 1       # Call before advancing predictive fields
ipdf_post_advance_fields = 2      # Call after advancing predictive fields
ipdf_pre_post_advance_fields = 3  # Call both before and after advancing
                                  # predictive fields

# Options for the two component normal (double Gaussian) PDF type to use for
# the w, rt, and theta-l (or w, chi, and eta) portion of CLUBB's multivariate,
# two-component PDF.
iiPDF_ADG1 = 1        # ADG1 PDF
iiPDF_ADG2 = 2        # ADG2 PDF
iiPDF_3D_Luhar = 3    # 3D Luhar PDF
iiPDF_new = 4         # new PDF
iiPDF_TSDADG = 5      # new TSDADG PDF
iiPDF_LY93 = 6        # Lewellen and Yoh (1993)
iiPDF_new_hybrid = 7  # new hybrid PDF

lapack = 1          # Use lapack library for matrix solves
penta_lu = 2        # Use penta_lu solver for 5 banded matrices
tridiag_lu = 2      # Use tridiag_lu solver for 3 banded matrices
penta_bicgstab = 3  # Use bicgstab to solve 5 banded matrices

l_gamma_Skw = True  # Use a Skw dependent gamma parameter

# Flag to advance xp3 using a simplified version of the d(xp3)/dt predictive
# equation or calculate it using a steady-state approximation.  When the flag
# is turned off, the Larson and Golaz (2005) ansatz to calculate xp3 after
# calculating Skx using the ansatz.
l_advance_xp3 = False

# Flag to use explicit turbulent advection in the xp2 and xpyp predictive
# equations.
l_explicit_turbulent_adv_xpyp = False
# Flag to use explicit turbulent advection in the wpxp predictive equation.
l_explicit_turbulent_adv_wpxp = False
# Flag to use explicit turbulent advection in the wp3 predictive equation.
l_explicit_turbulent_adv_wp3 = False

l_pos_def = False        # Flux limiting positive definite scheme on rtm
l_clip_turb_adv = False  # Corrects thlm/rtm when w'th_l'/w'r_t' is clipped

# Forces our matrices to be solved in descending mode, useful for the grid_generalization test
l_force_descending_solves = False
l_upwind_Kh_dp_term = False

# Options to set which algorithm the fill_holes routine uses to correct below threshold values
# in field solutions. The fill_holes method attempts to fill in a mass preserving way, in hopes
# of avoiding the need to perform blunt clipping, which can cause surious sources/sinks.
# An important consideration with these method is the locality - moving mass from one grid level
# to a far away one can create unintended non-local effects, so most methods attempt
# to fill with some degree of locality before relying on a global fill.
global_fill = 1          # Fast but minimally local, most methods use this as a last resort.
sliding_window = 2       # Attempt a highly local fill with a sliding window technique,
                         # and falls back to global if local pass failed.
widening_windows = 3     # Attempt to fill within fixed windows of a certain size, then
                         # repeat with increasaingly larger window sizes until all holes
                         # are filled. Window size can increase to entire domain, which
                         # is equivalent to a global fill.
smart_window = 4         # Uses lightweight hueristics to determine ranges to fill in one
                         # pass. This is highly local when possible, maintains some
                         # locality when wide hole ranges are encountered (if possible),
                         # and range can be the whole domain, which is equivalent to
                         # a global fallback. The gauranteed "one pass" feature seems
                         # to cause this to be the fastest method overall, at least with
                         # the current common hole patterns observed in CLUBB.
smart_window_smooth = 5  # Same as smart window, but contains fancy smoothing features.
                         # This could fail if the field is average above (but close to)
                         # threshold. The efficacy of the smoothing features is (currently)
                         # untested, and this is about 25% slower than smart_window without
                         # the smoothing, and has no global fallack (we could add one),
                         # but the smoothing could matter in theory, and looks nice.
parallel_fill = 6        # A parallelizable method that limits the mass each hole can
                         # steal, then considers each hole independently.
                         # Despite the parallizability being an attractive GPU
                         # feature, current timing results suggest this is the slowest
                         # method on a GPU (and CPU).

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

# Advance subroutine ordering variables
order_xm_wpxp = 1
order_xp2_xpyp = 2
order_wp2_wp3 = 3
order_windm = 4

# These are the integer constants that represent the various saturation
# formulas. To add a new formula, add an additional constant here,
# add the logic to check the strings for the new formula in clubb_core and
# this module, and add logic in saturation to call the proper function--
# the control logic will be based on these named constants.
saturation_bolton = 1  # Constant for Bolton approximations of saturation
saturation_gfdl = 2    # Constant for the GFDL approximation of saturation
saturation_flatau = 3  # Constant for Flatau approximations of saturation
saturation_lookup = 4  # Use a lookup table for mixing length
                       # saturation vapor pressure calculations

# Local constants used in advance_clubb_core
tau_const = 1000.0

ufmin = 0.01  # minimum value of friction velocity     [m/s]

# Setting l_use_invrs_tau_N2_iso = true will not change anything unless
l_use_invrs_tau_N2_iso = False

# whether to apply smooth min and max function
l_smooth_min_max = False
smth_type = 2  # Lscale_width_vert_avg smoothing type
below_grnd_val = 0.01  # Below-ground value for vertical averaging [s^-2]
