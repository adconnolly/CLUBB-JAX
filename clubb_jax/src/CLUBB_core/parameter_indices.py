"""JAX mirror of CLUBB_core/parameter_indices.F90 — the 1-based indices into the tunable-parameter array.

parameter_indices.F90 is a routine-less Fortran module of `integer, parameter :: iC1 = 1, …` enumerating each
tunable parameter's column in `clubb_params(ngrdcol, nparams)`. These had lived in constants_clubb.py; relocated
here to their Fortran home (iter 607). constants_clubb.py re-exports them (`from parameter_indices import *`) so
`from constants_clubb import iC1/ic_K1/…` keeps working. Pure constants → no imports → no circular dependency.

NOTE: Fortran uses 1-based indices into clubb_params(ngrdcol, nparams); the Python clubb_params array uses the SAME
1-based convention (column 0 is unused / padding) so these constants are used directly.
"""

nparams = 102

ic_K = 37
ic_K1 = 38
ic_K2 = 40
ic_K6 = 42
ic_K8 = 44
ic_K9 = 46
ic_K10 = 74
ic_K10h = 75
ithlp2_rad_coef = 76   # parameter_indices.F90 line 121

iC1 = 1
iC1b = 2
iC1c = 3          # Slope of C1 skewness function (added in later Fortran)
iC2rt = 4          # C2 for rtp2 dissipation  (was 3 before iC1c was added)
iC2thl = 5         # C2 for thlp2 dissipation
iC2rtthl = 6       # C2 for rtpthlp dissipation (was 7)
iC4 = 7            # Return-to-isotropy in wp2 (was 8; Fortran replaced iC5 with iC_uu_*)
iC_uu_shr = 8      # Shear term in wp2 pressure (was called iC5 in older Python)
iC_uu_buoy = 9     # Buoyancy term in wp2 pressure
iC6rt = 10
iC6rtb = 11
iC6rtc = 12
iC6thl = 13
iC6thlb = 14
iC6thlc = 15
iC7 = 16
iC7b = 17
iC7c = 18          # Slope of C7 skewness function (was missing, shifted iC8+)
iC8 = 19           # (was 18)
iC8b = 20          # (was 19)
iC10 = 21          # (was 20)
iC11 = 22          # (was 21)
iC11b = 23         # (was 22)
iC11c = 24         # (was 23)
iC12 = 25          # (was 24)
iC13 = 26          # (was 25)
iC14 = 27          # (was 26)
iC_wp2_pr_dfsn = 28
iC_wp3_pr_tp   = 29
iC_wp3_pr_turb = 30
iC_wp3_pr_dfsn = 31
iC_wp2_splat = 32
iSkw_max_mag   = 79
iSkw_denom_coef = 73
imult_coef = 67
inu1   = 39
inu2   = 41
inu6   = 43
inu8   = 45
inu9   = 47
inu10  = 48
inu_hm = 52

igamma_coef = 57
igamma_coefb = 58
igamma_coefc = 59
imu = 60
ibeta = 61
ilmin_coef = 62
iomicron = 63
izeta_vrnce_rat = 64
ilambda0_stability_coef = 66

itaumin = 68
itaumax = 69

iup2_sfc_coef = 78

ixp3_coef_base = 90
ixp3_coef_slope = 91

iC_invrs_tau_bkgnd          = 80
iC_invrs_tau_sfc            = 81
iC_invrs_tau_shear          = 82
iC_invrs_tau_N2             = 83
iC_invrs_tau_N2_wp2         = 84
iC_invrs_tau_N2_xp2         = 85
iC_invrs_tau_N2_wpxp        = 86
iC_invrs_tau_N2_clear_wp3   = 87
iC_invrs_tau_wpxp_Ri        = 88
iC_invrs_tau_wpxp_N2_thresh = 89
ialtitude_threshold         = 92
iwpxp_Ri_exp                = 101
iz_displace                 = 102

iCx_min = 94
iCx_max = 95
iRichardson_num_min = 96
iRichardson_num_max = 97
ia3_coef_min = 98
ia_const = 99
ibv_efold = 100

iLscale_mu_coef   = 70
iLscale_pert_coef = 71
