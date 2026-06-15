# CLUBB Fortran → JAX translation status

A static map from each in-scope Fortran oracle (`clubb_release/src/**.{F90,F,f90}`) to its JAX
counterpart (`clubb_jax/src/**.py`). The naming mirror is 1:1 where possible
(`clubb_jax/src/CLUBB_core/x.py` ↔ `clubb_release/src/CLUBB_core/x.F90`), but some Fortran is **folded**
into a differently-named JAX module (benchmark cases dispatch through `prescribe_forcings.py`;
`stats_netcdf.F90` → `io/stats_writer.py`; BUGSrad LW/SW split across files). "Ported" means *translated into
JAX* — distinct from *bit-faithful*, which is a per-case property (see DESIGN.md).

**Legend:** ✅ 1:1 mirror · 🔁 ported but folded into a differently-named module · ◐ in-scope routines ported,
documented remainder folded/deferred · ➖ not a target (LAPACK/Numerical-Recipes libs, build/IO/test infra, SILHS
RNG, or an alternative scheme the gated config never selects) · ❌ genuinely unported physics.

## Status

- **The model runs 100% in JAX** — zero Fortran calls per timestep (machine-enforced by
  `test_no_dead_imports.py::test_src_has_no_fortran_runtime_import`). The Fortran remains only as the compiled
  comparison oracle and the porting reference.
- **`python clubb_jax/run_scripts/mirror_audit.py` reports PASS** (MISSING=0, CASING=0, MISPLACED=0,
  UNMIRRORED_FILES=0, MISPLACED_FILES=0, REDUNDANT_TOL=0, JAX_ALIAS=0) at the routine **and** file-name levels across
  all three mirrored subsystems (CLUBB_core, Microphys, Radiation).
- **One DEFERRED routine:** `pdf_closure_driver_zm` (the zm-grid PDF closure, `pdf_closure_module.F90:4654`),
  gated by `l_call_pdf_closure_twice` which no case sets — so there is no validated case to port against and no f2py
  oracle exposes it. `clubb_driver` fail-loud rejects the flag, making the gap explicit and safe.
- **Faithfulness:** 20 cases are bit-faithful (`compare_cases.py` DEFAULT_CASES) and all 19+ cases are whole-driver
  `jax.grad`-differentiable (`compare_grad.py`). Remaining non-bit-faithful cases are FP-limited (chaotic cloud
  onset) or oracle-limited, not bugs.
- **Genuinely unported (❌):** no-oracle / impractical subsystems only — COAMPS microphysics, the GFDL
  `aer_ccn_act_wpdf_k` 5-D lookup, the SCM aerosol-activation subsystem, `pdf_hydromet_microphys_wrapper` (0 output
  for every gated case), and a few SILHS/COAMPS-blocked benchmark cases.
- ~100 named mirrors carry direct f2py-Fortran-oracle unit tests (bit-shadow on synthetic inputs; SKIP-clean when
  the oracle is unbuilt), proving behavioral — not just same-named — faithfulness.

**Audit accounting:** 296 Fortran source files are by-design scoped out of MISSING (lapack_blas=169,
coamps_microphys=47, g_unit_tests=15, numerical_recipes=13, silhs_sampling=13, io_readers=9, tuner_infra=7,
bugsrad_altsolver=6, scm_host_microphys=6, aerosol_activation=5, case_setups=3, state_api_types=3); everything in
scope is mirrored.

---

## `CLUBB_core/`  (✅49 · 🔁7 · ◐3 · ❌0 · ➖8) — **fully ported**

| Fortran | | JAX / note |
|---|:-:|---|
| `Nc_Ncn_eqns.F90` | ✅ | both directions (Nc↔Ncn); all 6 functions, f2py bit-match |
| `T_in_K_module.F90` | ✅ | `thlm2T_in_K` + inverse `T_in_K2thlm` (round-trip ~4e-16) |
| `adg1_adg2_3d_luhar_pdf.F90` | ✅ | all 2-component PDF drivers (ADG1/ADG2/Luhar 3D), f2py bit-match ≤1e-11 |
| `advance_clubb_core_module.F90` | ✅ | `advance_clubb_core` driver + `compute_diagnostic_cache` + `set_sfc_value_of_flux_profiles`; orchestration calls the extracted advance/pdf_closure modules |
| `advance_helper_module.F90` | ✅ | `smooth_min`/`calc_xpwp`/`pvertinterp`/`smooth_heaviside_peskin`/`vertical_avg`/`calc_wp3_on_wp2`/`calc_Ri_zm`/etc. (f2py bit-exact) |
| `advance_windm_edsclrm_module.F90` | ✅ | um/vm solve/rhs/lhs + `compute_uv_tndcy`. No-op for bit-gate cases (`l_predict_upwp_vpwp` default); `windm_edsclrm_implicit_stats` unported (no budget stats) |
| `advance_wp2_wp3_module.F90` | ✅ | `advance_wp2_wp3` driver + named `wp2_term_*`/`wp3_term_*` LHS/RHS builders + coupled `wp23_rhs`/`wp23_lhs`/`wp23_solve` |
| `advance_xm_wpxp_module.F90` | ✅ | whole `advance_xm_wpxp` driver + TA/LHS terms + solve + clipping + `diagnose_upxp` + sponge. `xm_correction_wpxp_cl` mirrored+tested but **not wired** (gated on `l_clip_turb_adv`, off in validated config — see memory `xm-correction-wpxp-cl-gated-off`) |
| `advance_xp3_module.F90` | ✅ | `compute_xp3`/`advance_xp3`/`advance_xp3_simplified` + terms. Non-ADG1 path gated off (documents an apparent Fortran `max(k+1,nzt)` typo the JAX avoids with `min`) |
| `calc_pressure.F90` | ✅ | `init_pressure` (lax.scan hydrostatic) + `calculate_thvm` (f2py 5.7e-14) |
| `clip_explicit.F90` | ✅ | `clip_covar`/`clip_variance`/`clip_skewness`(+core)/`clip_covars_denom`/`clip_rcm`, f2py bit-exact |
| `constants_clubb.F90` | ✅ | constants_clubb.py |
| `corr_varnce_module.F90` | ✅ | `def_corr_idx`/`get_corr_var_index`/`set_corr_arrays_to_default`/`print_corr_matrix` + KK prescribed correlations |
| `diffusion.F90` | ✅ | `diffusion_zt_lhs`/`diffusion_zm_lhs` |
| `fill_holes.F90` | ✅ | `fill_holes_vertical`/`_wp2_from_horz_tke`/`_global`/`_sliding_window` + hydromet clip |
| `grid_class.F90` | ✅ | `derived_types/grid_class.py` (Grid + `setup_grid` + zt↔zm weight builders + `flip`) and `CLUBB_core/grid_class.py` (zm2zt/zt2zm/ddzm/ddzt) |
| `matrix_solver_wrapper.F90` | ✅ | thin dispatch over the LU solver modules |
| `mixing_length.F90` | ✅ | `calc_Lscale` dispatcher + `compute_mixing_length`/`calc_Lscale_directly`/`diagnose_Lscale_from_tau`/`set_Lscale_max` |
| `model_flags.F90` | ✅ | `get_default_config_flags` (pure-JAX) |
| `mono_flux_limiter.F90` | ✅ | `monotonic_turbulent_flux_limit` (whole, f2py 2.49e-16) + mfl lhs/rhs/solve + `calc_turb_adv_range` + mean up/down w |
| `numerical_check.F90` | ✅ | all 7 validation checks + `check_nan`/`check_negative` + `calculate_spurious_source` (f2py 3.6e-15) |
| `parameters_tunable.F90` | ✅ | `init_clubb_params`/`calc_derrived_params`/`check_parameters`/`get_param_names` (102-param ordering single source) |
| `pdf_utilities.F90` | ✅ | `calc_comp_corrs_binormal`/`smooth_corr_quotient`/`compute_variance_binormal` (f2py ~4e-16) |
| `precipitation_fraction.F90` | ✅ | `precip_fraction` (case-validated; f2py oracle SIGFPEs) + component split + assert check |
| `saturation.F90` | ✅ | SVP dispatcher + flatau/bolton + `sat_mixrat_liq`/`_ice` + `rcm_sat_adj` (f2py 3.5e-17). gfdl/lookup formulas unported (non-default) |
| `setup_clubb_pdf_params.F90` | ✅ | comp mu/sigma + corr-array family + Cholesky + `compute_rtp2_from_chi` |
| `sfc_varnce_module.F90` | ✅ | sfc_varnce_module.py |
| `sigma_sqd_w_module.F90` | ✅ | `compute_sigma_sqd_w` (f2py bit-exact) |
| `sponge_layer_damping.F90` | ✅ | `sponge_damp_xm` wired; `_xp2`/`_xp3` ported+tested but not wired (no case enables; driver fail-loud rejects) |
| `Skx_module.F90` | ✅ | `Skx_func`/`compute_gamma_Skw`/`LG_2005_ansatz`/`xp3_LG_2005_ansatz` (f2py bit-match) |
| `advance_xp2_xpyp_module.F90` | ✅ | whole `advance_xp2_xpyp` driver (5 moments) + TA/dp1/pr terms + solve + shared budget-finalize kernels |
| `array_index.F90` | 🔁 | `derived_types/sclr_idx.py` + constants_clubb.py |
| `clubb_precision.F90` | 🔁 | x64 config |
| `err_info_type_module.F90` | 🔁 | `derived_types/err_info.py` |
| `error_code.F90` | 🔁 | `derived_types/err_info.py` |
| `interpolation.F90` | ✅ | `lin_interpolate_two_points`/`mono_cubic_interp`/`linear_interp_factor`/`zlinterp_fnc`/`plinterp_fnc`/`lin_interp_between_grids` (f2py ~4e-16). `binary_search` folded into jnp primitives |
| `mean_adv.F90` | ✅ | `term_ma_zt_lhs_jax` (flag-selected centered/upwind) + `term_ma_zm_lhs_jax` |
| `parameter_indices.F90` | 🔁 | parameters_tunable.py / constants_clubb.py |
| `pdf_closure_module.F90` | ◐ | moment-integral closures + chi-eta/cloud-frac/buoyancy components + `pdf_closure_driver` (all f2py-validated). ◐ = the deferred `pdf_closure_driver_zm` (gated, no oracle); `compute_cloud_cover`/orphan cluster are not-targets |
| `pdf_parameter_module.F90` | 🔁 | `derived_types/pdf_params.py` |
| `penta_lu_solver.F90` | ✅ | `penta_lu_solve` (LU + lax.scan) |
| `stats_clubb_utilities.F90` | ◐ | `stats_accumulate`; init/begin/end + NetCDF writer live in io/stats_writer.py |
| `stats_netcdf.F90` | 🔁 | io/stats_writer.py |
| `tridiag_lu_solver.F90` | ✅ | `tridiag_lu_solve` (Thomas/LU + lax.scan) |
| `turbulent_adv_pdf.F90` | ✅ | all 4 subroutines (centered/upwind flag-selected + Godunov variants), f2py bit-exact |
| `calc_roots.F90` | ✅ | cubic/quadratic/cube_root (sign-preserving), f2py sorted-real shadow |
| `diagnose_correlations_module.F90` | ✅ | `diagnose_correlations` (Larson 2011) + Cholesky family + assertion checks (f2py ~1e-15) |
| `hydromet_pdf_parameter_module.F90` | ✅ | dataclasses + init/zero helpers |
| `pos_definite_module.F90` | ✅ | Smolarkiewicz `pos_definite_adj` (f2py rel 0 + conservation) |
| `LY93_pdf.F90` | ✅ | alternative PDF (unused by ADG1), fully ported, f2py ≤1.3e-15 |
| `calendar.F90` | ✅ | all 5 date helpers for the solar-zenith path (JDN round-trip) |
| `index_mapping.F90` | ✅ | all 5 PDF↔hydromet index maps |
| `matrix_operations.F90` | ◐ | `Cholesky_factor` (f2py 1.1e-16) + `mirror_lower_triangular_matrix`; eigen/LAPACK helpers are infra |
| `new_hybrid_pdf.F90` / `new_hybrid_pdf_main.F90` / `new_pdf.F90` / `new_pdf_main.F90` / `new_tsdadg_pdf.F90` | ✅ | alternative PDF closures (unused by ADG1) — all fully ported, f2py end-to-end bit-match ~1e-14 |
| `remapping_module.F90` | ✅ | both methods (Ullrich-linear + E3SM PPM), f2py same-grid bit-exact + mass conservation. kord≥7 Huynh branch omitted |
| `penta_bicgstab_solver.F90` | ➖ | alternative iterative solver (method 3); LU is default+only path, driver rejects others |
| `clubb_api_module` / `code_timer_module` / `endian` / `file_functions` / `lapack_interfaces` / `lapack_wrap` / `grid_adaptation_module` | ➖ | Fortran build/IO/API/infra or unused-by-gated-config |

## `Microphys/`  (✅16 · 🔁4 · ◐3 · ❌3 · ➖9)

| Fortran | | JAX / note |
|---|:-:|---|
| `KK_microphys/KK_Nrm_tendencies.F90` | ✅ | KK_Nrm_tendencies.py |
| `KK_microphys/KK_upscaled_covariances.F90` | ✅ | quadrivar/trivar covar eqs + `covar_*_KK_{auto,accr}` drivers (oracle-confirmed) |
| `KK_microphys/KK_upscaled_means.F90` | ✅ | upscaled mean drivers + integrals + exponents |
| `KK_microphys/KK_upscaled_turbulent_sed.F90` | ✅ | bivar LL covar kernels + `KK_sed_vel_covars` driver |
| `KK_microphys/KK_utilities.F90` | ✅ | `G_T_p`; `Dv_fnc` absorbed into `parabolic_cylinder.py` |
| `KK_microphys/Parabolic.f90` (+constants) | 🔁 | `parabolic_cylinder.py` — D_v reimplemented via 1F1 series + asymptotics (float64, differentiable; better than scipy.pbdv) |
| `KK_microphys/PDF_integrals_covar.F90` | ✅ | trivar/quadrivar covar integrals + const_* specializations |
| `KK_microphys/PDF_integrals_means.F90` | ✅ | PDF_integrals_means.py |
| `KK_microphys/parameters_KK.F90` | ✅ | parameters_KK.py (KK exponents + constants) |
| `KK_microphys/KK_local_means.F90` | ✅ | grid-mean KK rates (analytic-validated, no oracle) |
| `KK_microphys/KK_upscaled_variances.F90` | ✅ | `variance_KK_mvr` (closed-form rel 0 + Monte-Carlo) |
| `KK_microphys/PDF_integrals_all_MM.F90` | ✅ | all 8 mixed-moment integrals (analytic + Monte-Carlo validated, differentiable) |
| `Morrison_microphys/module_mp_graupel.F90` | ✅ | module_mp_graupel.py (the WRF-M2005 scheme) |
| `Morrison_microphys/microphysics.F90` | 🔁 | module_mp_graupel.py |
| `advance_microphys_module.F90` | ✅ | per-hydrometeor lhs/rhs/solve + sed operators + `advance_one_hydrometeor`. Upwind-sed/cloud-top paths gated off |
| `cloud_sed_module.F90` | ✅ | cloud_sed_module.py |
| `KK_microphys_module.F90` | ◐ | `KK_microphys_adjust` + `KK_sedimentation` + the inline `kk_evap_coef`/`kk_auto_coef`; per-process rates + orchestration in `kk_microphys_driver.py` (rico-oracle GREEN) |
| `microphys_driver.F90` | ◐ | `calc_microphys_scheme_tendcies` dispatch; per-scheme impls in kk/morrison_microphys_step.py |
| `morrison_microphys_module.F90` | ◐ | `morrison_microphys_driver` (CLUBB↔M2005 interface); WRF scheme in module_mp_graupel.py |
| `microphys_init_cleanup.F90` | 🔁 | advance_clubb_to_end.py / clubb_driver.py |
| `parameters_microphys.F90` | 🔁 | Microphys/* constants |
| `coamps_microphys_driver_module.F90` | ❌ | COAMPS microphysics — Fortran fatal-errors, no oracle |
| `gfdl_activation.F90` | ❌ | CLUBB-side orchestration ported (`erff`/`updraft_weights`/`aer_act_clubb_ndrop`); only the 5-D aerosol-activation lookup table remains impractical |
| `pdf_hydromet_microphys_wrapper.F90` | ❌ | hydromet-PDF wrapper — 0 output for every gated case |
| `mixed_moment_PDF_integrals.F90` | ✅ | all 8 functions (closed-form + Monte-Carlo + literal-loop validated) |
| `ice_dfsn_module.F90` | ✅ | `ice_dfsn` (rel 1.2e-16 vs literal loop) |
| `Microphys_utils/microphys_stats_vars_module` / `SCM_Activation/*` (5) / `estimate_scm_microphys_module` / `lh_microphys_driver_module` / `silhs_category_variance_module` | ➖ | stats bookkeeping, aerosol activation, SILHS — not targets |

## `Radiation/`  (✅12 · 🔁4 · ◐1 · ❌0) — **fully ported**

| Fortran | | JAX / note |
|---|:-:|---|
| `BUGSrad/bugsrad_physconst.F90` | ✅ | bugsrad_physconst.py |
| `BUGSrad/bugsrad_planck.F90` | ✅ | bugsrad_planck.py |
| `BUGSrad/gases_ckd.F90` | ✅ | correlated-k routines; `.h` coefficient header → `gases_ckd_data.py` |
| `BUGSrad/newexp.F90` | ✅ | BUGSrad fast-exp rational approx (bit-replicated; `exp = newexp` alias) |
| `bugsrad_driver.F90` | ✅ | bugsrad_driver.py |
| `soil_vegetation.F90` | ✅ | soil_vegetation.py |
| `BUGSrad/kinds.F90` | 🔁 | x64 config |
| `BUGSrad/two_rt_lw.F` / `two_rt_sw.F` | ✅ | base two-stream LW/SW solvers (`-Dnooverlap` build path) |
| `cos_solar_zen_module.F90` | ✅ | `cos_solar_zen` + date helpers |
| `extended_atmosphere_module.F90` | ◐ | `determine_extended_atmos_bounds`; `finalize_extended_atm` deallocation has no JAX analog; std-atm loaders live in sounding.py |
| `parameters_radiation.F90` | 🔁 | radiation_module.py constants |
| `rad_lwsw_module.F90` | ✅ | `sunray_sw` (Delta-Eddington SW, f2py 1.35e-15) |
| `radiation_module.F90` | 🔁 | `advance_clubb_radiation` → `radiation_driver` → simplified/BUGSrad branches; gabls3 BUGSrad bit-faithful |
| `radiation_variables_module.F90` | 🔁 | radiation_module.py + state dict |
| `simple_rad_module.F90` | ✅ | `simple_rad` + `liq_water_path` + bomex variant + `simple_rad_lba` table reader |
| `BUGSrad/cloud_correlate.F90` | ✅ | `bugs_ctot`/`bugs_cloudfit` (optional, -Dnooverlap excludes; bit-faithful) |

## `Benchmark_cases/`  (✅22 · 🔁4 · ◐1 · ❌6)

Per-case modules expose the Fortran `<case>_tndcy`/`<case>_sfclyr` routines; `prescribe_forcings.py` dispatches.
All bare-named (no `_jax` suffix). Surface/forcing routines are bit-exact vs literal Fortran + differentiable;
the case-level bit-faithfulness verdict is a separate per-case property noted where relevant.

| Fortran | | JAX / note |
|---|:-:|---|
| `arm.F90` | ✅ | `arm_sfclyr` + forcing parser (bit gate ProgFail 0) |
| `atex.F90` / `atex_long.F90` | ✅ | `*_tndcy` (inversion-based subsidence) + `calc_forcings` + `*_sfclyr` (bit-faithful) |
| `bomex.F90` | ✅ | `bomex_tndcy` + `bomex_sfclyr` (bit-faithful) |
| `cobra.F90` | ✅ | `cobra_sfclyr` (ARM-variant, z0=1.75; bit-faithful) |
| `diag_ustar_module.F90` | ✅ | `diag_ustar` (Monin-Obukhov u*) |
| `dycoms2_rf01.F90` / `dycoms2_rf02.F90` | ✅ | both `*_tndcy` + `*_sfclyr` (bit-faithful) |
| `ekman.F90` / `neutral_case.F90` | ✅ | `*_sfclyr` (bit-faithful) |
| `fire.F90` | ✅ | `fire_sfclyr` (bulk-aerodynamic, bit-faithful) |
| `gabls2.F90` / `gabls3.F90` / `gabls3_night.F90` | ✅ | `*_tndcy`/`*_sfclyr` + Businger-Dyer stability fns; all bit-faithful |
| `jun25.F90` | 🔁 | prescribe_forcings.py (`jun25_altocu`) |
| `mpace_a.F90` | ✅ | `mpace_a_init`/`_tndcy`/`_sfclyr`. **Case is Tier-C PASS / FP-limited** past the strict 1e-6 bit gate (see memory `mpace-a-preexisting-regression`) |
| `nov11.F90` | ✅ | `nov11_altocu_rtm_adjust`; case is Morrison/ice FP-limited |
| `prescribe_forcings.F90` | 🔁 | `prescribe_forcings_arm` + `prescribe_forcings_generic` + `read_surface_var_for_bc` |
| `rico.F90` | ✅ | `rico_tndcy`/`rico_sfclyr`; case is KK-microphysics-limited / Tier-C |
| `sfc_flux.F90` | ✅ | shared surface-flux helpers (ubar/momentum/heat/moisture conversions) — all routines mirrored |
| `spec_hum_to_mixing_ratio.F90` | ✅ | flux + forcing q_t→r_t conversions |
| `time_dependent_input.F90` | ◐ | whole forcing lifecycle (load/parse/`time_select`/apply) incl. pressure-coord + `T_f` + nudging paths; readers folded |
| `wangara.F90` | ✅ | `wangara_sfclyr` + `wangara_tndcy` (bit-faithful) |
| `astex_a209.F90` | ✅ | `astex_a209_sfclyr`; case KK-microphysics-limited (not bit-faithful) |
| `clex9_nov02.F90` / `clex9_oct14.F90` | 🔁 | via prescribe_forcings.py (Morrison pre-activation); **bit-faithful gate members** |
| `cloud_feedback.F90` | ❌ | `cloud_feedback_sfclyr` ported (12 cgils/cloud_feedback cases). CGILS init+rad+forcing path fixed → cgils_s11/s12 Tier-C PASS + differentiable; cases are FP-limited at cloud onset |
| `lba.F90` | ✅ | `lba_sfclyr` + `lba_tndcy` (bit-exact); case SILHS-blocked end-to-end |
| `arm_97.F90` / `arm_0003.F90` / `arm_3year.F90` | ❌ | `*_sfclyr` ported + dispatch-wired; cases blocked (SILHS/COAMPS/missing data → no oracle) |
| `mpace_b.F90` / `twp_ice.F90` | ❌ | both/the sfclyr routines ported (bit-exact); cases SILHS-blocked |

## `Input_fields/`  (✅2 · 🔁1 · ◐2 · ➖7)

| Fortran | | JAX / note |
|---|:-:|---|
| `sounding.F90` | ✅ | `convert_pressure_sounding_to_z` (unblocks CGILS init) + std-atm/ozone loaders for the radiation extended atmosphere |
| `hydrostatic_module.F90` | ✅ | `hydrostatic`/`inverse_hydrostatic` + pressure-coord altitude integrators (round-trip 0.0) |
| `input_reader.F90` | ◐ | blank-fill numerics (`linear_fill_blanks`/`fill_blanks_two_dim_vars`); file parsers replaced by JAX `_parse_*` readers |
| `input_interpret.F90` | ◐ | `read_z_profile` pressure branch → `sounding.convert_pressure_sounding_to_z`; rest is GrADS/netCDF I/O not mirrored |
| `stat_file_module.F90` | 🔁 | io/stats_writer.py |
| `corr_varnce_input_reader` / `extrapolation` / `input_fields` / `input_grads` / `input_names` / `input_netcdf` / `stat_file_utils` | ➖ | Fortran file/namelist I/O readers (JAX uses Input_fields/*.py + netCDF4) |

## `src/` (root)  (✅2 · ➖9)

| Fortran | | JAX / note |
|---|:-:|---|
| `clubb_driver.F90` | ✅ | clubb_driver.py |
| `clubb_standalone.F90` | ✅ | clubb_standalone.py |
| `G_unit_tests` / `clubb_driver_test` / `clubb_thread_test` / `clubb_tuner` / `error` / `generalized_grid_test` / `int2txt` / `jacobian` / `text_writer` | ➖ | Fortran driver/test/tuner/utility programs (not physics) |

## Not-a-target subsystems (collapsed)

Whole directories where every file is ➖; rows omitted but counted above.

| Fortran directory | files | Why not translated |
|---|--:|---|
| `Lapack/` | 167 | LAPACK/BLAS reference library |
| `Numerical_recipes/` | 10 | Numerical Recipes utilities (quicksort, simulated annealing) |
| `G_unit_test_types/` | 15 | Fortran G-unit test harnesses (JAX has its own `clubb_jax/tests/`) |
| `SILHS/` | 11 | SILHS Latin-hypercube sampling — RNG-based, not bit-reproducible vs the Fortran RNG; not a target |
| `COAMPS_microphys/` | 47 | COAMPS microphysics — no oracle (Fortran fatal-errors) |
