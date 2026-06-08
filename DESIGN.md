# CLUBB-JAX Design

Goal: translate the CLUBB turbulence parameterization from Fortran to JAX for differentiable,
composable use in ML and autodiff workflows.

> **Mirror-refactor status (verified iter 243; per-file routine audit iters 262-275; alias-aware sweep iters 281-304; at-scale all-directory routine-diff + right-file misplacement audit iters 314-317; casing + file-name audits iter 318).**
> The file/routine-name mirror to the Fortran oracle is *comprehensively complete* for every in-scope, mirrorable routine: every JAX module
> sits at its Fortran file's relative path; every JAX routine that corresponds to a single Fortran
> `subroutine`/`function` now carries the bare Fortran name (the `_jax` suffix was fully retired iters 218-233 except
> the dual-structure jit-aliased raws — where the bare jitted alias already carries the Fortran name); exact-name
> "corrections" were reverted to mirror the Fortran (incl. its `respnder`/`derrived` typos); inlined Fortran-named
> routines were extracted to named functions (arm_sfclyr, atex/atex_long calc_forcings); routines were relocated to
> their Fortran-home module (e.g. precip_frac_double_delta_jax); and dead duplicates / dead jax-only helpers were purged.
> A systematic **per-file Fortran-subroutine-vs-JAX-def gap audit (iters 262-275)** then closed every remaining
> *exercised* gap it found: missing functions ported (numerical_check `length_check`/`pdf_closure_check`/`rad_check`;
> calendar `gregorian2julian_date`/`julian2gregorian_date`; Nc_Ncn forward `Ncnm_to_Nc_in_cloud`/`Ncnm_to_Ncm`/
> `bivar_NL_chi_Ncn_mean`; `T_in_K2thlm`; interpolation `lin_interp_between_grids`); inline code extracted to its Fortran
> subroutine (advance_xp3 `term_tp_rhs`/`term_ac_rhs`; precip_fraction `component_precip_frac_specify`; mono_flux_limiter
> `mfl_xm_lhs`/`mfl_xm_rhs`/`mfl_xm_solve`; corr_varnce `def_corr_idx`); de-inlined to the Fortran decomposition
> (calendar `compute_current_date` → JDN round trip); private→public promotions (`diagnose_corr`/`rearrange_corr_array`);
> `_jax`/`_api` alias vestiges retired (`get_default_config_flags`, `get_param_names`); and routine-name **casing**
> aligned to the exact Fortran (`Skx_func`, `KK_sedimentation`, `KK_microphys_adjust`, `KK_sed_vel_covars`,
> `Cholesky_factor`, `Diff_denom`). Cross-cutting scans confirmed **no misplaced routine** (every JAX def sits in its
> Fortran-home file — incl. `HmMetadata`/`NuVertResDep`/`ErrInfo` types in their Fortran modules, not the API's
> `derived_types/` extraction), **no removable progress-artifact** (`reset_clubb_core_state` is a legitimate reentrancy
> reset), and **no remaining mixed-case mismatch** (only the WRF-Morrison ALL-CAPS `POLYSVP`/`DERF1`, left lowercase per
> that restructured module's consistent convention). A follow-on **private→public promotion sweep (iters 281-284, 298,
> 300)** promoted 17 private `_`-prefixed helpers that mirror real (often `private`-in-Fortran or `result()`) Fortran
> routines to their bare names (gabls3_night `gm1`/`gh1`/`fm1`/`fh1`/`psi_h`; grid_class `calc_zt2zm_weights`/
> `calc_zm2zt_weights`; numerical_check `check_nan`/`check_negative`; remapping `kmppm`/`ppm2m`/`steepz`/`map1_ppm`;
> KK `bivar_LL_covar_partial`/`_rr`/`_Nr`/`_const_x2_partial`); and an **alias-aware per-file sweep (iters 296-304)** —
> which captures Fortran `result()`-functions, `private ::` routines, and JAX alias-/`__all__`-definitions the def-only
> scans had missed — closed the last genuine gaps: 2 missing functions (new_hybrid_pdf `calc_coefs_wpxpyp_semiimpl`),
> 1 dropped alias (PDF_integrals_covar `trivar_NNL_covar_const_all`), 2 divergent-name renames (`mpace_a_init`,
> `simple_rad_lba_init` — case/module forcing-table init readers), and completed the **validation-check class**
> (`length_check`/`pdf_closure_check`/`rad_check`/`sfc_varnce_check` + `corr_array_assertion_checks` +
> `precip_frac_assert_check` + `assert_corr_symmetric`). The **irreducible residual**
> (verified by-design via 10+ complementary scans) is: (1) **category-2 JAX decompositions** — functions that split or
> fold Fortran *inline* code with no single Fortran subroutine name (`calc_pdf_*`, `apply_lhs_band*`, `*_decomp`,
> `calc_xp2_xpyp_ta_*`, `calc_xpthvp_terms`, `solve_xp2_xpyp_jax`, `adg1_pdf_driver_zt_jax`, …), kept for
> differentiability and marked with `_jax`; (2) **JAX-infrastructure** (the `tracer_numpy` toolkit, `derived_types/`
> grouping mirroring the Gunther Python API, `_safe_div`/`_cardano_cbrt`/etc. numerical helpers); (3) **deliberate
> consolidations** (the Benchmark `prescribe_forcings` dispatch, sounding/input_interpret reading, `io/stats_writer`);
> and (4) **gated-off / no-oracle subsystems** — alternative schemes the validated config never selects, so they have
> no oracle to validate a port against: COAMPS, GFDL CCN lookup, SCM aerosol, SILHS RNG, `compute_cloud_cover`
> (l_use_cloud_cover=False), `trapezoidal_rule_*` (l_trapezoidal_rule_*=False), `wp3_term_ta_new_pdf_lhs`/
> `xpyp_term_ta_pdf_*_godunov` (non-ADG1 / explicit TA), `sed_upwind_diff_lhs` (l_upwind_diff_sed=False),
> `get_cloud_top_level` (l_prevent_hm_ta_above_cloud=False), the gfdl/lookup saturation formulae (saturation_formula=3),
> the BUGSrad `two_rt_*_{iter,sel,bs}` solver variants, and pure utilities with zero live callers (`calc_xp2`,
> `set_boundary_conditions_lhs/rhs`, `binary_search` — folded into jnp.searchsorted). These cannot be
> renamed/relocated to a Fortran name without an oracle or without undoing the differentiable architecture.
> **Their selector flags are fail-loud guarded (iters 346-348):** clubb_driver's `_check_unsupported_features` rejects
> `l_call_pdf_closure_twice` / `l_use_cloud_cover` / `l_trapezoidal_rule_zt`/`_zm` / `l_upwind_diff_sed` /
> `l_prevent_hm_ta_above_cloud` / `l_godunov_upwind_xpyp_ta` (all default-off, set by no case) with a clear
> "{flag} = true is not supported ({routine} not ported)" — so an unsupported config fails at validation rather than
> silently passing the flag through and computing default behavior. It also **rejects the FALSE setting** of two
> default-TRUE flags whose false branch advance_xm_wpxp doesn't implement (it hardcodes the default): `l_use_C7_Richardson`
> (hardcoded C7 = Cx_fnc_Richardson; the Skw-damped-C7 path unported) and `l_diag_Lscale_from_tau` (hardcoded constant C6;
> the Lscale-damped-C6 / `damp_coefficient` path unported). It further rejects non-default **PDF-closure placement**
> `ipdf_call_placement != 2` (iter 362): the pre-advance (1) / pre-post (3) PDF closures were never ported (they relied
> on the Fortran `clubb_python.clubb_api` fallback, absent in this tree). The init guard rejects them, and the
> pre-advance Block G `clubb_api.pdf_closure_driver` call was replaced with a fail-loud raise (iter 389), so **no
> `clubb_python` reference remains anywhere in the live driver**. It also
> rejects the **variance sponge** flags (wp2/wp3/up2_vp2 `l_sponge_damping`; `sponge_damp_xp2`/`xp3` are ported +
> unit-tested but not wired into the JAX advance, iter 364) and a non-LU **banded-solver method**
> `penta_solve_method`/`tridiag_solve_method != 2` (iter 370): the JAX `matrix_solver_wrapper` only implements the LU
> solvers and never reads the flag, so `penta_bicgstab` (= 3, the unported `penta_bicgstab_solver.F90`) would be
> silently ignored. An iter-371 "ConfigFlags field never read in src" sweep added **9 more guards** for flags the JAX
> never consults (so it hardcodes the default-off/on behavior): 8 default-FALSE reject-TRUE (`l_C2_cloud_frac`,
> `l_Lscale_plume_centered`, `l_do_expldiff_rtm_thlm`, `l_godunov_upwind_wpxp_ta`, `l_ho_trad_coriolis`,
> `l_partial_upwind_wp3`, `l_stability_correct_Kh_N2_zm`, `l_vert_avg_closure`) + `l_use_precip_frac` reject-FALSE.
> An **iter-497 wp2/wp3-closure sweep** added 5 more (the iter-371 heuristic had missed them because each appears in
> the `advance_wp2_wp3` solve docstring, so the "never referenced" check counted them as read): reject-TRUE
> `l_standard_term_ta` / `l_use_tke_in_wp2_wp3_K_dfsn` / `l_crank_nich_diff`, reject-FALSE
> `l_use_tke_in_wp3_pr_turb_term` / `l_damp_wp3_Skw_squared` — the JAX hardcodes the ARM/ADG1 closure config and these
> select unported branches; the *dispatched* siblings (`l_lmm_stepping`, `l_use_C11_Richardson`, `l_damp_wp2_using_em`,
> `l_tke_aniso`) were deliberately left unguarded.
> An iter-372 **integer-selector** sweep added an `iiPDF_type != 1` guard (only ADG1 is wired into the JAX
> `pdf_closure_driver`; ADG2/3D_Luhar/new/TSDADG/LY93/new_hybrid are ported as files but not wired) — the other
> integer selectors are already init-guarded or fail-loud at their use site (`grid_remap_method` →
> remapping_module ValueError, `fill_holes_type` → fill_holes_vertical NotImplementedError, `saturation_formula` →
> sat_mixrat_liq ValueError).
>
> A final **at-scale all-directory routine-diff + right-file misplacement audit (iters 307-317)** swept every
> `.py`↔`.F90` pair across CLUBB_core / Microphys / Radiation / Benchmark_cases / Input_fields / driver and cross-checked
> every routine's *home file*. It closed the last genuine gaps it found — the `sat_vapor_press_liq` dispatcher; a whole-file
> `index_mapping.py` mirror (+ the `mvr_*`/`cm3_per_m3` constants); `plinterp_fnc`; `get_corr_var_index`; `print_corr_matrix`;
> `compute_rtp2_from_chi`; the `bugs_ctot`/`bugs_cloudfit` divergent-name rename — and relocated the two KK process
> coefficients (`kk_evap_coef`, `kk_auto_coef`) to their Fortran-home `KK_microphys_module.py`. The misplacement audit
> confirmed **no remaining routine sits in the wrong file** (the only cross-file flags are the documented `pdf_params`↔
> `pdf_parameter_module` / `prescribe_forcings`↔`prescribe_forcings` renames and the `advance_clubb_to_end` split), and
> `derived_types/` is a complete 1:1 mirror of the Gunther API. The mirror is converged.
>
> The audit is now **reproducible**: `python clubb_jax/run_scripts/mirror_audit.py` (iter 331; pure-Python, no JAX/oracle
> needed) re-runs the JAX↔Fortran name diff comment-aware + typed-function-aware, scoped to mirrored files, with the
> documented fold/not-target/rename/casing exceptions enumerated in-code. The `_api` Gunther-wrapper fold is
> **precise** (iter 397): a `*_api` Fortran routine is excused from MISSING only when the JAX actually provides the
> de-api'd routine (bare name present), else it must be listed in the explicit `_API_DEFERRED` set (the reviewed
> wrappers with no bare-name mirror: err_info / config_flags / debug-level Gunther idioms, SILHS, and gated subsystems
> incl. `setup_corr_varnce_array_api` and the `setup_pdf_parameters_api` orchestration inlined in
> `kk_microphys_driver.py`) — so a *future* unmirrored `_api` routine surfaces as MISSING instead of being silently
> hidden by a blanket `_api$` regex. It checks **multiple dimensions**: the
> routine-name diff (MISSING / CASING / MISPLACED), a **file-name** diff (UNMIRRORED_FILES, iter 366: every
> `src/**/*.py` stem must match a Fortran source/header stem `.F90/.f90/.F/.f/.h/.inc`, a documented `_RENAMES`
> jax-side, or the `_JAX_ONLY_FILES` allowlist of intentional JAX-architecture files), a **directory-correspondence**
> diff (MISPLACED_FILES, iter 718: a JAX `.py` whose stem matches a Fortran *source* stem must also live in the
> oracle's directory — the basename-only UNMIRRORED check can't see a whole file moved to the wrong subdir; the one
> documented architectural split `grid_class`→`derived_types/` is in the `_DIR_SPLIT_OK` allowlist, itself liveness-
> guarded by `test_mirror_audit.py`), a tolerance-hygiene diff
> (REDUNDANT_TOL, iter 369: a `_NOT_TARGET` entry that is *actually* ported is flagged so the tolerance set can't
> accumulate stale entries that would mask a future regression), and a public-API-naming diff (JAX_ALIAS, iter 374:
> a `<name>_jax` def mirroring a Fortran routine must have a bare-name public alias, else the public name diverges
> to `_jax`). (Routine extraction itself was hardened iter 696 to parse continuation-style `subroutine NAME &`
> headers, so no routine — and thus no file — escapes the diff unseen.) It prints **PASS — MISSING=0 CASING=0
> MISPLACED=0 UNMIRRORED_FILES=0 MISPLACED_FILES=0 REDUNDANT_TOL=0 JAX_ALIAS=0** at
> convergence and exits 1 if a *new* genuine gap appears (a standing regression guard for the mirror, asserted by
> `tests/test_mirror_audit.py`).
>
> **Independent from-scratch reproduction (iters 971–980).** To corroborate the aggregate PASS rather than trust
> the counters, all seven dimensions were re-derived with standalone scanner code that imports only the audit's
> allowlist *data* (`_NOT_TARGET`/`_FOLD`/`_RENAMES`/`_DIR_SPLIT_OK`/`_JAX_ONLY_FILES`/`_CASING_OK`), not its
> logic: MISSING (every `.F90`/`.F`↔`.py` pair tree-wide; 133 CLUBB_core raw-misses classified to 0 genuine gaps;
> 0 across 62 non-CLUBB_core paired modules), reverse-naming (difflib fuzzy scan of JAX-only defs → 0 fixes),
> UNMIRRORED_FILES (147 `.py` all map; surfaced that BUGSrad uses fixed-form `.F`), MISPLACED (453 co-named pairs,
> 445 same-stem + 8 documented renames), CASING (453 pairs, exact oracle capitalization modulo `_CASING_OK`),
> REDUNDANT_TOL + MISPLACED_FILES (allowlist tight; dirs correspond), JAX_ALIAS (8 `_jax` mirrors all carry a
> bare alias) — all 0. The exercise also confirmed the audit correctly honors assignment aliases (`flip =
> flip_vertical`), the `grid_class` dir-split, and `.F`/`_api`/`_jax` folding.

> **Faithfulness-validation campaign (iters 408–446, saturated).** With the *name/file* mirror converged, a follow-on
> campaign closed the **behavioral**-validation gap: **~41 named mirrors** that had only weak/re-implemented references,
> property checks, or no validation at all now carry direct **f2py-Fortran-oracle** unit tests (bit-shadow on synthetic
> inputs, SKIP-clean when `clubb_f2py`/`clubb_python` are unbuilt). Coverage spans the calendar routines, the grid
> operators (`ddzt`/`ddzm`/`zt2zm`/`zm2zt` + round-trips), `diffusion_z{t,m}_lhs`, `zlinterp_fnc`/`mono_cubic_interp`,
> `quadratic_solve`, `cos_solar_zen`, the binormal/PDF utilities, `calc_stability_correction`/`calc_Ri_zm`,
> `vertical_avg/integral`, the clip family (`clip_rcm`/`clip_covar`/`clip_covars_denom`/`clip_variance`/`clip_skewness`),
> `calc_brunt_vaisala_freq_sqd`, the mixing-length routines, `smooth_{min,max,corr_quotient}`/`smooth_heaviside_peskin`,
> the saturation mixing ratios, `thlm2T_in_K`, hydrostatic pressure, `sat`/mean-advection operators, `xpyp_term_ta_pdf`
> (standard + Godunov), the `sponge_layer_damping` module, the `tridiag_lu`/`penta_lu` band solvers, the **whole**
> `monotonic_turbulent_flux_limit` end-to-end, `compute_Cx_fnc_Richardson` (production path), `sunray_sw`, the core
> `ADG1_pdf_driver` (25 outputs), and — most valuably — the **gated** routines `diagnose_Lscale_from_tau` and
> `calculate_thlp2_rad`, which no validated case exercises (previously *only* name-mirrored, behavior unverified).
>
> Two early classifications were later **corrected by re-test**: `fill_holes_vertical` is NOT an FP-floor (iter 437 — the
> apparent ~1e-8 residual was a float32 test-harness defect, missing `jax_enable_x64`; it bit-matches at 3.3e-16), and
> `compute_Cx_fnc_Richardson`/`sunray_sw` are NOT genuinely signature-divergent (iters 443–444 — their f2py wrappers
> merely hide dead/optional args or module-global params, recoverable). The lesson — **an apparent ~1e-8 mismatch is far
> more often a test-harness defect (missing x64, wrong hardcoded parameter, hidden flag) than a real divergence** — was
> banked and reused to reclaim the limiter/Cx/sunray from the "case-only" list. The genuinely case-level-only remainder
> is final and small: f2py wrappers that FPE-trap/core-dump (`precip_fraction`, `update_xp2_mc`, + the cnvg-test branch
> of `compute_Cx_fnc_Richardson`), the non-ADG1 branch of `compute_xp3` (uses `thvm`/Brunt-Vaisala — no case selects
> it; its **ADG1 path was reclaimed iter 490** by reconstructing the Fortran ADG1 branch from the bit-shadowed
> `f2py_zm2zt_2d` + `f2py_xp3_lg_2005_ansatz` primitives, `tests/test_compute_xp3.py`), and the 49-output monolithic
> `pdf_closure_driver` (its ADG1 core + every component closure are individually f2py-validated). The whole-driver
> `advance_*` are validated at the case level (bit-faithful `compare_cases`).
>
> **Structural-mirror hardening (iters 465–478).** A follow-on pass pinned the **load-bearing data structures** that the
> closures read — previously validated only *implicitly* (a single wrong index/value/enum would mis-tune or mis-dispatch
> the model, caught before only by a slow full case run). Now each has a direct, fast unit guard: the whole
> **tunable-parameter pipeline** (`test_param_names.py` — the 102 names + `i<name>` index constants vs `f2py_get_param_names`,
> the base values vs `clubb_api.init_clubb_params`, the derived `lmin`/`mixt_frac_max_mag` vs `f2py_calc_derrived_params`,
> all exact); the **config-flag defaults** (all 67 vs `clubb_api.get_default_config_flags`, + the existing 60-flag coverage
> check, `test_config_flags_complete.py`); the **physical/numerical constants** (33 literals parsed straight from
> `constants_clubb.F90`, `test_constants.py`); and the **dispatch enums** (`iiPDF_*` 1..7 and the saturation BOLTON/FLATAU
> codes, parsed from `model_flags.F90`). Source-parsing tests strip comments + skip expressions so they read the active
> standalone-build value, not a CESM-branch reference or a commented approximation. Extended (iters 475–478) to the
> **derived-type field mirrors**: the JAX `pdf_parameter` (49) / `implicit_coefs_terms` (30) / `SclrIdx` (6) NamedTuples
> are parsed-and-compared field-for-field, in order, against their Fortran `type … end type` definitions (the closure
> carries its whole state in these — a reordered/missing field would silently mis-populate it); and `ConfigFlags` is now
> **bidirectionally** checked (Fortran-case-settable ⊆ JAX coverage + JAX ⊆ Fortran no-spurious + all 67 default values).
> The restructured JAX types (`HmMetadata` dataclass, `Grid` computed-subset, `ErrInfo` simplified) are intentionally not
> field-mirrored. Net: every routine-validatable mirror is f2py-bit-shadowed and every load-bearing data structure is
> directly pinned — the bit-faithful cases are now the *third* line of defense, not the only one.
>
> **Validation saturation + continuous structural guards (iters 479–533).** The behavioral-validation campaign was
> driven to saturation across *every* directory (CLUBB_core, Microphys, Radiation, io, Input_fields, derived_types,
> Benchmark_cases): a routine-by-routine sweep confirmed every routine is now directly tested, on a bit-faithful
> full-case path, or transitively f2py-bit-shadowed via a tested driver. The last directly-untested mirrors were
> closed — the closure term-builders (wp2/wp3, xp2/xpyp, xm/wpxp per-level LHS/RHS), `ADG1_ADG2_responder_params`
> (exact F90 transcription), `plinterp_fnc` (f2py bit-shadow via the `zlinterp(−grid)` identity), `remap_vals_to_target`
> (two-grid mass conservation, exact), the `pdf_params` alloc+zero init, `time_select` (the time-dependent-forcing
> bracket selector, incl. its benign exact-node triple difference), and the `gabls3_night` Businger-Dyer stability
> functions — plus the fail-loud unsupported-config guards (`test_unsupported_config_guards.py`) extended to the 6
> infrastructure guards (SILHS/restart/input-fields/test-grid/grid-adapt) that *justify* the audit's `_NOT_TARGET`
> folds. Two convergence invariants were also made **continuously enforced** rather than re-checked by hand:
> `test_mirror_audit.py` (the full audit: MISSING/CASING/MISPLACED/UNMIRRORED/REDUNDANT/JAX_ALIAS all 0) and the new
> `test_routine_placement.py` (an audit-independent, source-parsed guard that every name-exact AND `_jax`-suffixed JAX
> routine lives in its Fortran home file). Independent reverse sweeps (Fortran→JAX MISSING, orphan-reachability,
> dead-JAX-only) confirm the sole non-folded unmirrored routine is the deliberately-DEFERRED `pdf_closure_driver_zm`
> (gated by `l_call_pdf_closure_twice` which no case sets, no f2py oracle, structurally non-reusable from the
> zt-specialized JAX `pdf_closure_driver` → a port would be unreachable, unvalidatable dead code; fail-loud guarded).
> Full unit suite green throughout (130+ files); both correctness gates (bit-faithful + whole-driver `jax.grad`) PASS.
>
> **Deep isolation-test campaign (iters 534–572).** With the mirror converged and saturation reached, a follow-on
> campaign drove *isolation-level* (mostly Monte-Carlo- or closed-form-independent) validation into the routines that
> were previously covered only end-to-end or transitively. Highlights: the whole `advance_windm_edsclrm` decomposition
> (Coriolis `compute_uv_tndcy`, the Crank-Nicholson tridiag `windm_edsclrm_lhs/rhs`); the **pdf_closure decomposition**
> — every higher-order PDF moment (`calc_wp4_pdf`, the binormal `<w'²x'>/<w'x'²>/<w'²x'²>`, the trinormal `<w'x'y'>`)
> Monte-Carlo-validated against the actual normal-mixture central moments, the liquid/ice cloud-fraction closures
> (`P(χ>0)`, `E[max(χ,0)]`, ice-supersat shifted CDF), `calc_pdf_chi_mean_var` (law of total variance), the `x'rc'`
> covariance fluxes, the Sommeria-Deardorff `χ/η` transform, the buoyancy-flux `<x'th_v'>` assembly, and the skewness
> diagnostics; the **KK upscaled-integral machinery** — the bivar/trivar/quadrivar mean+covariance variance-regime
> DISPATCH wiring (incl. the quadrivar x3↔x4/β↔γ symmetry swap), the LL mean/covariance partials (closed form == the
> tested general at the σ→0 limit), the N_r tendency formulas, and the `covar_*_KK_*` exact mixt_frac-linearity; the
> flux-limiter truncated means (`E[max(w,0)]`/`E[min(w,0)]`); and the mass-conserving hole-fill family
> (`fill_holes_global`/`_sliding_window`, ρ·dz conservation exact). A standing lesson reused throughout: when a closed
> form risks tautology, validate it INDEPENDENTLY — Monte-Carlo of the underlying distribution, the σ→0 limit of an
> already-tested general routine, or a conservation/total-variance identity. Suite now 158 files, ALL GREEN.

> **Call-structure mirroring + excusal hardening (iters 574–583).** With names, files, and locations converged, two
> closing fronts. (1) **Call-structure mirroring** — the budget-decomposition stats in the three moment-advance modules
> now obtain their contribution splits by the Fortran's own idiom (call the mirror-named term routine twice with one
> field/coefficient zeroed, or +1), not via JAX-only helpers or inlined formulas: iter 579 removed the JAX-only
> `term_tp_rhs_decomp_jax`/`term_pr1_decomp_jax` (now two `term_tp_rhs`/`term_pr1` calls with zeroed args,
> F90:3346-3352/3760-3770); iter 580 replaced the inlined `wpxp` bp/pr3 formulas with `wpxp_terms_bp_pr3_rhs(C7=0)`/`(C7+1)`
> (F90:1894-1913); a per-subroutine completeness audit of all five advance modules (iter 581) then confirmed every Fortran
> advance routine is a named mirror or a documented fold — no promotable inline remains — all bit-faithful across cases
> (bomex + dycoms2_rf01, stats rel ≤4e-13). (2) **Source-grounded excusal guards** — every non-trivial `mirror_audit`
> excusal is now a machine-checked tripwire against oracle drift rather than a bare human-asserted tolerance:
> `test_orphan_cluster_still_dead_in_fortran` (iter 486, the no-caller interp cluster), `..._boundary_condition_setters_still_dead...`
> (581, `set_boundary_conditions_{lhs,rhs}` uncalled), `..._pdf_closure_driver_zm_call_still_gated` (582, the SOLE
> `_DEFERRED` routine stays behind its `l_call_pdf_closure_twice` gate), and `..._compile_dead_parameter_gates_unchanged`
> (583, `l_explicit_turbulent_adv_wp3` stays a fixed `parameter=.false.`). A whole-source audit (582) + docstring-citation
> misplacement scan (583) reconfirmed the sole genuinely-unmirrored routine is `pdf_closure_driver_zm` — re-assessed as
> faithfully-portable-only-as-unreachable-oracle-less-dead-code (the JAX `calc_pdf_*` helpers are zt-specialized), so it
> stays deliberately deferred rather than reclassified to manufacture a "0 deferred" audit.

> **Audit completeness + de-scaffolding (iters 584–596).** Two finishing fronts over the converged mirror. (1) **The
> excusal-guard set was completed and the whole-file scoping made transparent.** Per-subsystem completeness audits
> extended to Morrison (585) and Radiation (586), each Fortran routine a named mirror or documented excusal; the two
> remaining excusal classes were source-grounded — `radiation_variables_module` stays state-management-only (586) and the
> `_jax_stems()` whole-file scoping is now visible (587: a "FORTRAN FILES scoped out of MISSING" INFO line), bucketed into
> 12 by-design-unmirrored subsystems matched by **dedicated directory** + non-physics name keyword (588/589), and guarded
> by `test_no_unrecognized_scoped_out_file` — so a new oracle *physics* file that is neither mirrored nor a recognized
> non-target fails loudly instead of being silently scoped out. Iter 590 added `test_src_has_no_fortran_runtime_import`
> (AST guard: zero executable `clubb_python` references in `src`), machine-enforcing the "100% JAX, zero Fortran calls per
> timestep" property. (2) **De-scaffolding** — `src` was swept free of all incremental-port progress-tracking residue with
> no Fortran analog: the last `IterNN:` development tags (591), the obsolete `Block M+N`/`M+10` block-numbering and
> "(Fortran oracle removed)" shadow-comparison markers (594), the orphaned `# JAX-only <routine>` section labels (596), the
> dormant env-gated `CLUBB_LEAK` debug hook (595), and the stale F2PY-era architecture docstring atop
> `advance_clubb_core_module.py` (593, rewritten to the accurate pure-JAX description). All comment/dormant-code only —
> the live arm path re-verified bit-running post-sweep (iter 597 JAX smoke run). The standing-guard set (10 `test_mirror_audit`
> checks — incl. the iter-719 `_DIR_SPLIT_OK` directory-split liveness guard + the iter-729 routine-less-module
> classification guard — + dead-import/function/config/param-roundtrip guards) is the drift-proof protective layer over the converged mirror.

> **Physical-constant deduplication (iters 598–604).** The Fortran does `use constants_clubb, only: …` per subroutine; the
> JAX had re-defined those values as local literals in many modules (a duplication + drift risk). Each was replaced with the
> named `from …constants_clubb import …`, all **bit-identical** (verified) and smoke/test-confirmed: `wpxp_terms_bp_pr3_rhs`'s
> grav default (598); `clubb_driver.py`'s 16-constant standalone block (599, dropping the unused intermediates Rv/ep);
> `saturation.py` Cp/Lv/T_freeze_K/ep (600); the arm/gabls2/cobra surface schemes' grav/p0/Rd/Cp/sec_per_hr — surfacing and
> removing a dead `_vonk` along the way (601/603); `diag_ustar`'s vonk, the simplified-radiation `_CP`, and the KK upscaled
> covariance/mean modules' Lv/Cp/chi_tol/rho_lw/mvr_rain_max (602/604). Intentionally NOT touched: the self-contained
> subsystems mirroring their OWN Fortran constants (BUGSrad `bugsrad_physconst`, WRF `module_mp_graupel`), and the few names
> the JAX `constants_clubb.py` deliberately never ported (Nc_tol/rr_tol — kept local with a comment). Enum-parameter dedup
> (iter 605): `numerical_check.py`'s `_order_*`/`_ipdf_*` — true duplications of constants_clubb's proper module constants —
> now import from constants_clubb (the iiPDF/saturation sets stay local, which the JAX constants_clubb subset lacks). A
> known remainder partially closed (iter 606): the `model_flags.F90` enum *parameters* the JAX had defined in
> `constants_clubb.py` (iiPDF_ADG1, ipdf_*, order_*, l_gamma_Skw, l_advance_xp3) were **moved to their Fortran home
> `model_flags.py`** (the JAX mirror of model_flags.F90), with `constants_clubb.py` re-exporting them so the 3 importers
> keep working (verified circular-free: constants_clubb→model_flags→config_flags-leaf; the 10 re-exports allowlisted in the
> dead-import guard). The saturation enums (`SATURATION_*` in saturation.py, no constants_clubb counterpart) remain the JAX
> home for that subsystem. (No
> *function* duplication exists — the 3 same-named-in-two-files routines, e.g. `term_tp_rhs` in advance_xp2_xpyp + advance_xp3,
> are faithful per-module mirrors of the Fortran, verified iter 605.)
>
> **Constants/parameters mirror completed (iters 607–638).** The `use constants_clubb` / `use parameters_KK` /
> `use clubb_api_module` fidelity sweep was carried to saturation: index *parameters* moved to a new
> `parameter_indices.py` (iter 607, mirror of parameter_indices.F90); every module-local literal that duplicated a
> `constants_clubb` value was replaced with the named import across CLUBB_core, Radiation, Microphys and
> Benchmark_cases (iters 612–630), and the genuinely-absent constants were *added* to `constants_clubb.py` first —
> `g_per_kg`, `omega_planet`, `stefan_boltzmann`, `rho_ice`, `pascal_per_mb`, `Nc_tol`, `rr_tol`, `eta_tol`,
> `parab_cyl_max_input`, `one_third`. A new mirror file **`Microphys/KK_microphys/parameters_KK.py`** was created
> (iter 631) holding the KK exponents + `r_0`/`C_evap`, with all `use parameters_KK` consumers repointed to import
> from it directly (iters 632–635); the KK `KK_tendency_coefs` coefficients `KK_ACCR_COEF`/`KK_MVR_COEF` were
> relocated to their Fortran-home `KK_microphys_module.py` alongside `kk_auto_coef`/`kk_evap_coef` (iters 633–634).
> A final value-scan confirms **zero** hardcoded `constants_clubb` values remain in code (only a docstring mention);
> the literals that stay are all faithful Fortran *local* `parameter`s (mixing_length `Lscale_sfclyr_depth=500`,
> ice_dfsn `N_i`/`k_u_coef`, Morrison `_KK_RHOW=997`/`cloud_frac_thresh`, the truncated `pi=3.141592654` deliberately
> kept full-precision). Bit-faithfulness preserved throughout (every change value-identical; arm/bomex/gabls2/gabls3/
> gabls3_night/dycoms2_rf01/wangara/atex/cobra bit-gates re-verified across the campaign).
>
> **`parameters_microphys.F90` — deliberately no `parameters_microphys.py` mirror (verified iter 728).** Unlike
> `parameters_KK.F90` (physical *constants* → mirrored file), `parameters_microphys.F90` is a routine-less module
> of per-case *configuration* (`microphys_scheme`, `l_ice_microphys`, `l_graupel`, `l_cloud_sed`, `Nc0_in_cloud`,
> the SILHS `lh_*` knobs, …) which the JAX faithfully represents as **runtime namelist config** read into
> `clubb_driver.py`/state — hardcoding it into a constants file would be wrong. Its only fixed *constants* are the
> `morrison_{no_aerosol,power_law,lognormal}` aerosol-type enums, used solely by the `specify_aerosol`
> Morrison-activation path = the scoped-out `SCM_Activation` subsystem (no validated case), so they are correctly
> absent. NB this file is one of the routine-less parameter modules (like `array_index.F90`) that are invisible to
> `mirror_audit`'s routine-based checks (0 routines → not MISSING; routine-less → never enters the scoped-out
> enumeration). That blind-spot is **now guarded** (iter 729): `mirror_audit._routineless_unclassified()` asserts
> every routine-less module is same-stem JAX-mirrored, a recognized subsystem, or in the documented
> `_ROUTINELESS_OK` allowlist (11 entries incl. `parameters_microphys`), so a NEW pure-parameter/type module in the
> oracle surfaces for review instead of hiding (tests/test_mirror_audit.py::test_no_unclassified_routineless_module).

---

## Repository Structure

```
CLUBB-JAX/
├── clubb_jax/              ← JAX translation (this repo)
│   ├── src/
│   │   ├── CLUBB_core/     ← physics modules, one file per Fortran oracle
│   │   ├── Benchmark_cases/
│   │   ├── Input_fields/
│   │   ├── Radiation/
│   │   ├── io/             ← stats_writer.py (pure Python NetCDF output)
│   │   ├── clubb_standalone.py  ← thin CLI frontend ↔ clubb_standalone.F90 (entry: python -m clubb_jax.src.clubb_standalone)
│   │   ├── clubb_driver.py      ← run_clubb / init_clubb_case / clean_up_clubb ↔ clubb_driver.F90
│   │   └── advance_clubb_to_end.py  ← timestep loop (advance_clubb_to_end subroutine of clubb_driver.F90)
│   ├── run_scripts/        ← test infrastructure
│   │   ├── compare_runs.py ← Fortran vs JAX comparison
│   │   └── run_scm.py      ← single-case runner
│   └── tests/              ← unit tests
└── clubb_release/          ← git submodule: larson-group/clubb_release (master)
    ├── src/                ← Fortran source oracle
    ├── input/              ← case setups, namelists, sounding files
    ├── clubb_python_api/   ← f2py compiled wrappers (not in submodule git)
    ├── bin/ install/       ← compiled Fortran binaries (not in submodule git)
    └── run_scripts/        ← upstream Fortran test scripts (not modified)
```

`clubb_jax/` and `clubb_release/` are siblings. The JAX package works against upstream
`clubb_release` master — it has no dependency on any custom branch.

---

## How to Test

**Prerequisites:** `clubb_release/bin/clubb_standalone` and `clubb_release/clubb_python_api/*.so`
must be present (compiled artifacts, not in git — copy from a build or compile from source).
The regression tests require `clubb_python_api` to be present (for the Fortran comparison run
and for non-ARM `prescribe_forcings`). The JAX driver itself has zero module-level Fortran imports
since Iter 73 — ARM runs do not need `clubb_python_api` to be importable at all.

```bash
# From CLUBB-JAX/ or any directory:

# Quick smoke test — JAX driver, no Fortran comparison run (~20s):
# (writes clubb_jax/output/arm_stats.nc — NOT the clubb_release/ oracle; see "Output-directory convention")
python clubb_jax/run_scripts/run_scm.py arm -jax -max_iters 3

# Full Fortran-vs-JAX regression for one case (30 timesteps):
python clubb_jax/run_scripts/compare_runs.py --case arm --max-iters 30

# Multi-case regression dashboard (generalises the test beyond ARM):
python clubb_jax/run_scripts/compare_cases.py --max-iters 30
python clubb_jax/run_scripts/compare_cases.py --cases arm,bomex,wangara --max-iters 15

# Whole unit-test suite in ONE command (Iter305) — runs every clubb_jax/tests/test_*.py, reports pass/fail
# per file + a summary; exit 0 iff all green. Tests vs an unavailable oracle SKIP cleanly (so it is portable).
python clubb_jax/run_scripts/run_all_tests.py            # all 21 files (~12 min: bugsrad+standalone are slow)
python clubb_jax/run_scripts/run_all_tests.py -k solver  # only files matching "solver"

# Divergence-onset classifier (Iter311) — after a compare_runs.py run, classify WHEN/HOW each failing
# prognostic diverges: bit-faithful / FP-growth (gradual, undamped-oscillator-style) / JUMP@N (discrete
# onset → rule out a term/threshold bug). Generalises the per-step decomposition done by hand on every case.
python clubb_jax/run_scripts/diagnose_divergence.py rico            # only failing prognostics
python clubb_jax/run_scripts/diagnose_divergence.py rico --var thlp2  # one variable, pass or fail

# Unit tests (pure JAX, no Fortran needed):
python clubb_jax/tests/test_solver.py
python clubb_jax/tests/test_diffusion.py
python clubb_jax/tests/test_penta_solver.py
python clubb_jax/tests/test_penta_faithful.py      # penta_lu solve == Fortran-order numpy replica (0 ULP eager)
python clubb_jax/tests/test_calc_roots.py          # calc_roots cubic/quadratic/cube_root vs polynomial residual + numpy.roots (completeness port)
python clubb_jax/tests/test_pos_definite.py        # pos_definite_adj (Smolarkiewicz limiter) BIT-EXACT vs f2py oracle + conservation (completeness port)
python clubb_jax/tests/test_diagnose_correlations.py # diagnose_correlations (Larson 2011) + PDF helpers — bit-match vs f2py oracle (completeness port)
python clubb_jax/tests/test_kk_local_means.py      # KK grid-mean (local) evap/auto/accr/mvr rates vs independent NumPy + branches (completeness port)
python clubb_jax/tests/test_kk_upscaled_variances.py # variance_KK_mvr vs independent closed-form lognormal moments (rel 0) + 4M-sample Monte-Carlo (rel 1.4e-4) (completeness port)
python clubb_jax/tests/test_ice_dfsn.py            # ice_dfsn vs literal NumPy loop (rel 1.2e-16) + thlm2T_in_K bit-exact vs f2py + cap/branch/grad (completeness port)
python clubb_jax/tests/test_hydromet_pdf_parameter.py # hydromet-PDF parameter containers: zero-init shapes/dims/round-trip (CLUBB_core now fully ported)
python clubb_jax/tests/test_mixed_moment_pdf_integrals.py # mixed_moment_PDF_integrals integrals/covariances vs binomial/tilting closed-forms (<1e-12) + Monte-Carlo (full port)
python clubb_jax/tests/test_hydrometeor_mixed_moments.py # hydrometeor_mixed_moments top driver vs literal Fortran-loop transcription (<1e-12) + grad
python clubb_jax/tests/test_pdf_integrals_all_mm.py # KK all-mixed-moment Dv integrals (trivar+quadrivar families, 8/8) vs analytic base cases + complex-branch MC
python clubb_jax/tests/test_cloud_correlate.py     # BUGSrad cloud-overlap (bugs_ctot + bugs_cloudfit) vs literal Fortran loops (rel 3e-16/1e-14) + invariants
python clubb_jax/tests/test_gfdl_activation.py     # GFDL erff (vs math.erf <1e-6) + updraft_weights (vs literal incl. Fortran quirk) (partial port)
python clubb_jax/tests/test_simple_rad_lba.py      # LBA prescribed radiation (table load + time/vertical interp) vs literal Fortran on real lba_rad.dat
python clubb_jax/tests/test_cloud_feedback_sfclyr.py # CGILS/cloud_feedback drag-law surface fluxes vs literal Fortran (rel 0) + physical invariants
python clubb_jax/tests/test_pressure_coord_forcing.py # Press[Pa]-coordinate time-dependent forcing (interp vs p_in_Pa) + height path byte-identical
python clubb_jax/tests/test_inverse_hydrostatic.py # inverse_hydrostatic (pressure-sounding altitudes) round-trip z→exner→z exact (5.5e-12 m) + literal + analytic
python clubb_jax/tests/test_lba_sfclyr.py          # LBA diurnal surface fluxes + MOST ustar vs literal Fortran + diurnal structure
python clubb_jax/tests/test_mpace_b_lba_tndcy.py   # M-PACE B large-scale subsidence/cooling forcing vs literal + invariants; LBA zero forcing
python clubb_jax/tests/test_silhs_surface_schemes.py # mpace_b/arm_97/twp_ice surface schemes vs literal Fortran (twp_ice == cloud_feedback drag law)
python clubb_jax/tests/test_f2py_advance_xm_wpxp.py # f2py advance_xm_wpxp .so directly callable (oracle unblocked; needs clubb_python_api)
python clubb_jax/tests/test_differentiability.py   # jax.grad through the building blocks (+ mixing-length reverse, REFACTOR B3)
python clubb_jax/tests/test_full_timestep_grad.py  # ★ REFACTOR B4: full-timestep jax.grad through advance_clubb_core (FD-correct)
python clubb_jax/tests/probe_driver_grad.py <case>  # ★ REFACTOR B5: WHOLE-driver jax.grad through advance_clubb_to_end (per case; FD-correct)
python clubb_jax/run_scripts/compare_grad.py        # ★ REFACTOR B5 GATE: whole-driver differentiability dashboard, all cases (grad analogue of compare_cases)
python clubb_jax/tests/test_mono_flux_limiter.py   # REFACTOR B2: JAX lax.scan flux limiter == NumPy (bit-exact) + grad
python clubb_jax/tests/test_invariants.py          # REFACTOR Tier-A: oracle-free conservation/positivity/Cauchy-Schwarz
python clubb_jax/tests/test_pdf_utilities.py       # lognormal<->normal moments — BIT-TO-BIT vs f2py
python clubb_jax/tests/test_kk_autoconversion.py   # KK rate functions vs quadrature/scipy
python clubb_jax/tests/test_kk_rico_oracle.py      # KK autoconv END-TO-END vs Fortran rico rrm_auto
python clubb_jax/tests/test_fill_holes_mean.py     # rtm_cl/thlm_cl mean-field fill (Iter186 fix guard)
python clubb_jax/tests/test_morrison_special.py    # Morrison POLYSVP/DERF1/GAMMA vs scipy (Iter190-192)
python clubb_jax/tests/test_morrison_rates.py      # Morrison warm-rain + ice rates vs nov11_altocu oracle (Iter193-201)
python clubb_jax/tests/test_morrison_differentiable.py  # Morrison rate library is jax.grad-able (Iter203)
python clubb_jax/tests/test_bugsrad.py             # 17 BUGSrad tests: RT machinery (vs Fortran replicas ≤2e-13) + bugs_rad + bugsrad_driver + advance_radiation dispatch (invariants) (Iter255-269)
python clubb_jax/tests/test_soil_vegetation.py     # soil_vegetation force-restore surface BC (gabls3 l_soil_veg): BIT-EXACT vs Fortran replica (Iter270)
python clubb_jax/tests/test_standalone_jax.py      # "entirely in JAX": ARM runs with clubb_python BLOCKED (find_spec import-blocker) (Iter280)
```

**Unit-test sweep convention.** Run each `python clubb_jax/tests/<t>.py` (or `run_all_tests.py`). Any
f2py-oracle import in a test MUST be guarded (try/except ImportError -> SKIP+return) so the JAX-only
assertions still run when `clubb_f2py` is unbuilt -- the suite is then a clean gate in any environment.

**★ Output-directory convention (Iter218; revised).** JAX-produced stats and Fortran-produced (oracle)
stats live in SEPARATE trees, so a JAX run can never clobber an oracle:
- **`run_scm.py <case> -jax`** defaults to **`clubb_jax/output/<case>_stats.nc`**.
- **`run_scm.py <case> -legacy`** (and `-exe`/default Fortran) defaults to **`clubb_release/output/<case>_stats.nc`** -- the oracle home, where the rate/stats tests read it.
- `-out_dir <dir>` overrides either default.
- `compare_runs.py` keeps each side in its own subdir -- Fortran `clubb_release/output/<case>_compare_fort/`, JAX `clubb_jax/output/<case>_compare_jax/`; `diagnose_divergence.py` reads those same paths.
Oracle-generation commands target `clubb_release/output/...` explicitly via `-legacy -out_dir` (e.g.
`rico_fort` = `run_scm.py rico -legacy -max_iters 10 -out_dir clubb_release/output/rico_fort`; `rico_long_fort`
= the same at `-max_iters 250` for developed-rain tests). Regenerate a clobbered oracle with `-legacy`.

**Verification oracles, in order of preference:**
1. **In-loop f2py shadow** -- most CLUBB_core routines are bit-to-bit verifiable via `clubb_f2py`. Caveats: some
   wrappers FPE-trap/core-dump (`f2py_precip_fraction`); the `.so` is also callable DIRECTLY past a stale wrapper
   using the `__doc__`-introspected signature (`run_scripts/debug/cmp_terms_f2py.py`, `debug/compare_xm_wpxp_f2py.py`) -- an
   input-matched per-term comparison the namelist A/B can't do.
2. **Case-stats oracle** (for unported subsystems the f2py API can't reach, e.g. microphysics): a Fortran SCM run
   writes both a rate's PDF-moment INPUTS and its rate OUTPUTS, so the JAX rate is verifiable in isolation by
   feeding the Fortran's own moments in (`test_kk_rico_oracle.py`, `test_morrison_rates.py`). **Timing confound:** a
   routine called mid-step sees inputs differing from the END-of-step stored stats, so rates depending on a field
   created during the step (accr/evap/deposition) match only to a few % -- validate the FORMULA by the median,
   defer bit-faithfulness to a running case; discrete tolerance tests can flip at boundaries, so validate on the
   well-resolved interior.
3. **Conservation contract** (oracle-free, for flux-form transport operators): the column-mass-weighted
   (sum rho_ds*dz*) tendency must equal the net boundary flux (machine precision), immune to the timing confound.
   **Gotcha:** a new microphysics module MUST `jax.config.update("jax_enable_x64", True)` at import -- float32
   silently passes relative-error tests but breaks a conservation (cancellation) contract at ~1e-5.
4. **Invariant** (when the Fortran can't be run for a bit-oracle): e.g. no-cloud -> all-sky fluxes == clear-sky.

**★ Regression gate.** `compare_runs.py --case X` runs Fortran and JAX independently and diffs the stats; all
PROGNOSTIC variables must PASS (rel 1e-6, abs floor 1e-12); diagnostic timing differences are expected.
`compare_cases.py` wraps it over all bit-faithful cases (one pass/fail line each) -- the generalized gate; run it
after any shared/core change. It auto-forces per-step output (stats_tsamp=stats_tout=dt_main) so the comparison
reflects physics, not stats-averaging windows. **Resource note: do NOT run multiple `compare_*` jobs in parallel**
-- concurrent JAX processes OOM-kill each other (looks like a spurious "JAX run failed rc=1"); run sequentially.

**★ Divergence diagnosis.** After a failing `compare_runs.py`, run `diagnose_divergence.py CASE`: per failing
prognostic it reports the onset step and classifies it -- **JUMP@N** (a sudden jump from machine-eps past the floor
= a discrete branch/threshold crossed, e.g. cloud/precip onset -> rule out a term bug) vs **FP-growth** (gradual
accumulation). It also prints a **sign tally** at the gate-cross: balanced/flipping -> FP/chaos; strongly
one-sided & persistent -> a systematic term/coefficient bug. General method that cracks most seeds:
**budget-decompose the prognostic** (`*_bt/_ma/_ta/_cl/_forcing/...` from the per-step stats) to pin the seed to
one term BEFORE chasing the solve, and **check the reference magnitude** (~ the tol floor => FP, not a bug).

**★ Durability & chaos horizon.** The 30-step gate MASKS late-activating events -- run `compare_cases.py
--max-iters 100` periodically. A case whose forcing/microphysics/event activates at a known time (`time >= ...`
gates, `microphys_start_time`, ice onset, diurnal sunset/sunrise where `amu0` crosses 0.01) must be verified PAST
that step. Conversely, **full-length bit-faithfulness is PHYSICALLY IMPOSSIBLE for chaotic turbulence**: two
bit-identical-start runs diverge after the Lyapunov time, so a >horizon FP-growth + sign-flipping failure is
physics, not a bug. Known chaos horizons: fire ~147, jun25_altocu ~200, atex_long ~305. The 100-step gate sits
within every case's horizon, so it is the right practical durability metric.

**★ Grid-type dimension.** Most bit-faithful cases are `grid_type=1` (uniform -> every zt<->zm weight exactly
0.5, so stretched-grid paths are NOT exercised). `rico` is `grid_type=2` (formula-stretched), `dycoms2_rf02_so`
is `grid_type=3` (file). A bug that vanishes when a case is switched to `grid_type=1` is a stretched-grid handling
bug -- the namelist A/B swap is the key localiser. (The one real stretched-grid bug found this way: the
`derived_types` `weights_zm2zt` columns were stored swapped vs Fortran, invisible on uniform grids; Iter151.)

**★ Differentiability / composability status.** Component-level differentiability + composability -- the
practical goal -- is DONE and tested (`tests/test_differentiability.py`): saturation, tridiag + penta solvers,
fill_holes, PDF cloud_frac, Brunt-Vaisala, the ADG1 w/full PDF closure, the KK rate drivers (auto/accr/evap,
full-array, edge-robust to rel ~1e-10), mixing-length forward-mode; the Iter290-291 core jits preserve grad.
Radiation too (`bugs_rad`, `soil_vegetation` are `jax.grad`-able). **Hardening convention:** a vanishing-denominator
quotient gives 0/0=nan whose VJP `jnp.where` masking does NOT fix (nan*0=nan) -- fix AT the operation (custom_jvp
safe-division, double-where safe_sqrt/_pos_pow, a D_v arg clamp), all forward-preserving. **★★ End-to-end
`jax.grad` through the core CLUBB timestep IS NOW AVAILABLE (REFACTOR B4, iter16-21).** `advance_clubb_core`
(the full closure + all prognostic solves + mixing length + flux limiter) is reverse-mode differentiable —
`jax.grad` w.r.t. the mean profile is finite and finite-difference-correct (rel 4.0e-10,
`tests/test_full_timestep_grad.py`). Achieved by **tracer-transparent numpy** (a drop-in shim that is jnp
under a JAX trace and *exactly* numpy otherwise, so the bit-faithful suite is unaffected): `_asarray`,
the `_xp` ufunc/`_like` shim, `_iset` (immutable-safe assignment), removal of dead shadow-comparison
scaffolding, and guarding the `_prev_adg1_j25` module-global under trace. The B2 flux limiter (iter11) and B3
mixing length (iter9) feed into this. **Convention (R6): hard min/max are differentiable (subgradient); only
`while_loop`/`np.asarray`/in-place-mutation/numpy-ufuncs break tracing — make them tracer-transparent, harden
`sqrt(maximum(0,·))` with `_safe_sqrt` and `maximum(x,0)**p` (p<1) with `_safe_pow` (clip-sqrt/clip-pow have
an inf reverse grad AT the clip — they nan only where the quantity actually reaches ≤0, so a stable case
passes while a convective one fails; audit every sqrt/fractional-pow on a possibly-≤0 quantity), and never
store a tracer in module-global state.** Grad uses the
standard differentiable-forward config (`debug_level=0`, `l_sample=False` — diagnostics/stats off). The shim
lives in `src/CLUBB_core/tracer_numpy.py` (`_asarray`/`_xp`/`_iset`/`_safe_sqrt`/`_is_tracer_arg`), shared by
the core and the driver. **★★ WHOLE-DRIVER `jax.grad` is now AVAILABLE for the arm case (REFACTOR B5,
iter25):** `jax.grad` through one full `advance_clubb_to_end` step (thvm + arm surface forcings + the core,
stats off) is finite + FD-correct (`tests/probe_driver_grad.py`; `d(½∑um²)/dum` rel 1.3e-8 exercises the
differentiable surface momentum-flux path). Two B5 patterns, both **bit-identical for concrete runs** (validated
arm Tier-B + Tier-C): **(R7) block-level tracer dispatch** — guard a small branchy block (Python `float()`/
`math`/`max`/fixed-point loops, e.g. the Monin-Obukhov `_diag_ustar`) with `if not _is_tracer_arg([...]):` →
exact original float path, `else:` → `jnp`/`jnp.where` mirror (`_diag_ustar_jax`); guard divisors a `where`
would otherwise leave `nan` in the unused branch (poisons reverse-mode grad even when masked). **(R8)
diagnostic-skip-under-trace** — pure NaN/Inf/stats checks that don't feed the prognostics early-`return`
unchanged when an input is a tracer (`parameterization_check`). **iter26 extended this to the
`prescribe_forcings` driver** (the path for ~17 cases): the generic surface scheme (incl. the convergence-test
`_mono_cubic_interp` BC and `_compute_ubar`) is now tracer-transparent and `d(½∑um²)/dum` whole-driver grad is
FD-correct for bomex. **iter27–29: bomex whole-driver grad is now COMPLETE (thlm + um both 87/87,
FD-correct rel ≤5.4e-7)** after hardening the inf-grad `sqrt`/`pow` sites that detonate in convective layers
(the binding one: `mixing_length.py:180` `sqrt(maximum(zero_threshold, bv_smth))` with `zero_threshold==0`,
pinned by **stop_gradient bisection**). So both the arm and prescribe_forcings whole-driver
paths are differentiable. **iters 30–33: whole-driver `jax.grad` is now finite for ~18 of 19 cases** —
all the major subsystems are tracer-transparent: simplified + BUGSrad radiation, soil-veg, KK + Morrison
microphysics (the post-core diagnostics use **detach-under-trace** — they feed only the next step, so are
dead for a single-step gradient; BUGSrad is also reverse-mode memory-prohibitive), the sponge layer
(vectorized to a no-op-outside-sponge form), cloud-droplet sedimentation, and the case surface schemes
(R7 `_diag_ustar_jax` dispatch). Some cases show a single-level FD kink at a hard physical threshold
(8e-3 `rtm` inversion) — a genuine non-smooth point, not a bug. **iter34: the last blocker (gabls3_night
`_landflx_scalar`, a Businger-Dyer land-surface MO scheme) is ported to `_landflx_jax` → ALL 19 cases now have
a finite whole-driver gradient.** `run_scripts/compare_grad.py` is the suite-wide differentiability GATE
(grad analogue of compare_cases); `tests/probe_driver_grad.py <case>` is the per-case validator;
`tests/_nanhunt.py` + stop_gradient bisection locate residual nan; clip-`sqrt`/fractional-`pow` →
`_safe_sqrt`/`_safe_pow`. **The B5 goal ("differentiable, entirely in JAX") is met suite-wide.**

**★ "Entirely in JAX."** The JAX driver references no `clubb_python` *anywhere* — not even the old lazy
forcing / pre-advance-PDF fallbacks (removed iters 388-389; both were dead since `clubb_python` is absent in this
tree). Verified by `tests/test_standalone_jax.py` (runs cases with `clubb_python` blocked). 19 cases have
entirely-in-JAX forcings (no Fortran fallback). "Faithful" != "entirely-in-JAX" != "bit-faithful full run" -- verify a ported
forcing via the STANDALONE (clubb_python-blocked) test, since a plain compare can PASS via the Fortran fallback
(false positive) and the fallback-hidden sfclyr often carries a bug. **Variants share a `runtype`** (rf02
nd/so/do/ds all `'dycoms2_rf02'`; rf01 vs rf01_fixed_sst by `sfctype` 0/1) -- key off `state['runtype']` + the
distinguishing flag, NOT the case-file name, and re-run the affected variant after a port/revert.

**★★ jit-recompilation -> unbounded compile-cache -> OOM (Iter290 root-cause + convention).** An eager
`lax.scan` whose body CLOSES OVER a concrete (non-tracer) array bakes that array's VALUES into the jaxpr as
constants, so XLA recompiles every timestep when the values change -> the compile cache grows without bound -> OOM
on long runs. **Rule: any per-timestep eager `lax.scan` (or a function containing one) should be `jax.jit`-wrapped
at a stable entry point** so captured arrays hoist to operands and it compiles once per aval. Diagnose with
`JAX_LOG_COMPILES=1` + `grep -c "Compiling jit(scan)"` -- a count that grows each step is this bug. Fixed for
`parabolic_cylinder.dv_parabolic_cylinder`, the tridiag/penta solvers, and `fill_holes_vertical` (rico
2165->381 total compiles, 137/step -> ~0; OOM gone, ~2x faster); re-verified bit-faithful. A `Killed`/EXIT=137
with no traceback is OOM (not a NaN) -- probe `resource.getrusage().ru_maxrss`. **Separate, still-open leak
(Iter323):** the per-step `l_sample=True` diagnostic path retains ~85 device buffers/step in the XLA backend (NOT
recompilation -- all compiles are at startup) -> a `compare_runs` per-step-stats Morrison run OOMs ~150-250 steps.
Workaround: run `advance_clubb_to_end` with `state['stats_writer']=None` and inspect the state dict, or sample at
the case default interval. Low priority (no current need for long Morrison compare runs).

**★ Precision convention — SUPERSEDED by the REFACTOR (numerical-accuracy standard).** Historically the
Fortran M2005 interface keeps `T_in_K`/`rcm_r4` in SINGLE precision (`real(...)`=REAL(4)), so its `thlm_mc`
carries a ~1e-7 single-precision round-trip residual even with zero microphysics tendencies; the JAX once
*replicated* the `real*4` casts to match it bit-for-bit (the sole reason mpace_a was "bit-faithful"). **Under
the relaxed numerical-accuracy standard (done; see "Correctness standard") we no longer reproduce the oracle's imprecisions.**
`module_mp_graupel.py` now computes `thlm_mc` in the algebraically-exact float64 form `(ten['T']−Lv/Cp·rcm_mc)/
exner` → clear-air `thlm_mc≈0` (correct; was a 2.9e-7 artifact). mpace_a is no longer bit-faithful but PASSES
Tier-C with large margin (means 70× / flux 21× / moment 104× / microphys 40×). **General rule going forward:
prefer float64 accuracy; validate within Tier-C rather than reproducing single-precision artifacts.**

4 persistent diagnostic-only (non-prognostic) differences, not fixable without matching Fortran FP ordering:
`rtm_spur_src` ~2e-16, `thlm_spur_src` ~2e-11, `rtp2_pd` ~7e-27, `up2_pd` ~1e-17.

**★ Previously-omitted default-on physics, now implemented.**
- **`l_rcm_supersat_adj` (default `.true.`; gap found iter 498, IMPLEMENTED + gate-verified iter 499).** The Fortran
  `pdf_closure` removes any *spurious* supersaturation remaining after the PDF call (pdf_closure_module.F90:4394 —
  where post-PDF `rel_humidity > 1`, it folds the excess `(rtm − rcm) − rsat` into `rcm`). This was originally
  omitted in the JAX (forward-identical for the bit-faithful suite, since the trigger never fires there — which is
  *why* the cases stayed bit-faithful despite the omission). Now implemented in advance_clubb_core Block U after the
  `pdf_closure_driver` call as `rcm = jnp.where(rel_humidity > 1, rcm + ((rtm−rcm)−rsat), rcm)` (T_in_K inlined as in
  Block O, via the validated `sat_mixrat_liq`). **Gate-verified iter 499:** all 19 strictly-bit-faithful
  `compare_cases` PASS ProgFail 0 (forward-identical), mpace_a PASS Tier-C — and it now makes the JAX faithful for a
  case that *does* supersaturate post-PDF. **R8-hardened iter 513:** the block is now skipped under a jax.grad trace
  (`and not _is_tracer_arg(...)`) — it is a no-op for the gradient of every differentiable case (none supersaturates),
  and the whole-driver grad probe runs one step from the saturated INITIAL state where the `sat_mixrat_liq` branch's
  grad is non-finite at extreme cold T; the concrete/forward path still applies it, so faithfulness is unchanged.
  (The `_rcm_supersat_adj` *diagnostic* field stays zeros — report-only, not gated.)
- **`clex9_oct14` grad — RESOLVED iter 515.** The iter-513 non-finite grad was **not** a nan but a
  `TracerArrayConversionError` in `sunray_sw` (the simplified-SW two-stream): a numpy/Python-native sequential routine
  (`np.exp`/`math.exp` + in-place flux loop), never tracer-transparent, that the daytime simplified-SW cases hit under
  a grad trace. Fixed by **skipping `sunray_sw` under a trace** (detach-under-trace at its radiation_module call site —
  radiation feeds only the next step, so it is dead for a single-step grad; forward path unchanged). clex9_oct14 is now
  grad-finite (184/184). The iter-514 `sat_mixrat_liq`/`sat_mixrat_ice` safe-division and the iter-513 R8 rcm guard
  remain as defensive grad-safety (real latent fixes, just not the cause here). `mpace_b` is **not** a grad bug — it
  sets `microphys_scheme='coamps'` (unported) and is init-rejected; `compare_grad` (iter 515) now classifies such
  unsupported/gated cases BLOCKED rather than counting them as gate failures. **`compare_grad` is back to PASS for all
  runnable cases.**

---

## Correctness standard (relaxed: numerical accuracy + differentiability)

The original gate was **bit-faithfulness** to the Fortran oracle (`compare_runs.py`, rel 1e-6 / abs 1e-12 on
prognostics). That was the right scaffolding for the incremental port — it caught real bugs (the stretched-grid
`weights_zm2zt` column-swap, the stale `wm_zm`, the KK covar driver) — but it outlived its use: it forced the
JAX to reproduce the Fortran's *imprecisions* (single-precision casts, the low-accuracy `expax`), it blocked
differentiability (hard min/max, `while_loop`, numpy round-trips), it produced brittle "failures" that are pure
FP/oracle artifacts at sharp edges, and trajectory-level bit agreement is *physically impossible* for chaotic
turbulence past the Lyapunov horizon anyway. The numerical-accuracy refactor (done on this branch) relaxed it to
a **tiered standard** — a change is correct if it passes the tiers appropriate to what it touches:

| Tier | Checks | Hardness / tool |
|---|---|---|
| **A. Invariants & conservation** | water/energy/mass conservation, positivity (`rrm,Nrm,rcm,…≥0`), bounded correlations (`\|corr\|≤1`), finiteness | **strict, oracle-free** — `tests/test_invariants.py`, `run_scripts/invariants.py` |
| **B. Golden-trajectory regression** | vs a stored **JAX reference run** per case, rel ~1e-9 | **strict-ish** — `run_scripts/golden.py`, `update_golden.py`, `validate_case.py --no-fortran` |
| **C. Physical fidelity vs Fortran** | windowed, field-scaled rel error within the chaos horizon (aggregate, not point bit-match) | **relaxed** — `compare_cases.py --tier physical`, `validation.py` |
| **D. Climatology / statistics** | time-mean & variance profiles, BL depth, cloud fraction past the chaos horizon | **statistical** (the honest gate for chaos-limited cases) |
| **E. Differentiability** | finite-difference grad checks; whole-driver `jax.grad` | **strict** — `compare_grad.py`, `probe_driver_grad.py`, `test_differentiability.py` |

**Tier-C field-class tolerances** (point-max `max|Δ|/(max|ref|+floor)`): means (`thlm,rtm,um,vm`) **1e-4**;
fluxes (`wpthlp,wprtp,upwp,vpwp`) **1e-3**; second moments (`wp2,wp3,rtp2,thlp2,em,…`) **3e-3**; microphysics
(`rrm,Nrm`) **1e-2**; diagnostics + `*_mc` tendencies **report-only** (timing-confounded). Bit-faithful cases
pass Tier-C by construction (rel ~1e-11 ≪ tol); calibrated against rico (near-worst FP case — dynamics PASS
2–10× margin) and arm/bomex (~1e7×).

**Status (this branch):** **20/20** `compare_cases` DEFAULT_CASES PASS Tier-C (19 strictly bit-faithful + mpace_a
within tolerance on its single-precision Morrison residual; clex9_nov02/oct14 added Iter313); **`compare_grad` is
27/28 whole-driver-`jax.grad`-grad-finite — gate PASS (iter 516)** (the 28th, mpace_b, is an unsupported
`microphys_scheme='coamps'` case the driver init-rejects, classified BLOCKED not FAIL; see "Differentiability status").
The accuracy-lowering contrivances were removed —
`parabolic_expax` (`epss=1e-4`), the Morrison `real*4` casts, BUGSrad `sngl`/float32-π — so the JAX is now
strictly *more* accurate there. **Preserve:** the Fortran oracle as a reference-within-tolerance (`--tier bit`
stays for debugging); golden refs as the regression net (re-baseline only via `update_golden.py`, deliberate +
reviewed); and **Tier A strict** — relaxed tolerances must never hide a conservation bug.

---

## Critical Conventions

**Band ordering:** Both Fortran and JAX use `lhs[0=super, 1=main, 2=sub]`. No flip needed
between diffusion output and solver input.

**Grid weights (`weights_zm2zt`):** Shape `(ngrdcol, nzt, 2)`. `[:,k,0]` = M_ABOVE (weight for
`zm[k]`), `[:,k,1]` = M_BELOW (weight for `zm[k+1]`). Fortran 1-indexed `m_above=1, m_below=2`.

**JAX x64 mode:** `jax.config.update("jax_enable_x64", True)` called at module load in
`advance_clubb_core_module.py`. All arrays must stay float64.

**Routine names mirror the Fortran subroutine; the `_jax` suffix retirement is COMPLETE for single-subroutine mirrors**
(iters 218-233; it was a shadow-comparison vestige). As of iter 234 the only residual `_jax`-suffixed routines are, by
design, two categories: **(1)** the dual-structure jit-aliased raws (see below — the bare jitted alias already carries
the Fortran name), and **(2)** JAX-specific helpers whose bare name is *not* a Fortran subroutine because the JAX
restructured the Fortran's inline code into a differently-decomposed function for differentiability (`calc_pdf_*`,
`apply_lhs_band*`, `*_decomp`, `solve_xp2_xpyp`, `calc_xp2_xpyp_ta_*`, `calc_xpthvp_terms`, `adg1_pdf_driver_zt`,
`precip_frac_double_delta`, `hydrometp2_zt`, `fill_holes_hydromet_clip`, `apply_sponge_field`, `get_param_names`,
`get_default_config_flags`). **Exception — the jit-alias dual structure:** a few leaf modules (`diffusion.py`, `mean_adv.py`) define a
raw un-jitted `<name>_jax` function AND a module-level `<name> = jit(<name>_jax)` alias that *already* carries the bare
Fortran name. Callers/tests import the **raw** `_jax` version on purpose — it accepts a plain grid object / non-pytree
`JaxGrid` and stays `jax.grad`-able, whereas the jitted alias rejects a non-pytree `gr`. There the Fortran name is
already mirrored, so do **not** collapse the two: only drop the `_jax` suffix in modules that have NO `jit()` alias of
the bare name (e.g. clip_explicit, Skx_module, advance_xp3). (mirror-refactor iters 218-220) NB `diffusion.py` had
accumulated **dead duplicate** routines (`term_dp1_*`/`xp2_xpyp_*`, `term_ma_zm_lhs`, `xpyp_term_ta_pdf_*`) — copies left
behind when those routines were "moved" to their Fortran homes (advance_xp2_xpyp_module.py / mean_adv.py /
turbulent_adv_pdf.py). All were purged at iters 228-229 (verified dead + AST-identical to the live copies);
`diffusion.py` is now a clean 1:1 mirror of diffusion.F90. Lesson: when relocating a routine, delete the original — a
left-behind copy silently diverges and confuses the location mirror. **Second clash type (iter 230):** before a blanket
`X_jax`→`X` rename, check the module doesn't already have a *distinct bare-named sibling* `X` (a standalone f2py-validation
form, or a NumPy reference impl). Collapsing them makes Python silently shadow one def with the other — harmless if the
bodies are numerically identical (pdf_closure's `calc_*_pdf`), but it can break a test that compared the two
(mono_flux_limiter's port-vs-reference check). Keep the live/Fortran-mirror impl on the bare name; give the
reference/standalone sibling a clearly-private name (e.g. `_monotonic_turbulent_flux_limit_numpy`).

**Index mapping (Fortran 1-based → Python 0-based):**
Interior loop `k=2..nzm-1` in Fortran → Python `[:,1:-1]` on zm-level arrays.

**`clubb_params` indexing:** Shape `(ngrdcol, 102)`, 0-based. Access as `clubb_params[:, iC2rt - 1]`.

---

## What Has Been Built

Each JAX module mirrors its Fortran oracle at the same relative path under `src/CLUBB_core/`.

| JAX Module | Fortran Oracle | Status |
|---|---|---|
| `grid_class.py` | `grid_class.F90` | `zm2zt`, `zt2zm`, `ddzm`, `ddzt`, `zm2zt2zm`, `zt2zm2zt` — unit tests pass |
| `diffusion.py` | `diffusion.F90` | `diffusion_zt/zm_lhs` — ≤ machine epsilon. (The `xpyp_term_ta_pdf_*` turbulent-adv terms are in `turbulent_adv_pdf.py`; `term_dp1`/`xp2_xpyp` assembly in `advance_xp2_xpyp_module.py`; `term_ma_*` in `mean_adv.py` — mirror-refactor) |
| `tridiag_lu_solver.py` / `penta_lu_solver.py` | `tridiag_lu_solver.F90` / `penta_lu_solver.F90` | `tridiag_lu_solve_jax` / `penta_lu_solve_jax` — bit-exact (split out of `matrix_solver_wrapper.py`, now a thin re-export dispatch, mirror-refactor) |
| `calc_roots.py` | `calc_roots.F90` | `cubic_solve` (Cardano, complex128 principal-branch), `quadratic_solve`, `cube_root` — polynomial residual ~4e-16 + numpy.roots set-match; differentiable. Completeness port (the gated ADG1 path doesn't call it; `new_pdf` does) |
| `pos_definite_module.py` | `pos_definite_module.F90` | `pos_definite_adj` — Smolarkiewicz (1989) flux-conservative positive-definite limiter (ascending grid). **Bit-exact vs the f2py oracle (rel 0)** + column-integral conservation; differentiable. Completeness port (gated by `l_pos_def`, off by default — the suite uses `mono_flux_limiter`) |
| `diagnose_correlations_module.py` | `diagnose_correlations_module.F90` | `diagnose_correlations` (Larson 2011 hydromet correlation diagnosis for SILHS: `rearrange_corr_array` + `diagnose_corr`) + PDF helpers `calc_mean`/`calc_varnce`/`calc_w_corr`. **Bit-match vs the f2py oracle (rel 1.6e-15)** across iiPDF_w edge cases; differentiable. Completeness port (gated config uses PRESCRIBED corr; `l_calc_w_corr=True` / approx_w_corr unported) |
| `Microphys/ice_dfsn_module.py` | `Microphys/ice_dfsn_module.F90` | `ice_dfsn` — depletion of cloud water by diffusional growth of ice (Larson 2006; R&Y Eq. 9.4) as a top-to-bottom falling-crystal mass-integration `lax.scan`; `diff_denom` helper. Validated vs a literal NumPy transcription (**rel 1.2e-16**), branch/over-depletion-cap coverage, differentiable. New helper `thlm2T_in_K` (T_in_K_module.py) is **bit-exact vs `f2py_thlm2t_in_k_1d`**. Completeness port (no f2py wrapper for ice_dfsn itself) |
| `Microphys/KK_microphys/KK_upscaled_variances.py` | `KK_microphys/KK_upscaled_variances.F90` | `variance_KK_mvr` — variance of the KK rain mean-volume radius `Var(R_vr)=E[R_vr²]−E[R_vr]²` over the 2-component in-precip bivariate-lognormal PDF (assembled from `bivar_LL_mean_eq` with doubled exponents). Validated against an independent closed-form lognormal-moment computation (**rel 0**) and a 4M-sample Monte-Carlo (**rel 1.4e-4**); differentiable. Completeness port (no f2py wrapper exposed) |
| `advance_xp2_xpyp_module.py` | `advance_xp2_xpyp_module.F90` | Full solve for rtp2/thlp2/rtpthlp/up2/vp2 — machine epsilon |
| `advance_xm_wpxp_module.py` | `advance_xm_wpxp_module.F90` | Full solve for wprtp/rtm/wpthlp/thlm/upwp/um/vpwp/vm — machine epsilon |
| `advance_wp2_wp3_module.py` | `advance_wp2_wp3_module.F90` | Full solve for wp2/wp3/wp2_zt — machine epsilon |
| `advance_windm_edsclrm_module.py` | `advance_windm_edsclrm_module.F90` | No-op for ARM (l_predict_upwp_vpwp=True) — bit-exact |
| `advance_xp3_module.py` | advance_xp3 + Skx_module | rtp3/thlp3/up3/vp3 (ADG1 path) — machine epsilon |
| `advance_helper_module.py` | `advance_helper_module.F90` | Skw, thvm, BV, Ri, Lscale/tau, splat, Cx — machine epsilon |
| `sfc_varnce_module.py` | `sfc_varnce_module.F90` | Surface second-order moments — sub-machine precision |
| `sigma_sqd_w_module.py` | `sigma_sqd_w_module.F90` | σ²_w PDF width parameter — bit-exact |
| `pdf_utilities.py` | `pdf_utilities.F90` | `mean_L2N`/`stdev_L2N` (lognormal->normal moments) **bit-to-bit vs f2py** (rel 0.0); `corr_NL2NN`/`corr_LL2NN` (vs Monte-Carlo); the inverses `corr_NN2NL`/`corr_NN2LL` and `calc_corr_chi_x`/`calc_corr_eta_x` (+ their round-trip inverses) **bit-to-bit vs f2py**. The lognormal-PDF inputs to the KK rate functions |
| `precipitation_fraction.py` | `precipitation_fraction.F90` | `precip_fraction` -- overall (downward cumulative-max) + per-component (`component_precip_frac_specify`) + max_hm limiter. **Bit-exact** vs the rico stats oracle on the well-resolved precip region. f2py wrapper FPE-traps -> stats oracle used |
| `setup_clubb_pdf_params.py` | `setup_clubb_pdf_params.F90` | `calc_comp_mu_sigma_hm` (in-precip component means/stdevs via a mean+variance-preserving quadratic solve, verified by its preservation contract); `compute_mean_stdev` + `norm_transform_mean_stdev` (the `setup_pdf_parameters` orchestration that stacks per-PDF-variable moments into the `(ngrdcol,nzt,pdf_dim)` arrays the rate functions index, iiPDF order [chi,eta,w,Ncn,hydrometeors], and transforms lognormal vars to log space). The KK driver assembles rr/Nr moments through these -- bit-identical + differentiable. Also `hydrometp2_zt_jax` (the overall precipitating-hydrometeor variance <hm'^2> = ((ratio+1)/precip_frac−1)·hm², F90:449; relocated here from kk_microphys_driver.py to its Fortran home, mirror-refactor iter 192 — the KK driver + the kk/morrison per-step paths import it). `tests/test_calc_comp_mu_sigma_hm.py` |
| `Nc_Ncn_eqns.py` | `Nc_Ncn_eqns.F90` | `Nc_in_cloud_to_Ncnm` (+ `Ncm_to_Ncnm`, `bivar_Ncnm_eqn_comp`) -- cloud-nuclei mean <Ncn> from in-cloud <Nc> and the chi PDF via the erfc integral. **Bit-to-bit vs f2py** (worst rel 2.4e-14); reproduces rico Ncnm exactly |
| `corr_varnce_module.py` | `corr_varnce_module.F90` | `set_corr_arrays_to_default` -- the prescribed in-cloud/below-cloud normal-space correlation arrays from the fixed 12x12 default tables (column-major reshape). The KK driver derives corr(chi,rr)/corr(chi,Nr)/corr(rr,Nr) from it instead of hardcoding (rico oracle bit-identical). `init_pdf_hydromet_arrays` + `HmMetadata` + `kk_hm_metadata` -- the per-hydrometeor PDF metadata (names/tols/flags, in-precip variance ratio, PDF-variable indices) the hydrometeor advance + setup consume. `tests/test_corr_varnce.py` |
| `Microphys/KK_microphys/KK_utilities.py` | `KK_microphys/KK_utilities.F90` + `KK_microphys_module.F90:1177` | `G_T_p` (drop-growth coefficient, Rogers&Yau) + `kk_evap_coef`. Validated via the rico rrm_evap oracle (T_liq=thlm*exner) |
| `fill_holes.py` | `fill_holes.F90` | `fill_holes_vertical`, `fill_holes_wp2_from_horz_tke` — machine epsilon |
| `clip_explicit.py` | `clip_explicit.F90` | `clip_variance`, `clip_skewness`, `clip_covar`, `clip_rcm`, `clip_covars_denom` — bit-exact |
| `adg1_adg2_3d_luhar_pdf.py` | `adg1_adg2_3d_luhar_pdf.F90` + `pdf_closure_module.F90` | Full ADG1 PDF closure — machine epsilon |
| `mixing_length.py` | `mixing_length.F90` | `diagnose_lscale_from_tau` + `compute_mixing_length` (Golaz 2002 nonlocal parcel) — machine epsilon |
| `saturation.py` | `saturation.F90` | `sat_mixrat_liq` (Flatau/Bolton), `rcm_sat_adj` (bisection) — machine epsilon |
| `sponge_layer_damping.py` | `sponge_layer_damping.F90` | `initialize_tau_sponge_damp` + `sponge_damp_xm` (xm fields rtm/thlm/uv) — wired into `advance_clubb_core`; ekman means bit-faithful. `sponge_damp_xp2`/`sponge_damp_xp3` (the variance/third-moment dampers) ARE ported + unit-tested (`tests/test_sponge_damp_xp23.py`) — name/file mirror complete — but **not wired** into the JAX `advance_xp2_xpyp`/`advance_wp2_wp3` (their wp2/wp3/up2_vp2 profiles aren't built in init, and no case enables the flags → no full-case oracle); `clubb_driver` fail-loud rejects the wp2/wp3/up2_vp2 sponge flags until that path is wired |
| `Microphys/cloud_sed_module.py` | `Microphys/cloud_sed_module.F90` | `cloud_drop_sed` (Stokes-regime cloud-droplet sedimentation, `l_cloud_sed`) — bit-faithful (`sed_rcm` ~1e-11); wired into the driver loop. Unblocked atex_long + dycoms2_rf02_so (Iter100) |
| `Microphys/KK_microphys/kk_microphys_driver.py` | (assembly) + `KK_microphys_module.F90:1196` | The three KK mass-tendency entry points `kk_autoconversion/accretion/evaporation_mean` (vs rico auto 4.7e-7, accr 6.1e-9, evap 3.3e-6); `kk_microphys_adjust` (rates -> rrm_mc/Nrm_mc/rvm_mc/rcm_mc/thlm_mc with source/evap over-depletion limiters); `compute_kk_microphysics` (the full standalone step: hydromet fields + PDF state -> tendencies). Differentiable |
| `Microphys/KK_microphys/KK_Nrm_tendencies.py` | `KK_microphys/KK_Nrm_tendencies.F90` | `KK_Nrm_auto_mean`, `KK_Nrm_evap_local_mean`, `KK_Nrm_evap_upscaled_mean` (vs rico Nrm_evap median 3.2e-6). **All KK rates ported+validated** (rrm auto/accr/evap, Nrm auto/evap, mvr) |
| `kk_microphys_driver.py::kk_sedimentation` | `KK_microphys_module.F90:1542` | KK mean sed velocities Vrr/VNr from the mean volume radius (KK00 Eq.37), clipped <=0, top zero-flux BC. **Bit-exact vs the rico oracle** (\|d\|max 1.1e-16 on rain points, via the bit-faithful `zt2zm`); differentiable. The V_hm input `advance_hydrometeor` needs |
| `Microphys/advance_microphys_module.py` | `advance_microphys_module.F90` | The full hydrometeor transport solve: `sed_centered_diff_lhs` + `term_turb_sed_lhs/rhs` (mean + turbulent sedimentation, flux-form), `microphys_lhs`/`microphys_rhs` (the implicit Crank-Nicholson tridiagonal: 1/dt + 1/2 diffusion_zt + term_ma_zt + sed + turb-sed), `advance_one_hydrometeor` (assemble + `tridiag_lu_solve`), and `calculate_K_hm` (the hydrometeor eddy diffusivity, capped at \|corr(w,hm)\|<=1). Verified by the conservation contract (~5e-15) + the rico `rrm_ma`/`rrm_ts` budgets; K_hm bit-validated vs the oracle's stored `K_hm_<hm>`. `tests/test_kk_rico_oracle.py` |
| `Microphys/KK_microphys/KK_upscaled_turbulent_sed.py` | `KK_microphys/KK_upscaled_turbulent_sed.F90` | `kk_sed_vel_covars` -- the rain sed-velocity covariances <V'r'>/<V'N'> (bivariate-lognormal, impc/expc semi-implicit split). **Bit-faithful-to-the-gate vs the rico oracle** (rel 4.5e-11, no timing confound); differentiable. Feeds `term_turb_sed_lhs` |
| `Microphys/KK_microphys/{parabolic_cylinder,PDF_integrals_means,KK_upscaled_means}.py` | `Microphys/KK_microphys/{KK_utilities,PDF_integrals_means,KK_upscaled_means}.F90` + `parameters_KK.F90` | **Complete upscaled-KK analytic means library** -- all 4 means (auto/accr via bivar_NL, evap via trivar_NLL over the chi<0 half, mvr via bivar_LL), built on the parabolic-cylinder D_v (1F1 series + optimally-truncated asymptotic, accurate float64 everywhere — the do/ds `epss=1e-4` `expax` reproduction was removed in the REFACTOR, A1). Verified vs scipy.pbdv + brute-force quadrature (<=3e-11); all jitted + differentiable |
| `Microphys/Morrison_microphys/module_mp_graupel.py` | `Morrison_microphys/module_mp_graupel.F90` | **Complete Morrison 2-moment (M2005) port.** Special functions `polysvp`/`derf1`/`gamma` (bit-exact / vs scipy ~1e-15). All process rates -- warm-rain (KK bulk PRC/PRA + evap PRE) + the full ice block (deposition/sublimation, snow/graupel collection, aggregation, nucleation, freezing, self-collection, melting), oracle-validated to a few % (timing confound) or bit-exact. The single-column step (`compute_m2005_rates` -> `m2005_step_tendencies`, cold/warm branches selected per level by T>=273.15 + the PCC saturation adjustment) verified by the **water-conservation contract** (sum mass tendencies ~1e-21). Sedimentation (`morrison_sedimentation`: rain/ice/snow/graupel/cloud, shared-NSTEP CFL, conservation-verified; vertical=LAST axis; the CLUBB<->M2005 grid index FLIP) + pre-rate slope clamps + the `morrison_microphys_driver` CLUBB interface (`hydromet_mc=(field_final-field_initial)/dt`; `thlm_mc` via the `real*4` round-trip, see the precision lesson). Wired via `morrison_microphys_step.py` (gated on `microphys_scheme=='morrison'`). Runs float64 except the deliberate single-precision interface casts. `tests/test_morrison_{special,rates,differentiable}.py` |
| `T_in_K_module.py` | `T_in_K_module.F90` | `thlm2T_in_K` — bit-exact (`calculate_thvm` moved to its Fortran home calc_pressure.py, mirror-refactor) |
| `calc_pressure.py` | `calc_pressure.F90` | `init_pressure` (via `jax.lax.scan`) + `calculate_thvm`. (`hydrostatic`/`inverse_hydrostatic`/`calc_ref_z_*` are in `Input_fields/hydrostatic_module.py` ↔ `hydrostatic_module.F90`, mirror-refactor) |
| `parameters_tunable.py` | `parameters_tunable.F90` | `init_clubb_params`, `calc_derrived_params` — bit-exact |
| `model_flags.py` | `model_flags.F90` | `get_default_config_flags` — all 88 flags |
| `numerical_check.py` | `numerical_check.F90` | `parameterization_check`, `check_clubb_settings` (`check_parameters` moved to its Fortran home parameters_tunable.py, mirror-refactor) |
| `Benchmark_cases/arm.py` | `arm.F90`, `prescribe_forcings.F90`, `time_dependent_input.F90`, `sfc_flux.F90`, `diag_ustar_module.F90` | Full ARM forcing (Monin-Obukhov, time-interpolated) |
| `io/stats_writer.py` | `stats_netcdf.F90` | Pure Python NetCDF stats output (StatsWriter) — bit-exact |
| `advance_clubb_core_module.py` | `advance_clubb_core_module.F90` | Full ARM timestep — **zero Fortran calls** |
| `src/clubb_standalone.py` | `clubb_standalone.F90` | Thin CLI frontend (argv → `run_clubb`) — mirrors the 88-line Fortran program. Entry point for `-jax` |
| `src/clubb_driver.py` | `clubb_driver.F90` | `run_clubb` (init → advance → cleanup) + `init_clubb_case` + `clean_up_clubb` — **zero Fortran API imports** |
| `src/derived_types/` | `clubb_python_api/clubb_python/derived_types/` | Pure-Python mirrors: ConfigFlags, ErrInfo, SclrIdx, Grid, pdf_parameter, implicit_coefs_terms |
| `src/Radiation/radiation_module.py` (+ `cos_solar_zen_module.py`, `rad_lwsw_module.py`, `simple_rad_module.py`) | `radiation_module.F90` (+ `cos_solar_zen_module.F90`, `rad_lwsw_module.F90`, `simple_rad_module.F90`) | radiation dispatch (renamed from `radiation.py`, mirror-refactor iter 111); `cos_solar_zen` / `sunray_sw` / `simple_rad`+`simple_rad_lba` now in their own same-named modules (mirror-refactor) — **zero Fortran imports** |
| `advance_clubb_to_end.py` | `clubb_driver.F90` (`advance_clubb_to_end` subroutine) | Timestep loop: stats + forcing → advance → stats. Kept in its own submodule for size; imported by `run_clubb`. **Zero module-level Fortran imports**; `prescribe_forcings` uses lazy import for non-ARM |

**ARM per-timestep Fortran calls: ZERO.** All prognostic state, diagnostics, forcings, and
stats output are pure JAX/Python.

**Module-level Fortran dependency status:**
- `clubb_standalone.py` / `clubb_driver.py`: zero (`derived_types` now local)
- `advance_clubb_to_end.py`: zero (lazy import only in non-ARM `_prescribe_forcings`)
- `radiation.py`: zero (`cos_solar_zen` and `sunray_sw` ported to pure Python)

---

## Cross-case bit-faithfulness status (vs Fortran, `compare_cases.py`)

"Runs" ≠ "bit-faithful". A case can run end-to-end in JAX yet diverge from the Fortran
oracle. Verified status (prognostic, rel tol 1e-6):

| Case | Status | Notes |
|---|---|---|
| arm | ✅ PASS (225 steps) | bit-faithful reference |
| wangara | ✅ PASS (30) | land case; uses `l_modify_bc_for_cnvg_test` 25 m BC |
| gabls2 | ✅ PASS (30) | land case |
| bomex | ✅ PASS (30) | fully bit-faithful (Iter79 BC + Iter80 thv + Iter81 rc_coef_zm) |
| dycoms2_rf01 | ✅ PASS (30) | fully bit-faithful (Iter80 thv + Iter81 rc_coef_zm) |
| atex | ✅ PASS (30) | bit-faithful since Iter84 (xm monotonic flux limiter wired in) |
| gabls2 | ✅ PASS (30) | bit-faithful (instantaneous output) |
| gabls3_night | ✅ PASS (30) | bit-faithful since Iter86 (um_f/ug time-dependent wind forcing) |
| fire | ✅ PASS (30) | bit-faithful since Iter87 (bulk surface scheme, sfctype=1) |
| neutral | ✅ PASS (30/55) | bit-faithful since Iter91 (neutral_case_sfclyr: ustar=0.5 + momentum flux) |
| ekman | ✅ PASS (30) | bit-faithful since Iter94 (sponge damping Iter93 + `ice_supersat_frac` at the cold 10 km top, Iter94) |
| cobra | ✅ PASS (40) | bit-faithful since Iter94 ice fix (Iter87-89 surface/wind/subsidence + the cold-cloud `ice_supersat_frac`); confirmed Iter96. Its step-14 cloud onset (T=266-270 K) was the SAME ice bug, not FP-boundary |
| dycoms2_rf02_nd | ✅ PASS (30) | bit-faithful (Iter96). "_nd" = **no drizzle** — a standard stratocumulus case, NOT the drizzle variant; was mislabeled blocked |
| dycoms2_rf01_fixed_sst | ✅ PASS (30) | bit-faithful (Iter98). dycoms2_rf01 variant with fixed SST surface |
| atex_long | ✅ PASS (30) | bit-faithful (Iter100). Unblocked by the `cloud_drop_sed` port (`l_cloud_sed`) |
| dycoms2_rf02_so | ✅ PASS (30) | bit-faithful (Iter100). Unblocked by `cloud_drop_sed`; do/ds variants still need drizzle microphysics |
| jun25_altocu | ✅ PASS (30) | bit-faithful (Iter188). Cold-cloud altocumulus + "simplified" radiation; unblocked by the per-step `wm_zm` (subsidence) recompute fix |
| gabls3 | ✅ PASS (30) | **bit-faithful (Iter273-274, 17th case)** — 0 prognostic failures at the full 30-step gate. Full BUGSrad correlated-k radiation + interactive soil_vegetation + gabls3 surface flux + omega subsidence. **bugs_rad is jitted (Iter274)** — fixes the eager-dispatch ~700 MB/call OOM-after-6-steps + ~2.4× faster (~6 s/step, JAX 30-step run 194 s); configured run is 1440 steps (24 h) |
| mpace_a | ✅ PASS (30) | bit-faithful (Iter299, 18th case). Morrison (l_ice_microphys) but clear/sub-saturated; the only Morrison signal is the clear-air single-precision thlm_mc round-trip residual |
| clex9_nov02 | ✅ PASS (30) | **bit-faithful (Iter313, 19th case)** — CLEX-9 cold-cloud altocumulus. Morrison configured but `microphys_start_time` (51411 s) is beyond the 30-step window → never activates; prognostically bit-exact + Tier-C clean once the pre-activation Ncm/Nc_in_cloud diagnostic was fixed to match `advance_microphys`'s early return |
| clex9_oct14 | ✅ PASS (30) | **bit-faithful (Iter313, 20th case)** — sibling of clex9_nov02 (same campaign, same Morrison pre-activation window) |

**Bit-faithful: 20 cases** (the table above; all pass `compare_cases.py` at 30 steps and the durability gate at
100). 9 are bit-faithful for their ENTIRE configured run (dycoms2_rf01, cobra, bomex, neutral, dycoms2_rf02_nd,
dycoms2_rf02_so, wangara, atex, dycoms2_rf01_fixed_sst). mpace_a (Iter299) is the first Morrison case made
faithful -- it stays clear/sub-saturated, so the only M2005 signal is the clear-air single-precision `thlm_mc`.

**48-case coverage (Iter153 survey): 20 run, 28 unsupported, 0 hard crashes.** Each of the 28 is blocked by one
unported SUBSYSTEM (the `_check_unsupported_features` message names it). NB the Iter153 "morrison microphysics --
~19 cases" lever is now **stale**: M2005 is a COMPLETE port (warm-rain + full ice/snow/graupel; the driver advances
rgm/Ngm as prognostic species — iter-373 reverified against morrison_microphys_module.py, so l_ice_microphys /
l_graupel / l_arctic_nucl are all supported). Those ~19 cases are now blocked by **bugsrad radiation** + **SILHS
interactive sampling** (not bit-reproducible vs the Fortran RNG -- not a target), NOT by the microphysics. The clean
radiation win was gabls3 (now faithful); **COAMPS microphysics** blocks arm_0003.

**Characterized not-bit-faithful cases (do NOT chase as bugs -- each is numerically/FP-limited, often because the
JAX is MORE accurate than the low-accuracy Fortran defaults):**
- **rico** (grid_type=2, KK): bit-faithful steps 1-4; from step 5 the near-zero rt-flux clip at the stretched dry
  top amplifies FP-level `rtp2` diffs (the dry-top rtp2 sits at the rt_tol^2 floor -- matching ~0 to rel-1e-6 is
  impossible). Grid verified bit-exact (Iter306); the step-1 seed (missing rtm/thlm `fill_holes`) was fixed Iter186.
- **coriolis_test**: an analytic Foucault-pendulum benchmark that zeroes nearly all closure constants and needs the
  nontraditional-Coriolis flag; the undamped oscillator accumulates FP noise (no decay). Step-1 faithful, no seed.
- **nov11_altocu** (Morrison + ice + SW radiation): bit-faithful through step 5; step 6 is the ice-cloud-edge FP
  floor (the `ice_supersat_frac` erf at near-zero scalar variance, then the `/0.001` Lscale ramp = a 1000x
  amplifier) -- every link verified faithful. Microphysics activates at step 60 (gated by `microphys_start_time`),
  past the FP floor, so the M2005 transport is validated by unit tests, not the full run.
- **dycoms2_rf02_morr** (warm Morrison, active from step 1): the M2005 hydrometeor transport (K_hm + sedimentation)
  is verified ~bit-exact (K_hm matches the oracle's stored `K_hm_<hm>`), but a tiny near-singular `rcm_mc` residual
  at the sharp cloud-top CF3D edge (the in-cloud /CF3D <-> grid-mean *CF3D conversion where cloud_frac->0) plus the
  M2005 single-precision floor keep it off the gate.
- **dycoms2_rf02_do / _ds** (KK, drizzle): the KK rt/thl covariance is physically correct but cancellation-amplifies
  the parabolic-cylinder `D_v`. The SCM oracle runs `parab` at `epss=1e-4`; the JAX uses the accurate float64 `D_v`,
  so the bit-gap WAS the oracle's deliberate low accuracy (proven with oracle numbers, Iter310). **REFACTOR A1
  (iter7): the `expax` reproduction of the oracle's epss=1e-4 artifact was DELETED** (`parabolic_expax.py` removed) —
  the JAX is now simply more accurate than the low-accuracy oracle. do/ds are not bit-faithful by design and are
  judged under Tier-C (dynamics) / Tier-D (drizzle hydrometeors), not against the oracle's imprecision.

**Durable lessons (the conventions these investigations produced):**
- **NEVER trust a default-vs-computed value** -- a Fortran line `x=default` may be overwritten later; verify the
  actual computed quantity (an unverified `precip_frac=1` cost 12 iterations chasing a 2x K_hm bug).
- **Decouple-the-oracle before blaming a subsystem** -- feed the Fortran's own field into the JAX subsystem (e.g.
  Fortran cloud into `_simple_rad_lw`) to exonerate it; "steep radiation" was a red herring for jun25 (the real
  seed was a stale per-step `wm_zm` -- a grid-staggered partner not recomputed when its `_zt` counterpart updated).
- A `*_forcing` stat that disagrees in a microphysics case is often `raw_forcing + lagged *_mc`, not a forcing bug;
  an isolated oracle debug build (`compile.py -install <scratch> -debug`) is the decisive tool when static analysis
  stalls. A `-jax` run dying with a Fortran `error stop`/no Python traceback is the unported-case `clubb_api`
  fallback -- port the case's tndcy/sfclyr to `prescribe_forcings.py`.

---

**★★ Numerical-accuracy refactor — COMPLETE (both criteria met).** **(b) Faithful:** all 20 `compare_cases`
DEFAULT_CASES PASS Tier-C (`--tier physical --max-iters 30`) — 19 stay strictly bit-faithful (0 prognostic
failures), mpace_a passes within tolerance (the intended A2 reclassification: float64 `thlm_mc` is more accurate
than the Fortran single-precision artifact). The accuracy-lowering contrivances (A1 expax, A2 Morrison real*4,
A3 BUGSrad sngl/float32-π) were removed and the differentiability work (B2–B5) was all forward-identical, so the
suite has ZERO faithfulness regression; Tier-B goldens baselined for all 18. **(a) Differentiable:** whole-driver
`jax.grad` through one `advance_clubb_to_end` step is finite + finite-difference-correct for **all 19 cases**
(`compare_grad.py`); see "Differentiability status". The sole case outside the faithful suite is **rico**, whose
KK rain-microphysics *transport+feedback* is a deliberately staged, gated-off port (`l_kk_micro_apply` default
off) — a pre-existing incomplete subsystem, not touched by this refactor.

---
## Remaining Work

**★ Achievable-state assessment -- read before picking the next piece.** The non-subsystem bit-faithful
frontier is nearly saturated (20 cases as of Iter313). Most remaining gains need a LARGE subsystem port with
poor ROI, because the cases they unblock are themselves numerically-limited (see the characterized cases above).
**BUT (Iter313) the frontier was NOT fully saturated:** clex9_nov02/oct14 were "unported" only because their
Morrison scheme never activates in the gate window — they are pure closure physics and were bit-faithful all
along, blocked only by a diagnostic-output mismatch (pre-activation Ncm). Lesson: before declaring a case
blocked by an unported subsystem, check whether that subsystem actually *runs* in the gate window. **Do NOT
chase the genuinely numerically-limited microphysics cases (rico, dycoms2_rf02_do/ds) as "bugs" -- they are
characterized.** Full 48-case completion is gated by Fortran numerical limits plus impractical ports.

**★ Completeness loop — final state (Iters 1–33).** A 33-iteration sweep ported every remaining **in-scope,
oracle-validatable, self-contained** routine and unit-tested each (differentiable; oracle = f2py bit-shadow
where exposed, else closed-form / Monte-Carlo / round-trip / literal-transcription). Highlights: the entire KK
PDF-integral mixed-moment machinery (`mixed_moment_PDF_integrals` + `PDF_integrals_all_MM`, both ✅), the BUGSrad
cloud-overlap (`cloud_correlate`, both subroutines → Radiation 100% ported), `ice_dfsn`, the GFDL droplet-activation
CLUBB-side (erff/updraft_weights/aer_act_clubb_ndrop), `inverse_hydrostatic`, the CGILS pressure-coordinate /
`T_f` / `um_ref` forcing-reader capability (guarded, gated cases byte-identical), and **all benchmark-case
surface/forcing schemes** (lba, mpace_b, arm_97, twp_ice, arm_3year, arm_0003, cloud_feedback). Verified
regression-free: 10/20 gated cases re-confirmed across every type (forcing-pipeline, sounding, analytic, cloud-sed,
Morrison); a test-infrastructure shadowing bug was found+fixed (iter 31).
The **genuinely remaining unported `.F90` (3 files, all impractical/out-of-scope)**: `coamps_microphys_driver`
(7000-line alternative microphysics the gated config never uses; the Fortran itself fatal-errors on `l_predict_Nc=F`
→ **no oracle**), `gfdl_activation`'s `aer_ccn_act_wpdf_k` lookup core (the ➖ `SCM_Activation` subsystem —
Gauss-Hermite + Köhler + 5-D single-precision lookup, no case exercises it), and `pdf_hydromet_microphys_wrapper`
(would wire `hydrometeor_mixed_moments` to compute `wp2hmp`/`rtphmp`/`thlphmp` — but those are **correctly zero for
all 20 gated cases** (no active hydrometeors), so the wiring has **zero validated payoff** and needs a
setup_pdf_parameters correlation-processing port; deferred). SILHS sampling is ➖ (a different RNG can't be
bit-matched). The differentiable+faithful JAX port is **complete for all tractable/in-scope code.**

**★ Post-loop completeness extensions (iters 81–96).** A further sweep closed the last in-scope, oracle-
validatable routines and extended faithfulness to the CGILS family:
- **All alternative PDF closures** end-to-end f2py-validated: ADG2, LY93, 3-D Luhar, new-TSDADG, new-pdf, and the
  full **new-hybrid driver** (`new_hybrid_pdf_main.py`, 1.15e-14). Plus `mirror_lower_triangular_matrix`, the
  Godunov-upwind `xpyp_term_ta`, and `sponge_damp_xp2/xp3` (all f2py bit-exact).
- **`remapping_module.F90` fully ported** (both methods): Ullrich-linear (eq. 30) + the E3SM **PPM** (map1_ppm/
  ppm2m/steepz/kmppm); f2py same-grid bit-exact + mass-conservation-rel-0 on a refined grid (`remapping_module.py`).
- **CGILS/cloud_feedback init+radiation fixed** → **cgils_s11 reaches Tier-C PASS** (was rel ~1e3): the Press[Pa]
  sounding→altitude conversion (`convert_pressure_sounding_to_z`), the absolute-temperature `T[K]`→θ init
  (clubb_driver.F90:5499-5524), and the case-specific radiation extended atmosphere from the deep sounding + ozone
  sounding (`convert_snd2extended_atm` → `build_case_extended_atmosphere`, gated on `l_use_default_std_atmosphere=
  .false.`). thlm is now bit-exact at init/step1; the residual is FP-limited (cloud-topped-BL chaos). Added to the
  `compare_cases.py --cases tier_c` physical-fidelity suite. All gated cases byte-untouched (the new paths are
  gated on Press[Pa]/T[K]/the std-atm flag). **Iter97** then fixed a systematic forcing-reader bug affecting the
  whole family: `_parse_forcings_file` edge-extrapolated the forcing outside its vertical range, but the Fortran's
  `zlinterp_fnc` (via read_to_grid) **zero-fills** — so cloud_feedback's out-of-range bottom levels got a spurious
  ≈−1.6e-5 thlm forcing. With `left=right=0` the cloud_feedback means → Tier-C PASS (moments now FP-limited at cloud
  onset); gated file-forcing cases (gabls3_night/…) byte-identical (their forcings cover the model range).
- **Last validation checks** ported (`assert_corr_symmetric`, `sfc_varnce_check`) — these have no observable f2py
  oracle (err_code not exposed), validated by transcription/behavior.
The genuinely-remaining unported `.F90` are unchanged (COAMPS, GFDL lookup core, pdf_hydromet_microphys_wrapper,
SILHS RNG) — all no-oracle/zero-payoff. **No in-scope, oracle-validatable Fortran routine remains.**

**★ Mirror-refactor loop (file/routine-name fidelity, ongoing).** A separate sweep (logged per-iteration in
CHANGELOG.md) makes the JAX *file and routine names* mirror the `.F90` oracle, without changing any numbers
(every step byte-identical: 5-case bit-faithful + 100-step durability + whole-driver `jax.grad`). Done so far:
(a) **relocated standalone subroutines** inlined in advance_clubb_core to their Fortran-home modules —
`diagnose_upxp`→advance_xm_wpxp, `calc_wp3_on_wp2`→advance_helper, `set_Lscale_max`→mixing_length,
`set_sfc_value_of_flux_profiles` (sibling fn in advance_clubb_core_module.py); (b) **un-inlined the pdf_closure
component physics** into pdf_closure_module.py (`transform_pdf_chi_eta_component`, `calc_{liquid,ice}_cloud_frac_
component`, `calc_xprcp_component`, `calc_xpthvp_terms`, the higher-order-moment integrals) and the advance_xp2_xpyp
LHS/RHS terms (`calc_{up2_vp2,xp2_xpyp}_lhs`, `calc_up2_vp2_rhs`, `solve_xp2_xpyp`, and the post-solve
`pos_definite_variances` hole-filling — iter 94, relocated from a local helper in advance_clubb_core); (c) routed hand-rolled inlines
back to their modules (`compute_gamma_Skw`→Skx_module, `clip_variance`/`clip_covar`→clip_explicit,
`mono_cubic_interp`→interpolation, `calc_xpwp`→advance_helper) and **fixed mislocations** (clip_covar was defined in
advance_xm_wpxp; smooth_heaviside_peskin in mixing_length; calc_comp_corrs_binormal duplicated in adg1 — each moved
to its real F90-home file); (d) **de-duplicated** the `_safe_sqrt`/`_ssqrt`/`_dvc`/`mean_vert_vel_up_down` copies to
one canonical definition; (e) **retired ALL iteration-tag / shadow-comparison local names** (iters 62, 65–74:
the `_jNN` ADG1-input tags → `_adg`; the `_11`/`_12`/`_13`/`_37` main-advance-loop shadow tags → routine-named
`_xw`/`_w23`/`_we`/`_uv`; the `_10`/`_36` xp2_xpyp blocks → `_x2`/`v2`; the `_60`/`_61`/`_69`/`_17` Block-U / stats
tags → `padv`/`_dg`/`_sf`; the small `_21`/`_24`/`_39`/`_68` families → descriptive names) — a whole-`src` scan
(iter 75) confirms none remain, and the vestigial `_advance_clubb_core`→`_python` shadow-dispatch wrapper was
collapsed; (f) **mirrored `compute_diagnostic_cache`** (iter 78, advance_clubb_core_module.F90:1752 →
`compute_diagnostic_cache`: thvm + em/sqrt_em_zt + ddzt_umvm_sqd, pulled out of the scattered Blocks I/J/K) —
**so advance_clubb_core_module.py now mirrors all three subroutines of its F90 file** (advance_clubb_core,
compute_diagnostic_cache, set_sfc_value_of_flux_profiles); (g) **moved the pdf_closure_driver body's physics into
pdf_closure_module.py** (iters 76–82: `_adg1_pdf_driver_zt`, `calc_pdf_liquid_cloud_frac_{jax,_components_jax}`,
`calc_pdf_ice_supersat_frac_jax`, `calc_pdf_xprcp_fluxes_jax`) so the post-advance Block U is now a *sequence of
pdf_closure_module routine calls* + plumbing. **Both of these have since been fully un-inlined** (this note
predates iters 139–142): the post-advance Block-U closure is now the module-level driver `pdf_closure_driver`,
un-inlined from advance_clubb_core via AST free-variable analysis and **relocated into pdf_closure_module.py**
(iters 141–142, its Fortran home) — advance_clubb_core now *calls* it, mirroring the Fortran
`call pdf_closure_driver`; and the whole 5-moment up2/vp2/rtp2/thlp2/rtpthlp solve + clips + interleaved budget
stats is the module-level driver `advance_xp2_xpyp`, un-inlined and **relocated into advance_xp2_xpyp_module.py**
(iters 139–140). So advance_clubb_core no longer inlines either driver. (The one remaining gated non-mirror is
`pdf_closure_driver_zm` — the second, zm-grid PDF closure, `l_call_pdf_closure_twice` off, no validated case;
fail-loud rejected at the driver.)
The xp2_xpyp budget-finalize stats correspond to the Fortran subroutine `stats_finalize_xp2_xpyp_terms`
(F90:2925); its shared 3-band-LHS-apply kernel is now the module routine `apply_lhs_band3_interior_jax`
(mirror-refactor iter 122), but a byte-identical `stats_finalize_xp2_xpyp_terms_jax` mirror is **not** available —
the JAX's per-group budget math diverged from the Fortran's uniform finalize (rtp2/thlp2/rtpthlp use
`dp1_ref*(threshold-mix)` for dp1; up2/vp2 use the C4/C14 split — two `term_pr1` calls with one coefficient zeroed,
mirroring the Fortran (iter 580); only dp2 is uniform), so unifying all five would change budget values.
(h) **per-case Benchmark_cases split** (iters 95–110): drove each case's forcing/surface routines out of the
monolithic `prescribe_forcings.py` into a per-`.F90` module mirroring the Fortran file + `use <case>` dispatch —
`spec_hum_to_mixing_ratio.py`, `sfc_flux.py`, `gabls3_night.py`, `bomex.py`, `dycoms2_rf01.py`, `wangara.py`,
`gabls2.py`, `gabls3.py`, `atex.py`, `atex_long.py`, `fire.py`, `dycoms2_rf02.py`, `mpace_a.py`, `rico.py`
(plus the pre-existing `arm.py`/`lba.py`/`arm_97.py`/`arm_0003.py`/`mpace_b.py` etc.). All byte-identical;
prescribe_forcings.py now imports each routine. The only Benchmark_cases routines still folded in prescribe_forcings
are the dispatcher-duplicate `_lba_sfclyr`/`_arm_variant_sfclyr` (whose pure routines are already in lba.py /
arm_97.py / arm_0003.py) — left as-is because those cases are SILHS-blocked and can't be gate-validated.
The shared `sfc_flux.F90` helpers were progressively extracted into `sfc_flux.py` and routed back: the
`convert_*_ht_*` unit conversions (iter 164) and the bulk-aerodynamic `compute_wpthlp_sfc`/`compute_wprtp_sfc`
(iter 180; the 8 case `*_sfclyr` paths that re-inlined `-Cd·ubar·(thlm−T_sfc/exner)`/`-Cd·ubar·(rtm−adj)` now call
them, mirroring the Fortran `use sfc_flux`) — byte-identical, only `compute_ht_mostr_flux` stays case-folded.
(i) **file-name + gated-subsystem mirroring** (iters 111–115): renamed `Radiation/radiation.py` →
`radiation_module.py` (mirrors radiation_module.F90); relocated each Fortran subroutine that the JAX had inlined
in the *wrong* module to a new file matching its `.F90` home — `KK_microphys_module.py`
(kk_microphys_adjust/kk_sedimentation, out of KK_microphys/kk_microphys_driver.py), `extended_atmosphere_module.py`
(determine_extended_atmos_bounds, out of bugsrad_driver.py), `morrison_microphys_module.py`
(morrison_microphys_driver, out of Morrison_microphys/module_mp_graupel.py), `CLUBB_core/calendar.py`
(gregorian2julian_day/leap_year/compute_current_date, out of cos_solar_zen_module.py). All byte-identical, each
validated by its unit test (test_bugsrad / test_morrison_rates / test_kk_rico_oracle) or the bit gate. A whole-tree
import sweep (iter 116) confirms all 135 src modules import clean. **The clean, validatable file/routine
relocations are now exhausted** — the residual non-mirroring is just intentional groupings (derived_types/pdf_params.py,
the advance_clubb_to_end glue) plus the gated `pdf_closure_driver_zm`. (The two then-entangled advance_clubb_core
wrappers this note originally flagged — `pdf_closure_driver` and `advance_xp2_xpyp` — were **subsequently un-inlined and
relocated** to pdf_closure_module.py / advance_xp2_xpyp_module.py at iters 139–142; advance_clubb_core now *calls* both,
mirroring the Fortran `call pdf_closure_driver` / `call advance_xp2_xpyp` (verified iter 458). They are no longer
residual.) (**The BUGSrad
files are NOT a divergence** — re-verified iter 456: each JAX BUGSrad file mirrors a real oracle file+subroutine
(`two_rt_lw.py` ↔ `two_rt_lw.F:two_rt_lw`, etc.), namely the variant the *default* config actually calls — `bugs_lwr.F`
guards `call two_rt_lw` (default) vs `call two_rt_lw_gsolap`/`_sel`/`_iter` (alternative cloud-overlap schemes behind
flags the standard build does not set); the `*_ocastrndm.F90` files hold the unported `two_rt_lw_gsolap`/`two_rt_sw_gsolap`
alternatives — correctly out of scope, like the SILHS RNG. The default path is what the bit-faithful BUGSrad cases run.
**Audit-robustness fix (iter 696):** `mirror_audit.py`'s `_ROUTINE` regex required a `(` or newline after a routine
name, so the `*_ocastrndm.F90` `two_rt_{lw,sw}_gsolap` declarations — written with a *continuation* header
(`subroutine NAME      &`) — were invisible to the audit, and so were their files. Broadening the terminator class
to `[(\n\r&]` makes the audit parse them and correctly surface both files in the scoped-out `bugsrad_altsolver`
bucket — the tool now *enforces* the out-of-scope classification this paragraph documents, rather than silently
not-seeing it. A bidirectional naive-vs-audit extraction re-diff (iter 697) then proved extraction complete on both
sides (0 Fortran routines unseen, 0 JAX defs uncounted).)
(j) **iteration-tag cleanup + the last advance_clubb_core helper relocations** (iters 117–120): stripped the
jax-only `IterNN:`/`IterNN shadow:` development-history comment tags (no Fortran analog) — 46 from
advance_clubb_core_module.py (iter 117) + 13 tree-wide (iter 118); comment-only, byte-identical. Then relocated
the final two cleanly-extractable inline helpers from advance_clubb_core to their Fortran homes:
`adg1_pdf_driver_zt_jax` (the regrid-zm→zt + ADG1_pdf_driver call) → `pdf_closure_module.py` (iter 119, mirrors
the ADG1 invocation inside pdf_closure_module.F90:pdf_closure_driver) and `apply_sponge_field_jax` (mean-field
sponge damping) → `advance_xm_wpxp_module.py` (iter 120, the sponge block at the tail of advance_xm_wpxp_module.F90).
Both byte-identical, bit gate PASS (iter 120 incl. ekman, `l_sponge_damping=.true.`). Then (iter 121) removed the
jax-only `_apply_mfl` partial-application closure, replacing its 4 call sites with direct
`monotonic_turbulent_flux_limit(...)` calls — mirroring the Fortran's per-field direct calls to
`monotonic_turbulent_flux_limit` (byte-identical; bit gate incl. atex, where the limiter fires). Likewise (iter
123) removed the jax-only `_pos_definite_clip_variance` combo-wrapper, replacing its 2 up2/vp2 call sites with the
two distinct Fortran calls it bundled — `pos_definite_variances` then `clip_variance` (byte-identical). The
shared implicit-budget-finalize kernel `_mm3` was relocated to advance_xp2_xpyp_module.py as
`apply_lhs_band3_interior_jax` (iter 122). The one remaining inline helper, `_clip_variance`, is the thin
tracer-convention adapter for the single Fortran `clip_variance` (6 sites) — kept, not a combo-wrapper.
(k) **budget-finalize kernel consolidation + the single_lhs solve-driver** (iters 122–129): every repeated
budget-finalize stencil across the three advance branches is now a shared, named kernel in
advance_xp2_xpyp_module.py — `apply_lhs_band3_interior_jax` (3-band implicit `lhs@field`, **15 sites**: xp2_xpyp 10,
wp2_wp3 2, xm_wpxp 3; iters 122/124/125), `apply_lhs_band2_zt2zm_interior_jax` (2-band zt→zm, wp2_ta + wprtp/wpthlp_tp,
iter 126), and `finalize_implicit_budget_interior_jax` (diagonal `rhs - lhs*field`, 5 wp2/wp3 pr1/pr2/dp1 sites, iter
127) — all byte-identical (the kernels are tracer-transparent, so the iter-128 differentiability gate confirmed the
grad is unperturbed). The rtp2/thlp2/rtpthlp solve is driven by `solve_xp2_xpyp_with_single_lhs` (iter 129,
mirroring the Fortran `solve_xp2_xpyp_with_single_lhs` call, F90:664). The then-remaining structural gap — the
inlined `advance_xp2_xpyp` solve → a single `advance_xp2_xpyp` driver (byte-identical only with the budget stats
moved *inside* it, returning the 5 variances, the ~25 solve internals staying local — the JAX's restructured budget
math blocks the Fortran solve/`stats_finalize_xp2_xpyp_terms` split), plus the pdf_closure_driver Block-U glue — was
**subsequently closed**: `advance_xp2_xpyp` extracted to advance_xp2_xpyp_module.py (iters 139–140) and
`pdf_closure_driver` to pdf_closure_module.py (iters 141–142). advance_clubb_core now *calls* both.
(l) **solve-wrapper mirroring + windm module + name/alias cleanups** (iters 131–138): added the named
Fortran-mirroring solve wrappers `xp2_xpyp_solve` (131), `xm_wpxp_solve`/`wp23_solve` (132),
`windm_edsclrm_solve` (133) so **all four advance branches route their solves through a named wrapper** (none
calls the generic LU solver directly from the advance code); mirrored the windm sub-routines `windm_edsclrm_rhs`
/`compute_uv_tndcy` (134, the former renamed from `_windm_rhs_jax`) and `windm_edsclrm_lhs` (136); renamed
`compute_shared_xm_wpxp_lhs_terms`→`calc_xm_wpxp_lhs_terms` (135); and removed the jax-only `_`-aliases that
masked Fortran-mirroring names (`smooth_max`, `term_ma_zt_lhs_centered_jax`, `advance_clubb_core`; iters 137–138).
**MILESTONE — the exercised modules' Fortran sub-routines are now fully name-mirrored**: a whole-tree sweep of the
advance/pdf/leaf modules + advance_clubb_core (which mirrors all 3 of its own F90 subroutines) + the leaf
`term_ma_*`/`xpyp_term_ta_*`/`clip_*`/`diffusion_*` found no remaining exercised name mismatches. (The windm module
is fully mirrored but **never exercised** — `l_predict_upwp_vpwp=True` default, no case override — so its routines
are byte-identical-by-construction + no-collateral-validated; its only un-mirrored subroutine is the unported
`windm_edsclrm_implicit_stats`.) **`advance_xp2_xpyp` is now DONE** (iters 139–140): the 424-line block was un-inlined into a module-level driver
via AST free-variable analysis (42 args, stats-inside, returns the 5 variances; advance_clubb_core now calls it
then does `clip_covars_denom`, mirroring the Fortran call chain) — byte-identical (bit gate + bomex grad) — **and
relocated to its Fortran-home file advance_xp2_xpyp_module.py** (iter 140, adding the constant/`clip_variance`/
`term_ma_zm_lhs` imports + the `_clip_variance` helper there). Full location + routine mirror achieved.
**`pdf_closure_driver` is now DONE too** (iter 141): the ~327-line Block-U post-advance PDF closure was
un-inlined into a module-level driver via the same AST analysis (34 args — incl. the 5 input-and-output fields
pdf_params/rtpthvp/thlpthvp/wp2thvp/wpthvp — `global _prev_adg1` for the cross-timestep carry, returns the 21
PDF-derived fields/moments); advance_clubb_core now CALLS it, mirroring the Fortran pdf_closure_driver call.
Byte-identical (bit gate + bomex grad). Iter 142 then made it **pure** (returns the ADG1 carry `_adg1` as a
22nd output instead of writing the `_prev_adg1` module-global; the caller does the tracer-guarded global write,
mirroring Fortran's stateless pdf_closure_driver) and **relocated it to its Fortran-home file pdf_closure_module.py**
(adding 8 imports there). **★ Both top-level driver wrappers are now extracted AND in their Fortran-home files**
(`advance_xp2_xpyp`→advance_xp2_xpyp_module.py, `pdf_closure_driver`→pdf_closure_module.py); advance_clubb_core
calls every top-level Fortran subroutine as a named JAX function in its proper module — no top-level inline driver
code remains.
(m) **wp2_wp3 post-solve tail folded into the driver** (iter 146): the ~195-line subroutine-tail orchestration that
stayed inline in advance_clubb_core's Block W — fill_holes (`fill_holes_vertical` +
`fill_holes_wp2_from_horz_tke`) + `clip_variance` + zm2zt + `clip_skewness` + the 21 budget
`stat_update`s — was moved **into `advance_wp2_wp3`**, so the JAX driver now does the complete work of its
Fortran namesake (Fortran's advance_wp2_wp3 calls fill_holes/clip/`stat_update` in-routine, not via a separate
`stats_finalize_*`). Verbatim block-move + name-remap: the caller's `_sd_w23['…']` stats-dict reads became direct
local references (those intermediates were already locals computed only to populate the dict);
`_wp2_pre_w23`→`wp2`/`_wp2_jax_w23_raw`→`wp2_new`/`dt_advance`→the `dt` arg (caller passes `dt=float(dt_advance)`,
so identical). The driver gained `flags`/`sfc_elevation`/`stats_writer`/`l_sample` args + fill_holes/clip_explicit/
grid + band-kernel imports, and **returns the clipped `(wp2, wp3, wp2_zt)`** instead of the raw solve + `stats_dict`.
Byte-identical (bit gate ProgFail 0, DiagFail unchanged) + bomex grad COMPLETE (5.4e-07, stats I/O guarded by
`l_sample` so the grad path is unperturbed).
(n) **xm_wpxp per-scalar clipping extracted to its named Fortran routine** (iter 147): the per-scalar post-solve
MFL + `fill_holes_vertical` + `clip_covar` that had been inlined twice in advance_clubb_core's Block V is now the
module routine `xm_wpxp_clipping_and_stats` (advance_xm_wpxp_module.py, mirroring F90:`xm_wpxp_clipping_and_stats`
:4410), called once per field — rt/thl scalars (iter 147) AND the um/vm wind components (iter 148: the routine gates
the mean-field fill_holes on `solve_type not in (MFL_UM,MFL_VM)`, mirroring the Fortran `solve_type/=um,vm` skip, so
all four advance_xm_wpxp post-solve clips route through the one named routine). Byte-identical (10-case bit gate
ProgFail 0 incl. atex/MFL-active + bomex grad COMPLETE). With the rt/thl clips + the iter-146 wp2_wp3 fill_holes both
moved out, advance_clubb_core no longer imports `fill_holes` at all (dead import removed); and with all four MFL
sites now inside the routine, `monotonic_turbulent_flux_limit` is no longer imported by advance_clubb_core either
— fill_holes + the limiter are `use`d only inside the advance-routine home modules now.
(o) **`xm_correction_wpxp_cl` mirrored, unit-tested, but config-gated-off** (iter 149): the Fortran
`xm_correction_wpxp_cl` (F90:5766) adjusts xm for the amount w'x' was clipped, but is gated on `l_clip_turb_adv`
which is OFF in the validated config. The covariance clip DOES fire here (it is not a no-op), so wiring the
correction into the live clip path fails the bit gate (ProgFail 16) — the bit-faithful JAX correctly omits it.
The routine is kept as a faithful, **unit-tested** (`tests/test_xm_correction_wpxp_cl.py` vs a NumPy reference —
f2py exposes only the whole advance_xm_wpxp) named mirror, NOT called from the live path (same
mirrored-but-not-exercised class as the windm module). `damp_coefficient` (F90:5990, the
`l_diag_Lscale_from_tau=.false.` Skw-damped C6/C7 path) is likewise unexercised by the gate cases. See memory
`xm-correction-wpxp-cl-gated-off`.
(p) **`calc_xm_wpxp_ta_terms` split into its own sibling routine** (iter 150): the ADG1 turbulent-advection LHS
operator `lhs_ta_wprtp` was computed inside `calc_xm_wpxp_lhs_terms`, but the Fortran computes it in a SEPARATE
call (`calc_xm_wpxp_ta_terms`, F90:1996) and passes it into the LHS assembly. Split it out into
`calc_xm_wpxp_ta_terms`; advance_clubb_core now calls the two as siblings (as the Fortran advance_xm_wpxp does)
and merges `lhs_ta_wprtp` into the shared-LHS dict. Byte-identical (bit gate ProgFail 0 + bomex grad COMPLETE).
(q) **`clip_skewness` split into the Fortran outer/core two-level structure** (iter 151): the JAX routine named
`clip_skewness` actually implemented the Fortran `clip_skewness_core` (the pure |Sk_w| clip). Renamed it to
`clip_skewness_core` and added the proper outer `clip_skewness` — the `wp3_cl`-budget wrapper that the
Fortran `clip_skewness` is. The wp2_wp3 fold now calls the wrapper (passing dt/stats_writer/l_sample) instead of
recording `wp3_cl` inline; byte-identical (bit gate DiagFail unchanged, so wp3_cl held; f2py clip_skewness test
retargeted to the core + still PASS; bomex grad COMPLETE).
(r) **mixing-length dispatcher `calc_Lscale` extracted** (iter 152): advance_clubb_core's inline Block L IS the
Fortran `calc_Lscale` (mixing_length.F90:19, dispatch on l_diag_Lscale_from_tau → diagnose_Lscale_from_tau |
calc_Lscale_directly). Relocated it verbatim (~90 lines, 26 AST-verified inputs, ~21 outputs returned as a dict the
caller unpacks) into mixing_length.py (its Fortran home), so advance_clubb_core calls one named mixing-length routine
instead of inlining the dispatch. mixing_length.py gained numpy/`_asarray`/`_xp`/`zt2zm`/`itaumax` imports for the
relocated orchestration; the Lscale/tau stat_updates stay at the caller. Byte-identical (10-case bit gate ProgFail 0,
DiagFail at baseline + bomex grad COMPLETE). The last large exercised structural item is the full
single-`advance_xm_wpxp` wind-path fold.
(s) **Leaf-routine name-mirror close-out + cumulative grad verification** (iters 151–157): split `clip_skewness` into
the Fortran outer/core two-level form (151), promoted the wp2_wp3 term-builders (154) + saturation vapor-pressure
helpers (155) + Skx LG-ansatz routines (156) + `calculate_spurious_source` (153) from private/imprecise names to the
precise public `_jax` Fortran-mirror convention, and extracted `calc_Lscale` (152). After an exhaustive name-gap
audit, **the in-scope EXERCISED file + routine name+location mirror is complete** — every exercised Fortran subroutine
has a same-named JAX function in its Fortran-home module (verified by the bit gate, which exercises the full advance
path). **★ The last in-scope structural item is now DONE (iter 160): the ~605-line `advance_xm_wpxp` whole-driver fold.**
Block V was relocated verbatim out of advance_clubb_core into a new whole-driver `advance_xm_wpxp` in
advance_xm_wpxp_module.py (62 AST-derived inputs, 9-field state dict return, the 2 nested closures moving with the
body), and the per-field function was renamed `advance_xm_wpxp` → `solve_xm_wpxp_with_single_lhs` (matching
what its docstring ports). advance_clubb_core now calls the one whole-driver routine (Block V collapsed ~605→~20
lines), mirroring the Fortran advance_clubb_core→advance_xm_wpxp chain. Byte-identical (10-case bit gate ProgFail 0,
DiagFail at baseline + bomex grad COMPLETE). **With this, every top-level Fortran advance/pdf subroutine
advance_clubb_core calls is a named JAX whole-driver in its Fortran-home module — the in-scope EXERCISED file +
routine name+location mirror is COMPLETE.** The only un-mirrored remainder is out-of-scope/unexercised/unported
(windm `_implicit_stats`, COAMPS, GFDL CCN, SILHS, edsclrm, gfdl/lookup saturation, non-ADG1 PDF variants, hydromet
setup) — none oracle-validatable. Cumulative
differentiability across the 146–157 streak: 5-case `compare_grad` grad-finite 5/5, PASS (dycoms2_rf01/cgils_s11 are
expected clip-point KINKs). Everything else un-mirrored is out-of-scope/unexercised (microphysics/SILHS setup,
gfdl/lookup saturation, debug checks, config-gated damp_coefficient/xm_correction_wpxp_cl). The then-inline Block-V code (pre-solve C6/C7/coef stats + the shared-LHS build + the full upwp/vpwp
wind-prediction path) and the full xm_wpxp solve-fold were **subsequently extracted** into the whole-driver
`advance_xm_wpxp` in advance_xm_wpxp_module.py (iter 160; advance_clubb_core now calls it). The other un-mirrored
items are the no-oracle/unported subsystems (windm `_implicit_stats`, COAMPS microphysics, the GFDL CCN lookup, SILHS
RNG, edsclrm) — all out of scope.
(t) **Benchmark_cases + time_dependent_input completion** (iters 180–202): finished the per-case/file mirror for the
forcing+surface subsystem. `sfc_flux.F90` fully mirrored (compute_ht_mostr_flux/compute_wpthlp_sfc/compute_wprtp_sfc +
the convert/momentum/ubar helpers — every bulk-aero surface flux now routes through the named sfc_flux routines). New
Fortran-home per-case modules: `neutral_case.py`, `ekman.py`, `cobra.py`, `astex_a209.py`, `nov11.py` (incl. the
previously-UNPORTED `nov11_altocu_rtm_adjust`); the dycoms2_rf01/rf02 + wangara `*_tndcy` extracted to their modules.
**`time_dependent_input.py` now mirrors the whole time_dependent_input.F90 lifecycle** — `time_select` +
`load_generic_forcings_data` (initialize) + the `_parse_forcings_file`/`_parse_sfc_file` table readers +
`apply_time_dependent_forcings`. `Input_fields/input_reader.py` holds the blank-fill numerics (linear_fill_blanks/
fill_blanks_two_dim_vars). Cross-module routines promoted to Fortran-mirror names (vertical_avg/integral,
smooth_heaviside_peskin, read_surface_var_for_bc, time_select); the BV-calc saturation + hydrometp2_zt relocated to
their Fortran homes; the `_lba_sfclyr`/`_arm_variant`/`_bulk_aero` JAX-only dispatch duplicates removed/routed to the
per-case modules. arm's forcing time-interp routes through `interpolation.linear_interp_factor`. All byte-identical
(per-iteration bit gates ProgFail 0; full suite 92/92 GREEN iter 198; 100-step durability iter 199). The residual
un-mirrored is the `_zero_flux_sfclyr`/`generic` JAX-only dispatch helpers (the zero-flux clex9/jun25 cases have no
Fortran sfclyr) + the Fortran-file read primitives (input_reader read_x_table, ➖ infra).
**★ Comprehensive subroutine-coverage audit (iters 206–207).** Scanned every `CLUBB_core/`, `Radiation/`, and
`Microphys/` `.F90` subroutine/function for a JAX counterpart. **No in-scope/exercised Fortran routine is unported.**
The only Fortran routines with no JAX mirror are: (a) generic-interface type variants the JAX collapses (grid_class
`gradzm_1/2`/`redirect_interpolated_*`, the smooth_min/max type family); (b) ➖ infra (LAPACK `*_wrap`, `index_mapping`,
`endian`, the `init_radiation`/`init_*` namelist readers the JAX reads via its own namelist path); (c) unported
*alternative* methods the gated config never selects (`fill_holes_smart_window`/`widening_windows`/`_wv`,
`plinterp_fnc`/`lin_interp_between_grids`); (d) out-of-scope DIAGNOSTICS (`module_mp_graupel.F90:calc_refl10cm`/
`rayleigh_soak_wetgraupel` — radar dBZ for the WRF post-processor, no CLUBB coupling, no oracle); (e) the
already-known no-oracle subsystems (COAMPS, SCM aerosol, SILHS RNG, GFDL CCN lookup). The BUGSrad JAX files
(`bugs_lwr`/`bugs_swr`/`bugs_rad`/`cloudg`/`comscp1`/`comscp2`/`gascon`) mirror the **upstream BUGSrad distribution**
file structure (finer than clubb_release's `two_rt_*_ocastrndm` consolidation; `comscp1.py`/`comscp2.py` split to match
the upstream `comscp1.F`/`comscp2.F` at mirror-refactor iter 247) — a deliberate, documented divergence.

**★ The strategic pivot (done) — the bit-faithful ceiling was an artifact of the *gate*, not the physics.**
Several "numerically-limited" cases were limited only because the JAX is MORE accurate than the low-accuracy
Fortran oracle, and a few modules existed solely to reproduce the oracle's imprecision (`parabolic_expax` at
`epss=1e-4`; the Morrison `real*4` casts; BUGSrad's `sngl`/float32). The **numerical-accuracy refactor** (this
branch) relaxed the gate to the tiered standard (see "Correctness standard"), which simultaneously simplified
the code, improved accuracy, and unlocked whole-driver `jax.grad`. **New work should judge correctness by the
tiered standard, not the bit-faithful frontier.**

**Subsystem status:**
- **KK microphysics (`khairoutdinov_kogan`)** -- COMPLETE and wired per-step: the full upscaled-mean/covariance
  analytic library + hydrometeor PDF setup + the `advance_one_hydrometeor` transport solve + `calculate_K_hm`.
  Unblocks rico, dycoms2_rf02_do/ds -- all three numerically-limited (above). The covar `expax` port closed the
  epss artifact.
- **Morrison 2-moment M2005 (`module_mp_graupel.py`)** -- COMPLETE: the special-function layer, all process rates
  (warm-rain + full ice block, oracle-validated), the single-column step assembly (water-conservation contract),
  sedimentation, the CLUBB<->M2005 interface, and the per-step wiring + hydrometeor transport. Runs float64 except
  the deliberate `real*4` interface casts. Faithful case: mpace_a; FP-limited: nov11, dycoms2_rf02_morr.
- **BUGSrad correlated-k radiation + `soil_vegetation`** -- COMPLETE and wired (`Radiation/BUGSrad/`,
  `bugsrad_driver.py`); `bugs_rad` is jitted (the eager dispatch leaked ~700 MB/call). gabls3 was the one
  clean radiation-only win (bit-faithful before the REFACTOR; now Tier-C). Notes: pass the constants the
  Fortran CALLER passes (constants_clubb grav/Cp, not BUGSrad's physconst); the build is `-Dradoffline
  -Dnooverlap -DCLUBB` (no ghost layer, simple two_rt called twice, `newexp` unused). **REFACTOR A3 (iter8):
  cloudg's deliberate float32 `sngl` truncation + float32-π were dropped (now float64) — ~1e-7 more accurate,
  within Tier-C; the JAX no longer reproduces those single-precision artifacts.**
- **COAMPS microphysics** (arm_0003) -- unported. **SILHS** interactive Latin-Hypercube sampling
  (rico_silhs/mpace_b/lba) -- random, not bit-reproducible; not a bit-faithfulness target.

**Microphysics-port roadmap (the pattern, for any future subsystem):** hydrometeor infrastructure first
(`hydromet_dim`, rrm/Nrm init, the `hydromet`/`wphydrometp`/`K_hm` transport) -- a rate can't be tested in-context
until the case RUNS (working rule 6) -- then the process rates (validatable via a case-stats oracle), then the
`advance_clubb_core` thv buoyancy coupling (the `{wpthvp,wp2thvp,thlpthvp,rtpthvp} -= thv_ds*...` hydrometeor terms,
a no-op for hydromet_dim=0). The f2py API exposes ZERO microphysics, so these are verifiable only by full-case
comparison once the whole subsystem is complete.

**Differentiability (secondary goal):** component-level DONE; full-timestep grad is blocked by the three coupled
items in the Differentiability status above (the all-or-nothing orchestration-numpy refactor, the numpy
`mono_flux_limiter`, the `mixing_length` while_loop).

**Minor unported pieces** (none of the current target cases need them): xp2/xp3 sponge damping for wp2/wp3/up2_vp2 —
the **functions `sponge_damp_xp2`/`sponge_damp_xp3` ARE ported and tested** (sponge_layer_damping.py;
`tests/test_sponge_damp_xp23.py`), only the *driver-level wiring* into the timestep loop is deferred, and
**verified iter 322: no `case_setups/*.in` enables `(wp2|wp3|up2_vp2)_sponge_damp_settings%l_sponge_damping`** (the
setting is absent everywhere → defaults false), so clubb_driver's defensive guard for it is never triggered. And the
`pdf_closure_driver` `ipdf_pre_advance_fields`/`ipdf_pre_post_advance_fields` path — the JAX `pdf_closure_driver` covers
only the default post-advance ADG1 closure; a non-default `ipdf_call_placement` is **fail-loud rejected** (the old lazy
`clubb_python.clubb_api.pdf_closure_driver` fallback was removed iters 388/389 — `advance_clubb_core_module.py:523-528`
now raises). **Verified iter 321: no `case_setups/*.in` overrides `ipdf_call_placement`** — every case uses the default
`ipdf_post_advance_fields` (=2), so the rejection never fires and the "100% JAX, zero Fortran calls per timestep" property
holds for every configured case (machine-guarded since iter 590: `test_no_dead_imports.py::test_src_has_no_fortran_runtime_import`
asserts `clubb_jax/src` has no executable `clubb_python` reference). (Porting the full pre-advance driver is a large
oracle-gated effort entangled with the gated hydromet-PDF / scalar / cloud-cover / `l_call_pdf_closure_twice` subsystems —
out of scope for the name-mirror, which is converged.)

The **`mirror_audit.py` DEFERRED set** (iter 337) enumerates the unmirrored *routine names* exactly — now just **one**:
`pdf_closure_driver_zm` (the second zm-grid PDF closure — `l_call_pdf_closure_twice` off, which no `case_setup` sets).
In Fortran it *looks* thin (`init_pdf_implicit_coefs_terms_api` + one `pdf_closure` call on zm-interpolated inputs;
its 362 lines are almost all declarations) because ONE grid-agnostic `pdf_closure` serves both grids. But the JAX
has **no monolithic `pdf_closure`** — it decomposed it into zt-SPECIALIZED pieces with the grid regrid baked in
(code-verified iter 381: `adg1_pdf_driver_zt_jax` regrids the zm moments → zt then calls ADG1_pdf_driver on zt;
`calc_pdf_higher_order_moments_jax` computes moments on zt then regrids → zm). So a faithful zm driver needs zm-grid
VARIANTS of every closure helper — **not** a thin wrapper — and (b) there is no way to validate them: no validated
case (`l_call_pdf_closure_twice` off, no `case_setup` sets it; the driver fail-loud rejects it, iter 346) AND the
f2py oracle does not expose `pdf_closure_driver_zm` or the monolithic `pdf_closure` (only `f2py_pdf_closure_driver` /
`f2py_pdf_closure_check`, verified iter 403), so the synthetic-input unit-test path used for the `calc_*_pdf` leaves
is unavailable too without rebuilding the Fortran f2py wrapper. Porting it would be ~unvalidatable dead code against
the "faithful AND differentiable" standard, so it stays deferred by design. The Fortran `advance_microphys`/`advance_hydrometeor`/`advance_Ncm` were **reclassified
out of DEFERRED (iter 337)**: the JAX mirrors that microphysics-advance flow by **restructuring it into the per-scheme
dispatch** — `advance_clubb_to_end` → `calc_microphys_scheme_tendcies` (microphys_driver.F90) → per-scheme
`advance_{morrison,kk}_microphysics`, each looping `advance_one_hydrometeor` (= the Fortran `advance_hydrometeor` solve)
+ sedimentation + Ncm. The **Morrison path is wired + validated** (the morrison cases run); the KK transport+feedback is
the staged part. So they are mirrored-by-restructuring (a documented per-scheme reorganization), not deferred gaps. The
single DEFERRED routine is the precise residual behind "the mirror is converged for all non-deferred routines."

---

## Agent Working Rules

**The Fortran→JAX port is complete.** Every `clubb_release/src/CLUBB_core/*.F90` has a JAX mirror (CLUBB_core
is now 100% ported, Iter312), the driver runs 100% in JAX, and the bit-faithful frontier is at 20 cases
(Iter313). The incremental
**shadow-comparison** workflow that built the port (run JAX beside the Fortran oracle in-loop, match to
machine epsilon, remove the Fortran call) is **retired** — there is nothing left to port that way. Most work
now is **refactoring, simplification, differentiability, and working under the numerical-accuracy
standard** (see "Correctness standard" above).

1. **Read `DESIGN.md` in full** at the start of every session. At the end, append one concise entry to
   `CHANGELOG.md`; do not read the full changelog history (it is the append-only work record).
2. **Keep the module-naming mirror.** Every `src/CLUBB_core/<name>.py` mirrors
   `clubb_release/src/CLUBB_core/<name>.F90` at the identical relative path; the Fortran stays the algorithm
   reference (now a *reference within tolerance*, not a per-timestep oracle). Export new public symbols from
   the relevant package `__init__.py`.
3. **Judge correctness by the tiered standard, not bit-faithfulness** (see "Correctness standard"):
   conservation / invariants (strict, Tier A), regression vs the golden JAX trajectory (Tier B),
   physical-fidelity vs Fortran within the field-scaled tolerance (Tier C), and — for any change to the core
   physics glue — a `jax.grad` / finite-difference differentiability check (Tier E). `compare_cases.py`
   (Tier C) + `compare_grad.py` (Tier E) are the gates; a NEW "failure" that is a known FP / oracle-precision
   artifact (sharp-edge sedimentation, covariance cancellation, single-precision residual) is
   **characterized, not chased**.
4. **Run the gate after any shared/core change**: `python clubb_jax/run_scripts/compare_cases.py
   --max-iters 30` (expect 0 prognostic failures across the bit-faithful cases), plus a periodic
   `--max-iters 100` durability pass. Re-baseline golden references only as a deliberate, reviewed step.
5. **Prefer the simpler / more-accurate / differentiable form.** When a faithfulness contrivance and a
   cleaner form differ only at the ULP level (smooth vs hard `min/max`, accurate vs oracle-truncated `D_v`,
   float64 vs replicated `real*4`, smooth vs NumPy flux limiter), take the cleaner one and re-validate under
   the tiered standard — that is the whole point of the numerical-accuracy refactor.
6. **Porting a genuinely new subsystem** (COAMPS or SILHS — the only unported pieces): the historical
   technique still applies — read the Fortran oracle, mirror its path under `src/`, and validate with a
   case-stats oracle (feed the Fortran's own state into the JAX routine) or a conservation contract, since
   the f2py API exposes no microphysics. See DESIGN.md "Verification oracles."
