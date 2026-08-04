#!/usr/bin/env python3
"""Reproducible file/routine-name mirror audit: JAX (clubb_jax/src) vs the Fortran oracle (clubb_release/src).

This encodes the mirror-refactor's verification methodology so the convergence claim is reproducibly checkable
(replacing the ad-hoc grep scans run across iters 314-329). It is COMMENT-AWARE (strips Fortran `!`-comments and
fixed-form col-1 `c/C/*` comments — the ad-hoc scans falsely flagged dead-commented subroutines like approx_w_corr/
rad_lwsw) and TYPED-FUNCTION-AWARE (matches `real(...) function NAME`, not just bare `function NAME` — the blind spot
that hid `invalid_model_arrays` until iter 324).

It reports four classes, each filtered against the documented fold/not-target categories (see _is_documented):
  1. MISSING   — a Fortran routine with no JAX def/alias of that name (the genuine-gap finder)
  2. CASING    — a JAX def matching a Fortran routine case-insensitively but with different exact case
  3. MISPLACED — a JAX def whose Fortran home file differs from the JAX file (modulo documented renames)
  4. JAX-ONLY  — a JAX public def with no Fortran routine of that name (info; expected to be category-2/helpers)

Exit 0 if MISSING and CASING and MISPLACED are all empty after filtering (the converged state); exit 1 otherwise.
Run: python clubb_jax/run_scripts/mirror_audit.py [--verbose]
"""
import os
import re
import sys
import glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))           # repo root (contains clubb_jax/, clubb_release/)
_JAX = os.path.join(_ROOT, "clubb_jax", "src")
_FORT = os.path.join(_ROOT, "clubb_release", "src")

# Post-name terminator includes `&` and `\r` so a continuation-style header (`subroutine NAME      &`, args on the
# next line — used by the BUGSrad ocastrndm alt-solvers `two_rt_{lw,sw}_gsolap`) is parsed, not silently dropped.
_ROUTINE = re.compile(r"\b(?:subroutine|function)\s+([a-zA-Z][a-zA-Z_0-9]*)\s*[(\n\r&]", re.I)
_PYDEF = re.compile(r"^\s*def\s+([a-zA-Z_0-9]+)", re.M)
_PYDEF_TOP = re.compile(r"^def\s+([a-zA-Z_0-9]+)", re.M)   # column-0 (module-public) defs only
_PYALIAS = re.compile(r"^([a-zA-Z_0-9]+)\s*=", re.M)

# Documented fold / not-target name patterns (see DESIGN.md "Mirror-refactor status" + CHANGELOG):
#   stats_* (io/stats_writer rearchitecture); _dp precision twins (JAX is x64);
#   rank-interface variants (_1d/_2d/_3d/_k/_sclr); generic check_nan_*; SILHS / radar-diagnostic / gated-alt names.
# NOTE (iter 397): `_api$` was REMOVED from this blanket fold. A Gunther `_api` wrapper is now folded only when
# the JAX actually provides the de-api'd routine (bare name present) — see `_is_documented`. The `_api` wrappers
# with NO bare-name JAX mirror are enumerated explicitly in `_API_DEFERRED` so that a FUTURE `_api` routine added
# without a JAX mirror surfaces as MISSING instead of being silently hidden.
_FOLD = re.compile(
    r"^stats_|_dp$|_(1d|2d|3d|k|sclr)$|^check_nan_(1d|2d|sclr|soln)"
    r"|godunov|_iter$|_sel$|_bs$|ocastrndm|_complex_|refl10cm|radar_init|rayleigh_soak",
    re.I,
)
# Gunther `_api` wrappers with NO bare-name JAX mirror (verified iter 397). Listed explicitly rather than
# blanket-folded so the audit catches a future unmirrored `_api` routine. Grouped by why no bare mirror exists:
_API_DEFERRED = {
    # err_info: JAX uses derived_types/err_info.py (Gunther short-name idiom), not the Fortran *_err_info_api shims
    "cleanup_err_info_api", "init_default_err_info_api", "init_err_info_api", "set_err_info_values_api",
    # config_flags: JAX uses derived_types/config_flags.py (get_default_config_flags etc.)
    "initialize_clubb_config_flags_type_api", "initialize_silhs_config_flags_type_api",
    "print_clubb_config_flags_api", "print_silhs_config_flags_api",
    "set_default_clubb_config_flags_api", "set_default_silhs_config_flags_api",
    # debug-level: the JAX has no clubb_debug_level system
    "clubb_at_least_debug_level_api", "set_clubb_debug_level_api",
    # SILHS (unported by design)
    "latin_hypercube_2d_output_api", "lh_microphys_var_covar_driver_api",
    # gated / unported subsystems (no validated case exercises these)
    "cleanup_grid_api", "finalize_tau_sponge_damp_api", "est_kessler_microphys_api",
    "fill_holes_driver_api", "fill_holes_hydromet_api", "lin_interpolate_on_grid_api",
    "setup_corr_varnce_array_api",   # SILHS/hydromet correlation+variance setup (corr_varnce_module.py docstring)
    "setup_pdf_parameters_api",      # driver orchestration inlined in Microphys/KK_microphys/kk_microphys_driver.py
}
# Specific documented not-target / dead / gated routine names (exact, lower-cased).
_NOT_TARGET = {
    # gated alternatives the JAX does not take. Two sub-classes (verified iters 377-378):
    #   (a) COMPILE-TIME-DEAD — gated by a Fortran `parameter` fixed to the non-active value, so the routine is
    #       unreachable in the ORACLE itself: wp3_term_ta_explicit_rhs (l_explicit_turbulent_adv_wp3=.false. param),
    #       component_precip_frac_weighted (precip_frac_calc_type=2 param → the ==1 branch is dead).
    #   (b) CONFIGURABLE-BUT-GUARDED — reachable if a namelist flag is set, but clubb_driver fail-loud rejects it:
    #       compute_cloud_cover (l_use_cloud_cover), trapezoidal_rule_* (l_trapezoidal_rule_*), sed_upwind_diff_lhs
    #       (l_upwind_diff_sed), get_cloud_top_level (l_prevent_hm_ta_above_cloud), wp3_term_ta_new_pdf_lhs +
    #       pdf_closure_driver_zm (iiPDF_new / l_call_pdf_closure_twice), damp_coefficient (l_diag_Lscale_from_tau),
    #       sat_vapor_press_liq_gfdl/lookup (saturation_formula).
    "damp_coefficient", "wp3_term_ta_explicit_rhs", "wp3_term_ta_new_pdf_lhs", "compute_cloud_cover",
    "trapezoidal_rule_zm", "trapezoidal_rule_zt", "calc_trapezoid_zm", "calc_trapezoid_zt",
    "component_precip_frac_weighted", "sat_vapor_press_liq_gfdl", "sat_vapor_press_liq_lookup",
    "sed_upwind_diff_lhs", "get_cloud_top_level", "pdf_closure_driver_zm",
    "interp_var_array", "var_subgrid_interp", "var_value_integer_height",   # no-caller orphan cluster
    "calc_xp2", "set_boundary_conditions_lhs", "set_boundary_conditions_rhs", "compute_rtp2_from_chi_stat",
    # monolith / category-2 (decomposed into *_jax pieces, kept split for differentiability)
    "pdf_closure", "calc_xp2_xpyp_ta_terms", "solve_xp2_xpyp_with_multiple_lhs", "solve_xm_wpxp_with_multiple_lhs",
    "m2005micro_graupel", "mp_graupel",
    # microphysics-advance flow restructured into the JAX's per-scheme dispatch (calc_microphys_scheme_tendcies ->
    # advance_{morrison,kk}_microphysics, each looping advance_one_hydrometeor + sed + Ncm); Morrison path wired+validated
    "advance_microphys", "advance_hydrometeor", "advance_ncm",
    # gated staged-KK orchestration / SILHS / aerosol / error / IO
    # (NB print_corr_matrix is NOT here: it IS ported — corr_varnce_module.py ↔ corr_varnce_module.F90 —
    #  so it is verified by name+placement match, not tolerated. Removed iter 368.)
    "error_prints_xm_wpxp", "write_adv_micro_errors", "print_morr_error_output",
    "silhs_radiation_driver", "update_radiation_variables", "setup_stats_names", "windm_edsclrm_implicit_stats",
    "init_radiation", "finalize_extended_atm", "graupel_init", "positive_qv_adj", "get_m_mix", "get_m_mix_nested",
    "approx_w_corr", "approx_w_covar", "set_w_corr", "unpack_correlations",
    "pack_hydromet_pdf_params", "pdf_param_hm_stats", "pdf_param_ln_hm_stats", "calc_refl10cm",
    "remap_vals_to_target_helper", "vertical_integral_conserve_mass", "remap_iv_from_var_name",
    "clip_hydromet_conc_mvr", "fill_holes_wv", "hole_filling_hm_one_lev",
    "fill_holes_parallel", "fill_holes_smart_window", "fill_holes_smart_window_smooth", "fill_holes_widening_windows",
    "rad_lwsw", "nov11_altocu_tndcy", "astex_a209_tndcy",   # dead-commented / obsolete Fortran
    "calc_surface_varnce",   # comment reference; surface variances set in the sfclyr path
    # remapping / matrix / lapack / grid-adaptation (SILHS / unported-by-design subsystems)
    "get_lower_triangular_matrix", "print_lower_triangular_matrix", "row_mult_lower_tri_matrix",
    "set_lower_triangular_matrix", "symm_covar_matrix_2_corr_matrix", "symm_matrix_eigenvalues",
    "check_conservation", "check_consistency", "check_monotonicity",
    "check_remap_matrix_conservation", "check_remap_matrix_consistency", "check_remap_matrix_monotonicity",
    "read_grid_heights", "setup_grid_heights",
    "cubic_interpolated_azm_2d", "cubic_interpolated_azt_2d", "linear_interpolated_azm_2d", "linear_interpolated_azt_2d",
    "redirect_interpolated_azm_1d", "redirect_interpolated_azm_2d", "redirect_interpolated_azm_k",
    "redirect_interpolated_azt_1d", "redirect_interpolated_azt_2d", "redirect_interpolated_azt_k",
    "gradzm_1d", "gradzm_2d", "gradzt_1d", "gradzt_2d", "calc_xpwp_1d", "calc_xpwp_2d",
    "smooth_max_arrays", "smooth_max_array_scalar", "smooth_max_array_1d_scalar", "smooth_max_scalars",
    "smooth_max_sclr_idx", "smooth_min_arrays", "smooth_min_array_scalar", "smooth_min_scalars", "smooth_min_sclr_idx",
    "band_solve_multiple_rhs_lhs", "band_solve_single_rhs_multiple_lhs", "tridiag_solve_multiple_rhs_lhs",
    "tridiag_solve_single_rhs_lhs", "tridiag_solve_single_rhs_multiple_lhs",
    "penta_lu_solve_multiple_rhs_lhs", "penta_lu_solve_single_rhs_multiple_lhs",
    "binary_search",   # folded into jnp primitives (searchsorted). plinterp_fnc is now ported
                       # (interpolation.py ↔ interpolation.F90) so it is verified by name-match, not tolerated here.
    "pack_parameters", "unpack_parameters", "read_param_constraints", "read_param_minmax", "set_default_parameters",
    # IO/namelist readers folded into JAX _parse_* / dedicated readers
    "read_one_dim_file", "read_two_dim_file", "count_columns", "deallocate_one_dim_vars", "deallocate_two_dim_vars",
    "fill_blanks_one_dim_vars", "get_target_index", "read_x_profile", "read_x_table",
    "read_sounding_file", "read_sclr_sounding_file", "read_edsclr_sounding_file", "read_profile",
    "nov11_altocu_read_t_dependent", "init_microphys", "setup_pdf_indices",
    "apply_time_dependent_forcings_from_array", "apply_time_dependent_forcings_from_dycore", "read_to_grid",
    "initialize_t_dependent_forcings", "initialize_t_dependent_input", "initialize_t_dependent_sfc",
    "finalize_t_dependent_forcings", "finalize_t_dependent_input", "finalize_t_dependent_sfc",
    "restart_clubb", "initialize_clubb", "initialize_clubb_variables", "set_case_initial_conditions",
    "aer_act_clubb_quadrature_gauss", "loading", "invalid_model_arrays_dummy",
    # stats_netcdf infra (JAX stats rearchitected into io/stats_writer)
    "add_expanded_def", "format_date", "is_line_comment", "parse_registry_line", "split_registry_fields",
    # single<->multi column pdf_params copies (JAX is always ngrdcol-batched)
    "copy_multi_pdf_params_to_single", "copy_single_pdf_params_to_multi",
    # KK sed covariances (JAX: bivar_LL_covar_partial_rr/_Nr) + KK utilities (Dv_fnc→parabolic_cylinder; factorial→jnp)
    "covar_nr_kk_mvr", "covar_nr_kk_mvr_coefa", "covar_nr_kk_mvr_termb",
    "covar_rr_kk_mvr", "covar_rr_kk_mvr_coefa", "covar_rr_kk_mvr_termb", "dv_fnc", "factorial", "get_unit",
    # gated staged-KK orchestration (KK_microphys_module.F90; l_kk_micro_apply, decomposed into kk_microphys_driver)
    "kk_local_microphys", "kk_microphys_init", "kk_microphys_output", "kk_stats_output", "kk_tendency_coefs",
    "kk_upscaled_microphys", "kk_upscaled_stats", "unpack_pdf_params_kk", "kk_upscaled_means_driver",
    # forcing dispatcher (JAX: prescribe_forcings_generic + per-case modules) + tridiag_lu solver multi-rhs/lhs variants
    "prescribe_forcings", "tridiag_lu_solve_multiple_rhs_lhs", "tridiag_lu_solve_single_rhs_lhs",
    "tridiag_lu_solve_single_rhs_multiple_lhs",
}
# Casing differences accepted by design (WRF-Morrison ALL-CAPS left lowercase per that module's convention).
_CASING_OK = {"derf1", "polysvp"}

# DEFERRED: genuinely-gated routine(s) with NO JAX equivalent and no validated case to verify a port against.
# Reported separately from the true folds so the genuine remaining work stays visible rather than silently suppressed.
# (NB iter 337: advance_microphys/advance_hydrometeor/advance_Ncm were RECLASSIFIED out of DEFERRED — they are
#  restructured into the JAX's per-scheme microphysics dispatch [calc_microphys_scheme_tendcies →
#  advance_{morrison,kk}_microphysics, each looping advance_one_hydrometeor + sed + Ncm], the Morrison path of which is
#  wired + validated; see _NOT_TARGET. So they are mirrored-by-restructuring, not deferred gaps.)
_DEFERRED = {
    # The Fortran routine is a thin wrapper: interpolate zt fields → zm, then `call pdf_closure(...)` on the zm
    # inputs (its 362 lines are almost all declarations). In Fortran ONE grid-agnostic `pdf_closure` serves both
    # grids. The JAX has NO such monolithic pdf_closure — it was decomposed into zt-SPECIALIZED pieces with the
    # grid regrid baked in (code-verified iter 381): `adg1_pdf_driver_zt_jax` regrids the zm moments → zt and calls
    # ADG1_pdf_driver on the zt grid; `calc_pdf_higher_order_moments_jax` computes the moments on zt then regrids
    # → zm. A faithful zm driver therefore needs zm-grid VARIANTS of every closure helper (not a thin wrapper), and
    # there is NO validated case to check them against — l_call_pdf_closure_twice defaults False, no case_setup
    # sets it (the driver fail-loud rejects it, iter 346). NOR is it unit-validatable: the f2py oracle exposes
    # `f2py_pdf_closure_driver`/`f2py_pdf_closure_check` but NOT `pdf_closure_driver_zm` (nor the monolithic
    # `pdf_closure`), so a synthetic-input unit test against the oracle (the path used for the calc_*_pdf leaves)
    # is impossible too without modifying + rebuilding the Fortran f2py wrapper (verified iter 403). Porting it
    # would mean ~unvalidatable dead code, against the "faithful AND differentiable" standard — so it stays
    # deferred by design.
    "pdf_closure_driver_zm": "second (zm-grid) PDF closure call (l_call_pdf_closure_twice=False; no case_setup sets it)",
}
# Documented file renames / splits: (jax_stem, fortran_stem) pairs that are NOT misplacements.
_RENAMES = {
    ("pdf_params", "pdf_parameter_module"),
    # (generic_forcings → prescribe_forcings rename retired iter 385: the file is now prescribe_forcings.py,
    #  matching prescribe_forcings.F90 directly.)
    ("advance_clubb_to_end", "clubb_driver"), ("stats_writer", "stats_netcdf"),
    # (corr_varnce_module → array_index rename removed iter 393: array_index.F90 is a pure constants/index
    #  module with NO subroutines/functions, so this routine-placement rename matched nothing — a no-op.
    #  array_index.F90's index constants are mirrored in derived_types/sclr_idx.py, not as routines.)
}

# (jax_stem, jax_reldir-basename) pairs where a stem-matching JAX file deliberately lives in a DIFFERENT
# directory than its Fortran oracle — the documented architectural splits. Used by the directory-correspondence
# check so a genuine whole-file dir-misplacement surfaces while these by-design placements do not.
#   grid_class: grid_class.F90 (CLUBB_core) is split into CLUBB_core/grid_class.py (operators) + the grid
#   derived-TYPE half derived_types/grid_class.py (the JAX derived-types architecture layer). (iter 717)
_DIR_SPLIT_OK = {("grid_class", "derived_types")}

# JAX source files (stem, lower-cased) that intentionally have NO Fortran *source-file* counterpart — the
# JAX-specific architecture layers. Used by the file-name mirror check so a NEW divergent filename (one that
# is neither a Fortran stem, a header stem, a documented _RENAMES jax-side, nor here) is surfaced. Each entry
# is a deliberate design choice, not a gap. (Keep in sync when a genuinely-new JAX-only file is added.)
_JAX_ONLY_FILES = {
    # JAX-only differentiability / typing infra (no Fortran equivalent)
    "tracer_numpy": "JAX-only differentiability tracer/numpy bridge",
    "common": "derived_types typing util (Array = np.ndarray)",
    # derived-type API layer (Gunther's Python-API style; the Fortran types live in *_module.F90 / array_index.F90)
    "err_info": "derived_types err_info container (API layer for err_info_type_module)",
    "config_flags": "derived_types ConfigFlags container (API layer; constants mirror model_flags.py)",
    "sclr_idx": "derived_types scalar-index container (API layer for array_index)",
    # JAX I/O groupings (Input_fields readers; Fortran input is split across input_reader/sounding/driver)
    "surface": "Input_fields surface-forcing reader (JAX I/O grouping)",
    "namelist": "Input_fields namelist parser (JAX I/O grouping)",
    "grid_file": "Input_fields grid-heights file reader (JAX I/O grouping)",
    # per-step microphysics wiring (mirror the clubb_driver call SEQUENCE, not a single Fortran file)
    "morrison_microphys_step": "per-step Morrison wiring (mirrors the driver call sequence)",
    "kk_microphys_step": "per-step KK wiring (mirrors the driver call sequence)",
    "coamps_microphys_step": "per-step COAMPS wiring (mirrors the driver call sequence)",
    "kk_microphys_driver": "JAX KK orchestration driver (composes the ported KK pieces)",
    # JAX differentiable replacement for Parabolic.f90's Algorithm-850 D_v (a different algorithm, not a port)
    "parabolic_cylinder": "JAX differentiable D_v(z) replacement for Parabolic.f90 (DLMF series, not a port)",
}

def _fortran_file_stems():
    """Lower-cased basenames (sans extension) of every Fortran SOURCE + header file, for the file-name check."""
    stems = set()
    for f in glob.glob(os.path.join(_FORT, "**", "*"), recursive=True):
        b = os.path.basename(f)
        m = re.match(r"(.+)\.(f90|f|h|inc)$", b, re.I)
        if m:
            stems.add(m.group(1).lower())
    return stems

def _unmirrored_files():
    """JAX src .py files whose stem matches neither a Fortran source/header stem, a documented _RENAMES
    jax-side, nor the _JAX_ONLY_FILES allowlist — i.e. a NEW divergent filename to review."""
    fort_stems = _fortran_file_stems()
    rename_jax = {j for j, _f in _RENAMES}
    out = []
    for f in glob.glob(os.path.join(_JAX, "**", "*.py"), recursive=True):
        b = os.path.basename(f)
        if b.startswith("__"):
            continue
        stem = b[:-3]
        sl = stem.lower()
        if sl in fort_stems or sl in rename_jax or sl in _JAX_ONLY_FILES:
            continue
        out.append(os.path.relpath(f, _JAX))
    return out

def _fortran_stem_dirs():
    """Lower-cased source-file stem -> set of reldirs (relative to clubb_release/src) where it lives."""
    dirs = {}
    for f in glob.glob(os.path.join(_FORT, "**", "*.[fF]*"), recursive=True):
        stem = re.sub(r"\.[fF](90)?$", "", os.path.basename(f)).lower()
        dirs.setdefault(stem, set()).add(os.path.basename(os.path.relpath(os.path.dirname(f), _FORT)))
    return dirs

def _misplaced_dir_files():
    """JAX .py files whose stem matches a Fortran SOURCE stem but whose DIRECTORY does not correspond to the
    oracle's (a whole-file directory misplacement the basename-only UNMIRRORED check cannot see). The
    `_DIR_SPLIT_OK` allowlist excuses the documented architectural splits. Returns (stem, jax_reldir,
    [fortran_dirs])."""
    fdirs = _fortran_stem_dirs()
    out = []
    for f in glob.glob(os.path.join(_JAX, "**", "*.py"), recursive=True):
        b = os.path.basename(f)
        if b.startswith("__"):
            continue
        sl = b[:-3].lower()
        if sl not in fdirs:
            continue   # jax-only / renamed file — covered by _unmirrored_files / _RENAMES
        jdir = os.path.relpath(os.path.dirname(f), _JAX)
        jbase = os.path.basename(jdir)
        if (sl, jbase) in _DIR_SPLIT_OK:
            continue
        if any(jbase == fd or jbase in fd or fd in jbase for fd in fdirs[sl]):
            continue
        out.append((sl, jdir, sorted(fdirs[sl])))
    return out

def _strip_fortran_comments(text, fixed_form):
    out = []
    for line in text.splitlines():
        s = line
        if fixed_form and s[:1] in ("c", "C", "*", "!"):
            continue
        # remove inline ! comment (naive; ignores ! inside strings — fine for routine-name extraction)
        i = s.find("!")
        if i >= 0:
            s = s[:i]
        if s.strip():
            out.append(s)
    return "\n".join(out)

def _fortran_routines():
    by_name = {}   # lower -> set(exact)
    homes = {}     # lower -> set(file stem lower)
    for f in glob.glob(os.path.join(_FORT, "**", "*.[fF]*"), recursive=True):
        stem = re.sub(r"\.[fF](90)?$", "", os.path.basename(f))
        # Fixed-form source is `.f`/`.F` (BUGSrad); free-form is `.f90`/`.F90`. `f.lower().endswith(".f")` is True
        # for both `.f` and `.F` (lowercased) and False for `.f90`/`.F90` — the correct fixed-vs-free split.
        txt = _strip_fortran_comments(open(f, errors="ignore").read(), fixed_form=f.lower().endswith(".f"))
        for m in _ROUTINE.finditer(txt):
            nm = m.group(1)
            by_name.setdefault(nm.lower(), set()).add(nm)
            homes.setdefault(nm.lower(), set()).add(stem.lower())
    return by_name, homes

def _jax_stems():
    """Set of JAX module stems (lower) — used to scope MISSING to files we actually intend to mirror,
    excluding the by-design-unmirrored Fortran subsystems (LAPACK, Numerical Recipes/nrutil/mt95/ran_state,
    the tuner + G_unit *_tests, SILHS, COAMPS microphysics, aerosol activation, the input/* readers, the
    Parabolic.f90 internals, AIRYfunction, etc.) which have no .py counterpart."""
    stems = set()
    for f in glob.glob(os.path.join(_JAX, "**", "*.py"), recursive=True):
        b = os.path.basename(f)
        if b.startswith("__"):
            continue
        stems.add(b[:-3].lower())
    # documented file renames make these Fortran stems "mirrored" too
    for _jx, fort in _RENAMES:
        stems.add(fort.lower())
    return stems

def _jax_alias_violations(fort):
    """A `_jax`-suffixed top-level def whose base name mirrors a real Fortran routine MUST have a bare-name
    public alias (e.g. `term_dp1_lhs = jit(term_dp1_lhs_jax)`) — that bare name is what mirrors the Fortran
    public API. Flag any `<name>_jax` def for which Fortran has `<name>` but no bare `<name>` def/alias exists
    in the same file (so the public name would diverge to `<name>_jax`). Convention verified clean iter 374."""
    DEF = re.compile(r"^def ([a-zA-Z_0-9]+)", re.M)
    ALIAS = re.compile(r"^([a-zA-Z_0-9]+)\s*=", re.M)
    out = []
    for f in glob.glob(os.path.join(_JAX, "**", "*.py"), recursive=True):
        if "__" in os.path.basename(f):
            continue
        txt = open(f, errors="ignore").read()
        names = set(DEF.findall(txt)) | set(ALIAS.findall(txt))
        for d in DEF.findall(txt):
            if d.endswith("_jax"):
                base = d[:-4]
                if base.lower() in fort and base not in names:
                    out.append((os.path.relpath(f, _JAX), d))
    return out

def _jax_names():
    defs = {}      # lower(base) -> set(exact base)
    homes = {}     # lower(base) -> set(file stem lower)
    aliases = set()
    for f in glob.glob(os.path.join(_JAX, "**", "*.py"), recursive=True):
        if "__" in os.path.basename(f):
            continue
        stem = os.path.basename(f)[:-3]
        txt = open(f, errors="ignore").read()
        for m in _PYDEF.finditer(txt):
            nm = m.group(1)
            base = nm[:-4] if nm.endswith("_jax") else nm
            defs.setdefault(base.lower(), set()).add(base)
            homes.setdefault(base.lower(), set()).add(stem.lower())
        for m in _PYALIAS.finditer(txt):
            nm = m.group(1)
            base = nm[:-4] if nm.endswith("_jax") else nm
            aliases.add(base.lower())
    return defs, homes, aliases

def _jax_toplevel_names():
    """Lower-cased base names of COLUMN-0 (module-public) JAX defs — excludes nested closures / class methods,
    so the JAX-ONLY count reflects the real public surface rather than `body`/`f`/`step` scan noise."""
    top = set()
    for f in glob.glob(os.path.join(_JAX, "**", "*.py"), recursive=True):
        if "__" in os.path.basename(f):
            continue
        for m in _PYDEF_TOP.finditer(open(f, errors="ignore").read()):
            nm = m.group(1)
            top.add((nm[:-4] if nm.endswith("_jax") else nm).lower())
    return top

def _redundant_tolerances(fort, fort_homes, jall, mirrored_stems):
    """_NOT_TARGET entries that are ACTUALLY ported (an exact JAX def/alias exists) AND are a real Fortran
    routine in a mirrored file — so the tolerance is redundant: the routine would be verified by name-match
    anyway. Keeping it silently weakens the audit (a real future regression on that name would be masked).
    Surfacing these keeps the tolerance set honest (cf. iter 363 plinterp_fnc, iter 368 print_corr_matrix)."""
    out = []
    for nm in sorted(_NOT_TARGET):
        if nm in jall and nm in fort:
            in_scope = any(_norm(s) in mirrored_stems or s in mirrored_stems for s in fort_homes.get(nm, []))
            if in_scope:
                out.append(nm)
    return out

def _is_documented(lname, jall):
    # A Gunther `_api` wrapper is out of MISSING scope iff the JAX provides the de-api'd routine (bare name),
    # or it is explicitly classified as a no-bare-mirror wrapper in _API_DEFERRED. Otherwise fall through to the
    # remaining class folds (^stats_, _dp, rank suffixes, …) and _NOT_TARGET.
    if lname.endswith("_api"):
        if lname[:-4] in jall or lname in jall or lname in _API_DEFERRED:
            return True
    return bool(_FOLD.search(lname)) or lname in _NOT_TARGET

def _norm(s):
    return s.replace("_module", "")

# Buckets the audit's scoped-out Fortran files (no JAX stem) by the by-design-unmirrored subsystem they belong to
# (reviewed iter 588, made directory-robust iter 589). The big library/test subsystems live in DEDICATED directories,
# so they are matched by PATH (precise — a new short physics file in a physics dir cannot masquerade as LAPACK); the
# scattered physics-directory residue is matched by NAME. A file matching NEITHER lands in `uncategorized` — a soft
# tripwire: a future Fortran physics file added to the oracle surfaces there for porting review rather than being
# silently scoped out. Guarded by tests/test_mirror_audit.py::test_no_unrecognized_scoped_out_file.
# (bucket, dir-substring-or-None, name-regex-or-None). A file is bucketed if its reldir contains the directory
# substring OR its stem matches the name regex. The big library/test subsystems live in DEDICATED directories
# (matched by PATH — precise) and ALSO carry a name keyword (lapack/coamps/silhs/_test) so the few that live outside
# their main dir (e.g. CLUBB_core/lapack_wrap, Microphys/coamps_microphys_driver_module) are still recognized; the
# scattered physics-directory residue is matched by NAME only. The keywords (lapack/coamps/silhs) and the explicit
# stem lists are all non-physics, so a genuine new physics file matches NOTHING → `uncategorized` (the tripwire).
_BUCKETS = [
    ("lapack_blas",        "lapack",            r"lapack"),
    ("numerical_recipes",  "numerical_recipes", r"^(jacobian|airyfunction|parabolic|nr|nrtype|nrutil|gamm?a|airy)$"),
    ("silhs_sampling",     "silhs",             r"silhs"),
    ("aerosol_activation", "scm_activation",    r"^(aer_\w+|aerosol\w*)$"),
    ("coamps_microphys",   "coamps_microphys",  r"coamps"),
    ("g_unit_tests",       "g_unit_test_types", r"_tests?$|generalized_grid"),
    ("bugsrad_altsolver",  "bugsrad",           None),
    ("scm_host_microphys", None,                r"^(est_kessler_microphys_module|estimate_scm_microphys_module|"
                                                r"microphysics|microphys_init_cleanup|microphys_stats_vars_module|"
                                                r"pdf_hydromet_microphys_wrapper|lh_microphys_\w+|transform_to_pdf_module|"
                                                r"generate_uniform_sample_module|math_utilities|cloud_feedback)$"),
    ("io_readers",         None,                r"^(input_\w+|file_functions|endian|stat_file_utils|text_writer|"
                                                r"stats_netcdf|corr_varnce_input_reader)$"),
    ("tuner_infra",        None,                r"^(clubb_tuner|code_timer_module|error|error_code|extrapolation|"
                                                r"grid_adaptation_module|penta_bicgstab_solver|enhanced_simann|"
                                                r"corr_cholesky_mtx_tests)$"),
    ("case_setups",        None,                r"^(clex9_\w+|jun25)$"),
    ("state_api_types",    None,                r"(_type_module|_api_module)$|^clubb_api_module$|^radiation_variables_module$"),
]
_BUCKET_RX = [(b, ds, re.compile(nx, re.I) if nx else None) for b, ds, nx in _BUCKETS]
_ALL_BUCKETS = [b for b, _, _ in _BUCKETS]

# Routine-LESS Fortran modules (0 subroutines/functions) are invisible to BOTH the routine-based MISSING check
# AND the routine-bearing scoped-out enumeration (iter 728 found this re parameters_microphys.F90). Each such
# module must be same-stem JAX-mirrored, a recognized by-design-unmirrored subsystem (_BUCKET_RX), or in this
# allowlist of deliberate dispositions; a NEW unclassified one is a pure-parameter/type module to evaluate.
# (iter 729; guarded by tests/test_mirror_audit.py::test_no_unclassified_routineless_module.)
_ROUTINELESS_OK = {
    "array_index":          "index constants → derived_types/sclr_idx.py",
    "clubb_precision":      "kind defs (core_rknd/dp/sp) → JAX native float64 (x64)",
    "parameters_microphys": "per-case config → runtime namelist (clubb_driver/state); aerosol enums → scoped-out SCM_Activation",
    "parameters_radiation": "radiation scheme-name/config params → runtime radiation dispatch",
    "input_names":          "input var-name string constants → IO readers",
    "stat_file_module":     "stats file derived types → io/stats_writer (rearchitected stats)",
    "parabolic_constants":  "ACM-850 D_v constants → replaced by parabolic_cylinder.py (DLMF series)",
    "kinds":                "BUGSrad kind defs → JAX native",
    "driver_read":          "BUGSrad standalone-driver IO (scoped-out subsystem)",
    "nrtype":               "Numerical Recipes kind module (scoped out)",
    "int2txt":              "standalone integer→text build utility (no physics, no JAX need)",
}

def _routineless_unclassified():
    """Routine-less Fortran modules not same-stem JAX-mirrored, not a recognized by-design-unmirrored subsystem
    (_BUCKET_RX), and not in _ROUTINELESS_OK — i.e. a NEW pure-parameter/type module the routine-based checks
    cannot see. Returns sorted "reldir/stem" list (should stay empty)."""
    jstems = {os.path.basename(f)[:-3].lower()
              for f in glob.glob(os.path.join(_JAX, "**", "*.py"), recursive=True)
              if not os.path.basename(f).startswith("__")}
    out = set()
    for f in glob.glob(os.path.join(_FORT, "**", "*.[fF]*"), recursive=True):
        txt = _strip_fortran_comments(open(f, errors="ignore").read(), fixed_form=f.lower().endswith(".f"))
        if _ROUTINE.search(txt):
            continue   # routine-bearing — covered by MISSING / scoped-out
        stem = re.sub(r"\.[fF](90)?$", "", os.path.basename(f))
        sl = stem.lower()
        if sl in jstems or sl in _ROUTINELESS_OK:
            continue
        reldir = os.path.relpath(os.path.dirname(f), _FORT)
        rl = reldir.replace("\\", "/").lower()
        if any((ds and ds in rl) or (nx and nx.search(stem)) for _b, ds, nx in _BUCKET_RX):
            continue
        out.add(f"{reldir}/{stem}")
    return sorted(out)


def scoped_out_breakdown(entries):
    """Partition the scoped-out Fortran files into the by-design-unmirrored subsystem buckets above. `entries` is a
    list of (reldir, stem); a file is bucketed by its dedicated directory OR a non-physics name keyword. Returns
    (counts_dict, uncategorized_list of "reldir/stem"). `uncategorized` should stay empty: a non-empty entry is a
    Fortran file with no JAX mirror that is ALSO not a recognized non-target — i.e. a candidate port the whole-file
    scoping would otherwise hide."""
    counts = {b: 0 for b in _ALL_BUCKETS}
    uncategorized = []
    for reldir, stem in entries:
        rl = reldir.replace("\\", "/").lower()
        hit = next((b for b, ds, nx in _BUCKET_RX
                    if (ds and ds in rl) or (nx and nx.search(stem))), None)
        if hit is None:
            uncategorized.append(f"{reldir}/{stem}")
        else:
            counts[hit] += 1
    return counts, uncategorized


def scoped_out_entries():
    """The Fortran source files (reldir, stem) scoped OUT of the MISSING check (no JAX `.py` mirror). reldir is
    relative to clubb_release/src; one entry per routine-bearing file. Exposed for the breakdown + guard."""
    _, fort_homes = _fortran_routines()
    mirrored_stems = _jax_stems()
    scoped_stems = {s for homes in fort_homes.values() for s in homes
                    if _norm(s) not in mirrored_stems and s not in mirrored_stems}
    entries, seen = [], set()
    for f in glob.glob(os.path.join(_FORT, "**", "*.[fF]*"), recursive=True):
        stem = re.sub(r"\.[fF](90)?$", "", os.path.basename(f))
        if stem.lower() not in scoped_stems:
            continue
        reldir = os.path.relpath(os.path.dirname(f), _FORT)
        key = (reldir, stem.lower())
        if key not in seen:
            seen.add(key)
            entries.append((reldir, stem))
    return sorted(entries)


def main():
    verbose = "--verbose" in sys.argv
    fort, fort_homes = _fortran_routines()
    jdefs, jhomes, jaliases = _jax_names()
    jall = set(jdefs) | jaliases
    mirrored_stems = _jax_stems()   # scope MISSING to in-scope (mirrored) Fortran files only

    missing, casing, misplaced, jaxonly, deferred = [], [], [], [], []
    for ln, exacts in sorted(fort.items()):
        if ln in jall:
            # casing: in jax defs but exact case differs
            if ln in jdefs and not (jdefs[ln] & exacts):
                if not _is_documented(ln, jall) and ln not in _CASING_OK:
                    casing.append((sorted(jdefs[ln])[0], sorted(exacts)))
            # misplacement: jax home shares no normalized stem with fortran home (modulo renames)
            if ln in jhomes:
                js = {_norm(s) for s in jhomes[ln]}
                fs = {_norm(s) for s in fort_homes[ln]}
                if not (js & fs) and not any(a in b or b in a for a in js for b in fs):
                    if not any((j, f) in _RENAMES for j in jhomes[ln] for f in fort_homes[ln]):
                        misplaced.append((ln, sorted(jhomes[ln]), sorted(fort_homes[ln])))
        else:
            # only a gap if the routine's Fortran home file is one we intend to mirror
            in_scope = any(_norm(s) in mirrored_stems or s in mirrored_stems for s in fort_homes[ln])
            if not in_scope:
                continue
            if ln in _DEFERRED:
                deferred.append((ln, _DEFERRED[ln]))
            elif not _is_documented(ln, jall):
                missing.append((ln, sorted(fort_homes[ln])))
    for ln, exacts in sorted(jdefs.items()):
        if ln not in fort and not ln.startswith("_"):
            jaxonly.append(sorted(exacts)[0])

    def show(title, items, limit=None):
        print(f"\n{title}: {len(items)}")
        for it in (items if (verbose or limit is None) else items[:limit]):
            print(f"  {it}")

    unmirrored = _unmirrored_files()
    misplaced_dirs = _misplaced_dir_files()
    redundant = _redundant_tolerances(fort, fort_homes, jall, mirrored_stems)
    alias_viol = _jax_alias_violations(fort)

    # INFO: the Fortran source files whose routines are scoped OUT of the MISSING check because the file has no
    # JAX `.py` mirror (the `_jax_stems()` by-design-unmirrored set — LAPACK/Numerical-Recipes, SILHS, COAMPS,
    # aerosol activation, the input/* readers, radiation state modules, BUGSrad alt-solvers, the tuner/G_unit
    # tests, etc.). Emitting them keeps the whole-file scoping VISIBLE: a future Fortran file that SHOULD be
    # mirrored would appear here rather than being silently excused. Not a failure (does not count toward `bad`).
    scoped_entries = scoped_out_entries()
    scoped_out_files = [f"{d}/{s}" for d, s in scoped_entries]
    scope_counts, scope_uncategorized = scoped_out_breakdown(scoped_entries)

    print("=== CLUBB-JAX mirror audit (comment-aware, typed-function-aware) ===")
    show("MISSING (genuine-gap candidates, after fold filter)", missing)
    show("CASING mismatches", casing)
    show("MISPLACED (wrong-file, modulo documented renames)", misplaced)
    show("UNMIRRORED FILES (jax .py with no Fortran source/header stem, not in the JAX-only allowlist)", unmirrored)
    show("MISPLACED FILES (stem matches a Fortran source but in a non-corresponding dir, modulo _DIR_SPLIT_OK)", misplaced_dirs)
    show("REDUNDANT TOLERANCES (_NOT_TARGET entries that are actually ported — remove to verify by name-match)", redundant)
    show("JAX-ALIAS violations (_jax def mirroring a Fortran routine but missing its bare-name public alias)", alias_viol)
    show("DEFERRED (gated routine[s] with no validated case to verify a port against; not a regression)", deferred)
    _top = _jax_toplevel_names()
    jaxonly_top = [nm for nm in jaxonly if nm.lower() in _top]
    _nested = len(jaxonly) - len(jaxonly_top)
    if verbose:
        show("JAX-ONLY public defs (info) [top-level module functions]", jaxonly_top)
        print(f"  (+ {_nested} nested-closure / class-method defs, not module-public)")
        show("FORTRAN FILES scoped out of MISSING (info; no JAX mirror — by-design-unmirrored subsystems)",
             scoped_out_files)
    else:
        print(f"\nJAX-ONLY public defs (info): {len(jaxonly_top)} top-level module functions "
              f"(+ {_nested} nested/method defs)  (use --verbose to list)")
        print(f"FORTRAN FILES scoped out of MISSING (info): {len(scoped_out_files)} by-design-unmirrored "
              f"source files with no JAX stem  (use --verbose to list)")
    # per-subsystem breakdown of the scoped-out set + the uncategorized soft-tripwire
    _bk = "  ".join(f"{k}={c}" for k, c in scope_counts.items() if c)
    print(f"  scoped-out by subsystem: {_bk}")
    if scope_uncategorized:
        print(f"  ⚠ UNRECOGNIZED scoped-out file(s) — not matched by any by-design-unmirrored subsystem pattern; "
              f"REVIEW for porting: {scope_uncategorized}")
    # routine-less (pure-parameter/type) Fortran modules: invisible to MISSING + the routine-bearing scoped-out
    # enumeration, so checked separately (iter 729). Soft tripwire — should stay empty.
    routineless_unclassified = _routineless_unclassified()
    print(f"ROUTINE-LESS modules (info): {len(_ROUTINELESS_OK)} documented dispositions; "
          f"{len(routineless_unclassified)} unclassified")
    if routineless_unclassified:
        print(f"  ⚠ UNCLASSIFIED routine-less module(s) — a pure-parameter/type Fortran module with no JAX mirror "
              f"that is neither a recognized subsystem nor in _ROUTINELESS_OK; classify it: {routineless_unclassified}")

    bad = (len(missing) + len(casing) + len(misplaced) + len(unmirrored) + len(misplaced_dirs)
           + len(redundant) + len(alias_viol))
    print(f"\n{'PASS' if bad == 0 else 'REVIEW'}: MISSING={len(missing)} CASING={len(casing)} "
          f"MISPLACED={len(misplaced)} UNMIRRORED_FILES={len(unmirrored)} MISPLACED_FILES={len(misplaced_dirs)} "
          f"REDUNDANT_TOL={len(redundant)} JAX_ALIAS={len(alias_viol)}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
