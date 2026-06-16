"""CLUBB driver — mirrors ``clubb_release/src/clubb_driver.F90``.

Holds the case orchestration the Fortran keeps in the ``clubb_driver`` module:
``run_clubb`` (init -> advance -> cleanup), ``init_clubb_case``, and ``clean_up_clubb``.
The timestep loop ``advance_clubb_to_end`` is the ``advance_clubb_to_end`` subroutine of
the same Fortran module, kept in its own submodule (``advance_clubb_to_end.py``) for size
and imported by ``run_clubb``. The thin command-line frontend lives in ``clubb_standalone.py``
(mirroring ``clubb_standalone.F90``)."""
import math
import os
from pathlib import Path

# clubb_jax/ and clubb_release/ are siblings under the same parent directory.
# All input data, Fortran binaries, and run_scripts live inside clubb_release/.
_CLUBB_RELEASE_ROOT = Path(__file__).resolve().parents[2] / "clubb_release"

import jax.numpy as jnp
import numpy as np

# JAX core
from clubb_jax.src.CLUBB_core.saturation import sat_mixrat_liq, rcm_sat_adj
from clubb_jax.src.CLUBB_core.calc_pressure import calculate_thvm
from clubb_jax.src.CLUBB_core.grid_class import zt2zm
from clubb_jax.src.CLUBB_core.sponge_layer_damping import initialize_tau_sponge_damp
from clubb_jax.src.Input_fields.hydrostatic_module import hydrostatic
from clubb_jax.src.CLUBB_core.parameters_tunable import (
    init_clubb_params,
    calc_derrived_params,
    get_param_names,
    check_parameters,
)
from clubb_jax.src.CLUBB_core.model_flags import get_default_config_flags
from clubb_jax.src.CLUBB_core.numerical_check import check_clubb_settings
from clubb_jax.src.derived_types.config_flags import ConfigFlags
from clubb_jax.src.derived_types.grid_class import setup_grid
from clubb_jax.src.derived_types.sclr_idx import SclrIdx
from clubb_jax.src.derived_types.err_info import ErrInfo
from clubb_jax.src.derived_types.pdf_params import (
    init_pdf_implicit_coefs_terms_api,
    init_pdf_params,
)

# I/O
from clubb_jax.src.io.stats_writer import StatsWriter
from clubb_jax.src.Input_fields.grid_file import read_grid_file
from clubb_jax.src.Input_fields.namelist import read_namelist
from clubb_jax.src.Input_fields.sounding import (
    read_sounding,
    interpolate_sounding,
    convert_pressure_sounding_to_z,
    read_scalar_sounding,
    interpolate_scalar_sounding,
)
from clubb_jax.src.Input_fields.surface import read_surface
from clubb_jax.src.Benchmark_cases.arm import load_arm_forcings_data
from clubb_jax.src.Benchmark_cases.time_dependent_input import load_generic_forcings_data

# ── Physical constants — mirror the Fortran clubb_driver.F90 `use constants_clubb` rather than re-defining the
#    standalone block here (all 16 values verified bit-identical to constants_clubb, iter 599) ──────
from clubb_jax.src.CLUBB_core.constants_clubb import (
    Cp, Lv, Rd, ep1, ep2, kappa, grav, p0,
    rt_tol, thl_tol, w_tol, em_min, cloud_frac_min, radians_per_deg, omega_planet,
)
Nc0_in_cloud = 100.0e6      # [num/m^3] (driver-local)

_CLOUD_FEEDBACK_CASES = {
    "cloud_feedback_s6",
    "cloud_feedback_s6_p2k",
    "cloud_feedback_s11",
    "cloud_feedback_s11_p2k",
    "cloud_feedback_s12",
    "cloud_feedback_s12_p2k",
}


def _initialize_em_profile(runtype: str, gr, um: np.ndarray):
    """Mirror initialize_clubb() case-based em setup from Fortran."""
    runtype = str(runtype).strip()
    zm = gr.zm
    ngrdcol, nzm = zm.shape
    em = np.full((ngrdcol, nzm), em_min, dtype=np.float64)
    um_out = um.copy()

    def _set_cloud_top_profile(cloud_top: float, em_max_val: float):
        e = np.where(zm < cloud_top, em_max_val, em_min)
        if nzm > 1:
            e[:, 0] = e[:, 1]
        e[:, -1] = em_min
        return e

    em_min_cases = {"bomex", "ekman", "atex_long", "arm"}
    em_one_topmin_cases = {
        "generic", "arm_97", "twp_ice", "arm_0003", "arm_3year",
        "dycoms2_rf02", "gabls3",
    } | _CLOUD_FEEDBACK_CASES
    em_point1_topmin_cases = {"lba", "cobra"}
    fixed_cloud_top_cases = {
        "astex_a209": (700.0, 1.0),
        "fire": (700.0, 4.5),
        "dycoms2_rf01": (800.0, 1.1),
        "mpace_a": (2000.0, 1.0),    # clubb_driver.F90:5131 (em=1 below 2 km cloud top, em_min above)
        "mpace_b": (1300.0, 1.0),
        "rico": (1500.0, 1.0),
    }
    offset_cloud_top_cases = {
        "nov11_altocu": 2800.0,
        "clex9_nov02": 2200.0,
        "clex9_oct14": 3500.0,
    }

    if runtype in em_one_topmin_cases:
        em[:, :] = 1.0
        em[:, -1] = em_min
    elif runtype in em_min_cases:
        em[:, :] = em_min
    elif runtype == "atex":
        um_out = np.maximum(um_out, -8.0)
        em[:, :] = em_min
    elif runtype in fixed_cloud_top_cases:
        cloud_top, em_max = fixed_cloud_top_cases[runtype]
        em = _set_cloud_top_profile(cloud_top, em_max)
    elif runtype in offset_cloud_top_cases:
        em = _set_cloud_top_profile(offset_cloud_top_cases[runtype] + float(zm[0, 0]), 0.01)
    elif runtype == "jun25_altocu":
        em[:, :] = 0.01
        if nzm > 1:
            em[:, 0] = em[:, 1]
        em[:, -1] = em_min
    elif runtype in em_point1_topmin_cases:
        em[:, :] = 0.1
        em[:, -1] = em_min
    elif runtype == "gabls2":
        cloud_top = 800.0
        em = np.where(zm < cloud_top, 0.5 * (1.0 - (zm / cloud_top)), em_min)
        if nzm > 1:
            em[:, 0] = em[:, 1]
        em[:, -1] = em_min
    elif runtype == "gabls3_night":
        em[:, :] = 1.0
    elif runtype == "coriolis_test":
        depth = (zm[:, -1] - zm[:, 0])[:, None]
        em = np.sin(np.pi * zm / depth) * (w_tol**2) * 6.0

    return em, um_out


def _initialize_turbulence_state(runtype: str, gr, dt_main: float,
                                 fcor_y: np.ndarray, l_tke_aniso: bool,
                                 um: np.ndarray):
    """Mirror initialize_clubb() em/wp2/up2/vp2/upwp initialization."""
    em, um_adj = _initialize_em_profile(runtype, gr, um)

    if l_tke_aniso:
        wp2 = (2.0 / 3.0) * em
        up2 = (2.0 / 3.0) * em
        vp2 = (2.0 / 3.0) * em
        upwp = np.zeros_like(em)

        if str(runtype).strip() == "coriolis_test":
            w_tol_sqd = w_tol**2
            wp2 = (1.0 / 3.0) * em + w_tol_sqd
            up2 = (3.0 / 3.0) * em + w_tol_sqd
            vp2 = (2.0 / 3.0) * em + w_tol_sqd
            em = em + 1.5 * w_tol_sqd
            upwp = 0.5 * dt_main * fcor_y[:, None] * (up2 - wp2)
    else:
        wp2 = (2.0 / 3.0) * em
        up2 = np.zeros_like(em)
        vp2 = np.zeros_like(em)
        upwp = np.zeros_like(em)

    return em, wp2, up2, vp2, upwp, um_adj


def run_clubb(namelist_path: str, l_stdout: bool = True, max_steps: int | None = None):
    """Run CLUBB standalone for a case described by a namelist file.

    Mirrors ``run_clubb`` in clubb_driver.F90: init -> advance -> cleanup.

    Args:
        namelist_path: path to *_model.in file
        l_stdout: print timestep info to stdout
        max_steps: optional cap on the number of timesteps (JAX extension; None = run to time_final)
    """
    from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end

    state = init_clubb_case(namelist_path)
    advance_clubb_to_end(state, l_stdout=l_stdout, max_steps=max_steps)
    clean_up_clubb(state)
    return state


def _resolve_stats_registry_path(namelist_path: str, cfg: dict) -> Path:
    """Resolve the stats registry file path.

    Priority:
      1) explicit namelist key `stats_registry`
      2) the runfile itself, if it contains `&clubb_stats_nl`
      3) repository default `input/stats/standard_stats.in`
    """
    configured = str(cfg.get('stats_registry', '')).strip()
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            p = Path(namelist_path).resolve().parent / p
        return p.resolve()

    runfile = Path(namelist_path).resolve()
    if '&clubb_stats_nl' in runfile.read_text().lower():
        return runfile

    repo_root = _CLUBB_RELEASE_ROOT
    return repo_root / "input" / "stats" / "standard_stats.in"


def _resolve_case_input_path(namelist_dir: Path, runtype: str, suffix: str) -> Path:
    """Resolve case input files for either model.in or aggregated CASE.in runs."""
    candidate = namelist_dir / f"{runtype}{suffix}"
    if candidate.exists():
        return candidate

    repo_root = _CLUBB_RELEASE_ROOT
    fallback = repo_root / "input" / "case_setups" / f"{runtype}{suffix}"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Required case input file not found: {candidate} (also checked {fallback})"
    )


def _clean_namelist_path(path_value) -> str:
    """Normalize namelist path strings (strip quotes and whitespace)."""
    return str(path_value).strip().strip("'\"")


def _validate_scalar_column_names(names, idx_rt: int, idx_thl: int, idx_co2: int, label: str):
    """Validate scalar column order against namelist scalar-index mapping."""
    for col_idx, name in enumerate(names, start=1):
        if name == 'CO2[ppmv]' and idx_co2 > 0 and col_idx != idx_co2:
            raise ValueError(f"{label}: iisclr/iiedsclr_CO2 index does not match column order.")
        if name == 'rt[kg/kg]' and idx_rt > 0 and col_idx != idx_rt:
            raise ValueError(f"{label}: iisclr/iiedsclr_rt index does not match column order.")
        if name in {'thm[K]', 'thlm[K]', 'T[K]'} and idx_thl > 0 and col_idx != idx_thl:
            raise ValueError(f"{label}: iisclr/iiedsclr_thl index does not match column order.")


def _resolve_grid_file_path(namelist_dir: Path, grid_path_value) -> Path:
    """Resolve a grid filename from namelist conventions to an existing path."""
    raw = _clean_namelist_path(grid_path_value)
    if not raw:
        raise ValueError("Grid filename is empty.")

    p = Path(raw)
    repo_root = _CLUBB_RELEASE_ROOT
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(namelist_dir / p)
        candidates.append(repo_root / p)
        if raw.startswith("../input/"):
            candidates.append(repo_root / raw[3:])

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Grid file not found: {raw}. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


# =========================================================================
# Feature gate
# =========================================================================

def _check_unsupported_features(cfg: dict, flags, microphys_scheme: str,
                                rad_scheme: str, l_calc_thlp2_rad: bool):
    """Check for namelist settings that the Python driver does not support.

    Raises ValueError with a clear message listing all unsupported features
    that are enabled, so the user can fix them all at once.
    """
    errors = []

    # --- Microphysics ---
    # 'none', 'khairoutdinov_kogan' (KK), and 'morrison' (M2005 2-moment) are supported. The Morrison
    # rate library + driver + CLUBB interface (morrison_microphys_driver) are ported and validated on
    # nov11_altocu fields; the per-step call is wired in advance_clubb_to_end (gated by the scheme).
    if microphys_scheme not in ("none", "khairoutdinov_kogan", "morrison"):
        errors.append(
            f"microphys_scheme = '{microphys_scheme}' is not supported "
            "(only 'none', 'khairoutdinov_kogan', and 'morrison' are implemented)."
        )

    # (NB: the JAX Morrison is a COMPLETE M2005 port — warm-rain + the full ice/snow/graupel block; the driver
    #  advances rgm/Ngm as prognostic species [morrison_microphys_module.py], so l_ice_microphys/l_graupel/
    #  l_arctic_nucl are all supported. The Morrison *cases* arm_97/twp_ice/lba/mc3e are BLOCKED only because they
    #  additionally need BUGSrad + SILHS, not because of the microphysics. So no microphysics-flag guard is added.)

    # --- Cloud water sedimentation ---
    # cloud_drop_sed is now ported (Microphys/cloud_sed_module.py) and called from
    # the Python driver loop, so l_cloud_sed is supported.

    # --- Radiation ---
    supported_rad = {"none", "simplified", "simplified_bomex", "bugsrad"}
    if rad_scheme not in supported_rad:
        errors.append(
            f"rad_scheme = '{rad_scheme}' is not supported "
            f"(supported: {', '.join(sorted(supported_rad))})."
        )

    if l_calc_thlp2_rad and rad_scheme == "none":
        errors.append(
            "l_calc_thlp2_rad = true is incompatible with rad_scheme = 'none'."
        )

    # --- Sponge damping ---
    # Sponge-layer damping: the xm fields (thlm/rtm/uv) are ported AND wired
    # (sponge_layer_damping.sponge_damp_xm, applied in the advance path).
    # The variance/third-moment routines sponge_damp_xp2/xp3 ARE ported and
    # unit-tested (sponge_layer_damping.py, tests/test_sponge_damp_xp23.py), but
    # they are NOT wired into the JAX advance_xp2_xpyp (up2/vp2) / advance_wp2_wp3
    # (wp2/wp3): the wp2/wp3/up2_vp2 damping profiles are not built in init and no
    # case_setup enables these flags, so there is no validated full-case oracle.
    # Fail loud (rather than silently ignore the flag) until that path is wired.
    _UNWIRED_SPONGE_FIELDS = ["wp2", "wp3", "up2_vp2"]
    sponge_enabled = [
        f for f in _UNWIRED_SPONGE_FIELDS
        if bool(cfg.get(f'{f}_sponge_damp_settings%l_sponge_damping', False))
    ]
    if sponge_enabled:
        names = ", ".join(sponge_enabled)
        errors.append(
            f"Sponge damping is enabled for [{names}] but the variance sponge "
            "(sponge_damp_xp2/xp3) is ported + unit-tested yet not wired into the "
            "JAX advance_xp2_xpyp / advance_wp2_wp3 (no validated full-case oracle)."
        )

    # --- SILHS / Latin Hypercube sampling ---
    lh_type = str(cfg.get('lh_microphys_type', 'disabled')).strip().lower()
    if lh_type != "disabled":
        errors.append(
            f"lh_microphys_type = '{lh_type}' is not supported "
            "(SILHS sampling is not implemented in the Python driver)."
        )
    if bool(cfg.get('l_silhs_rad', False)):
        errors.append("l_silhs_rad = true is not supported (SILHS is not available).")

    # --- Soil / vegetation --- (soil_vegetation.py + gabls3 sfclyr)

    # --- Restarts ---
    if bool(cfg.get('l_restart', False)):
        errors.append("l_restart = true is not supported (no GrADS restart I/O).")

    # --- Input fields (time-dependent forcing from files) ---
    if bool(cfg.get('l_input_fields', False)):
        errors.append("l_input_fields = true is not supported.")

    # --- Second (zm-grid) PDF closure ---
    # When l_call_pdf_closure_twice=true the Fortran calls pdf_closure_driver_zm to compute a second,
    # zm-grid PDF closure (pdf_params_zm). That routine is not ported (gated, no validated case), so the
    # JAX would read an uncomputed pdf_params_zm — fail loud rather than silently use zeros.
    if bool(getattr(flags, 'l_call_pdf_closure_twice', False)) or bool(cfg.get('l_call_pdf_closure_twice', False)):
        errors.append(
            "l_call_pdf_closure_twice = true is not supported "
            "(the second zm-grid PDF closure, pdf_closure_driver_zm, is not ported)."
        )

    # --- PDF-closure call placement ---
    # Only ipdf_post_advance_fields (=2, the default) is ported. The pre-advance / pre-and-post
    # placements (ipdf_pre_advance_fields=1, ipdf_pre_post_advance_fields=3) take a path in
    # advance_clubb_core that lazily imports the Fortran clubb_python (an unavailable module in this
    # tree), so they would crash with a cryptic ModuleNotFoundError rather than run. No case_setup
    # sets a non-default placement; fail loud if one ever does.
    _ipdf_placement = int(getattr(flags, 'ipdf_call_placement',
                                  cfg.get('ipdf_call_placement', 2)))
    if _ipdf_placement != 2:
        errors.append(
            f"ipdf_call_placement = {_ipdf_placement} is not supported "
            "(only ipdf_post_advance_fields = 2 is ported; the pre-advance / pre-post placements "
            "fall back to the unavailable Fortran clubb_python)."
        )

    # --- PDF type ---
    # The JAX pdf_closure_driver wires ONLY ADG1 (iiPDF_type == iiPDF_ADG1 == 1, the default): the block is
    # `if flags.iiPDF_type == iiPDF_ADG1:` with `_adg1 = None` otherwise, so a non-ADG1 type would skip the
    # closure and then crash downstream (None used as the ADG1 dict). The other PDF-type modules (ADG2 / 3D_Luhar /
    # new / TSDADG / LY93 / new_hybrid) are ported as files but not wired into the driver. No case_setup sets a
    # non-default iiPDF_type; fail loud at init if one ever does.
    _iipdf = int(getattr(flags, 'iiPDF_type', cfg.get('iipdf_type', 1)))
    if _iipdf != 1:
        errors.append(
            f"iiPDF_type = {_iipdf} is not supported "
            "(only iiPDF_ADG1 = 1 is wired into the JAX pdf_closure_driver; the ADG2/3D_Luhar/new/TSDADG/"
            "LY93/new_hybrid PDF modules are ported but not wired)."
        )

    # --- Banded-matrix solver method ---
    # The JAX matrix_solver_wrapper only implements the LU solvers (penta_lu / tridiag_lu, both == 2,
    # the default). The Fortran also offers penta_bicgstab (= 3, penta_bicgstab_solver.F90), which is NOT
    # ported. The JAX wrapper does not read penta_solve_method / tridiag_solve_method, so a non-LU request
    # would be silently ignored (the LU solver used instead) — fail loud. No case_setup sets a non-default
    # method. (penta_lu = tridiag_lu = 2 in model_flags; penta_bicgstab = 3.)
    for _flag in ('penta_solve_method', 'tridiag_solve_method'):
        _method = int(getattr(flags, _flag, cfg.get(_flag, 2)))
        if _method != 2:
            errors.append(
                f"{_flag} = {_method} is not supported "
                "(only the LU solver = 2 is ported; penta_bicgstab = 3 / penta_bicgstab_solver is not)."
            )

    # --- Other gated PDF-closure / microphysics features that the JAX passes through but does NOT implement ---
    # (all default-off; the JAX silently ignores them, so fail loud if a case ever turns one on rather than
    #  producing default behavior under a non-default request — same footgun class as l_call_pdf_closure_twice).
    for _flag, _routine in (
        ('l_use_cloud_cover', 'compute_cloud_cover'),
        ('l_trapezoidal_rule_zt', 'trapezoidal_rule_zt'),
        ('l_trapezoidal_rule_zm', 'trapezoidal_rule_zm'),
        ('l_upwind_diff_sed', 'sed_upwind_diff_lhs'),
        ('l_prevent_hm_ta_above_cloud', 'get_cloud_top_level'),
        ('l_godunov_upwind_xpyp_ta', 'xpyp_term_ta_pdf_lhs/rhs_godunov'),
    ):
        if bool(getattr(flags, _flag, False)) or bool(cfg.get(_flag, False)):
            errors.append(f"{_flag} = true is not supported ({_routine} is not ported).")

    # --- Default-FALSE flags the JAX never reads (it hardcodes the default-off behavior) ---
    # Found by the iter-371 "ConfigFlags field never read in src" sweep: these select alternate closure/numerics
    # branches the JAX does not implement (it is bit-faithful to the default-off path, so the on-path is genuinely
    # absent). No case_setup sets any of them; fail loud if one ever does rather than silently ignore the request.
    for _flag, _what in (
        ('l_C2_cloud_frac', 'cloud-fraction-weighted C2 closure coefficient'),
        ('l_Lscale_plume_centered', 'plume-centered mixing-length (Lscale) computation'),
        ('l_do_expldiff_rtm_thlm', 'explicit diffusion of rtm/thlm'),
        ('l_godunov_upwind_wpxp_ta', 'Godunov-upwind wpxp turbulent advection'),
        ('l_ho_trad_coriolis', 'higher-order traditional Coriolis terms'),
        ('l_partial_upwind_wp3', 'partial-upwind wp3 turbulent advection'),
        ('l_stability_correct_Kh_N2_zm', 'N^2 stability correction to Kh on zm'),
        ('l_vert_avg_closure', 'vertically-averaged (rather than pointwise) PDF closure'),
    ):
        if bool(getattr(flags, _flag, False)) or bool(cfg.get(_flag, False)):
            errors.append(f"{_flag} = true is not supported (the {_what} path is not ported).")

    # --- wp2/wp3-closure config the JAX hardcodes to the ARM/ADG1 defaults (iter 497) ---
    # advance_wp2_wp3 (solve_xp2_xpyp_jax docstring) assumes a fixed closure config; these flags are NOT read in
    # src to dispatch (verified iter 497 — distinct from l_lmm_stepping / l_use_C11_Richardson / l_damp_wp2_using_em,
    # which ARE dispatched). No case_setup sets any of them. The iter-371 never-read sweep missed them because each
    # appears in the solve docstring (so the "never referenced" heuristic counted them as read). Default-FALSE here,
    # True-branch unported → fail loud on a True request.
    for _flag, _what in (
        ('l_standard_term_ta', 'standard wp3 turbulent-advection discretization (the JAX uses the non-standard ADG1 form)'),
        ('l_use_tke_in_wp2_wp3_K_dfsn', 'TKE in the wp2/wp3 eddy-diffusion (K) term'),
        ('l_crank_nich_diff', 'Crank-Nicolson diffusion in the wp2/wp3 solve'),
    ):
        if bool(getattr(flags, _flag, False)) or bool(cfg.get(_flag, False)):
            errors.append(f"{_flag} = true is not supported (the {_what} path is not ported).")

    # --- Default-TRUE flags whose FALSE branch the JAX does not implement (it hardcodes the default) ---
    # advance_xm_wpxp.py hardcodes C7 = Cx_fnc_Richardson (l_use_C7_Richardson=True) and C6 = const
    # (l_diag_Lscale_from_tau=True); the false branches (Skw-damped C7, Lscale-damped C6 via damp_coefficient)
    # are not ported, so a case turning either off would silently get the default — fail loud. Both default true,
    # set false by no case.
    for _flag, _why in (
        ('l_use_C7_Richardson', 'the JAX hardcodes C7 = Cx_fnc_Richardson; the Skw-damped-C7 path is not ported'),
        ('l_diag_Lscale_from_tau', 'the JAX hardcodes C6 = const; the Lscale-damped-C6 (damp_coefficient) path is not ported'),
        # iter 371 never-read sweep: l_use_precip_frac (default true) — the JAX always uses precipitation fractions;
        # the no-precip-frac path is not ported.
        ('l_use_precip_frac', 'the JAX always uses precipitation fractions; the l_use_precip_frac=false path is not ported'),
        # iter 497 wp2/wp3-closure sweep (default-TRUE, FALSE branch unported; not dispatched in src):
        ('l_use_tke_in_wp3_pr_turb_term', 'the JAX uses the TKE form of the wp3 pr_turb term; the Kh/buoyancy-shear (false) form is not ported'),
        ('l_damp_wp3_Skw_squared', 'the JAX hardcodes the Skw^2-damped wp3 pr1 term; the false branch is not ported'),
    ):
        # default True; only fire if explicitly set false
        _val = getattr(flags, _flag, cfg.get(_flag, True))
        if _val is False:
            errors.append(f"{_flag} = false is not supported ({_why}).")

    # --- Generalized grid test ---
    if bool(cfg.get('l_test_grid_generalization', False)):
        errors.append("l_test_grid_generalization = true is not supported.")

    # --- Host dynamical-core grid (iter 498) ---
    # l_add_dycore_grid (clubb_driver.F90:2201) builds a separate host dycore grid + remaps onto it — a host-coupling
    # (e.g. CAM) feature with no meaning in the standalone SCM; the JAX driver never reads it (default-False path).
    if bool(getattr(flags, 'l_add_dycore_grid', False)) or bool(cfg.get('l_add_dycore_grid', False)):
        errors.append("l_add_dycore_grid = true is not supported (the host dycore-grid remap is a host-coupling "
                      "feature, not implemented in the standalone JAX driver).")

    # --- Adaptive gridding ---
    # grid_adapt_in_time_method > 0 means some form of adaptation is active.
    grid_adapt = int(cfg.get('grid_adapt_in_time_method', 0))
    if grid_adapt > 0:
        errors.append(
            f"grid_adapt_in_time_method = {grid_adapt} is not supported "
            "(only 0 / no adaptation is implemented)."
        )

    if errors:
        msg = "Python driver does not support the following enabled features:\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)


# =========================================================================
# Initialization
# =========================================================================

def init_clubb_case(namelist_path: str) -> dict:
    """Initialize a CLUBB case from a namelist file.

    Returns a dict containing all model state arrays and config.
    """
    # Reset cross-timestep core state so each init starts clean (reentrancy — allows running
    # multiple cases in one process without the previous case's ADG1 state leaking, Iter281).
    from clubb_jax.src.CLUBB_core.advance_clubb_core_module import reset_clubb_core_state
    reset_clubb_core_state()
    # Resolve to absolute path before any chdir so subsequent uses remain valid.
    namelist_path = str(Path(namelist_path).resolve())
    cfg = read_namelist(namelist_path)
    namelist_dir = Path(namelist_path).parent
    # Fortran time_dependent_input uses hardcoded "../input/case_setups/" relative path.
    # CWD must be a sibling of "input/" (e.g. run_scripts/) so "../input/..." resolves.
    os.chdir(_CLUBB_RELEASE_ROOT / "run_scripts")

    # Unpack key config values
    ngrdcol = cfg['ngrdcol']
    nzmax = cfg['nzmax']
    grid_type = cfg['grid_type']
    dt_main = cfg['dt_main']
    dt_rad = cfg['dt_rad']
    runtype = cfg['runtype']
    sclr_dim = cfg['sclr_dim']
    edsclr_dim = cfg['edsclr_dim']

    # ── 1. Get config flags ─────────────────────────────────────────────────
    flags = get_default_config_flags()
    # Override from namelist (configurable_clubb_flags_nl)
    flag_overrides = {}
    for name in ConfigFlags._fields:
        if name.lower() in cfg:
            flag_overrides[name] = cfg[name.lower()]
    if flag_overrides:
        d = flags._asdict()
        d.update(flag_overrides)
        flags = ConfigFlags(**d)

    saturation_formula = flags.saturation_formula
    microphys_scheme = str(cfg.get('microphys_scheme', 'none')).strip().strip("'\"").lower()
    l_cloud_sed = bool(cfg.get('l_cloud_sed', False))
    sigma_g = float(cfg.get('sigma_g', 1.5))
    rad_scheme = str(cfg.get('rad_scheme', 'none')).strip().strip("'\"").lower()
    l_calc_thlp2_rad = bool(cfg.get('l_calc_thlp2_rad', flags.l_calc_thlp2_rad))

    _check_unsupported_features(cfg, flags, microphys_scheme, rad_scheme, l_calc_thlp2_rad)

    # ── 3. Read sounding ────────────────────────────────────────────────
    snd_path = _resolve_case_input_path(namelist_dir, runtype, "_sounding.in")
    snd = read_sounding(str(snd_path))
    # Pressure-coordinate soundings (CGILS/cloud_feedback): derive level altitudes hydrostatically from the
    # sounding's own thermodynamics before interpolating to the grid. Height-coordinate cases are untouched.
    if str(snd['alt_type']).strip().lower() == 'press[pa]':
        snd = convert_pressure_sounding_to_z(
            snd, float(cfg['p_sfc_nl']), float(cfg['zm_init_nl']), flags.saturation_formula)
    theta_type = snd['theta_type']
    subs_type = snd['subs_type']

    # ── 4. Set up grid ──────────────────────────────────────────────────
    deltaz = np.full(ngrdcol, cfg['deltaz_nl'])
    zm_init = np.full(ngrdcol, cfg['zm_init_nl'])
    zm_top = np.full(ngrdcol, cfg['zm_top_nl'])
    sfc_elevation = np.full(ngrdcol, cfg['sfc_elevation_nl'])

    zt_grid_fname = _clean_namelist_path(cfg.get('zt_grid_fname', ''))
    zm_grid_fname = _clean_namelist_path(cfg.get('zm_grid_fname', ''))

    # For grid_type 1 (even spacing), heights are computed by setup_grid.
    # For stretched grids, these arrays are loaded from *.grd files.
    momentum_heights = None
    thermodynamic_heights = None

    if grid_type == 1:
        if zt_grid_fname or zm_grid_fname:
            raise ValueError(
                "grid_type=1 requires both zt_grid_fname and zm_grid_fname to be empty."
            )
    elif grid_type == 2:
        if zm_grid_fname:
            raise ValueError("grid_type=2 requires zm_grid_fname to be empty.")
        if not zt_grid_fname:
            raise ValueError("grid_type=2 requires zt_grid_fname.")

        zt_grid_path = _resolve_grid_file_path(namelist_dir, zt_grid_fname)
        zt_levels = read_grid_file(str(zt_grid_path))
        expected = nzmax - 1
        if zt_levels.size != expected:
            raise ValueError(
                f"zt grid file {zt_grid_path} has {zt_levels.size} levels; "
                f"expected nzmax-1={expected}."
            )
        thermodynamic_heights = np.tile(zt_levels[None, :], (ngrdcol, 1))
    elif grid_type == 3:
        if zt_grid_fname:
            raise ValueError("grid_type=3 requires zt_grid_fname to be empty.")
        if not zm_grid_fname:
            raise ValueError("grid_type=3 requires zm_grid_fname.")

        zm_grid_path = _resolve_grid_file_path(namelist_dir, zm_grid_fname)
        zm_levels = read_grid_file(str(zm_grid_path))
        expected = nzmax
        if zm_levels.size != expected:
            raise ValueError(
                f"zm grid file {zm_grid_path} has {zm_levels.size} levels; "
                f"expected nzmax={expected}."
            )
        momentum_heights = np.tile(zm_levels[None, :], (ngrdcol, 1))
    else:
        raise ValueError(f"Unsupported grid_type: {grid_type}")

    gr = setup_grid(
        ngrdcol=ngrdcol,
        deltaz=deltaz,
        zm_init=zm_init,
        zm_top=zm_top,
        l_ascending_grid=True,
        grid_type=grid_type,
        momentum_heights=momentum_heights,
        thermodynamic_heights=thermodynamic_heights,
    )

    # Use grid dimensions from Python grid construction.
    nzm = gr.nzm
    nzt = gr.nzt

    print(f"nzm = {nzm} -- nzt = {nzt}")

    # ── 5. Interpolate sounding onto grid ───────────────────────────────
    # Use first column's zt for interpolation
    zt_1d = gr.zt[0, :]  # (nzt,)
    use_cubic_ic = bool(cfg.get('l_modify_ic_with_cubic_int', False))
    snd_interp = interpolate_sounding(snd, zt_1d, use_cubic=use_cubic_ic)

    # Build 2D arrays (ngrdcol, nzt) by broadcasting 1D profile
    thlm = np.tile(snd_interp['theta'], (ngrdcol, 1))
    # Absolute-temperature sounding column (theta_type 'T[K]', the CGILS/cloud_feedback cases): convert
    # T → potential temperature θ using the sounding's OWN pressure interpolated to the grid, BEFORE the
    # hydrostatic bootstrap — faithful to clubb_driver.F90:5499-5524 (thlm = thlm/(p_in_Pa/p0)**kappa). After
    # this θ sits in `thlm` and the `theta_type in ('thm[K]','T[K]')` branch below treats it exactly like a
    # thm[K] (θ) sounding. The Fortran errors on a z-coordinate T[K] sounding, so this only fires for pressure
    # soundings (which carry snd['p_in_Pa'] from convert_pressure_sounding_to_z). Without it thlm was left as
    # the raw absolute temperature → a ~69 K init error (Iter90 cgils_s11 diagnosis).
    if str(theta_type).strip() == 'T[K]' and snd.get('p_in_Pa') is not None:
        p_snd_zt = np.interp(zt_1d, np.asarray(snd['z']), np.asarray(snd['p_in_Pa']))
        thlm = thlm / (p_snd_zt[np.newaxis, :] / p0) ** kappa
    rtm = np.tile(snd_interp['rt'], (ngrdcol, 1))
    # NO IC floor: the Fortran does NOT floor the initial rtm — its rtm_old entering step 1 is
    # the bare sounding value (~0/2e-19 at rico's dry top, NOT 1e-8). It is the per-step fill_holes inside
    # advance_xm_wpxp (rtm_cl budget, advance_clubb_core_module.py:~1700) that raises the sub-rt_tol dry top
    # to 1e-8 each step, mass-conservingly pulling a tiny amount (~4.5e-11) from the topmost moist level.
    # The old Iter141 `rtm=max(rtm,rt_tol)` floor pre-empted that fill → it never fired → the donor-level
    # mass transfer (rico's step-1 k51 rtm seed) was missing. No-op for the 15 cases (rtm >= rt_tol at IC).
    um = np.tile(snd_interp['u'], (ngrdcol, 1))
    vm = np.tile(snd_interp['v'], (ngrdcol, 1))
    ug = np.tile(snd_interp['ug'], (ngrdcol, 1))
    vg = np.tile(snd_interp['vg'], (ngrdcol, 1))
    wm_zt = np.tile(snd_interp['w'], (ngrdcol, 1))
    p_in_Pa = np.zeros((ngrdcol, nzt))  # will be computed

    # ── 6. Initialize pressure / thermodynamic variables ────────────────
    p_sfc = np.full(ngrdcol, cfg['p_sfc_nl'])
    T0 = cfg['t0']
    fcor_nl = cfg['fcor_nl']
    lat_vals = cfg['lat_vals']

    fcor = np.full(ngrdcol, fcor_nl)
    fcor_y = np.full(ngrdcol, 2.0 * omega_planet * math.cos(lat_vals * radians_per_deg))

    # Compute initial thvm (approximation: thvm = thlm * (1 + ep1 * rv))
    # where rv = rtm / (1 + rtm)
    thvm = thlm * (1.0 + ep1 * (rtm / (1.0 + rtm)))

    # Hydrostatic pressure (hydrostatic)
    _hyd = hydrostatic(jnp.asarray(thvm), jnp.asarray(p_sfc), gr)
    p_in_Pa, p_in_Pa_zm, exner, exner_zm, rho, rho_zm = [
        np.array(x, dtype=np.float64) for x in _hyd
    ]

    # Convert temperature type
    if theta_type in ('thm[K]', 'T[K]'):
        # theta sounding — need to compute rcm and convert to thlm
        thm = thlm.copy()
        # rcm = max(rtm - rsat(p, T), 0)
        T_in_K = thm * exner
        # sat_mixrat_liq replaced with sat_mixrat_liq
        rsat = np.asarray(sat_mixrat_liq(
            jnp.asarray(p_in_Pa), jnp.asarray(T_in_K), saturation_formula,
        ), dtype=np.float64)
        rcm = np.maximum(rtm - rsat, 0.0)
        thlm = thm - Lv / (Cp * exner) * rcm
    elif theta_type == 'thlm[K]':
        # Already liquid potential temperature — rcm_sat_adj
        rcm = np.asarray(rcm_sat_adj(
            jnp.asarray(thlm), jnp.asarray(rtm),
            jnp.asarray(p_in_Pa), jnp.asarray(exner),
            saturation_formula,
        ), dtype=np.float64)
        thm = thlm + Lv / (Cp * exner) * rcm
    else:
        raise ValueError(f"Unknown theta_type: {theta_type}")

    # Recompute thvm and hydrostatic with corrected thlm
    # NOTE: Fortran passes thm (not thlm) as the 5th arg to calculate_thvm
    # calculate_thvm replaced with calculate_thvm
    _thv_ds_zt = thm * (1.0 + ep2 * (rtm - rcm))**kappa
    thvm = np.asarray(calculate_thvm(
        jnp.asarray(thlm), jnp.asarray(rtm), jnp.asarray(rcm),
        jnp.asarray(exner), jnp.asarray(_thv_ds_zt),
    ), dtype=np.float64)
    # second hydrostatic call also replaced with hydrostatic
    _hyd2 = hydrostatic(jnp.asarray(thvm), jnp.asarray(p_sfc), gr)
    p_in_Pa, p_in_Pa_zm, exner, exner_zm, rho, rho_zm = [
        np.array(x, dtype=np.float64) for x in _hyd2
    ]

    # Compute dry static density (anelastic base state)
    # NOTE: thm was already computed before the 2nd hydrostatic call
    # (from sounding for thm[K] case, or from thlm+Lv/(Cp*exner)*rcm for
    # thlm[K] case). Do NOT recompute it here with the updated exner —
    # the Fortran uses the original thm throughout.
    rv = rtm - rcm  # water vapor mixing ratio
    p_dry = p_in_Pa / (1.0 + ep2 * rv)
    exner_dry = (p_dry / p0)**kappa
    th_dry = thm * (1.0 + ep2 * rv)**kappa
    rho_dry = p_dry / (Rd * th_dry * exner_dry)

    rho_ds_zt = rho_dry.copy()
    thv_ds_zt = th_dry.copy()
    invrs_rho_ds_zt = 1.0 / rho_ds_zt

    # Momentum level versions via zt2zm interpolation
    rv_zm = np.maximum(np.asarray(zt2zm(gr.nzm, gr.nzt, gr.ngrdcol, gr, jnp.asarray(rv)), dtype=np.float64), 0.0)
    thm_zm = np.asarray(zt2zm(gr.nzm, gr.nzt, gr.ngrdcol, gr, jnp.asarray(thm)), dtype=np.float64)

    # rtm_sfc: linearly interpolate sounding rt to the zm surface level,
    # matching Fortran read_sounding which interpolates to gr%zm(1).
    zm_sfc = gr.zm[0, 0]  # surface momentum level (z=0 typically)
    z_snd = snd['z']
    rt_snd = snd['rt']
    valid_rt = rt_snd > -998.0
    if np.sum(valid_rt) >= 2 and zm_sfc >= z_snd[valid_rt][0]:
        rtm_sfc = float(np.interp(zm_sfc, z_snd[valid_rt], rt_snd[valid_rt]))
    else:
        rtm_sfc = float(rtm[0, 0])  # fallback: use lowest zt level
    pd_sfc = p_sfc / (1.0 + ep2 * rtm_sfc)

    p_dry_zm = p_in_Pa_zm / (1.0 + ep2 * rv_zm)
    p_dry_zm[:, 0] = pd_sfc
    exner_dry_zm = (p_dry_zm / p0)**kappa
    th_dry_zm = thm_zm * (1.0 + ep2 * rv_zm)**kappa
    rho_dry_zm = p_dry_zm / (Rd * th_dry_zm * exner_dry_zm)

    rho_ds_zm = rho_dry_zm.copy()
    thv_ds_zm = th_dry_zm.copy()
    invrs_rho_ds_zm = 1.0 / rho_ds_zm

    # ── 7. Subsidence / vertical wind ───────────────────────────────────
    # Boundary handling is subs_type-dependent, faithful to clubb_driver.F90:4748-4763:
    #   wm[m/s]:     wm_zm = zt2zm(wm_zt); zero BOTH boundaries.
    #   omega[Pa/s]: wm_zt = -wm_zt/(grav*rho), wm_zt[top]=0; wm_zm = zt2zm(wm_zt); zero TOP ONLY
    #                (the bottom keeps the interpolated value — e.g. jun25's 6.94e-5). The old code
    #                always zeroed both, which was unfaithful for omega cases with init subsidence.
    if subs_type == 'omega[Pa/s]':
        wm_zt = -wm_zt / (grav * rho)
        wm_zt[:, -1] = 0.0
        wm_zm = np.array(zt2zm(gr.nzm, gr.nzt, gr.ngrdcol, gr, jnp.asarray(wm_zt)), dtype=np.float64)
        wm_zm[:, -1] = 0.0
    else:
        wm_zm = np.array(zt2zm(gr.nzm, gr.nzt, gr.ngrdcol, gr, jnp.asarray(wm_zt)), dtype=np.float64)
        wm_zm[:, 0] = 0.0
        wm_zm[:, -1] = 0.0

    # ── 8. Initialize PDF and tunable parameters ────────────────────────
    # Use __file__-relative path so it works regardless of namelist location.
    _tunable_params_path = str(
        (_CLUBB_RELEASE_ROOT / "input/tunable_parameters/tunable_parameters.in").resolve()
    )
    clubb_params = init_clubb_params(ngrdcol, filename=_tunable_params_path)
    # Apply per-column parameter overrides from &clubb_params_nl (multicol runs).
    # cfg contains lowercased keys; build case-insensitive name→index map.
    # Scalars from the namelist apply to all columns (ngrdcol=1 or uniform override).
    # Lists of length ngrdcol give per-column values.
    _param_names = get_param_names()
    _name_to_idx = {n.lower(): i for i, n in enumerate(_param_names)}
    for _key, _val in cfg.items():
        if _key not in _name_to_idx:
            continue
        _pidx = _name_to_idx[_key]
        if isinstance(_val, list) and len(_val) == ngrdcol:
            for _col in range(ngrdcol):
                clubb_params[_col, _pidx] = float(_val[_col])
        elif not isinstance(_val, list) and isinstance(_val, (int, float)):
            # Scalar override applies to all columns
            clubb_params[:, _pidx] = float(_val)
    pdf_params = init_pdf_params(nzt, ngrdcol)
    pdf_params_zm = init_pdf_params(nzm, ngrdcol)   # NB: Fortran uses nzm for pdf_params_zm
    pdf_implicit_coefs_terms = init_pdf_implicit_coefs_terms_api(nzt, ngrdcol, sclr_dim)

    # Scalar indices (mirror initialize_clubb defaults/namelist overrides).
    iisclr_rt = int(cfg.get('iisclr_rt', -1))
    iisclr_thl = int(cfg.get('iisclr_thl', -1))
    iisclr_co2 = int(cfg.get('iisclr_co2', -1))
    iiedsclr_rt = int(cfg.get('iiedsclr_rt', -1))
    iiedsclr_thl = int(cfg.get('iiedsclr_thl', -1))
    iiedsclr_co2 = int(cfg.get('iiedsclr_co2', -1))
    sclr_idx = SclrIdx(
        iisclr_rt=iisclr_rt,
        iisclr_thl=iisclr_thl,
        iisclr_CO2=iisclr_co2,
        iiedsclr_rt=iiedsclr_rt,
        iiedsclr_thl=iiedsclr_thl,
        iiedsclr_CO2=iiedsclr_co2,
    )
    nu_vert_res_dep, lmin, mixt_frac_max_mag = calc_derrived_params(
        gr=gr,
        ngrdcol=ngrdcol,
        grid_type=grid_type,
        deltaz=deltaz,
        clubb_params=clubb_params,
        l_prescribed_avg_deltaz=False,
    )
    err_info = ErrInfo(ngrdcol=ngrdcol)

    err_info = check_clubb_settings(
        ngrdcol=ngrdcol,
        params=clubb_params,
        config_flags=flags,
        err_info=err_info,
        l_implemented=False,
        l_input_fields=False,
    )
    err_info = check_parameters(
        ngrdcol=ngrdcol,
        clubb_params=clubb_params,
        lmin=lmin,
        err_info=err_info,
    )
    # Reset after validation warnings (all warnings are non-fatal for default ARM config)
    err_info = ErrInfo(ngrdcol=ngrdcol)

    # ── 9. Initialize TKE / variances (case-specific, Fortran-like) ────
    em, wp2, up2, vp2, upwp, um = _initialize_turbulence_state(
        runtype=runtype,
        gr=gr,
        dt_main=dt_main,
        fcor_y=fcor_y,
        l_tke_aniso=bool(flags.l_tke_aniso),
        um=um,
    )

    # ── 10. Initialize remaining prognostic arrays to zero ──────────────

    # Reference profiles (matches initialize_clubb logic in Fortran driver,
    # clubb_driver.F90:5298-5316). The reference is the initial sounding profile
    # and is captured only when the field's sponge (or uv nudge) is active.
    uv_sponge_enabled   = bool(cfg.get('uv_sponge_damp_settings%l_sponge_damping', False))
    thlm_sponge_enabled = bool(cfg.get('thlm_sponge_damp_settings%l_sponge_damping', False))
    rtm_sponge_enabled  = bool(cfg.get('rtm_sponge_damp_settings%l_sponge_damping', False))
    if flags.l_uv_nudge or uv_sponge_enabled:
        um_ref = um.copy()
        vm_ref = vm.copy()
    else:
        um_ref = np.zeros((ngrdcol, nzt))
        vm_ref = np.zeros((ngrdcol, nzt))
    thlm_ref = thlm.copy() if thlm_sponge_enabled else np.zeros((ngrdcol, nzt))
    rtm_ref  = rtm.copy()  if rtm_sponge_enabled  else np.zeros((ngrdcol, nzt))

    # Sponge-layer damping config: precompute the per-field damping-timescale
    # profile once (sponge_layer_damping.initialize_tau_sponge_damp). Applied
    # inside advance_clubb_core after each xm solve. Only the xm-field (rtm/thlm/uv)
    # profiles are built + wired; the variance sponge (wp2/wp3/up2_vp2 → sponge_damp_xp2/xp3)
    # is ported + unit-tested but unwired (no profiles built here), so it is gated off in
    # _check_unsupported_features above.
    sponge_cfg = {}
    _zt_col = np.asarray(gr.zt, dtype=np.float64)[0, :]
    _zm_top = float(np.asarray(gr.zm, dtype=np.float64)[0, -1])
    for _key, _on in (('rtm', rtm_sponge_enabled), ('thlm', thlm_sponge_enabled),
                      ('uv', uv_sponge_enabled)):
        if not _on:
            continue
        _tau_min = float(cfg.get(f'{_key}_sponge_damp_settings%tau_sponge_damp_min', 60.0))
        _tau_max = float(cfg.get(f'{_key}_sponge_damp_settings%tau_sponge_damp_max', 1800.0))
        _depth_f = float(cfg.get(f'{_key}_sponge_damp_settings%sponge_damp_depth', 0.25))
        _tau, _depth = initialize_tau_sponge_damp(
            _zt_col, float(dt_main), _zm_top, _tau_min, _tau_max, _depth_f)
        sponge_cfg[_key] = {'tau': _tau, 'depth': _depth}

    # Cloud properties
    nc0_in_cloud = float(cfg.get('nc0_in_cloud', Nc0_in_cloud))
    Nc_in_cloud = nc0_in_cloud / rho
    cloud_frac = np.zeros((ngrdcol, nzt))
    Ncm = np.where(rcm > 0, Nc_in_cloud, Nc_in_cloud * cloud_frac_min)

    # Scalar/hydromet arrays carry a padded trailing extent (max(dim, 1)) so a dim=0 case still has a
    # well-formed (…, 1) trailing axis; the logical *_dim values remain authoritative for the active extent.
    # KK microphysics predicts 2 hydrometeors (rrm at idx 0, Nrm at idx 1); all other supported
    # cases are hydromet_dim=0. The hydromet means are initialised to 0 (rico's sounding has no rain).
    hm_metadata = None
    if microphys_scheme == "khairoutdinov_kogan":
        from clubb_jax.src.CLUBB_core.corr_varnce_module import kk_hm_metadata
        hm_metadata = kk_hm_metadata()
        hydromet_dim = hm_metadata.hydromet_dim   # = 2
    elif microphys_scheme == "morrison":
        from clubb_jax.src.CLUBB_core.corr_varnce_module import morrison_hm_metadata
        hm_metadata = morrison_hm_metadata()
        hydromet_dim = hm_metadata.hydromet_dim   # = 8 (rr/Nr/ri/Ni/rs/Ns/rg/Ng)
    else:
        hydromet_dim = 0
    hm_dim_transport = max(hydromet_dim, 1)
    hydromet = np.zeros((ngrdcol, nzt, hm_dim_transport))   # mean hydrometeors (rrm, Nrm)
    l_mix_rat_hm = (np.asarray(hm_metadata.l_mix_rat_hm) if hm_metadata is not None
                    else np.zeros((hm_dim_transport,), dtype=bool))
    wphydrometp = np.zeros((ngrdcol, nzm, hm_dim_transport))
    wp2hmp = np.zeros((ngrdcol, nzt, hm_dim_transport))
    rtphmp_zt = np.zeros((ngrdcol, nzt, hm_dim_transport))
    thlphmp_zt = np.zeros((ngrdcol, nzt, hm_dim_transport))

    sc_dim_transport = max(sclr_dim, 1)
    edsc_dim_transport = max(edsclr_dim, 1)
    # sclr_tol_nl may be parsed as a scalar float (single value, e.g. astex_a209) or a list
    # (e.g. atex: "1.e-2, 1.e-8"); normalise to a 1-D array before slicing.
    sclr_tol = np.atleast_1d(np.asarray(cfg.get('sclr_tol_nl', []), dtype=np.float64)).ravel()[:sclr_dim]
    if len(sclr_tol) < sclr_dim:
        sclr_tol = np.pad(sclr_tol, (0, sclr_dim - len(sclr_tol)), constant_values=1e-8)
    sclrm = np.zeros((ngrdcol, nzt, sc_dim_transport))
    sclrp2 = np.zeros((ngrdcol, nzm, sc_dim_transport))
    if sclr_dim > 0:
        sclrp2[:, :, :sclr_dim] = sclr_tol[:sclr_dim].reshape(1, 1, sclr_dim) ** 2
    sclrp3 = np.zeros((ngrdcol, nzt, sc_dim_transport))
    sclrprtp = np.zeros((ngrdcol, nzm, sc_dim_transport))
    sclrpthlp = np.zeros((ngrdcol, nzm, sc_dim_transport))
    sclrpthvp = np.zeros((ngrdcol, nzm, sc_dim_transport))
    wpsclrp = np.zeros((ngrdcol, nzm, sc_dim_transport))
    sclrm_forcing = np.zeros((ngrdcol, nzt, sc_dim_transport))
    wpsclrp_sfc = np.zeros((ngrdcol, sc_dim_transport))

    edsclrm = np.zeros((ngrdcol, nzt, edsc_dim_transport))
    edsclrm_forcing = np.zeros((ngrdcol, nzt, edsc_dim_transport))
    wpedsclrp_sfc = np.zeros((ngrdcol, edsc_dim_transport))

    # Initialize scalar means from dedicated scalar sounding files.
    if sclr_dim > 0:
        sclr_path = _resolve_case_input_path(namelist_dir, runtype, "_sclr_sounding.in")
        sclr_raw = read_scalar_sounding(str(sclr_path), sclr_dim)
        _validate_scalar_column_names(
            sclr_raw['names'], iisclr_rt, iisclr_thl, iisclr_co2, label='sclr_sounding'
        )
        sclr_zt = interpolate_scalar_sounding(
            snd['z'], sclr_raw['data'], zt_1d, use_cubic=use_cubic_ic
        )
        sclrm[:, :, :sclr_dim] = np.tile(sclr_zt[None, :, :], (ngrdcol, 1, 1))

    if edsclr_dim > 0:
        edsclr_path = _resolve_case_input_path(namelist_dir, runtype, "_edsclr_sounding.in")
        edsclr_raw = read_scalar_sounding(str(edsclr_path), edsclr_dim)
        _validate_scalar_column_names(
            edsclr_raw['names'], iiedsclr_rt, iiedsclr_thl, iiedsclr_co2, label='edsclr_sounding'
        )
        edsclr_zt = interpolate_scalar_sounding(
            snd['z'], edsclr_raw['data'], zt_1d, use_cubic=use_cubic_ic
        )
        edsclrm[:, :, :edsclr_dim] = np.tile(edsclr_zt[None, :, :], (ngrdcol, 1, 1))

    # ── 11. Read surface file ───────────────────────────────────────────
    sfc_path = None
    try:
        sfc_path = _resolve_case_input_path(namelist_dir, runtype, "_sfc.in")
    except FileNotFoundError:
        sfc_path = None
    sfc_data = None
    if sfc_path is not None and sfc_path.exists():
        sfc_data = read_surface(str(sfc_path))

    # ── 12. Time controls ───────────────────────────────────────────────
    time_initial = cfg['time_initial']
    time_final = cfg['time_final']
    ifinal = int(math.floor((time_final - time_initial) / dt_main))
    stats_nsamp = int(round(cfg['stats_tsamp'] / dt_main))
    stats_nout = int(round(cfg['stats_tout'] / dt_main))

    # ── 13. Initialize stats ────────────────────────────────────────────
    l_stats = bool(cfg['l_stats'])
    repo_root = _CLUBB_RELEASE_ROOT
    stats_registry_path = _resolve_stats_registry_path(namelist_path, cfg)
    stats_prefix = str(cfg.get('fname_prefix', '')).strip() or runtype
    output_dir_raw = str(cfg.get("output_dir", "")).strip().strip("'\"")
    if output_dir_raw:
        output_dir_path = Path(output_dir_raw)
        if not output_dir_path.is_absolute():
            output_dir_path = (namelist_dir / output_dir_path).resolve()
    else:
        output_dir_path = repo_root / "output"
    stats_output_path = output_dir_path / f"{stats_prefix}_stats.nc"

    stats_writer = None
    if l_stats:
        if not stats_registry_path.exists():
            raise FileNotFoundError(f"Stats registry file not found: {stats_registry_path}")
        stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        stats_writer = StatsWriter(
            registry_path=str(stats_registry_path),
            output_path=str(stats_output_path),
            nzt=nzt,
            nzm=nzm,
            ngrdcol=ngrdcol,
            zt=gr.zt[0, :],
            zm=gr.zm[0, :],
            stats_tsamp=float(cfg['stats_tsamp']),
            stats_tout=float(cfg['stats_tout']),
            dt_main=float(dt_main),
            day=int(cfg['day']),
            month=int(cfg['month']),
            year=int(cfg['year']),
            time_initial=float(time_initial),
            clubb_params_vals=clubb_params,
            param_names=get_param_names(),
            sclr_dim=sclr_dim,
            edsclr_dim=edsclr_dim,
        )

    # ── 14. Zero PDF params ─────────────────────────────────────────────
    pdf_params = init_pdf_params(nzt, ngrdcol)
    pdf_params_zm = init_pdf_params(nzm, ngrdcol)

    # ── Build state dict ────────────────────────────────────────────────
    state = dict(
        # Config
        cfg=cfg, flags=flags, gr=gr, namelist_dir=str(namelist_dir),
        runtype=runtype, ngrdcol=ngrdcol, nzt=nzt, nzm=nzm,
        dt_main=dt_main, dt_rad=dt_rad,
        time_initial=time_initial, time_final=time_final,
        ifinal=ifinal, l_stats=l_stats,
        stats_nsamp=stats_nsamp, stats_nout=stats_nout,
        stats_registry_path=str(stats_registry_path),
        stats_output_path=str(stats_output_path),
        saturation_formula=saturation_formula,
        sfctype=int(cfg['sfctype']),
        microphys_scheme=microphys_scheme,
        # Apply the full KK rain microphysics (rates + hydrometeor transport) for KK cases.
        l_kk_micro_apply=(microphys_scheme == "khairoutdinov_kogan"),
        # The microphysics is skipped until this time (microphys_driver.F90:389; nov11=64800 → 60-step
        # spinup); default 0 = active from the start.
        microphys_start_time=float(cfg.get('microphys_start_time', 0.0)),
        l_cloud_sed=l_cloud_sed,
        sigma_g=sigma_g,
        nc0_in_cloud=nc0_in_cloud,
        rad_scheme=rad_scheme,
        l_calc_thlp2_rad=l_calc_thlp2_rad,
        hydromet_dim=hydromet_dim, hydromet=hydromet, hm_metadata=hm_metadata,
        sclr_dim=sclr_dim, edsclr_dim=edsclr_dim,
        T0=T0, lmin=lmin, mixt_frac_max_mag=mixt_frac_max_mag,
        ts_nudge=cfg['ts_nudge'],
        rtm_min=cfg['rtm_min'],
        rtm_nudge_max_altitude=cfg['rtm_nudge_max_altitude'],
        l_t_dependent=bool(cfg.get('l_t_dependent', False)),
        l_ignore_forcings=bool(cfg.get('l_ignore_forcings', False)),
        l_input_xpwp_sfc=bool(cfg.get('l_input_xpwp_sfc', False)),
        iisclr_rt=iisclr_rt, iisclr_thl=iisclr_thl, iisclr_co2=iisclr_co2,
        iiedsclr_rt=iiedsclr_rt, iiedsclr_thl=iiedsclr_thl, iiedsclr_co2=iiedsclr_co2,
        sclr_idx=sclr_idx,
        nu_vert_res_dep=nu_vert_res_dep,
        pdf_params=pdf_params,
        pdf_params_zm=pdf_params_zm,
        pdf_implicit_coefs_terms=pdf_implicit_coefs_terms,
        err_info=err_info,
        l_modify_bc_for_cnvg_test=bool(cfg.get('l_modify_bc_for_cnvg_test', False)),
        sfc_data=sfc_data,
        # 1D arrays
        fcor=fcor, fcor_y=fcor_y, sfc_elevation=sfc_elevation,
        p_sfc=p_sfc,
        host_dx=np.full(ngrdcol, 1.0e6),
        host_dy=np.full(ngrdcol, 1.0e6),
        upwp_sfc_pert=np.zeros((ngrdcol,)),
        vpwp_sfc_pert=np.zeros((ngrdcol,)),
        sclr_tol=sclr_tol,
        l_mix_rat_hm=l_mix_rat_hm,
        clubb_params=clubb_params,
        # Prognostic zt (ngrdcol, nzt)
        um=um, vm=vm, thlm=thlm, rtm=rtm,
        up3=np.zeros((ngrdcol, nzt)), vp3=np.zeros((ngrdcol, nzt)),
        rtp3=np.zeros((ngrdcol, nzt)), thlp3=np.zeros((ngrdcol, nzt)),
        wp3=np.zeros((ngrdcol, nzt)),
        p_in_Pa=p_in_Pa, exner=exner, rcm=rcm,
        cloud_frac=cloud_frac,
        wp2thvp=np.zeros((ngrdcol, nzt)), wp2up=np.zeros((ngrdcol, nzt)),
        wp2rtp=np.zeros((ngrdcol, nzt)), wp2thlp=np.zeros((ngrdcol, nzt)),
        wpup2=np.zeros((ngrdcol, nzt)), wpvp2=np.zeros((ngrdcol, nzt)),
        ice_supersat_frac=np.zeros((ngrdcol, nzt)),
        um_pert=np.zeros((ngrdcol, nzt)), vm_pert=np.zeros((ngrdcol, nzt)),
        # Prognostic zm (ngrdcol, nzm)
        upwp=upwp, vpwp=np.zeros((ngrdcol, nzm)),
        up2=up2, vp2=vp2,
        wprtp=np.zeros((ngrdcol, nzm)), wpthlp=np.zeros((ngrdcol, nzm)),
        rtp2=np.full((ngrdcol, nzm), rt_tol**2),
        thlp2=np.full((ngrdcol, nzm), thl_tol**2),
        rtpthlp=np.zeros((ngrdcol, nzm)),
        wp2=wp2,
        wpthvp=np.zeros((ngrdcol, nzm)), rtpthvp=np.zeros((ngrdcol, nzm)),
        thlpthvp=np.zeros((ngrdcol, nzm)),
        uprcp=np.zeros((ngrdcol, nzm)), vprcp=np.zeros((ngrdcol, nzm)),
        rc_coef_zm=np.zeros((ngrdcol, nzm)),
        wp4=np.zeros((ngrdcol, nzm)),
        wp2up2=np.zeros((ngrdcol, nzm)), wp2vp2=np.zeros((ngrdcol, nzm)),
        upwp_pert=np.zeros((ngrdcol, nzm)), vpwp_pert=np.zeros((ngrdcol, nzm)),
        # Forcing arrays
        thlm_forcing=np.zeros((ngrdcol, nzt)), rtm_forcing=np.zeros((ngrdcol, nzt)),
        um_forcing=np.zeros((ngrdcol, nzt)), vm_forcing=np.zeros((ngrdcol, nzt)),
        wprtp_forcing=np.zeros((ngrdcol, nzm)), wpthlp_forcing=np.zeros((ngrdcol, nzm)),
        rtp2_forcing=np.zeros((ngrdcol, nzm)), thlp2_forcing=np.zeros((ngrdcol, nzm)),
        rtpthlp_forcing=np.zeros((ngrdcol, nzm)),
        # Meteorological profiles
        wm_zt=wm_zt, wm_zm=wm_zm,
        rho=rho, rho_zm=rho_zm,
        Ncm=Ncm, Nc_in_cloud=Nc_in_cloud,
        rho_ds_zt=rho_ds_zt, rho_ds_zm=rho_ds_zm,
        invrs_rho_ds_zt=invrs_rho_ds_zt, invrs_rho_ds_zm=invrs_rho_ds_zm,
        thv_ds_zt=thv_ds_zt, thv_ds_zm=thv_ds_zm,
        thvm=thvm, radht=np.zeros((ngrdcol, nzt)),
        rcm_mc=np.zeros((ngrdcol, nzt)),
        thlm_mc=np.zeros((ngrdcol, nzt)),
        rfrzm=np.zeros((ngrdcol, nzt)),
        # Reference profiles
        um_ref=um_ref, vm_ref=vm_ref,
        thlm_ref=thlm_ref, rtm_ref=rtm_ref,
        sponge=sponge_cfg,
        ug=ug, vg=vg,
        # Hydromet
        wphydrometp=wphydrometp,
        wp2hmp=wp2hmp, rtphmp_zt=rtphmp_zt, thlphmp_zt=thlphmp_zt,
        # Scalars
        sclrm=sclrm, sclrp2=sclrp2, sclrp3=sclrp3,
        sclrprtp=sclrprtp, sclrpthlp=sclrpthlp, sclrpthvp=sclrpthvp,
        wpsclrp=wpsclrp,
        sclrm_forcing=sclrm_forcing,
        wpsclrp_sfc=wpsclrp_sfc,
        edsclrm=edsclrm, edsclrm_forcing=edsclrm_forcing,
        wpedsclrp_sfc=wpedsclrp_sfc,
        # Surface fluxes (will be set by forcings)
        wpthlp_sfc=np.zeros((ngrdcol,)),
        wprtp_sfc=np.zeros((ngrdcol,)),
        upwp_sfc=np.zeros((ngrdcol,)),
        vpwp_sfc=np.zeros((ngrdcol,)),
        T_sfc=np.full(ngrdcol, float(cfg.get('t_sfc_nl', 288.0))),
        sens_ht=float(cfg.get('sens_ht', 0.0)),
        latent_ht=float(cfg.get('latent_ht', 0.0)),
        # Output / diagnostic
        thlprcp=np.zeros((ngrdcol, nzm)),
        # Python stats writer (replaces Fortran stats API)
        stats_writer=stats_writer,
    )

    # ── Case forcing data: pre-load and vertically interpolate ──────────────
    if runtype == 'arm':
        arm_forcings_path = _resolve_case_input_path(namelist_dir, 'arm', '_forcings.in')
        arm_sfc_path      = _resolve_case_input_path(namelist_dir, 'arm', '_sfc.in')
        state['_arm_forcings_data'] = load_arm_forcings_data(
            str(arm_forcings_path), str(arm_sfc_path), gr.zt[0, :]
        )
    else:
        # Generic cases: load {runtype}_forcings.in and {runtype}_sfc.in if present
        case_setups_dir = str(_CLUBB_RELEASE_ROOT / "input" / "case_setups")
        state['_forcings_data'] = load_generic_forcings_data(
            runtype, case_setups_dir, gr.zt[0, :], p_in_Pa=np.asarray(p_in_Pa)[0, :]
        )

    # Radiation extended atmosphere from the case's OWN sounding + ozone sounding when
    # l_use_default_std_atmosphere=.false. (CGILS/cloud_feedback/astex/twp_ice). The Fortran
    # (convert_snd2extended_atm) builds the above-model-top T/q/p/o3 radiation profile from the deep case
    # sounding + {case}_ozone_sounding.in rather than the default US-standard atmosphere; without this those
    # cases get a model-top radht bias (Iter91 diagnosis). Gated cases keep the flag true → std atmosphere.
    if rad_scheme == 'bugsrad' and not bool(cfg.get('l_use_default_std_atmosphere', True)) \
            and snd.get('p_in_Pa') is not None:
        _oz_path = _CLUBB_RELEASE_ROOT / "input" / "case_setups" / f"{runtype}_ozone_sounding.in"
        if _oz_path.exists():
            from clubb_jax.src.Input_fields.sounding import (
                read_ozone_sounding, convert_snd2extended_atm)
            _o3l = read_ozone_sounding(str(_oz_path))
            state['_rad_ext_atm'] = convert_snd2extended_atm(
                snd['z'], snd['theta'], theta_type, snd['rt'], snd['p_in_Pa'],
                float(cfg['p_sfc_nl']), _o3l)

    print(f"Initialized {runtype} case: nzm={nzm}, nzt={nzt}, ngrdcol={ngrdcol}")
    print(f"  dt_main={dt_main}s, time={time_initial}s to {time_final}s, {ifinal} steps")

    return state

def clean_up_clubb(state: dict):
    """Clean up state."""
    if state['l_stats']:
        sw = state.get('stats_writer')
        if sw is not None:
            sw.finalize()
    print("CLUBB cleanup complete.")
