"""Regression guard for the fail-loud unsupported-config validation (clubb_driver iters 346-349).

The JAX passes several gated CLUBB flags through to the closure/microphysics routines but does NOT
implement their non-default behavior (it hardcodes the validated default). To avoid silently mis-computing
when a case turns one on/off, `clubb_driver._check_unsupported_features` raises ValueError. This test pins
that each of the 9 guarded flags trips the error — so a future change that drops a guard is caught.

Pure-Python (no JAX physics run), never SKIPs.
"""
import os
import sys
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _p in (_ROOT + "/clubb_release", _ROOT + "/clubb_release/clubb_python_api"):
    if _p not in sys.path:
        sys.path.append(_p)

from clubb_jax.src.clubb_driver import _check_unsupported_features as _chk
from clubb_jax.src.CLUBB_core.model_flags import get_default_config_flags

_BASE = dict(microphys_scheme="none", rad_scheme="none", l_calc_thlp2_rad=False)

# (flag, set-via-cfg-as-True) for the reject-TRUE guards
_REJECT_TRUE = [
    "l_call_pdf_closure_twice", "l_use_cloud_cover", "l_trapezoidal_rule_zt", "l_trapezoidal_rule_zm",
    "l_upwind_diff_sed", "l_prevent_hm_ta_above_cloud", "l_godunov_upwind_xpyp_ta",
    # iter-371 never-read sweep: default-FALSE flags the JAX never reads (alternate branches unported)
    "l_C2_cloud_frac", "l_Lscale_plume_centered", "l_do_expldiff_rtm_thlm", "l_godunov_upwind_wpxp_ta",
    "l_ho_trad_coriolis", "l_partial_upwind_wp3", "l_stability_correct_Kh_N2_zm", "l_vert_avg_closure",
    # iter-497 wp2/wp3-closure sweep (default-FALSE, true-branch unported, not dispatched in src)
    "l_standard_term_ta", "l_use_tke_in_wp2_wp3_K_dfsn", "l_crank_nich_diff",
    # iter-498: host dycore-grid coupling (default-FALSE, standalone-N/A)
    "l_add_dycore_grid",
]
# reject-FALSE guards (default-TRUE flags whose false branch is unimplemented)
_REJECT_FALSE = ["l_use_C7_Richardson", "l_diag_Lscale_from_tau", "l_use_precip_frac",
                 # iter-497 wp2/wp3-closure sweep (default-TRUE, false-branch unported)
                 "l_use_tke_in_wp3_pr_turb_term", "l_damp_wp3_Skw_squared"]


def test_baseline_default_config_passes():
    flags = get_default_config_flags()
    _chk({}, flags, **_BASE)  # must NOT raise
    print("  baseline default config (no gated flags) validates clean  PASS")


def test_reject_true_guards_fire():
    flags = get_default_config_flags()
    for flag in _REJECT_TRUE:
        try:
            _chk({flag: True}, flags, **_BASE)
        except ValueError as e:
            assert flag in str(e), f"{flag}=true error didn't mention {flag}: {e}"
        else:
            raise AssertionError(f"{flag}=true was NOT rejected (guard missing/broken)")
    print(f"  reject-TRUE guards: all {len(_REJECT_TRUE)} gated-on flags rejected  PASS")


def test_reject_false_guards_fire():
    flags = get_default_config_flags()
    for flag in _REJECT_FALSE:
        f2 = flags._replace(**{flag: False})
        try:
            _chk({}, f2, **_BASE)
        except ValueError as e:
            assert flag in str(e), f"{flag}=false error didn't mention {flag}: {e}"
        else:
            raise AssertionError(f"{flag}=false was NOT rejected (guard missing/broken)")
    print(f"  reject-FALSE guards: all {len(_REJECT_FALSE)} default-true flags rejected when off  PASS")


def test_call_pdf_closure_twice_rejected_on_flags_object():
    """The `l_call_pdf_closure_twice` guard (clubb_driver.py) checks BOTH the cfg dict AND the flags object
    (`getattr(flags, ...) or cfg.get(...)`) — because it is a `ConfigFlags` field, normally set on the flags
    object, not the cfg dict. test_reject_true_guards_fire only exercises the cfg-dict half; this pins the
    flags-OBJECT half, the branch that actually fires for a real case turning the flag on. It is the invariant
    that keeps the unported second zm-grid closure (pdf_closure_driver_zm) safely deferred: if a refactor dropped
    the `getattr(flags, ...)` half, a case setting the flag on the object would silently read an uncomputed
    pdf_params_zm instead of failing loud. The error must name the unported routine. (iter 487)"""
    flags = get_default_config_flags()._replace(l_call_pdf_closure_twice=True)
    try:
        _chk({}, flags, **_BASE)  # flag set on the OBJECT, cfg dict empty
    except ValueError as e:
        assert "l_call_pdf_closure_twice" in str(e) and "pdf_closure_driver_zm" in str(e), \
            f"flags-object rejection didn't name flag + unported routine: {e}"
        print("  l_call_pdf_closure_twice set on the flags object is rejected (names pdf_closure_driver_zm)  PASS")
        return
    raise AssertionError("l_call_pdf_closure_twice=True on the flags object was NOT rejected (getattr-half guard missing)")


def test_iipdf_type_guard_fires():
    """Only ADG1 (iiPDF_type=1, default) is wired into the JAX pdf_closure_driver; the other PDF types
    (2..7) are ported but not wired, so a non-ADG1 request must fail loud. Default (1) must pass."""
    flags = get_default_config_flags()
    _chk({}, flags._replace(iiPDF_type=1), **_BASE)  # ADG1 default passes
    for t in (2, 3, 4, 5, 6, 7):
        f2 = flags._replace(iiPDF_type=t)
        try:
            _chk({}, f2, **_BASE)
        except ValueError as e:
            assert "iiPDF_type" in str(e), f"iiPDF_type={t} error didn't mention iiPDF_type: {e}"
        else:
            raise AssertionError(f"iiPDF_type={t} was NOT rejected (guard missing/broken)")
    print("  iiPDF_type guard: non-ADG1 (2..7) rejected, ADG1(1) passes  PASS")


def test_solver_method_guard_fires():
    """Only the LU banded solvers (penta_lu/tridiag_lu = 2) are ported; penta_bicgstab (=3) is not, and the
    JAX wrapper ignores the method flag — so a non-LU request must fail loud. Default (2) must pass."""
    flags = get_default_config_flags()
    _chk({}, flags._replace(penta_solve_method=2, tridiag_solve_method=2), **_BASE)  # default passes
    for fld, val in (("penta_solve_method", 3), ("tridiag_solve_method", 3)):
        f2 = flags._replace(**{fld: val})
        try:
            _chk({}, f2, **_BASE)
        except ValueError as e:
            assert fld in str(e), f"{fld}={val} error didn't mention {fld}: {e}"
        else:
            raise AssertionError(f"{fld}={val} was NOT rejected (guard missing/broken)")
    print("  solver-method guard: non-LU penta/tridiag rejected, LU(2) passes  PASS")


def test_ipdf_call_placement_guard_fires():
    """Non-default PDF-closure placement (pre-advance=1 / pre-post=3) is unported and must be rejected;
    the default post-advance=2 must pass."""
    flags = get_default_config_flags()
    # default (post-advance = 2) passes
    _chk({}, flags._replace(ipdf_call_placement=2), **_BASE)
    # pre-advance (1) and pre-post (3) are rejected
    for placement in (1, 3):
        f2 = flags._replace(ipdf_call_placement=placement)
        try:
            _chk({}, f2, **_BASE)
        except ValueError as e:
            assert "ipdf_call_placement" in str(e), \
                f"placement={placement} error didn't mention ipdf_call_placement: {e}"
        else:
            raise AssertionError(f"ipdf_call_placement={placement} was NOT rejected (guard missing/broken)")
    print("  ipdf_call_placement guard: pre-advance(1)/pre-post(3) rejected, post-advance(2) passes  PASS")


def test_variance_sponge_guard_fires():
    """The variance sponge (sponge_damp_xp2/xp3) is ported + unit-tested but NOT wired into the JAX
    advance loop; enabling the wp2/wp3/up2_vp2 sponge flags must fail loud (no validated full-case oracle)."""
    flags = get_default_config_flags()
    for field in ("wp2", "wp3", "up2_vp2"):
        cfg = {f"{field}_sponge_damp_settings%l_sponge_damping": True}
        try:
            _chk(cfg, flags, **_BASE)
        except ValueError as e:
            assert "sponge_damp_xp2/xp3" in str(e) and "not wired" in str(e), \
                f"{field} sponge error didn't mention the unwired variance sponge: {e}"
        else:
            raise AssertionError(f"{field} sponge flag was NOT rejected (guard missing/broken)")
    print("  variance sponge guard: wp2/wp3/up2_vp2 sponge flags rejected (ported-but-unwired)  PASS")


def test_infrastructure_guards_fire():
    """The closure/PDF guards above are covered by _REJECT_TRUE/_REJECT_FALSE + the dedicated enum tests, but the
    SIX *infrastructure* guards in _check_unsupported_features had no coverage: the SILHS sampler (lh_microphys_type
    != disabled, l_silhs_rad), GrADS restart I/O (l_restart), file-driven time-dependent forcing (l_input_fields),
    the generalized-grid test (l_test_grid_generalization), and in-time grid adaptation (grid_adapt_in_time_method
    > 0). Each gates an un-ported subsystem the JAX silently ignores, so dropping a guard would let an unimplemented
    config run and mis-compute. These are all driven off the cfg DICT (not ConfigFlags fields), so they are exercised
    by passing the offending key in cfg. (iter 523)"""
    flags = get_default_config_flags()
    # (cfg-override, substring the raised error must contain)
    cases = [
        ({"lh_microphys_type": "cluster"}, "lh_microphys_type"),
        ({"l_silhs_rad": True},            "l_silhs_rad"),
        ({"l_restart": True},              "l_restart"),
        ({"l_input_fields": True},         "l_input_fields"),
        ({"l_test_grid_generalization": True}, "l_test_grid_generalization"),
        ({"grid_adapt_in_time_method": 1}, "grid_adapt_in_time_method"),
    ]
    for cfg, needle in cases:
        try:
            _chk(cfg, flags, **_BASE)
        except ValueError as e:
            assert needle in str(e), f"{cfg} error didn't mention {needle}: {e}"
        else:
            raise AssertionError(f"{cfg} was NOT rejected (infrastructure guard missing/broken)")
    print(f"  infrastructure guards: all {len(cases)} (SILHS/restart/input-fields/test-grid/grid-adapt) rejected  PASS")


def test_pdf_type_enum_values_match_fortran():
    """The `iiPDF_<type>` enum VALUES (ADG1=1 … new_hybrid=7) must match model_flags.F90 — they select the PDF closure
    and drive the guards above, so a drifted value would dispatch the wrong closure / mis-fire the guard. Source-grounded:
    parses the `iiPDF_<name> = <n>` integer parameters straight from model_flags.F90 and compares to the JAX
    (numerical_check `_iiPDF_<name>`, the full 1..7 set + constants_clubb `iiPDF_ADG1`). SKIPs if the F90 source is absent.
    (iter 470)"""
    import re
    f90 = os.path.join(_ROOT, "clubb_release", "src", "CLUBB_core", "model_flags.F90")
    if not os.path.exists(f90):
        print("  iiPDF enum values vs Fortran: SKIP (model_flags.F90 absent)")
        return
    fort = {}
    for raw in open(f90):
        line = raw.split("!", 1)[0]
        m = re.match(r"\s*(iiPDF_[A-Za-z0-9_]+)\s*=\s*([0-9]+)[\s,&]*$", line)
        if m and m.group(1) not in fort:
            fort[m.group(1)] = int(m.group(2))
    assert len(fort) >= 7, f"parsed only {len(fort)} iiPDF enums from model_flags.F90 — extraction broke"
    from clubb_jax.src.CLUBB_core import numerical_check as nc
    from clubb_jax.src.CLUBB_core.constants_clubb import iiPDF_ADG1
    mism = []
    for name, fv in fort.items():
        jv = getattr(nc, "_" + name, None)
        if jv is None:
            mism.append(f"{name}: not defined in numerical_check (Fortran={fv})")
        elif jv != fv:
            mism.append(f"{name}: JAX {jv} vs Fortran {fv}")
    if iiPDF_ADG1 != fort.get("iiPDF_ADG1"):
        mism.append(f"constants_clubb.iiPDF_ADG1 {iiPDF_ADG1} vs Fortran {fort.get('iiPDF_ADG1')}")
    assert not mism, "iiPDF enum value(s) diverge from model_flags.F90:\n  " + "\n  ".join(mism)
    print(f"  iiPDF_* enum values (1..7) match model_flags.F90 exactly ({len(fort)} types)  PASS")


if __name__ == "__main__":
    print("test_unsupported_config_guards:")
    test_baseline_default_config_passes()
    test_reject_true_guards_fire()
    test_reject_false_guards_fire()
    test_call_pdf_closure_twice_rejected_on_flags_object()
    test_ipdf_call_placement_guard_fires()
    test_variance_sponge_guard_fires()
    test_infrastructure_guards_fire()
    test_solver_method_guard_fires()
    test_iipdf_type_guard_fires()
    test_pdf_type_enum_values_match_fortran()
    print("Done.")
