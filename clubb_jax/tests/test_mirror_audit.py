"""Standing regression guard: the JAX↔Fortran file/routine-name mirror stays converged.

Runs the reproducible audit (run_scripts/mirror_audit.py) and asserts it reports PASS
(MISSING=0, CASING=0, MISPLACED=0, UNMIRRORED_FILES=0, REDUNDANT_TOL=0, JAX_ALIAS=0). If a future change
introduces a Fortran routine with no JAX counterpart (MISSING), a case-mismatched name (CASING), moves a
routine to the wrong file (MISPLACED), adds a JAX source file whose name mirrors no Fortran source/header file
(UNMIRRORED_FILES), leaves a `_NOT_TARGET` tolerance for a routine that is actually ported (REDUNDANT_TOL), or
adds a `_jax`-suffixed routine mirroring a Fortran routine without its bare-name public alias (JAX_ALIAS)
— without updating the documented fold/not-target/rename/JAX-only-file exceptions in mirror_audit.py
— this test fails, flagging the regression.

Pure-Python (no JAX / no Fortran oracle needed), so it never SKIPs.
"""
import os
import sys

_RUN_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "run_scripts")
sys.path.insert(0, os.path.abspath(_RUN_SCRIPTS))

import mirror_audit

# The "no-caller orphan cluster" the audit excuses from MISSING (mirror_audit.py `_NOT_TARGET`): three interpolation
# helpers in pdf_closure_module.F90 that are referenced ONLY inside their own contiguous definition block (3345–3591)
# — interp_var_array calls var_value_integer_height/var_subgrid_interp, and NOTHING calls interp_var_array. They are
# excused precisely because they are dead in the ORACLE; that excusal is only valid while they STAY uncalled.
_ORPHAN_CLUSTER = ("interp_var_array", "var_value_integer_height", "var_subgrid_interp")
_ORPHAN_HOME = "CLUBB_core/pdf_closure_module.F90"  # their sole reference site (relative to clubb_release/src)


def test_orphan_cluster_still_dead_in_fortran():
    """Source-grounded guard for the audit's orphan-cluster excusal. The three interp helpers are excused from MISSING
    because they have NO live caller in the Fortran oracle (verified iter 486 — referenced only inside their own
    def block). If a future `clubb_release` update wires any of them into live code, they become a real mirror gap
    that must be ported — but the hardcoded `_NOT_TARGET` excusal would keep silently hiding it. This asserts they
    remain uncalled: every reference (comment-stripped) lives inside the cluster's own def block in pdf_closure_module.F90.
    A new reference anywhere else fails loudly → re-port the now-live routine. Pure source scan; SKIPs if the Fortran
    submodule is absent. (iter 486)"""
    import re
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clubb_release", "src"))
    if not os.path.isdir(src_root):
        print("  orphan-cluster liveness: SKIP (clubb_release/src absent)")
        return
    home = os.path.join(src_root, _ORPHAN_HOME)
    # the cluster's own contiguous def block (function interp_var_array … end function var_subgrid_interp)
    block_lo, block_hi = 3345, 3591
    pats = {n: re.compile(r"(?<![A-Za-z0-9_])" + n + r"(?![A-Za-z0-9_])", re.IGNORECASE) for n in _ORPHAN_CLUSTER}
    offenders = []
    for dirpath, _dirs, files in os.walk(src_root):
        for fn in files:
            if not fn.lower().endswith((".f90", ".f", ".inc")):
                continue
            path = os.path.join(dirpath, fn)
            in_home = os.path.normpath(path) == os.path.normpath(home)
            with open(path, errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    code = line.split("!", 1)[0]  # strip Fortran comment
                    if in_home and block_lo <= lineno <= block_hi:
                        continue  # the cluster's own definitions / internal references — allowed
                    for name, pat in pats.items():
                        if pat.search(code):
                            offenders.append(f"{os.path.relpath(path, src_root)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "An orphan-cluster routine (" + ", ".join(_ORPHAN_CLUSTER) + ") is now REFERENCED outside its dead def block "
        "in the Fortran oracle — it is no longer a no-caller orphan and must be ported to JAX (then removed from the "
        "mirror_audit `_NOT_TARGET` cluster):\n  " + "\n  ".join(offenders))
    print("  orphan-cluster (interp_var_array/var_value_integer_height/var_subgrid_interp) still dead in Fortran — excusal valid  PASS")


# The `set_boundary_conditions_{lhs,rhs}` routines (advance_helper_module.F90) set Dirichlet BCs on a LAPACK
# banded matrix / RHS. The audit excuses them from MISSING (mirror_audit.py `_NOT_TARGET`) because they have NO
# `call` site anywhere in the oracle (verified iter 581) — the JAX faithfully inlines the boundary rows directly
# into each band-LHS assembly on the actual execution path, so there is nothing live to mirror. That excusal is
# only valid while they STAY uncalled.
_DEAD_BC_ROUTINES = ("set_boundary_conditions_lhs", "set_boundary_conditions_rhs")


def test_boundary_condition_setters_still_dead_in_fortran():
    """Source-grounded guard for the audit's `set_boundary_conditions_{lhs,rhs}` excusal. They are excused from
    MISSING because they are never CALLED in the Fortran oracle (the standalone path inlines the matrix boundary
    rows). If a future `clubb_release` update wires either into live code, it becomes a real mirror gap that must
    be ported — but the hardcoded `_NOT_TARGET` excusal would keep silently hiding it. This asserts no
    `call set_boundary_conditions_{lhs,rhs}` statement exists anywhere in the oracle source (comment-stripped).
    A new call site fails loudly → port the now-live routine. Pure source scan; SKIPs if the Fortran submodule is
    absent. (iter 581, mirrors the iter-486 orphan-cluster guard.)"""
    import re
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clubb_release", "src"))
    if not os.path.isdir(src_root):
        print("  boundary-condition-setter liveness: SKIP (clubb_release/src absent)")
        return
    # a live invocation is `call set_boundary_conditions_lhs(` / `..._rhs(` (Fortran is case-insensitive)
    pats = {n: re.compile(r"(?<![A-Za-z0-9_])call\s+" + n + r"\s*\(", re.IGNORECASE) for n in _DEAD_BC_ROUTINES}
    offenders = []
    for dirpath, _dirs, files in os.walk(src_root):
        for fn in files:
            if not fn.lower().endswith((".f90", ".f", ".inc")):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    code = line.split("!", 1)[0]  # strip Fortran comment
                    for name, pat in pats.items():
                        if pat.search(code):
                            offenders.append(f"{os.path.relpath(path, src_root)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "A boundary-condition setter (" + ", ".join(_DEAD_BC_ROUTINES) + ") is now CALLED in the Fortran oracle — it "
        "is no longer dead and must be ported to JAX (then removed from the mirror_audit `_NOT_TARGET` set):\n  "
        + "\n  ".join(offenders))
    print("  set_boundary_conditions_lhs/rhs still uncalled in Fortran — excusal valid  PASS")


# `pdf_closure_driver_zm` (pdf_closure_module.F90) is the audit's SOLE `_DEFERRED` routine — the second (zm-grid) PDF
# closure call. The audit excuses it (not a regression) because its only call site is gated by `l_call_pdf_closure_twice`,
# a flag the JAX `clubb_driver` fail-loud rejects (so the path is unreachable) AND it cannot be faithfully+validatably
# ported (the JAX decomposed `pdf_closure` into zt-specialized `calc_pdf_*` helpers; a zm-native re-derivation would be
# unreachable, oracle-less dead code). That excusal is only valid while the Fortran keeps the call gated.
_DEFERRED_ZM_CALLER = "CLUBB_core/pdf_closure_module.F90"
_DEFERRED_ZM_GATE = "l_call_pdf_closure_twice"


def test_pdf_closure_driver_zm_call_still_gated():
    """Source-grounded guard for the audit's sole `_DEFERRED` excusal. `pdf_closure_driver_zm` is excused from being a
    mirror gap because its ONE call site (pdf_closure_module.F90, inside `pdf_closure_driver`) sits inside an
    `if ( l_call_pdf_closure_twice )` block — a flag the JAX rejects, so the routine is unreachable on every supported
    config. If a future `clubb_release` update calls it UNCONDITIONALLY (un-gated), it becomes reachable and a real
    re-port obligation — but the hardcoded `_DEFERRED` excusal would keep silently hiding it. This asserts every
    `call pdf_closure_driver_zm` is governed by the `l_call_pdf_closure_twice` gate (the nearest preceding enclosing
    `if (...) then`, before any `else`/`end if`, tests that flag). A new un-gated call fails loudly. Pure source scan;
    SKIPs if the Fortran submodule is absent. (iter 582, mirrors the iter-486/581 liveness guards.)"""
    import re
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clubb_release", "src"))
    if not os.path.isdir(src_root):
        print("  pdf_closure_driver_zm gating: SKIP (clubb_release/src absent)")
        return
    home = os.path.join(src_root, _DEFERRED_ZM_CALLER)
    call_pat = re.compile(r"(?<![A-Za-z0-9_])call\s+pdf_closure_driver_zm\s*\(", re.IGNORECASE)
    if_pat = re.compile(r"(?<![A-Za-z0-9_])if\s*\(", re.IGNORECASE)
    gate_pat = re.compile(r"(?<![A-Za-z0-9_])" + _DEFERRED_ZM_GATE + r"(?![A-Za-z0-9_])", re.IGNORECASE)
    boundary_pat = re.compile(r"(?<![A-Za-z0-9_])(else|end\s*if|end\s+subroutine)(?![A-Za-z0-9_])", re.IGNORECASE)
    with open(home, errors="replace") as fh:
        lines = [ln.split("!", 1)[0] for ln in fh]  # comment-stripped
    call_lines = [i for i, ln in enumerate(lines) if call_pat.search(ln)]
    assert call_lines, ("pdf_closure_driver_zm is no longer called in " + _DEFERRED_ZM_CALLER + " — the oracle changed; "
                        "re-verify the _DEFERRED excusal.")
    offenders = []
    for ci in call_lines:
        gated = False
        for j in range(ci - 1, max(ci - 60, -1), -1):
            code = lines[j]
            if if_pat.search(code):           # nearest preceding `if (` governing this call
                gated = bool(gate_pat.search(code))
                break
            if boundary_pat.search(code):     # hit a block boundary first → not directly gated by an if-head
                break
        if not gated:
            offenders.append(f"{_DEFERRED_ZM_CALLER}:{ci + 1}: {lines[ci].strip()}")
    assert not offenders, (
        "A `call pdf_closure_driver_zm` is no longer governed by an `if ( " + _DEFERRED_ZM_GATE + " )` gate in the "
        "Fortran oracle — the sole _DEFERRED routine may now be reachable and must be ported to JAX (or the excusal "
        "re-justified):\n  " + "\n  ".join(offenders))
    print("  pdf_closure_driver_zm call still gated by l_call_pdf_closure_twice — _DEFERRED excusal valid  PASS")


# COMPILE-TIME-DEAD `_NOT_TARGET` routines are unreachable in the ORACLE ITSELF because a Fortran `parameter`
# (a compile-time constant, NOT a namelist flag) is fixed to the non-active value. `wp3_term_ta_explicit_rhs`
# (advance_wp2_wp3_module.F90) lives entirely inside `if ( l_explicit_turbulent_adv_wp3 )`, and
# `l_explicit_turbulent_adv_wp3` is declared `logical, parameter :: ... = .false.` in model_flags.F90 — so the
# branch is dead at compile time. The audit excuses it precisely because of that fixed parameter; the excusal is
# valid only while the parameter stays `parameter ... = .false.` (vs becoming `.true.` or a settable namelist flag).
_COMPILE_DEAD_PARAM_GATES = {
    # parameter name : (declaring file relative to clubb_release/src, expected fixed value, the dead routine it gates)
    "l_explicit_turbulent_adv_wp3": ("CLUBB_core/model_flags.F90", ".false.", "wp3_term_ta_explicit_rhs"),
}


def test_compile_dead_parameter_gates_unchanged():
    """Source-grounded guard for the audit's COMPILE-TIME-DEAD `_NOT_TARGET` excusals. A routine like
    `wp3_term_ta_explicit_rhs` is excused because the Fortran `parameter l_explicit_turbulent_adv_wp3 = .false.`
    makes its `if`-branch unreachable at compile time. If a future `clubb_release` update flips that parameter to
    `.true.`, or demotes it from a `parameter` to a settable namelist flag, the dead routine becomes reachable and a
    real re-port obligation — but the hardcoded `_NOT_TARGET` excusal would keep silently hiding it. This asserts each
    such gate is still declared inside a `... parameter ...` declaration AND fixed to its non-activating value. Pure
    source scan; SKIPs if the Fortran submodule is absent. (iter 583, mirrors the iter-486/581/582 liveness guards.)"""
    import re
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clubb_release", "src"))
    if not os.path.isdir(src_root):
        print("  compile-dead parameter gates: SKIP (clubb_release/src absent)")
        return
    problems = []
    for gate, (relfile, expect_val, dead_routine) in _COMPILE_DEAD_PARAM_GATES.items():
        path = os.path.join(src_root, relfile)
        with open(path, errors="replace") as fh:
            lines = [ln.split("!", 1)[0] for ln in fh]  # comment-stripped
        assign = re.compile(r"(?<![A-Za-z0-9_])" + gate + r"\s*=\s*(\.[a-z]+\.)", re.IGNORECASE)
        hit = None
        for i, ln in enumerate(lines):
            m = assign.search(ln)
            if m:
                hit = (i, m.group(1).lower()); break
        if hit is None:
            problems.append(f"{gate}: no `{gate} = .<value>.` assignment found in {relfile} (oracle changed)")
            continue
        i, val = hit
        if val != expect_val.lower():
            problems.append(f"{gate}: value is {val}, expected {expect_val} — {dead_routine} may now be live")
        # walk back to the declaration head (the line bearing `::`) and require `parameter`
        decl_head = None
        for j in range(i, max(i - 30, -1), -1):
            if "::" in lines[j]:
                decl_head = lines[j]; break
        if decl_head is None or not re.search(r"(?<![A-Za-z0-9_])parameter(?![A-Za-z0-9_])", decl_head, re.IGNORECASE):
            problems.append(f"{gate}: no `parameter` in its declaration head — demoted to a settable flag? "
                            f"{dead_routine} may now be reachable")
    assert not problems, ("A COMPILE-TIME-DEAD parameter gate changed in the Fortran oracle — the routine it gates "
                          "is no longer dead and must be ported (or the _NOT_TARGET excusal re-justified):\n  "
                          + "\n  ".join(problems))
    print("  compile-dead parameter gates (l_explicit_turbulent_adv_wp3=.false. param) unchanged — excusals valid  PASS")


# WHOLE-FILE excusal: `radiation_variables_module.F90` has no JAX `.py` mirror, so `mirror_audit._jax_stems()`
# scopes it out of the MISSING check (a by-design-unmirrored file). That is legitimate because the file is pure
# module-level STATE MANAGEMENT — allocate/zero/deallocate of the Fortran radiation arrays — which the JAX replaces
# with dict-based state, so there is no physics to mirror. That excusal is valid only while the file stays
# state-management-only; a future PHYSICS routine added there would be silently scoped out (no JAX stem → not checked).
_RAD_STATE_MODULE = "Radiation/radiation_variables_module.F90"
_STATE_MGMT_RE = r"^(setup|reset|cleanup)_"


def test_radiation_state_module_is_state_management_only():
    """Source-grounded guard for the whole-file excusal of `radiation_variables_module.F90`. The audit scopes it out
    of MISSING because it has no JAX mirror file — valid only because every routine is `(setup|reset|cleanup)_*`
    state management (allocate/zero/deallocate of the module radiation arrays), which the JAX expresses as dict
    state. If a future `clubb_release` update adds a non-state (physics) routine to this file, it would be silently
    scoped out by the no-JAX-stem rule — this asserts every `subroutine`/`function` in the file matches the
    state-management name pattern, failing loudly on any new physics routine so it is evaluated for porting. Pure
    source scan; SKIPs if the Fortran submodule is absent. (iter 586, mirrors the iter-486/581/582/583 guards.)"""
    import re
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clubb_release", "src"))
    if not os.path.isdir(src_root):
        print("  radiation state-module liveness: SKIP (clubb_release/src absent)")
        return
    path = os.path.join(src_root, _RAD_STATE_MODULE)
    rtn = re.compile(r"^\s*(?:pure\s+|elemental\s+|recursive\s+)*(?:subroutine|function)\s+([A-Za-z_0-9]+)", re.I)
    state_mgmt = re.compile(_STATE_MGMT_RE, re.I)
    offenders = []
    with open(path, errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            code = line.split("!", 1)[0]
            m = rtn.match(code)
            if m and not state_mgmt.match(m.group(1)):
                offenders.append(f"{_RAD_STATE_MODULE}:{lineno}: {m.group(1)}")
    assert not offenders, (
        "radiation_variables_module.F90 now has a non-state-management (physics?) routine — the whole-file mirror "
        "excusal (no JAX stem → scoped out of MISSING) may be hiding a real port obligation; evaluate it:\n  "
        + "\n  ".join(offenders))
    print("  radiation_variables_module.F90 is state-management-only (setup/reset/cleanup) — whole-file excusal valid  PASS")


def test_no_unrecognized_scoped_out_file():
    """Soft-tripwire guard with teeth for the WHOLE-FILE scoping CLASS (generalizing the radiation guard above). Every
    Fortran file with no JAX mirror is scoped out of MISSING by `_jax_stems()`; iter 587 made that set visible and iter
    588 bucketed it by by-design-unmirrored subsystem (LAPACK/BLAS, Numerical Recipes, SILHS, COAMPS, aerosol, SCM/host
    microphys, IO readers, tuner/infra, G-unit tests, case setups, BUGSrad alt-solvers, state/api types). A scoped-out
    file matching NONE of those buckets is a Fortran file that is neither mirrored NOR a recognized non-target — i.e. a
    candidate port the whole-file scoping would otherwise hide. This asserts that `uncategorized` stays empty: a new
    physics file in the oracle fails loudly here (add its JAX mirror, or extend the documented bucket patterns in
    mirror_audit._BUCKETS if it is genuinely a new non-target). SKIPs if the Fortran submodule is absent. (iter 588)"""
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clubb_release", "src"))
    if not os.path.isdir(src_root):
        print("  scoped-out categorization: SKIP (clubb_release/src absent)")
        return
    entries = mirror_audit.scoped_out_entries()
    _counts, uncategorized = mirror_audit.scoped_out_breakdown(entries)
    assert not uncategorized, (
        "Fortran file(s) with no JAX mirror are also NOT a recognized by-design-unmirrored subsystem — the whole-file "
        "scoping (no JAX stem → not MISSING-checked) may be hiding a real port obligation. Mirror them in JAX, or if "
        "genuinely a new non-target extend mirror_audit._BUCKETS:\n  " + "\n  ".join(uncategorized))
    print(f"  all {len(entries)} scoped-out Fortran files are recognized non-target subsystems (0 unrecognized)  PASS")


def test_no_unclassified_routineless_module():
    """Soft-tripwire guard WITH teeth for the routine-LESS-module class (iter 729, generalizing the iter-728
    parameters_microphys finding). A Fortran module with 0 subroutines/functions is invisible to the routine-based
    MISSING check AND the routine-bearing scoped-out enumeration — so a NEW pure-parameter/type module added to the
    oracle (e.g. a new `parameters_*`/`*_constants` module) would silently have no JAX mirror and no reviewer signal.
    This asserts `mirror_audit._routineless_unclassified()` stays empty: every routine-less module is same-stem
    JAX-mirrored, a recognized by-design-unmirrored subsystem, or in the documented `_ROUTINELESS_OK` allowlist.
    A new one fails loudly here (mirror it in JAX, or classify it in `_ROUTINELESS_OK`). SKIPs if the Fortran
    submodule is absent."""
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clubb_release", "src"))
    if not os.path.isdir(src_root):
        print("  routine-less classification: SKIP (clubb_release/src absent)")
        return
    unclassified = mirror_audit._routineless_unclassified()
    assert not unclassified, (
        "A routine-less (pure-parameter/type) Fortran module is neither JAX-mirrored, a recognized non-target "
        "subsystem, nor in mirror_audit._ROUTINELESS_OK — it is invisible to MISSING/scoped-out and may be hiding a "
        "real port obligation. Mirror it, or classify it in _ROUTINELESS_OK:\n  " + "\n  ".join(unclassified))
    print(f"  all routine-less Fortran modules classified ({len(mirror_audit._ROUTINELESS_OK)} documented, 0 unclassified)  PASS")


def test_mirror_is_converged():
    rc = mirror_audit.main()
    assert rc == 0, ("mirror_audit reported a NEW name/file-mirror gap "
                     "(MISSING/CASING/MISPLACED/UNMIRRORED_FILES/MISPLACED_FILES/REDUNDANT_TOL/JAX_ALIAS != 0). Either "
                     "mirror the new Fortran routine/file in JAX, or — if it is a fold/not-target/rename/JAX-only-file/"
                     "dir-split — add it to the documented exceptions in run_scripts/mirror_audit.py (remove any "
                     "tolerance for a routine that is actually ported; give any new _jax routine its bare-name alias).")
    print("  mirror_audit: MISSING=0 CASING=0 MISPLACED=0 UNMIRRORED_FILES=0 MISPLACED_FILES=0 REDUNDANT_TOL=0 JAX_ALIAS=0 — converged  PASS")


def test_dir_split_allowlist_still_live():
    """Source-grounded liveness guard for the iter-718 `_DIR_SPLIT_OK` allowlist (the directory-correspondence
    check). Each entry excuses a JAX file that deliberately lives in a non-oracle directory; if the split is later
    undone (the file deleted/moved), the entry would silently keep excusing nothing — or worse, mask a real future
    misplacement of a same-named file. This asserts every `_DIR_SPLIT_OK` (stem, jax_dir) still corresponds to a
    real JAX file at clubb_jax/src/<jax_dir>/<stem>.py AND a Fortran oracle <stem>.F90/.F still exists, so the
    excusal stays justified. Pure existence scan; SKIPs if clubb_release/src is absent. (iter 719)"""
    import glob as _glob
    jsrc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    fsrc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clubb_release", "src"))
    if not os.path.isdir(fsrc):
        print("  _DIR_SPLIT_OK liveness: SKIP (clubb_release/src absent)")
        return
    offenders = []
    for stem, jdir in sorted(mirror_audit._DIR_SPLIT_OK):
        jax_path = os.path.join(jsrc, jdir, stem + ".py")
        fort_hits = _glob.glob(os.path.join(fsrc, "**", stem + ".[fF]*"), recursive=True)
        if not os.path.isfile(jax_path):
            offenders.append(f"{stem}: JAX split file {jdir}/{stem}.py no longer exists — remove the _DIR_SPLIT_OK entry")
        elif not fort_hits:
            offenders.append(f"{stem}: Fortran oracle {stem}.F90 gone — the dir-split excusal is stale")
    assert not offenders, (
        "An iter-718 _DIR_SPLIT_OK directory-split excusal is no longer justified — it could mask a real whole-file "
        "misplacement. Update mirror_audit._DIR_SPLIT_OK:\n  " + "\n  ".join(offenders))
    print(f"  all {len(mirror_audit._DIR_SPLIT_OK)} _DIR_SPLIT_OK dir-split excusal(s) still live (jax file + oracle exist)  PASS")


if __name__ == "__main__":
    print("test_mirror_audit:")
    test_mirror_is_converged()
    test_orphan_cluster_still_dead_in_fortran()
    test_boundary_condition_setters_still_dead_in_fortran()
    test_pdf_closure_driver_zm_call_still_gated()
    test_compile_dead_parameter_gates_unchanged()
    test_radiation_state_module_is_state_management_only()
    test_no_unrecognized_scoped_out_file()
    test_dir_split_allowlist_still_live()
    test_no_unclassified_routineless_module()
    print("Done.")
