#!/usr/bin/env python3
"""compare_cases.py — multi-case Fortran-vs-JAX regression dashboard.

Runs `compare_runs.py` across a list of cases and prints a one-line-per-case
summary of prognostic bit-faithfulness. This generalises the regression test
beyond the single ARM case (see DESIGN.md "How to Test").

Usage:
    python run_scripts/compare_cases.py [--max-iters N] [--cases a,b,c]
    python run_scripts/compare_cases.py --list      # show the default case set

Default case set is the group of cases whose JAX driver runs end-to-end
without unported microphysics/radiation (see DESIGN.md "Remaining Work").

Exit code 0 if every case has 0 prognostic failures, else 1.
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import subprocess
import sys

RUN_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
JAX_ROOT = os.path.dirname(os.path.dirname(RUN_SCRIPTS))
_CASE_SETUPS = os.path.join(JAX_ROOT, "clubb_release", "input", "case_setups")


def discover_cases() -> list:
    """All SCM cases, from clubb_release/input/case_setups/*_model.in."""
    return sorted(os.path.basename(f)[:-len("_model.in")]
                  for f in glob.glob(os.path.join(_CASE_SETUPS, "*_model.in")))


# Feature-blocker patterns (the unported subsystem each unsupported case needs).
# Keep ordered most-specific first. Used by --survey to categorise without a full run.
_FEATURE_RE = re.compile(
    r"(morrison|coamps|bugsrad|SILHS sampling is not implemented|"
    r"lh_microphys_type = 'interactive'|is not supported|is not implemented)",
    re.IGNORECASE)


def smoke_case(case: str, max_iters: int = 3, timeout: int = 120) -> dict:
    """Fast JAX-only smoke run to categorise RUNS / UNSUPPORTED(feature) / ERROR.

    Does NOT run the Fortran oracle — just checks whether the JAX driver gets
    through `max_iters` steps, or which unported feature blocks it.
    """
    cmd = [sys.executable, os.path.join(RUN_SCRIPTS, "run_scm.py"),
           case, "-jax", "-max_iters", str(max_iters)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"case": case, "kind": "TIMEOUT", "feature": None}
    out = proc.stdout + proc.stderr
    if "Completed" in out and "timesteps" in out:
        return {"case": case, "kind": "RUNS", "feature": None}
    if re.search(r"not supported|not implemented", out, re.IGNORECASE):
        feat = "morrison" if "morrison" in out else \
               "coamps" if "coamps" in out else \
               "bugsrad" if "bugsrad" in out else \
               "SILHS" if re.search(r"SILHS|interactive", out) else "other"
        return {"case": case, "kind": "UNSUPPORTED", "feature": feat}
    return {"case": case, "kind": "ERROR", "feature": None}


def run_survey(max_iters: int):
    """Auto-discover all cases and print a RUNS/UNSUPPORTED/ERROR coverage map.

    Run sequentially (NOT in parallel) — concurrent JAX runs each compile + run
    in ~100s and trip short timeouts under contention, giving false ERRORs."""
    cases = discover_cases()
    print(f"=== Coverage survey: {len(cases)} cases (JAX smoke, {max_iters} steps) ===\n")
    buckets = {"RUNS": [], "UNSUPPORTED": [], "ERROR": [], "TIMEOUT": []}
    feat_count: dict = {}
    for c in cases:
        r = smoke_case(c, max_iters)
        buckets[r["kind"]].append(c)
        if r["feature"]:
            feat_count[r["feature"]] = feat_count.get(r["feature"], 0) + 1
        tag = r["kind"] + (f"({r['feature']})" if r["feature"] else "")
        print(f"  {c:<26} {tag}")
    print("\n--- Summary ---")
    for k in ("RUNS", "UNSUPPORTED", "ERROR", "TIMEOUT"):
        if buckets[k]:
            print(f"  {k}: {len(buckets[k])}")
    if feat_count:
        print("  Blocking features (coverage levers):")
        for f, n in sorted(feat_count.items(), key=lambda kv: -kv[1]):
            print(f"    {f:<10} {n} cases")
    print(f"\n  RUNS: {', '.join(buckets['RUNS'])}")
    if buckets["ERROR"]:
        print(f"  ERROR (investigate): {', '.join(buckets['ERROR'])}")

# Cases whose JAX driver runs end-to-end (no unported microphysics/radiation).
# ARM is the bit-faithful reference; the rest are verified here for the first
# time against the Fortran oracle.
# Cases verified bit-faithful (0 prognostic failures) at 30 timesteps. This is the
# regression gate — all must stay PASS. NOTE: verify at 30 (or more), not 15 — gabls2
# passed at 15 but fails intermittently at 10/20/30 (a false positive when checked short).
DEFAULT_CASES = [
    "arm",
    "bomex",
    "dycoms2_rf01",
    "wangara",
    "atex",          # bit-faithful since Iter84 (xm monotonic flux limiter wired in)
    "gabls2",        # bit-faithful (physics, instantaneous output)
    "gabls3_night",  # bit-faithful since Iter86 (um_f/ug time-dependent wind forcing)
    "fire",          # bit-faithful since Iter87 (bulk surface scheme, sfctype=1)
    "neutral",       # bit-faithful since Iter91 (neutral_case_sfclyr: ustar=0.5 + momentum flux)
    "ekman",         # bit-faithful since Iter94 (sponge damping + ice_supersat_frac at cold top)
    "cobra",         # bit-faithful since Iter94 ice fix (cold cloud layer T=266-270K); confirmed Iter96
    "dycoms2_rf02_nd",  # bit-faithful (Iter96): "_nd" = no-drizzle stratocumulus, no special physics
    "dycoms2_rf01_fixed_sst",  # bit-faithful (Iter98): dycoms2_rf01 variant with fixed SST surface
    "atex_long",        # bit-faithful (Iter100): unblocked by cloud_drop_sed port (l_cloud_sed)
    "dycoms2_rf02_so",  # bit-faithful (Iter100): unblocked by cloud_drop_sed port (l_cloud_sed)
    "jun25_altocu",     # bit-faithful (Iter188): cold-cloud altocumulus + simplified radiation; unblocked by the per-step wm_zm (subsidence) recompute fix
    "gabls3",           # bit-faithful (Iter273, 17th case): full BUGSrad correlated-k radiation + interactive soil_veg + omega subsidence. SLOW (~6 s/step → ~3 min at 30 iters) — the BUGSrad path's regression guard.
    "mpace_a",          # bit-faithful (Iter299, 18th case): Morrison (l_ice_microphys) but clear/sub-saturated — the only Morrison signal is the clear-air thlm_mc, which is the Fortran's SINGLE-PRECISION thlm<->T_in_K round-trip residual (module_mp_graupel.py reproduces the real*4 cast).
    "clex9_nov02",      # bit-faithful (19th case): cold-cloud altocumulus, Morrison microphysics but microphys_start_time=51411 s is BEYOND the 30-step window, so Morrison never activates (clear/closure-only physics, prognostic bit-exact). Tier-C clean once the Ncm/Nc_in_cloud pre-activation diagnostic was fixed to match advance_microphys's early return (no stats written before start time).
    "clex9_oct14",      # bit-faithful (20th case): sibling of clex9_nov02 (same altocumulus campaign, Morrison pre-activation). Prognostic bit-exact + Tier-C clean at 30 steps.
]

# Cases that are NOT strictly bit-faithful but ARE physically faithful (Tier-C PASS vs the Fortran oracle).
# They are FP-limited, not bug-limited: their init + first-step physics are bit-exact, and the residual past
# step 1 is genuine FP/chaos amplification (cloud-topped boundary layers magnify a sub-tolerance cloud
# difference). Run with `--cases tier_c` (forces --tier physical).
TIER_C_CASES = [
    "cgils_s11",   # Iter89/90/92: Press[Pa]-sounding→z + T[K]→θ init (clubb_driver.F90:5499-5524) +
                   # convert_snd2extended_atm radiation (case ozone/deep sounding). thlm bit-exact at step 1;
                   # FP-balanced residual (surface radht / cloud chaos) from step 2. Shared init+rad path for
                   # the 12 CGILS/cloud_feedback cases.
    "cgils_s12",   # Iter102: same init+rad path; the Iter97 forcing zero-fill fix was what tipped it to Tier-C
                   # PASS (its forcing tops out at 101687 Pa, below the model bottom → out-of-range levels that
                   # were previously edge-extrapolated). FP-limited moments beyond cloud onset.
]

# Cases whose JAX driver runs but which are NOT yet bit-faithful, with the missing
# feature that blocks them. Run explicitly with --cases to track progress.
BLOCKED_CASES = {
    "coriolis_test": "analytic Foucault-pendulum benchmark; zeroes nearly all closure constants. Iter102 ported the missing uv-nudging term (fixed um drift); Iter103 ported the nontraditional Coriolis terms (l_ho_nontrad_coriolis: 4 RHS/forcing terms). With them ON (--override l_ho_nontrad_coriolis=.true.) it passes at 2 steps with ALL moments incl. the Foucault upwp oscillator at machine-eps. Residual at >=10 steps is upwp FP ACCUMULATION (rel ~4e-6) in the undamped zeroed-closure benchmark — the physics is faithful, the degenerate setup is FP-limited (the undamped oscillator does not decay errors).",
    "dycoms2_rf02_do": "KK (khairoutdinov_kogan) drizzle microphysics — PORTED + wired (Iter106-174): means, "
        "transport, and the second-moment covariance driver. STEP 1 IS FULLY BIT-FAITHFUL (Iter174 fixed the "
        "fill_holes threshold: must be zero_threshold not hydromet_tol). Step 2+ is NOT achievable because the "
        "FORTRAN ORACLE ITSELF IS INACCURATE at the rt-covariance second-moment source (Iter175): the rt covar is "
        "an extreme ~850x cancellation, and the Fortran's ACM-850 parabolic-cylinder Dv at the alpha+1 order "
        "(~1e-3-accurate) gets amplified to ~15x. A 20M-sample Monte-Carlo of the full covariance confirms the JAX "
        "(rel 1.2e-3 to MC) is CORRECT and the Fortran (15x off) is wrong. Matching it would require degrading the "
        "JAX. Secondary: the active-rain rrm transport carries ~2.7% (within-step K_hm timing, Iter140). "
        "ORACLE-LIMITED — not a JAX bug.",
    "dycoms2_rf02_ds": "KK drizzle microphysics, same scheme as dycoms2_rf02_do — same oracle-limit at the rt-covar "
        "cancellation (see dycoms2_rf02_do).",
    "rico":          "KK microphysics. ★★ Iter307 FIXED TWO REAL KK BUGS (rico 5x closer; the 18 bit-faithful cases "
        "don't use KK so untouched; rico N=2 still passes): (1) the JAX was MISSING the `hydromet_tol` clip that "
        "fill_holes_driver_api applies (fill_holes.F90:2444 — zero hydromet<=tol, return mass to vapor + latent thlm "
        "adjust); without it sub-tol rrm ACCUMULATED → premature precip onset. (2) the KK rt tendency dropped `rvm_mc` "
        "(rain-evap→vapor) — clubb_driver.F90:3337 does `rtm_forcing += rcm_mc + rvm_mc` (Morrison path already did; KK "
        "added only rcm_mc), =0 until rain forms ~step 5 = where rico diverged. Both in kk_microphys_step.py. Worst N=10 "
        "error 1.97e-5 → 3.7e-6. STILL FP-limited at precip onset: the residual thl 2nd moments (thlp2/wpthlp ~2-4e-6) "
        "are the KK covar source `wpthlp_mc` (∝ precip_frac) amplifying a sub-tolerance rrm difference where rrm first "
        "crosses the precip threshold (a discrete one-step-early activation) — precipitation_fraction + KK_covar_driver "
        "both pass their unit tests, so this residual is genuine FP, no clean fix. Grid hypothesis ruled out (the rico "
        "zt grid is bit-exact). (do is grid_type=1, _so is grid_type=3 + bit-faithful.) ★ Iter308 CONFIRMED the two "
        "fixes fully resolved the SYSTEMATIC bug: rrm AND Nrm are now bit-exact through step 6 (was diverging from "
        "step 5) and FP-balanced after (no sign bias) — the residual is genuine FP, rico is now FP-limited not bug-limited.",
    "arm_97":        "SILHS interactive sampling (lh_microphys_type='interactive', lh_num_samples=8) — not "
        "bit-reproducible vs the Fortran RNG, not a target. (NB morrison microphysics AND bugsrad radiation, which "
        "arm_97 also uses, are BOTH supported now — SILHS is the sole blocker; iter-373 reverify.)",
    "arm_0003":      "COAMPS microphysics — and arm_0003 FATAL-ERRORS in the Fortran itself ('COAMPS does not "
        "support l_predict_Nc=F'), so there is NO ORACLE to validate against. COAMPS is also ~7000 lines (adjtq 1609 "
        "+ conice + ~45 process-rate eqaXX.F files + iterative nrmtqw/nrmtqi saturation adjustments) of ice/snow/"
        "graupel/rain bulk microphysics — impractical for bit-faithful porting (the iterative adjustments are FP-"
        "sensitive). Unviable (no oracle + impractical).",
    "mc3e":          "SILHS interactive sampling (lh_microphys_type='interactive') — not a target; morrison + "
        "bugsrad (also used) are supported, so SILHS is the blocker (iter-373 reverify).",
    # mpace_a: BIT-FAITHFUL since Iter299 (18th case, moved to DEFAULT_CASES). It is Morrison (l_ice_microphys)
    # but stays clear/sub-saturated, so the only Morrison signal is the clear-air thlm_mc — which is the Fortran's
    # SINGLE-PRECISION thlm<->T_in_K round-trip residual (morrison_microphys_module.F90 keeps T_in_K/rcm_r4 in
    # real*4 via `real(...)`). The JAX float64 round-trip gave EXACTLY 0; module_mp_graupel.py now reproduces the
    # real*4 cast (Iter299). NOTE: the Iter293-296 "forcing-time" analysis was misdirected by the thlm_forcing STAT
    # = raw_LS_forcing + lagged thlm_mc; the LS-forcing port was always faithful (Iter297 oracle debug build).
    # jun25_altocu: BIT-FAITHFUL since Iter188 (moved to DEFAULT_CASES). The Iter97-99 "near-gate FP
    # amplification, no fixable bug" conclusion was WRONG — the seed was the stale per-step wm_zm (the
    # subsidence on momentum levels was never recomputed when the forcing updated wm_zt → the xp2/xpyp
    # mean-advection by subsidence was missing). A reminder that "exhaustive diagnosis, FP-limited" can
    # still hide a real bug; budget-decompose (thlp2_ma=0 → wm_zm=0) before concluding FP.
}

_RESULT_RE = re.compile(r"Prognostic failures:\s*(\d+)\s+Diagnostic failures.*?:\s*(\d+)")


def run_case(case: str, max_iters: int, tier: str = "bit") -> dict:
    """Run compare_runs.py for one case; parse prognostic/diagnostic failure counts.

    With tier='physical', the case PASS/FAIL is the Tier-C verdict (compare_runs' exit code);
    the parsed prognostic-failure count (the legacy bit-gate line, always printed) is still
    reported as informational."""
    # compare_runs.py auto-forces per-step output (stats_tsamp = stats_tout = dt_main),
    # so the gate compares the PHYSICS directly — immune to stats-averaging-window
    # artifacts for cases whose default stats output is averaged (gabls2, gabls3_night).
    cmd = [sys.executable, os.path.join(RUN_SCRIPTS, "compare_runs.py"),
           "--case", case, "--max-iters", str(max_iters), "--tier", tier]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    m = _RESULT_RE.search(out)
    if m:
        return {"case": case, "rc": proc.returncode,
                "prog_fail": int(m.group(1)), "diag_fail": int(m.group(2)),
                "ran": True, "tail": ""}
    # Did not reach the comparison table — capture a short reason.
    tail = " | ".join(l.strip() for l in out.strip().splitlines()[-3:])
    return {"case": case, "rc": proc.returncode, "prog_fail": None,
            "diag_fail": None, "ran": False, "tail": tail[:160]}


def main():
    p = argparse.ArgumentParser(description="Multi-case Fortran-vs-JAX regression dashboard")
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument("--cases", default=None,
                   help="comma-separated case list (default: built-in runnable set)")
    p.add_argument("--list", action="store_true", help="print default case set and exit")
    p.add_argument("--survey", action="store_true",
                   help="auto-discover ALL cases and categorise RUNS/UNSUPPORTED/ERROR (JAX smoke)")
    p.add_argument("--tier", choices=["bit", "physical"], default="bit",
                   help="correctness standard (REFACTOR.md §2): 'bit' = legacy machine-precision gate "
                        "(default); 'physical' = Tier-C field-class tolerances (the numerical-accuracy gate).")
    args = p.parse_args()

    if args.survey:
        run_survey(max_iters=3)
        return

    if args.list:
        print("Bit-faithful regression set:", ", ".join(DEFAULT_CASES))
        print("\nTier-C physical-fidelity set (run: --cases tier_c):", ", ".join(TIER_C_CASES))
        print("\nBlocked cases (run/diverge; missing feature):")
        for c, why in BLOCKED_CASES.items():
            print(f"  {c:<18} {why}")
        return

    if args.cases == "tier_c":
        # Physical-fidelity suite (Tier-C-faithful, FP-limited). Force the physical tier.
        cases = TIER_C_CASES
        args.tier = "physical"
    else:
        cases = args.cases.split(",") if args.cases else DEFAULT_CASES

    print(f"=== Multi-case regression ({args.max_iters} iters) ===\n")
    print(f"{'Case':<16}  {'Status':<8}  {'ProgFail':>8}  {'DiagFail':>8}  Notes")
    print("-" * 70)

    all_pass = True
    for case in cases:
        r = run_case(case, args.max_iters, tier=args.tier)
        if not r["ran"]:
            all_pass = False
            print(f"{case:<16}  {'ERROR':<8}  {'-':>8}  {'-':>8}  {r['tail']}")
            continue
        # tier='physical' → case verdict is the Tier-C result (compare_runs exit code rc==0);
        # tier='bit' → verdict is 0 prognostic failures.
        case_ok = (r["rc"] == 0) if args.tier == "physical" else (r["prog_fail"] == 0)
        status = "PASS" if case_ok else "FAIL"
        if not case_ok:
            all_pass = False
        print(f"{case:<16}  {status:<8}  {r['prog_fail']:>8}  {r['diag_fail']:>8}")

    print("-" * 70)
    _ok_msg = ("PASS — all cases bit-faithful (prognostic)" if args.tier == "bit"
               else "PASS — all cases within Tier-C numerical-accuracy tolerances")
    print(f"\nResult [{args.tier}]: {_ok_msg if all_pass else 'FAIL — see above'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
