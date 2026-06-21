#!/usr/bin/env python3
"""Flip CLUBB config flags to non-default values and check JAX still agrees with Fortran.

For each requested flag override this:
  1. copies the default `configurable_model_flags.in`, flips the one flag,
  2. runs the JAX driver and the Fortran standalone with that flags file (same input both sides),
  3. diffs the stats with run_bindiff_all.py and classifies the outcome.

Because a flag changes the *algorithm in both* runs, the bit-faithful gate still applies: a flip
whose JAX path is a faithful port stays within tolerance; a flip that runs but diverges is a real
porting bug in that flag's path; a flip that fail-louds in JAX is an unsupported/gated path.

Usage:
  python clubb_jax/run_scripts/compare_flag_variants.py --case bomex --max-iters 30 \
      --flags l_godunov_upwind_wpxp_ta=.true. l_use_cloud_cover=.true. ...
  (omit --flags to run the built-in DEFAULT_SWEEP set)
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLUBB_RELEASE = REPO / "clubb_release"
RUN_SCM = REPO / "clubb_jax" / "run_scripts" / "run_scm.py"
RUN_BINDIFF = CLUBB_RELEASE / "run_scripts" / "run_bindiff_all.py"
DEFAULT_FLAGS = CLUBB_RELEASE / "input" / "tunable_parameters" / "configurable_model_flags.in"
STATS = CLUBB_RELEASE / "input" / "stats" / "standard_stats.in"
HR_SPEC = "C8/0.2:0.8/4"

# Non-default closure-flag flips worth exercising (default -> flipped). Microphysics / radiation /
# the deferred l_call_pdf_closure_twice are deliberately excluded.
DEFAULT_SWEEP = [
    "l_godunov_upwind_wpxp_ta=.true.",
    "l_godunov_upwind_xpyp_ta=.true.",
    "l_upwind_xpyp_ta=.false.",
    "l_upwind_xm_ma=.false.",
    "l_standard_term_ta=.true.",
    "l_partial_upwind_wp3=.true.",
    "l_use_cloud_cover=.true.",
    "l_C2_cloud_frac=.true.",
    "l_diffuse_rtm_and_thlm=.true.",
    "l_stability_correct_Kh_N2_zm=.true.",
    "l_use_C11_Richardson=.true.",
    "l_brunt_vaisala_freq_moist=.true.",
    "l_trapezoidal_rule_zt=.true.",
    "l_trapezoidal_rule_zm=.true.",
    "l_vert_avg_closure=.true.",
    "l_tke_aniso=.false.",
]


def flip_flag(text: str, flag: str, value: str) -> str:
    """Replace `flag = <old>` with `flag = <value>` in a namelist body; raise if not present."""
    if value.startswith("."):  # logical
        pat = re.compile(rf"^(\s*{re.escape(flag)}\s*=\s*)\.(?:true|false)\.", re.MULTILINE | re.IGNORECASE)
    else:                       # integer
        pat = re.compile(rf"^(\s*{re.escape(flag)}\s*=\s*)-?\d+", re.MULTILINE)
    new, n = pat.subn(rf"\g<1>{value}", text)
    if n == 0:
        raise KeyError(f"flag {flag!r} not found in flags file")
    return new


def run(cmd, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n\n")
        p = subprocess.run([str(c) for c in cmd], cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def tail(log: Path, n: int = 8) -> str:
    if not log.exists():
        return ""
    return "\n".join(log.read_text(errors="replace").splitlines()[-n:])


def bindiff_exceed(log: Path) -> int | None:
    """Return the count of variables exceeding threshold, or None if not parseable."""
    if not log.exists():
        return None
    for line in log.read_text(errors="replace").splitlines():
        if "Variables exceeding threshold" in line:
            try:
                return int(line.split(":")[-1].strip())
            except ValueError:
                return None
    return None


def one_variant(flag: str, value: str, case: str, iters: int, thresh: float, work: Path) -> dict:
    vdir = work / f"{flag}_{value.strip('.')}"
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True)
    flags_file = vdir / "flags.in"
    flags_file.write_text(flip_flag(DEFAULT_FLAGS.read_text(), flag, value))

    jax_out, f90_out = vdir / "jax", vdir / "fort"
    common = [sys.executable, RUN_SCM, "-stats", STATS, "-multicol", HR_SPEC,
              "-debug", "0", "-flags", flags_file, "-max_iters", str(iters)]
    t0 = time.time()
    jax_rc = run([*common, "-jax", "-out_dir", jax_out, case], vdir / "jax.log")
    if jax_rc != 0:
        return {"flag": flag, "value": value, "status": "JAX_FAIL_LOUD",
                "elapsed": time.time() - t0, "note": tail(vdir / "jax.log")}
    f90_rc = run([*common, "-out_dir", f90_out, case], vdir / "fort.log")
    if f90_rc != 0:
        return {"flag": flag, "value": value, "status": "FORTRAN_FAIL",
                "elapsed": time.time() - t0, "note": tail(vdir / "fort.log")}
    diff_rc = run([sys.executable, RUN_BINDIFF, "-v", "2", "-case", case,
                   "-t", str(thresh), "-pt", str(thresh), jax_out, f90_out], vdir / "diff.log")
    exceed = bindiff_exceed(vdir / "diff.log")
    status = "MATCH" if (diff_rc == 0 or exceed == 0) else "DIFF"
    return {"flag": flag, "value": value, "status": status, "exceed": exceed,
            "elapsed": time.time() - t0, "note": "" if status == "MATCH" else tail(vdir / "diff.log", 12)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="bomex")
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=1e-6)
    ap.add_argument("--flags", nargs="+", default=None, help="flag=value overrides (default: built-in sweep)")
    ap.add_argument("--work", default=str(REPO / "flag_variant_results"))
    args = ap.parse_args()

    overrides = args.flags if args.flags else DEFAULT_SWEEP
    work = Path(args.work)
    print(f"Flag-variant sweep: case={args.case} iters={args.max_iters} thresh={args.threshold} "
          f"n={len(overrides)}", flush=True)
    results = []
    for spec in overrides:
        flag, value = spec.split("=", 1)
        try:
            r = one_variant(flag, value, args.case, args.max_iters, args.threshold, work)
        except KeyError as e:
            r = {"flag": flag, "value": value, "status": "NO_SUCH_FLAG", "note": str(e), "elapsed": 0.0}
        results.append(r)
        exc = f" exceed={r.get('exceed')}" if r.get("exceed") is not None else ""
        print(f"  [{r['status']:14}] {flag}={value}{exc} ({r['elapsed']:.0f}s)", flush=True)
        if r["status"] in ("DIFF", "JAX_FAIL_LOUD") and r.get("note"):
            print("      " + r["note"].replace("\n", "\n      "), flush=True)

    print("\nSummary:", flush=True)
    for st in ("MATCH", "DIFF", "JAX_FAIL_LOUD", "FORTRAN_FAIL", "NO_SUCH_FLAG"):
        names = [f"{r['flag']}={r['value']}" for r in results if r["status"] == st]
        if names:
            print(f"  {st}: {len(names)}  {names}", flush=True)
    return 1 if any(r["status"] == "DIFF" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
