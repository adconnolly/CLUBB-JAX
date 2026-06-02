# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Start of every session

1. **Read `DESIGN.md` in full.** It contains the repository structure, test instructions, critical conventions, the full table of what has been built, remaining work, and the agent working rules. Do not proceed without reading it.
2. **Append to `CHANGELOG.md` at the end of the session** — one concise entry summarising what changed. Do not read the full changelog history; it is append-only. Periodically condense newer CHANGELOG.md entries.

---

## Running tests

The correctness gate is the **multi-case Fortran-vs-JAX regression** over
all bit-faithful cases (18 as of Iter299), plus a periodic **durability** run that catches
late-activating events the short gate masks. See DESIGN.md ("How to Test") for the full rationale.

```bash
# ── Primary gate: multi-case regression dashboard (one pass/fail line per case) ──
python clubb_jax/run_scripts/compare_cases.py --max-iters 30          # all bit-faithful cases
python clubb_jax/run_scripts/compare_cases.py --cases arm,bomex,gabls3 --max-iters 15

# ── Durability gate (run periodically): 100+ steps surfaces time-gated forcings, cloud/precip
#    onset, solar transitions — events the 30-step gate misses. ──
python clubb_jax/run_scripts/compare_cases.py --max-iters 100

# Single-case regression + divergence diagnosis when a case fails:
python clubb_jax/run_scripts/compare_runs.py --case arm --max-iters 30
python clubb_jax/run_scripts/diagnose_divergence.py arm   # classifies onset: bit-faithful / FP-growth / JUMP@N

# Quick smoke test — JAX driver, no Fortran comparison run (~20s):
python clubb_jax/run_scripts/run_scm.py arm -jax -max_iters 3
#   JAX runs default to clubb_jax/output/<case>_stats.nc; the Fortran oracle lives in
#   clubb_release/output/. A bare -jax run no longer clobbers the oracle.
#   See DESIGN.md "Output-directory convention".

# ── Unit tests ── whole suite in one command (exit 0 iff all green; oracle-less tests SKIP cleanly):
python clubb_jax/run_scripts/run_all_tests.py
python clubb_jax/run_scripts/run_all_tests.py -k solver   # only files matching "solver"
# …or a single file directly:
python clubb_jax/tests/test_solver.py
```

`compare_*` and `run_scm.py -jax` require compiled artifacts in `clubb_release/` — see DESIGN.md.
Unit tests require only JAX (those comparing against the f2py oracle SKIP when it is unbuilt).

---

## Architecture

The translation strategy is **incremental replacement with shadow comparison**: each Fortran function is ported to JAX, verified bit-for-bit alongside the Fortran oracle inside the running timestep loop, then the Fortran call is removed. **18 cases are now bit-faithful and run 100% in JAX/Python** — zero Fortran calls per timestep (19 cases have entirely-in-JAX forcings). ARM was the original focus; the gate has since generalized across cases (see DESIGN.md for the per-case status table and the irreducible FP/oracle/unported-subsystem limits on the rest). The Fortran remains essential as (a) the compiled bit-comparison oracle and (b) the porting source reference.

**Execution flow:**

```
src/clubb_standalone.py                ← entry point (python -m clubb_jax.src.clubb_standalone) ↔ clubb_standalone.F90: thin argv frontend → run_clubb
  src/clubb_driver.py                  ← run_clubb / init_clubb_case / clean_up_clubb ↔ clubb_driver.F90
  src/advance_clubb_to_end.py          ← timestep loop (advance_clubb_to_end subroutine of clubb_driver.F90): forcings → advance → stats
    src/CLUBB_core/advance_clubb_core_module.py  ← physics: all 16 prognostic variables
    src/Benchmark_cases/generic_forcings.py      ← large-scale forcings/surface for most cases
    src/Benchmark_cases/arm.py         ← ARM-specific forcings (prescribe_forcings_arm)
    src/io/stats_writer.py             ← NetCDF output (StatsWriter)
```

`state` is a plain Python dict passed through the call stack. It holds all prognostic arrays (shape `(ngrdcol, nzm)` or `(ngrdcol, nzt)`), grid object, flags, params, and the `stats_writer`.

**JAX vs NumPy:** JAX arrays are used for all physics computations inside `advance_clubb_core_module.py`. Arrays enter as NumPy (from Fortran initialisation or NetCDF), are converted to JAX at each call site with `jnp.asarray(...)`, and results are written back to the state dict as NumPy via `np.asarray(...)`. x64 mode is enabled globally in `advance_clubb_core_module.py`.

**`clubb_release/` submodule:** provides the Fortran oracle source (`src/CLUBB_core/*.F90`), input files (`input/case_setups/`, `input/tunable_parameters/`), and compiled binaries. `_CLUBB_RELEASE_ROOT` in `src/clubb_standalone.py` resolves all paths to it as a sibling of `clubb_jax/`.

**Module naming convention:** every `src/CLUBB_core/*.py` file mirrors its Fortran oracle at the identical relative path — `src/CLUBB_core/diffusion.py` ↔ `clubb_release/src/CLUBB_core/diffusion.F90`. Look up the Fortran source first when porting.
