# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Start of every session

1. **Read `DESIGN.md` in full.** It contains the repository structure, test instructions, critical conventions, the full table of what has been built, remaining work, and the agent working rules. Do not proceed without reading it.
2. **Append to `CHANGELOG.md` at the end of the session** — one concise entry summarising what changed. Do not read the full changelog history; it is append-only. Periodically condense newer CHANGELOG.md entries.

---

## Running tests

Correctness is a **tiered standard** (DESIGN.md "Correctness standard"), not pure bit-matching. The dual goal is
**faithful** (vs the Fortran oracle, within the chaos horizon) **and differentiable** (whole-driver `jax.grad`),
so there are two gates:

```bash
# ── Faithfulness gate: multi-case Fortran-vs-JAX regression (one pass/fail line per case) ──
python clubb_jax/run_scripts/compare_cases.py --max-iters 30                  # 20 bit-faithful DEFAULT_CASES (default --tier bit)
python clubb_jax/run_scripts/compare_cases.py --tier physical --max-iters 30  # Tier-C field-scaled tolerances (the numerical-accuracy gate)
python clubb_jax/run_scripts/compare_cases.py --cases tier_c                  # physically-faithful-but-FP-limited suite (cgils_s11, cgils_s12)
python clubb_jax/run_scripts/compare_cases.py --cases arm,bomex,gabls3 --max-iters 15
python clubb_jax/run_scripts/compare_cases.py --list                          # DEFAULT / TIER_C / BLOCKED case sets

# ── Differentiability gate: whole-driver jax.grad must be finite for every case ──
python clubb_jax/run_scripts/compare_grad.py                                  # dashboard (grad analogue of compare_cases)
python clubb_jax/run_scripts/compare_grad.py --cases bomex,dycoms2_rf01,cgils_s11

# Single-case regression + divergence diagnosis when a case fails:
python clubb_jax/run_scripts/compare_runs.py --case arm --max-iters 30
python clubb_jax/run_scripts/diagnose_divergence.py arm   # classifies onset: bit-faithful / FP-growth (chaos) / JUMP@N (term/threshold bug)

# Quick smoke test — JAX driver, no Fortran comparison run (~20s):
python clubb_jax/run_scripts/run_scm.py arm -jax -max_iters 3
#   JAX runs default to clubb_jax/output/<case>_stats.nc; the Fortran oracle lives in clubb_release/output/.
#   (compare_runs.py writes to output/<case>_compare_{jax,fort}/ — see DESIGN.md "Output-directory convention".)

# ── Unit tests ── whole suite in one command (exit 0 iff all green; oracle-less tests SKIP cleanly):
python clubb_jax/run_scripts/run_all_tests.py             # ~91 files (bugsrad/standalone files are slow)
python clubb_jax/run_scripts/run_all_tests.py -k solver   # only files matching "solver"
python clubb_jax/tests/test_solver.py                     # …or a single file directly
```

`compare_*`, `compare_grad.py`, and `run_scm.py -jax` require the compiled `clubb_release/` artifacts (Fortran
binary + f2py `.so`) — see DESIGN.md. Unit tests need only JAX (f2py-oracle tests SKIP when it is unbuilt).

**Operational note:** long runs auto-backgrounded by the harness write their logs to a `tasks/` tmpfs that
intermittently reports ENOSPC (a quota artifact) and silently truncates the output. For a long `compare_*` /
`run_all_tests` run, launch it detached with output redirected to the working dir —
`(python … > out.txt 2>&1 &)` — and read `out.txt`, rather than relying on the backgrounded stdout.

---

## Architecture

The translation strategy was **incremental replacement with shadow comparison**: each Fortran routine was ported to JAX and verified against the Fortran oracle inside the running timestep loop, then the Fortran call removed. The **numerical-accuracy refactor** (DESIGN.md "Correctness standard") then relaxed the original strict bit-faithfulness gate to a **tiered standard** that favors *differentiability* and accuracy — bit-faithfulness is preserved where it's achievable and kept as a debugging tool (`--tier bit`), but the goal is now **faithful AND differentiable**, not byte-for-byte (trajectory bit-agreement is physically impossible past the turbulence Lyapunov horizon anyway).

**State of the port:** the model runs **100% in JAX** — zero Fortran calls per timestep. **20 cases are bit-faithful** (`compare_cases.py` DEFAULT_CASES) and **all 19+ cases are whole-driver `jax.grad`-differentiable** (`compare_grad.py`). The CGILS/cloud_feedback family was brought in on this branch — Press[Pa]-sounding→altitude + absolute-`T[K]`→θ init, the case-specific BUGSrad extended atmosphere (from the case's deep sounding + ozone sounding), and the forcing out-of-range zero-fill fix — so cgils_s11/s12 are now both Tier-C-faithful **and** differentiable. The remaining non-bit-faithful cases are FP-limited (chaotic cloud-onset sensitivity) or oracle-limited (the Fortran's own single-precision imprecision), not bugs — see DESIGN.md. Every in-scope, oracle-validatable Fortran routine is now ported; the only unported `.F90` are no-oracle/impractical subsystems (COAMPS microphysics, the GFDL CCN 5-D lookup, SCM aerosol activation, SILHS RNG). The Fortran remains essential as (a) the compiled comparison oracle and (b) the porting source reference.

**Execution flow:**

```
src/clubb_standalone.py                ← entry point (python -m clubb_jax.src.clubb_standalone) ↔ clubb_standalone.F90: thin argv frontend → run_clubb
  src/clubb_driver.py                  ← run_clubb / init_clubb_case / clean_up_clubb ↔ clubb_driver.F90
    · init: reads the sounding (Input_fields/sounding.py; Press[Pa] soundings → convert_pressure_sounding_to_z),
      builds the grid + hydrostatic pressure (calc_pressure.py), the forcings/surface, and (BUGSrad) the radiation grid.
  src/advance_clubb_to_end.py          ← timestep loop (advance_clubb_to_end of clubb_driver.F90): forcings → +micro/rad tendencies → advance → radiation → microphysics → stats
    src/Benchmark_cases/generic_forcings.py / arm.py  ← large-scale forcings + surface (prescribe_forcings; generic reader + analytic/ARM cases)
    src/CLUBB_core/advance_clubb_core_module.py       ← the closure physics: all 16 prognostic moments (PDF closure, the advances, mixing length, clipping)
    src/Radiation/radiation.py                        ← radiation dispatch (BUGSrad correlated-k / simplified rad), every dt_rad
    src/Microphys/{morrison,kk}_microphys_step.py     ← microphysics dispatch (Morrison M2005 / Khairoutdinov-Kogan), per microphys_scheme, after microphys_start_time
    src/io/stats_writer.py                            ← NetCDF output (StatsWriter)
```

`state` is a plain Python dict passed through the call stack. It holds all prognostic arrays (shape `(ngrdcol, nzm)` or `(ngrdcol, nzt)`), grid object, flags, params, and the `stats_writer`.

**JAX vs NumPy:** JAX arrays are used for all physics computations inside `advance_clubb_core_module.py`. Arrays enter as NumPy (from Fortran initialisation or NetCDF), are converted to JAX at each call site with `jnp.asarray(...)`, and results are written back to the state dict as NumPy via `np.asarray(...)`. x64 mode is enabled globally in `advance_clubb_core_module.py`.

**`clubb_release/` submodule:** provides the Fortran oracle source (`src/CLUBB_core/*.F90`), input files (`input/case_setups/`, `input/tunable_parameters/`), and compiled binaries. `_CLUBB_RELEASE_ROOT` in `src/clubb_standalone.py` resolves all paths to it as a sibling of `clubb_jax/`.

**Module naming convention:** every `src/CLUBB_core/*.py` file mirrors its Fortran oracle at the identical relative path — `src/CLUBB_core/diffusion.py` ↔ `clubb_release/src/CLUBB_core/diffusion.F90`. Look up the Fortran source first when porting.
