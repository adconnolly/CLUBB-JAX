# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

---

## Start of every session

1. **Read `DESIGN.md` in full** — repository structure, test instructions, critical conventions, what has been
   built, remaining work, and the agent working rules.
2. **At the end, append one concise entry to `CHANGELOG.md`** (under "Recent work"). Do not read the full changelog
   history; it is condensed and the full record is in git. Periodically condense newer entries.

Per-file Fortran↔JAX port status lives in `TRANSLATION_STATUS.md` (kept current by `mirror_audit.py`).

---

## Running tests

Correctness is a **tiered standard** (DESIGN.md "Correctness standard"), not pure bit-matching. The dual goal is
**faithful** (vs the Fortran oracle, within the chaos horizon) **and differentiable** (whole-driver `jax.grad`), so
there are two gates:

```bash
# ── Faithfulness gate: multi-case Fortran-vs-JAX regression (one pass/fail line per case) ──
python clubb_jax/run_scripts/compare_cases.py --max-iters 30                  # 20 bit-faithful DEFAULT_CASES (default --tier bit)
python clubb_jax/run_scripts/compare_cases.py --tier physical --max-iters 30  # Tier-C field-scaled tolerances
python clubb_jax/run_scripts/compare_cases.py --cases tier_c                  # FP-limited suite (cgils_s11, cgils_s12)
python clubb_jax/run_scripts/compare_cases.py --list                          # DEFAULT / TIER_C / BLOCKED case sets

# ── Differentiability gate: whole-driver jax.grad must be finite for every case ──
python clubb_jax/run_scripts/compare_grad.py                                  # dashboard (grad analogue of compare_cases)

# Single-case regression + divergence diagnosis when a case fails:
python clubb_jax/run_scripts/compare_runs.py --case arm --max-iters 30
python clubb_jax/run_scripts/diagnose_divergence.py arm   # onset: bit-faithful / FP-growth (chaos) / JUMP@N (term/threshold bug)

# Quick smoke test — JAX driver, no Fortran (~20s; writes clubb_jax/output/<case>_stats.nc):
python clubb_jax/run_scripts/run_scm.py arm -jax -max_iters 3

# Unit tests — whole suite in one command (exit 0 iff all green; oracle-less tests SKIP cleanly):
python clubb_jax/run_scripts/run_all_tests.py             # ~165 files (bugsrad/standalone are slow)
python clubb_jax/run_scripts/run_all_tests.py -k solver   # only files matching "solver"
python clubb_jax/tests/test_solver.py                     # …or a single file directly

# Mirror-name audit (pure-Python, no JAX/oracle needed):
python clubb_jax/run_scripts/mirror_audit.py
```

`compare_*`, `compare_grad.py`, and `run_scm.py -jax` require the compiled `clubb_release/` artifacts (Fortran
binary + f2py `.so`). Unit tests need only JAX (f2py-oracle tests SKIP when it is unbuilt).

**Operational note:** long backgrounded runs write logs to a `tasks/` tmpfs that intermittently reports ENOSPC (a
quota artifact) and silently truncates output. For a long `compare_*` / `run_all_tests` run, launch it detached
with output to the working dir — `(python … > out.txt 2>&1 &)` — and read `out.txt`.

---

## Architecture

The port was built by **incremental replacement with shadow comparison** (port each Fortran routine, verify against
the in-loop oracle to machine epsilon, remove the Fortran call), then a **numerical-accuracy refactor** relaxed the
strict bit-faithfulness gate to a **tiered standard** favoring differentiability and accuracy. The goal is now
**faithful AND differentiable**, not byte-for-byte (trajectory bit-agreement is physically impossible past the
turbulence Lyapunov horizon). Bit-faithfulness is preserved where achievable and kept as a debug tool (`--tier
bit`). See DESIGN.md "Correctness standard".

**State of the port:** runs **100% in JAX** — zero Fortran calls per timestep. **20 cases are bit-faithful** and
**all 19+ cases are whole-driver `jax.grad`-differentiable**. Every in-scope, oracle-validatable Fortran routine is
ported; the only unported `.F90` are no-oracle/impractical subsystems (COAMPS microphysics, the GFDL CCN 5-D
lookup, SCM aerosol activation, SILHS RNG). The Fortran remains the compiled comparison oracle and the porting
reference.

**Execution flow:**

```
src/clubb_standalone.py        ← CLI frontend (python -m clubb_jax.src.clubb_standalone) ↔ clubb_standalone.F90
  src/clubb_driver.py          ← run_clubb / init_clubb_case / clean_up_clubb ↔ clubb_driver.F90
                                  init: sounding (Input_fields/sounding.py) → grid + hydrostatic pressure
                                  (calc_pressure.py) → forcings/surface → (BUGSrad) radiation grid
  src/advance_clubb_to_end.py  ← timestep loop: forcings → +micro/rad tendencies → advance → radiation → microphysics → stats
    src/Benchmark_cases/prescribe_forcings.py   ← large-scale forcings + surface (generic dispatch + per-case modules)
    src/CLUBB_core/advance_clubb_core_module.py ← closure physics: all prognostic moments (PDF closure, advances, mixing length, clipping)
    src/Radiation/radiation_module.py           ← radiation dispatch (BUGSrad correlated-k / simplified), every dt_rad
    src/Microphys/{morrison,kk}_microphys_step.py ← microphysics dispatch (M2005 / Khairoutdinov-Kogan)
    src/io/stats_writer.py                      ← NetCDF output (StatsWriter)
```

`state` is a plain Python dict passed through the call stack — all prognostic arrays (shape `(ngrdcol, nzm)` or
`(ngrdcol, nzt)`), grid object, flags, params, and the `stats_writer`.

**JAX vs NumPy:** physics computations inside `advance_clubb_core_module.py` use JAX arrays. Arrays enter as NumPy
(from init or NetCDF), convert to JAX at each call site (`jnp.asarray`), and write back as NumPy (`np.asarray`).
x64 mode is enabled globally in `advance_clubb_core_module.py`.

**`clubb_release/` submodule** provides the Fortran oracle source, input files (`input/case_setups/`,
`input/tunable_parameters/`), and compiled binaries; `_CLUBB_RELEASE_ROOT` in `clubb_standalone.py` resolves paths
to it as a sibling of `clubb_jax/`.

**Module naming convention:** every `src/CLUBB_core/*.py` mirrors its Fortran oracle at the identical relative path
(`diffusion.py` ↔ `clubb_release/src/CLUBB_core/diffusion.F90`). Look up the Fortran source first when porting;
`mirror_audit.py` enforces the mirror.
