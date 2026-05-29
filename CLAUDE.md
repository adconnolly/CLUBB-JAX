# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Start of every session

1. **Read `DESIGN.md` in full.** It contains the repository structure, test instructions, critical conventions, the full table of what has been built, remaining work, and the agent working rules. Do not proceed without reading it.
2. **Append to `CHANGELOG.md` at the end of the session** — one concise entry summarising what changed. Do not read the full changelog history; it is append-only.

---

## Running tests

```bash
# Full Fortran-vs-JAX regression (the primary correctness gate):
python clubb_jax/run_scripts/compare_runs.py --case arm --max-iters 30

# Quick smoke test — JAX only, no comparison (3 timesteps, ~20s):
python clubb_jax/run_scripts/run_scm.py arm -jax -max_iters 3

# Run a single unit test file:
python clubb_jax/tests/test_solver.py
python clubb_jax/tests/test_diffusion.py
python clubb_jax/tests/test_penta_solver.py
```

`compare_runs.py` requires compiled artifacts in `clubb_release/` — see DESIGN.md. Unit tests require only JAX.

---

## Architecture

The translation strategy is **incremental replacement with shadow comparison**: each Fortran function is ported to JAX, verified bit-for-bit alongside the Fortran oracle inside the running timestep loop, then the Fortran call is removed. The ARM case is now 100% JAX/Python — zero Fortran calls per timestep.

**Execution flow:**

```
clubb_jax/clubb_standalone.py          ← entry point (python -m clubb_jax.clubb_standalone)
  src/clubb_standalone.py              ← init_clubb_case(): reads namelists, initialises state dict
  src/advance_clubb_to_end.py          ← timestep loop: forcings → advance → stats
    src/CLUBB_core/advance_clubb_core_module.py  ← physics: all 16 prognostic variables
    src/Benchmark_cases/arm.py         ← ARM forcings (prescribe_forcings_arm)
    src/io/stats_writer.py             ← NetCDF output (StatsWriter)
```

`state` is a plain Python dict passed through the call stack. It holds all prognostic arrays (shape `(ngrdcol, nzm)` or `(ngrdcol, nzt)`), grid object, flags, params, and the `stats_writer`.

**JAX vs NumPy:** JAX arrays are used for all physics computations inside `advance_clubb_core_module.py`. Arrays enter as NumPy (from Fortran initialisation or NetCDF), are converted to JAX at each call site with `jnp.asarray(...)`, and results are written back to the state dict as NumPy via `np.asarray(...)`. x64 mode is enabled globally in `advance_clubb_core_module.py`.

**`clubb_release/` submodule:** provides the Fortran oracle source (`src/CLUBB_core/*.F90`), input files (`input/case_setups/`, `input/tunable_parameters/`), and compiled binaries. `_CLUBB_RELEASE_ROOT` in `src/clubb_standalone.py` resolves all paths to it as a sibling of `clubb_jax/`.

**Module naming convention:** every `src/CLUBB_core/*.py` file mirrors its Fortran oracle at the identical relative path — `src/CLUBB_core/diffusion.py` ↔ `clubb_release/src/CLUBB_core/diffusion.F90`. Look up the Fortran source first when porting.
