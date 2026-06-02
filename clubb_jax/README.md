# CLUBB-JAX

A JAX translation of the CLUBB single-column turbulence model, for differentiable,
composable use in ML and autodiff workflows.

This is a genuine JAX/Python implementation — **not** a wrapper around the Fortran. The
physics runs entirely in JAX: 18 benchmark cases are bit-for-bit faithful to the Fortran
reference and execute with **zero Fortran calls per timestep**. The original Fortran
(`../clubb_release/`) is kept only as (a) the bit-comparison oracle and (b) the porting
source reference.

> For the full design, the per-case status table, testing strategy, and conventions, see
> [`../DESIGN.md`](../DESIGN.md). This file is just install-and-run.

## Layout

```
clubb_jax/
├── src/
│   ├── clubb_standalone.py   ← CLI entry point (thin frontend ↔ clubb_standalone.F90)
│   ├── clubb_driver.py       ← run_clubb / init_clubb_case / clean_up_clubb ↔ clubb_driver.F90
│   ├── advance_clubb_to_end.py  ← the timestep loop
│   ├── CLUBB_core/           ← physics modules, one file per Fortran oracle
│   ├── Benchmark_cases/      ← per-case forcings & surface
│   ├── Radiation/            ← simple radiation + BUGSrad
│   └── io/                   ← NetCDF stats output
├── run_scripts/              ← run + comparison + test harnesses
└── tests/                    ← unit tests
```

Each `src/CLUBB_core/<name>.py` mirrors `../clubb_release/src/CLUBB_core/<name>.F90` at the
same relative path.

## Requirements

- **Python with JAX** (x64 mode is enabled by the code). NumPy and netCDF4. This is all the
  pure-JAX driver needs to run a case.
- **A sibling `clubb_release/` checkout** (`CLUBB-JAX/clubb_release/`). The run scripts read
  case namelists, soundings, and tunable parameters from `clubb_release/input/`.
- **Compiled Fortran artifacts in `clubb_release/`** — `bin/clubb_standalone` and
  `clubb_python_api/*.so` — are needed to (a) generate/compare against the Fortran oracle and
  (b) for the launcher's environment setup. They are build outputs, not in git; build them
  from the repo root with:
  ```bash
  ./compile.py [-debug] -python
  ```
  Note: the compiled API is the *oracle* for comparison — the JAX physics does not call it
  per timestep.

## Run a case

From the repository root:

```bash
# Run ARM in pure JAX for 30 steps:
python clubb_jax/run_scripts/run_scm.py arm -jax -max_iters 30
```

`-jax` writes stats to **`clubb_jax/output/<case>_stats.nc`** (the Fortran oracle lives
separately in `clubb_release/output/`, so a JAX run never clobbers it). Drop `-jax` to run
the compiled Fortran instead (`-legacy`).

You can also invoke the driver module directly on an already-aggregated namelist:

```bash
python -m clubb_jax.src.clubb_standalone clubb_jax/output/arm.in
```

## Compare against Fortran & run tests

```bash
# Single case: run Fortran and JAX, diff the prognostic stats (must be 0 failures):
python clubb_jax/run_scripts/compare_runs.py --case arm --max-iters 30

# All bit-faithful cases, one pass/fail line each:
python clubb_jax/run_scripts/compare_cases.py --max-iters 30

# Diagnose where/how a failing case diverges:
python clubb_jax/run_scripts/diagnose_divergence.py <case>

# Unit-test suite (pure JAX; oracle-dependent tests skip cleanly if unbuilt):
python clubb_jax/run_scripts/run_all_tests.py
```

The comparison scripts require the compiled Fortran artifacts above; the unit tests need only
JAX. See [`../DESIGN.md`](../DESIGN.md) for the testing rationale, durability gates, and the
list of which cases are bit-faithful.
