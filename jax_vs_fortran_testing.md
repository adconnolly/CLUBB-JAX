# JAX vs Fortran Testing Quick Start

Run the adapted comparison runner from the repo root:

```bash
python clubb_jax/run_scripts/run_jax_vs_fortran_cases.py \
  --stats input/stats/standard_stats.in \
  --debug 0 \
  --cases fire
```

Run this with whatever Python environment is appropriate for the checkout. The
runner uses `sys.executable` for child JAX and Fortran runs, so the Python used
to launch the runner is also the Python used for the underlying case runs.

## Common Commands

Run one case:

```bash
python clubb_jax/run_scripts/run_jax_vs_fortran_cases.py \
  --stats input/stats/standard_stats.in \
  --debug 0 \
  --cases arm
```

Run selected failing cases:

```bash
python clubb_jax/run_scripts/run_jax_vs_fortran_cases.py \
  --stats input/stats/standard_stats.in \
  --debug 0 \
  --cases arm gabls3_night
```

Limit timesteps for faster debugging:

```bash
python clubb_jax/run_scripts/run_jax_vs_fortran_cases.py \
  --stats input/stats/standard_stats.in \
  --debug 0 \
  --max-iters 30 \
  --cases fire
```

Use a different stats file:

```bash
python clubb_jax/run_scripts/run_jax_vs_fortran_cases.py \
  --stats input/stats/multi_col_stats.in \
  --debug 0 \
  --cases bomex
```

## Speed Levers

`--cases`: biggest lever. Run only the failing case or cases, not the full
suite.

`--max-iters N`: caps timesteps for all selected cases.

- `--max-iters 1`: catches initialization and stats wiring problems.
- `--max-iters 5` or `30`: catches early timestep drift without waiting for a
  full case.
- Omit it to use the runner's per-case defaults.

`--stats`: smaller or more focused stats files reduce stats overhead. Use a
small stats file while debugging, then rerun with `standard_stats.in` before
trusting the fix.

`--debug 0`: faster and quieter. The runner defaults to debug 2 unless
specified.

## Outputs To Check

Main files:

```text
jax_driver_test_results/final_bindiff.log
jax_driver_test_results/case_compare_summary.json
jax_driver_test_results/<case>_run_jax.log
jax_driver_test_results/<case>_run_fortran.log
jax_driver_test_results/<case>_bindiff.log
```

Typical loop:

1. Read `jax_driver_test_results/final_bindiff.log`.
2. Identify failing cases and variables.
3. Rerun only those cases with `--cases`.
4. For quick debugging, add `--max-iters 30`.
5. Once fixed, rerun the failing cases without `--max-iters`.

A successful case can still say linux diff detected differences because NetCDF
metadata or ordering can differ. What matters is:

```text
Variables exceeding threshold: 0
bindiff_rc=0
```
