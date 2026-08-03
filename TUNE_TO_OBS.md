# tune-to-obs — differentiable parameter tuning of CLUBB-JAX to observations

Branch `tune-to-obs` (off `comble-case`). Goal: use the whole-driver `jax.grad`
of the JAX CLUBB port to fit model parameters to observed profiles by gradient
descent (Adam over a short reverse-mode trajectory).

## Status (2026-08-03)

**DONE — CLUBB closure-coefficient tuning, end to end.**
- 5 coefficients tuned in log space: `C1, C11, c_K, gamma_coef, mult_coef`
  (they flow as a traced `state['clubb_params']` array — no plumbing needed).
- Loss: per-field-normalized MSE over mean profiles + turbulence moments.
  Optimizer: hand-rolled Adam + lr decay (`clubb_jax/run_scripts/tune_coeffs.py`).
- Whole-driver `jax.grad` was hardened to be finite + FD-correct (commit
  `bece0db` + deep-convection fixes): the pattern is `_safe_sqrt`/`_safe_pow`
  for `sqrt`/`**frac` that hit 0, a safe denominator for `where(c, a/b, 0)`
  double-wheres, and dropping the `C_varying` compute-shortcut `where` (it
  zeroed the gradient at `C==Cb`). Grad requires `fill_holes_type=global_fill`.
- Synthetic-recovery validation: loss ↓1e4×, coeffs recovered to <8%.
- **Real obs**: tunes `mc3e` to the real **ARM VARANAL** observed profiles.
  `obs_target.py` loads an ARM VARANAL netCDF (T/q/u/v) onto the model grid;
  `tune_coeffs.py --obs <file.nc>`. Obs = MC3E VARANAL
  `sgp180iopsndgv3varanaC1.c1` (2011-04-22 IOP; MC3E case start == obs start).
- **BUGSrad wired into the driver**: `advance_clubb_to_end` now uses
  `radiation_module.advance_clubb_radiation` (was the limited `radiation.py`).
  BUGSrad was already ported; it just wasn't reachable. arm stays `Result[bit]`
  PASS. SILHS stays disabled (unported) → mc3e = morrison + bugsrad,
  `lh_microphys_type=disabled` (working namelist
  `clubb_jax/output/mc3e_compare_jax/mc3e.in`).

**IN PROGRESS — Morrison microphysics-parameter tuning** (target `Nc_in_cloud`).
See `MORRISON_PORT.md` for the full runbook. Morrison is detached under
`jax.grad` by design; its scheme (`module_mp_graupel.py`, 1336 lines pure-jnp)
must be hardened op-by-op to be differentiable. First batch done (slope
`_safe_pow`s); large multi-session remainder.

**Caveat (science, not pipeline):** over a short deep-convection window the loss
is nearly flat in the parameter directions (forcing/IC-dominated), so params
drift. Meaningful tuning needs box constraints + regularization + a longer
window with active precipitation.

## Key files
- `clubb_jax/run_scripts/tune_coeffs.py` — Adam tuner (`--obs` for real obs).
- `clubb_jax/run_scripts/obs_target.py` — ARM VARANAL netCDF → model-grid target.
- `clubb_jax/tests/probe_coeff_grad.py` — validate coeff grad (finite + FD).
- `clubb_jax/tests/probe_morrison_grad.py` — Morrison grad probe (`--nanhunt`).
- `clubb_jax/src/CLUBB_core/tracer_numpy.py` — `_safe_sqrt`, `_safe_pow`, `_iset`.

Reference: JAX-SCM v1.0 (arXiv 2605.24544) — adjacent differentiable-SCM prior art.
