# tune-to-obs — differentiable coefficient tuning of CLUBB-JAX to observations

Branch `tune-to-obs` (off `comble-case`). Goal: use the whole-driver `jax.grad`
of the JAX CLUBB port to fit a small set of high-impact closure coefficients to
observed boundary-layer structure by gradient descent.

## Decisions (from the task interview)

**Data route — pipeline-first on a light case.** Stand up the full differentiable
Adam tuning loop on an existing case (`arm` 1997 or `wangara`) using obs that can
be fetched cleanly (mean profiles + surface fluxes + BL depth). Add turbulence-
variance obs (ARM SGP Doppler-lidar `wp2`/skewness, 2010+, e.g. via a LASSO
bundle) as a richer target once the loop works. LASSO (freely downloadable SGP
shallow-convection bundles, obs+forcing+LES) is the intended richer second target;
it needs a free ARM account and a COMBLE-sized new-case build.

**Coefficients (5, high-impact for CBL structure), tuned in transformed/box-constrained space:**
| param | default | lever |
|-------|--------:|-------|
| `C1`         | 1.00 | wp2 dissipation timescale → TKE magnitude, BL depth |
| `C11`        | 0.40 | wp3 buoyancy → skewness, entrainment |
| `c_K`        | 0.20 | eddy diffusivity (wp2) → flux profiles |
| `gamma_coef` | 0.25 | skewness function → third-moment transport |
| `mult_coef`  | 0.50 | mixing length → BL depth / entrainment |

**Loss:** per-field-normalized RMSE over three groups, equal group weights —
(a) mean profiles thlm, rt, u, v; (b) turbulence moments/fluxes wp2, wpthlp,
wprtp; (c) surface sensible/latent flux + BL depth (inversion height).

**Optimizer / horizon:** Adam; reverse-mode gradient through a short ~30–60-step
(~0.5–1 h) window with `jax.checkpoint`/remat to bound memory.

## De-risking (do before wiring the optimizer)
1. Confirm the whole-driver `jax.grad` is finite + finite-difference-correct through
   30–60 steps (it is only *validated* for 1 step today; `compare_grad.py`).
2. Expose the 5 coefficients as **traceable inputs** to the JAX computation (not
   baked into init) so `jax.grad` w.r.t. them works. Check how tunable params flow
   from `tunable_parameters.in` into `advance_clubb_core_module` and thread them.

## Open items
- Pick + fetch the light-case obs (arm-1997 GCSS validation vs wangara published
  profiles) — verify availability first.
- Later: LASSO bundle download + new-case build for the turbulence-rich target.
- Reference: JAX-SCM v1.0 (arXiv 2605.24544) — adjacent differentiable-SCM prior art.
