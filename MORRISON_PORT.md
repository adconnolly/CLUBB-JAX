# MORRISON_PORT.md — runbook: make Morrison microphysics differentiable

**One-line goal:** make the JAX Morrison scheme produce a finite, FD-correct
reverse-mode gradient so its parameters (`Nc_in_cloud`, `C_evap`, autoconversion
coeffs) can be tuned by gradient descent — **without changing the forward
result** (the Morrison rate/sedimentation oracle tests must stay PASS).

This is a **loop-until-done** task. Each iteration = find the next NaN-producing
op → replace it with a forward-identical safe form → verify forward unchanged →
commit. Repeat until the probe reports a finite, FD-correct gradient. Then do
Phase 2 (wire the wrapper) and Phase 3 (tune).

---

## Environment (every session)
```bash
source /burg/home/ac5006/scratch/jaxenv/bin/activate 2>/dev/null || true   # jax 0.10.2 CPU venv
export PYTHONPATH=/burg-archive/glab/users/ac5006/CLUBB-JAX:/burg-archive/glab/users/ac5006/CLUBB-JAX/clubb_release
export JAX_PLATFORMS=cpu
PY=/burg/home/ac5006/scratch/jaxenv/bin/python
# For running the Fortran oracle (forward-faithfulness checks) also:
export LD_LIBRARY_PATH=/burg-archive/opt/intel-oneAPI-toolkit/HPCKit_p_2023.2.0.49440/compiler/2023.2.0/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH
```
Read `DESIGN.md` first (correctness standard, tracer-transparent toolkit,
conventions). The safe-op toolkit is `clubb_jax/src/CLUBB_core/tracer_numpy.py`.

---

## Phase 1 — harden the driver gradient (the loop)  ✅ DONE (2026-08-03)
Driver grad w.r.t. Ncm is finite + FD-correct (rel 8.8e-7) at an active-precip
state. Fixes: slope `_safe_pow` (commits `3aa842e`-style) + gamma `custom_jvp`
(`7e7a206`). The probe now forces active cloud+precip (it must — step-4 mc3e has
none, so FD=0/grad=0 is a false "clean"). If more NaNs appear at other operating
points, resume this loop; otherwise Phase 1 is complete → go to Phase 2.

**Probe (the iteration driver):**
```bash
# NaN attribution (finds the next singular op's file:line):
CLUBB_JAX_NO_JIT_CACHE=1 CLUBB_JAX_NO_WHOLE_STEP_JIT=1 $PY clubb_jax/tests/probe_morrison_grad.py --nanhunt
# Progress check (is the gradient finite + FD-correct yet?):
CLUBB_JAX_NO_JIT_CACHE=1 $PY clubb_jax/tests/probe_morrison_grad.py
```
The probe grads `sum(*_mc tendencies)` w.r.t. the droplet number `Ncm` from a
warmed `mc3e` state. `--nanhunt` enables `jax_debug_nans` and prints the deepest
`module_mp_graupel.py` frame of the first NaN.

**Attribution caveat:** functions are `@jax.jit`-decorated, so `debug_nans` may
report the *jit call site* rather than the exact inner line, and the empirical
line can be a *reverse-attribution* of a NaN that originates in a **downstream**
consumer (the NaN cotangent surfaces at the first un-jitted op it reaches). If a
reported op is fine in isolation, un-jit it via its `.__wrapped__` in the probe
(see how `nanhunt_mc3e` did it in git history) OR look downstream. Also: the
same singular op appears in MANY functions (24 fractional powers, ~180
divisions, 121 `where`s, 23 gamma calls) — fixing a *shared helper* (e.g. the
generic `cloud_slope`/`_slope_np` slope) clears many call sites at once.

**Fix patterns** (all forward-identical for a non-negative base / floored arg):
- `x ** (1/3)`, `x ** 0.5`, `x ** frac` where `x` can be 0 (hydrometeor mixing
  ratios `qr/qi/qs/qg` and numbers are 0 in clear air) → `_safe_pow(x, frac)`.
  SAFE already (do NOT wrap): `(_M_RHOSU/rho)**0.54`, `sc**(1/3)` — bases are
  strictly positive (air density, Schmidt number).
- `jnp.sqrt(x)` with `x` possibly ≤0 → `_safe_sqrt(x)`.
- `a / b` inside `jnp.where(cond, a/b, else)` where `b==0` on the masked branch
  (e.g. `/ lambda**b` with `lambda=0` in clear air) → give the division a safe
  denominator on the masked branch:
  `bs = jnp.where(cond, b, 1.0); jnp.where(cond, a/bs, else)`. OR make the
  producing slope helper return a safe nonzero (`1.0`) where off instead of `0`
  (many consumers mask their own output, so this is forward-identical and fixes
  all `/lambda**b` at once — high leverage; verify against the oracle test).
- Custom `gamma(x)` (module_mp_graupel.py ~L176): if ITS gradient is NaN (a
  lookup/branch approximation), it needs a differentiable reformulation or a
  `jax.custom_jvp` with the analytic `gamma(x)*digamma(x)`. Check with an
  isolated `jax.grad(gamma)(x)` first.
- `jnp.where(cond, f(x), g(x))` "double-where" where the NOT-taken branch has an
  inf/NaN derivative → make that branch's argument safe (as above); the outer
  `where` alone does NOT stop the NaN cotangent.

**Forward-faithfulness gate (MUST pass after every batch):**
```bash
$PY clubb_jax/tests/test_morrison_rates.py    # rate + sedimentation oracle tests must stay PASS
# (the trailing test_morrison_hm_metadata pdf_dim==4 assertion is a PRE-EXISTING
#  unrelated failure — ignore it; the rate/sed/conservation tests are what matter)
```
Commit each forward-faithful batch with a message naming the op class fixed.

**Phase-1 DONE when:** `probe_morrison_grad.py` prints
`d/d(Nc_scale)=<finite>` and `rel < 1e-2` (use `eps=1e-3`; microphysics is stiff,
so FD needs a larger step and rel ~1e-2 is acceptable). Use a warmed state where
precip is ACTIVE (advance more steps, or a rainier case) so FD is non-zero.

---

## Phase 2 — run Morrison under trace (wrapper)  🟡 IN PROGRESS

**Wrapper wired** (commit after this): `morrison_microphys_step.py` now runs the
driver under trace (`under_trace` includes `Nc_in_cloud`; jnp inputs; stores
`_morr_*_mc` jnp; skips transport under trace). Concrete path byte-identical.

**OPEN — the trajectory grad w.r.t. `Nc_in_cloud` is DETACHED** (analytic 0, but
FD=1.30 through ~6-8 mc3e steps; probe `clubb_jax/tests/probe_nc_traj_grad.py`).
Nc reaches thlm via TWO routes, both computing `_mc` tendencies applied to the
next step's forcing (advance_clubb_to_end.py:133-142):
  1. `_cloud_drop_sed` (advance_clubb_to_end.py:179 → cloud_sed_module.py) —
     `ncm = Nc_in_cloud*cloud_frac`, jnp, stores state['rcm_mc']/['thlm_mc'].
  2. Morrison dispatch (microphys_driver.py → morrison_microphys_step) —
     stores `_morr_*_mc` (gated by `time_current >= microphys_start_time`).
Both look jnp-clean, yet the gradient is a HARD 0 → a `np.asarray`/`stop_gradient`
detaches the Nc→`_mc`→forcing→thlm path somewhere. NEXT: isolate — grad of
`sum(state['rcm_mc'])` after ONE step w.r.t. Nc (does route 1 flow?), then
`sum(state['_morr_thlm_mc'])` (route 2). Find the np conversion on whichever is
cut. Likely candidates: how `state['Ncm']` / `Nc_in_cloud` is set before the
microphys phase, or a detach in the forcing carry between steps.

### Wiring reference (for reading)

`clubb_jax/src/Microphys/morrison_microphys_step.py` currently **returns early
under a jax.grad trace** (line ~28, `if _is_tracer_arg([...]): return`). Replace
the early return with a differentiable tendency path:
- Build inputs with `jnp.asarray(state[k])` (NOT `np.asarray` — that detaches).
- Call `morrison_microphys_driver(...)` (already jnp) with `Ncm` traceable.
- Store `state['_morr_rcm_mc'/'_morr_thlm_mc'/'_morr_rvm_mc']` as jnp (trace-safe).
- **Skip the hydrometeor transport under trace** (lines ~61+) — it does not feed
  thlm/rtm (the obs) at first order; keep it only on the concrete path.
Keep the concrete (non-trace) path byte-identical.

Validate: `probe_coeff_grad.py mc3e 10` style but tuning `Nc_in_cloud` — grad
should now flow through the full trajectory (finite + FD-correct).

---

## Phase 3 — tune Morrison params to obs

Expose `Nc_in_cloud` (and optionally `C_evap`) as traceable inputs; extend
`tune_coeffs.py` with a `--morrison` mode that injects `state['Nc_in_cloud']`
(check it is not used concretely elsewhere first). Tune to the MC3E VARANAL obs
(`--obs`) over a **longer window with active precip** (short windows are
insensitive — FD≈0 early). Add box constraints + regularization.

---

## Guardrails
- Never change a forward value. If `test_morrison_rates.py` regresses, the fix
  was not forward-identical — revert and use the safe-branch form.
- `mc3e` working namelist has SILHS off + bugsrad on (already set). Grad needs
  `fill_holes_type=global_fill` (the probe sets it).
- Commit small, forward-faithful batches. One op-class per commit.
- Progress state + history: `git log` on `tune-to-obs`; the CLUBB-coeff analogue
  (commits `bece0db`, `3aa842e`, `8230df2`) shows the exact fix style.
