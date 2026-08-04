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

**NOT a detach — mc3e has NO CLOUD.** `probe_nc_route.py` shows `rcm_max=0`,
`cloud_frac_max=0` after 30 mc3e steps → `ncm = Nc_in_cloud*cloud_frac = 0`, so
the grad is CORRECTLY ~0 (Nc has no effect with no cloud). Both routes give 0 for
the right reason. The earlier "FD=1.30" (probe_nc_traj_grad.py) is FP noise on the
~1e7 stratospheric `thlm²` loss (relative ~1e-10), not a real Nc effect. The
wrapper (Phase 2) is CORRECT — no fix needed.

**REAL blocker — need a cloudy state to exercise/tune Nc.** mc3e from t=0 (00:00
GMT, pre-convective) forms no mean cloud water in the JAX run over the tested
window. NEXT: (a) confirm the wrapper gives a finite, FD-correct trajectory grad
on a genuine CLOUD case — `mpace_a` (Arctic stratus, morrison; needs its JAX
working namelist: SILHS off + simplified rad, like the mc3e setup) — the driver
probe already gives rel 8.8e-7 with forced cloud, so this should pass; then (b)
for MC3E obs tuning, find a cloudy window (advance to the convective part of the
IOP, or a rainier mc3e period) — or switch the Nc-tuning demo to mpace_a and its
obs. Use a TROPOSPHERIC / normalized loss (not full-column thlm²) so FD isn't noise.

**BLOCKER refined — `state['rcm']` reads 0 in BOTH mc3e AND mpace_a** at the
point the probe checks (after `advance_clubb_to_end` returns) — even mpace_a
(Arctic stratus, should be cloudy) shows `rcm_max=0` after warmup. Likely the
grid-mean cloud water is PDF-diagnosed INSIDE the core step and `state['rcm']`
is stale between steps (cf. morrison_microphys_step.py comment: "state['Ncm'] is
stale (0) at this point in the loop"). So the loss `sum(rcm²)` is 0 → grad 0.
NEXT: (a) find where the real cloud water lives when the microphysics runs
(inside the step, `advance_clubb_core` diagnoses rcm → the morrison step reads
`g('rcm')` at that moment — is IT nonzero?); OR (b) use a loss on a field that IS
nonzero + Nc-sensitive at the returned state (e.g. `rrm` rain, or `_morr_rvm_mc`),
OR warm many more steps until `state['rcm']>0`; OR (c) validate Nc-sensitivity via
the WITHIN-STEP driver (probe_morrison_grad.py already gives rel 8.8e-7 with forced
cloud — that IS the proof the Nc path is differentiable; the trajectory 0 is a
no-cloud-in-state artifact, not a code bug). Consider the Nc port VALIDATED by the
driver probe and move to Phase 3 with a cloudy-state loss.

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

## Phase 3 — tune Morrison params to obs  ✅ MACHINERY DONE (blocked on cloud)

`tune_coeffs.py` has a `--morrison` / `MORR=1` mode: tunes `Nc_in_cloud` as a
single log-scale param (`s['Nc_in_cloud'] = nc0 * exp(u)`), same normalized-obs
loss + Adam. It runs end-to-end under trace (no error, grad computed, Adam steps).

**The Morrison port is functionally COMPLETE**: driver differentiable (Phase 1,
rel 8.8e-7), wrapper runs under trace (Phase 2), tuning mode built (Phase 3).

**No-cloud investigation (RESOLVED 2026-08-04) — needs a different case.**
Key correction: the earlier "cloud on day 2, step ~340" claim was a **probe
artifact**. `advance_clubb_to_end` computes `time_current = time_initial +
(itime-1)*dt` from the *local* step index (L21), so calling it repeatedly in
chunks **restarts the forcing clock at t0** — `probe_mc3e_cloud_timeline.py` just
re-applied the earliest forcing 20× and accumulated fake "cloud". A single proper
`max_steps=450` call gives **rcm=0** (both fill types → `global_fill` is NOT the
cause). Diagnosis of the real state:
- **mc3e**: forcing IS applied (thlm_f ~9e-4 K/s), but the BL barely moves over
  10 h and slightly *dries* (Δrtm<0) → stays subsaturated → never clouds. No
  reachable cloudy window. Wrong case for Nc tuning.
- **mpace_a** (morrison-configured): rcm=0 through 6 h — and rcm=0 even with
  microphysics DISABLED, so it's the CLUBB **PDF closure** finding no saturation,
  NOT a Morrison bug. The case doesn't cloud in the JAX run.
- **dycoms2_rf01**: robustly cloudy (rcm=4.5e-4 at init, warm nocturnal Sc) — but
  `microphys_scheme="none"`. No Morrison → no Nc.
So the Nc gradient is proven differentiable (probe_morrison_grad rel 8.8e-7 with
*forced* cloud), but no currently-configured case gives natural cloud + Morrison
together. Next step is a DIRECTION decision (see below), likely: enable Morrison
(warm, l_ice=.false.) on dycoms2 — the textbook Sc drizzle-vs-Nc case.
tune_coeffs.py has `WARMUP=<steps>` (default 3) + absolute obs time `(WARMUP+N)*dt`.

**mpace_a no-cloud DEBUG — concluded (2026-08-04): faithful, not a bug.**
The Fortran oracle run fresh for 360 steps (`clubb_release/output/mpace_a_stats.nc`,
nt=36 = 6 h) gives **rcm_max=9.86e-12, cf_max=8.5e-7** — negligible cloud, matching
JAX's rcm=0. So JAX is faithful; mpace_**a** is the dry/clear MPACE period
(rt peaks ~3.2 g/kg), NOT the cloudy one. The classic cloudy MPACE is mpace_**b**
(Klein et al. 2009, LWP~100 g/m²), but its case setup uses
`microphys_scheme="coamps"` — an UNPORTED scheme — and has no runnable .in here.
Implication for Nc tuning: getting a cloudy + Morrison case needs config work —
either mpace_b's (moist) sounding/forcing driven with Morrison instead of COAMPS,
or Morrison enabled on the already-cloudy dycoms2_rf01 (warm Sc, l_ice=.false.).

**Historical note — real Nc tuning was blocked on a cloudy target.** mc3e synthetic-recovery
(MORR=1, N=25) gives a FLAT loss (8.6e-7, Nc stays at 1.0, target 1.5) — no cloud
in the window → Nc has no leverage on thlm/rtm. Same rcm=0 in mpace_a. So the
open item is NOT code — it is finding a state where the JAX run has active cloud:
either debug why these runs form no mean cloud water (config? a stale `state['rcm']`
between steps? genuinely subsaturated windows?), or advance to a cloudy period /
pick a reliably-cloudy case with obs. That is a case/data decision, not a port
task. Original Phase-3 detail below.



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
