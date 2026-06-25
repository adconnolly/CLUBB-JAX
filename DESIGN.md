# CLUBB-JAX Design

Goal: translate the CLUBB turbulence parameterization from Fortran to JAX for differentiable,
composable use in ML and autodiff workflows.

**State of the port.** The model runs **100% in JAX** — zero Fortran calls per timestep. Every in-scope,
oracle-validatable Fortran routine is ported; the file/routine-name mirror to the oracle is converged
(`python clubb_jax/run_scripts/mirror_audit.py` → PASS, all dimensions 0) with a single deliberately-deferred
routine, `pdf_closure_driver_zm` (gated by `l_call_pdf_closure_twice`, which no case sets — no validated case
and no f2py oracle to port against; the driver fail-loud rejects the flag). Per-file mapping lives in
`TRANSLATION_STATUS.md`. The Fortran remains essential as (a) the compiled comparison oracle and (b) the
porting reference.

---

## Repository Structure

```
CLUBB-JAX/
├── clubb_jax/              ← JAX translation (this repo)
│   ├── src/
│   │   ├── CLUBB_core/     ← physics modules, one file per Fortran oracle
│   │   ├── Benchmark_cases/
│   │   ├── Input_fields/
│   │   ├── Radiation/
│   │   ├── derived_types/  ← pure-Python type mirrors (ConfigFlags, Grid, pdf_parameter, …)
│   │   ├── io/             ← stats_writer.py (pure-Python NetCDF output)
│   │   ├── clubb_standalone.py  ← thin CLI frontend ↔ clubb_standalone.F90 (entry: python -m clubb_jax.src.clubb_standalone)
│   │   ├── clubb_driver.py      ← run_clubb / init_clubb_case / clean_up_clubb ↔ clubb_driver.F90
│   │   └── advance_clubb_to_end.py  ← timestep loop (advance_clubb_to_end subroutine of clubb_driver.F90)
│   ├── run_scripts/        ← test infrastructure (compare_*, run_scm, mirror_audit, diagnose_divergence)
│   └── tests/              ← unit tests
└── clubb_release/          ← git submodule: larson-group/clubb_release (master)
    ├── src/                ← Fortran source oracle
    ├── input/              ← case setups, namelists, sounding files
    ├── clubb_python_api/   ← f2py compiled wrappers (not in submodule git)
    └── bin/ install/       ← compiled Fortran binaries (not in submodule git)
```

`clubb_jax/` and `clubb_release/` are siblings. The JAX package works against upstream `clubb_release` master —
no dependency on any custom branch.

---

## How to Test

**Prerequisites:** `clubb_release/bin/clubb_standalone` and `clubb_release/clubb_python_api/*.so` must be present
(compiled artifacts, not in git). The regression tests need `clubb_python_api` (for the Fortran comparison run and
non-ARM `prescribe_forcings`). The JAX driver itself has zero module-level Fortran imports — ARM runs need no oracle.

```bash
# Quick smoke test — JAX driver, no Fortran (~20s; writes clubb_jax/output/arm_stats.nc):
python clubb_jax/run_scripts/run_scm.py arm -jax -max_iters 3

# Faithfulness gate — Fortran-vs-JAX regression (one pass/fail line per case):
python clubb_jax/run_scripts/compare_cases.py --max-iters 30                  # 20 bit-faithful DEFAULT_CASES
python clubb_jax/run_scripts/compare_cases.py --tier physical --max-iters 30  # Tier-C field-scaled tolerances
python clubb_jax/run_scripts/compare_cases.py --cases tier_c                  # FP-limited suite (cgils_s11/s12)
python clubb_jax/run_scripts/compare_cases.py --list                          # DEFAULT / TIER_C / BLOCKED sets
python clubb_jax/run_scripts/compare_runs.py --case arm --max-iters 30        # single case
python clubb_jax/run_scripts/diagnose_divergence.py rico                      # classify divergence onset

# Differentiability gate — whole-driver jax.grad must be finite for every case:
python clubb_jax/run_scripts/compare_grad.py                                  # dashboard
python clubb_jax/run_scripts/probe_driver_grad.py bomex                       # per-case validator

# Unit tests — whole suite in one command (exit 0 iff all green; oracle-less tests SKIP cleanly):
python clubb_jax/run_scripts/run_all_tests.py            # ~165 files (bugsrad/standalone are slow)
python clubb_jax/run_scripts/run_all_tests.py -k solver  # only files matching "solver"
python clubb_jax/tests/test_solver.py                    # …or a single file directly

# Mirror-name audit (pure-Python, no JAX/oracle): JAX↔Fortran name/file/dir diff
python clubb_jax/run_scripts/mirror_audit.py
```

**Unit-test convention.** Any f2py-oracle import in a test MUST be guarded (try/except ImportError → SKIP) so the
JAX-only assertions still run when `clubb_f2py` is unbuilt — the suite is then a clean gate in any environment.

**Output-directory convention.** JAX and Fortran (oracle) stats live in SEPARATE trees so a JAX run can never
clobber an oracle: `run_scm.py <case> -jax` → `clubb_jax/output/<case>_stats.nc`; `-legacy` (Fortran) →
`clubb_release/output/<case>_stats.nc` (the oracle home the rate/stats tests read); `-out_dir <dir>` overrides.
`compare_runs.py` keeps each side in its own subdir (`clubb_release/output/<case>_compare_fort/`,
`clubb_jax/output/<case>_compare_jax/`); `diagnose_divergence.py` reads those. Regenerate a clobbered oracle with
`-legacy`.

**Operational note.** Long runs auto-backgrounded by the harness write logs to a `tasks/` tmpfs that intermittently
reports ENOSPC (a quota artifact) and silently truncates output. For a long `compare_*` / `run_all_tests` run,
launch it detached with output redirected to the working dir — `(python … > out.txt 2>&1 &)` — and read `out.txt`.

### Verification oracles (in order of preference)

1. **In-loop f2py shadow** — most CLUBB_core routines are bit-to-bit verifiable via `clubb_f2py`. Caveats: some
   wrappers FPE-trap/core-dump (`f2py_precip_fraction`); the `.so` is also callable directly past a stale wrapper
   via the `__doc__`-introspected signature (`run_scripts/debug/cmp_terms_f2py.py`).
2. **Case-stats oracle** (for subsystems the f2py API can't reach, e.g. microphysics): a Fortran SCM run writes both
   a rate's PDF-moment INPUTS and its OUTPUTS, so the JAX rate is verifiable in isolation. **Timing confound:** a
   rate depending on a field created mid-step (accr/evap/deposition) matches the END-of-step stored stats only to a
   few % — validate the FORMULA by the median, defer bit-faithfulness to a running case; validate on the
   well-resolved interior (discrete tolerance tests flip at boundaries).
3. **Conservation contract** (oracle-free, for flux-form transport): column-mass-weighted (Σ ρ_ds·dz·) tendency =
   net boundary flux to machine precision, immune to the timing confound. **Gotcha:** a microphysics module MUST
   `jax.config.update("jax_enable_x64", True)` at import — float32 passes relative-error tests but breaks a
   conservation (cancellation) contract at ~1e-5.
4. **Invariant** (when the Fortran can't be run): e.g. no-cloud → all-sky fluxes == clear-sky.

### Gates & diagnosis

**Regression gate.** `compare_runs.py --case X` runs Fortran and JAX independently and diffs the stats; all
PROGNOSTIC variables must PASS (rel 1e-6, abs floor 1e-12); diagnostic timing differences are expected.
`compare_cases.py` wraps it over the bit-faithful cases — the generalized gate; run after any shared/core change.
It auto-forces per-step output so the comparison reflects physics, not stats-averaging windows. **Do NOT run
multiple `compare_*` jobs in parallel** — concurrent JAX processes OOM-kill each other (looks like a spurious "JAX
run failed rc=1").

**Divergence diagnosis.** After a failing `compare_runs.py`, run `diagnose_divergence.py CASE`: per failing
prognostic it reports the onset step and classifies it — **JUMP@N** (sudden jump from machine-eps past the floor =
a discrete branch/threshold crossed, e.g. cloud/precip onset → rule out a term bug) vs **FP-growth** (gradual
accumulation). It also prints a sign tally: balanced/flipping → FP/chaos; strongly one-sided & persistent → a
systematic term/coefficient bug. General method: **budget-decompose the prognostic** (`*_bt/_ma/_ta/_cl/_forcing`
from per-step stats) to pin the seed BEFORE chasing the solve, and **check the reference magnitude** (~ the tol
floor ⇒ FP, not a bug).

**Durability & chaos horizon.** The 30-step gate masks late-activating events — run `compare_cases.py
--max-iters 100` periodically. A case with a time-gated forcing/microphysics/event (`microphys_start_time`, ice
onset, diurnal sunset) must be verified past that step. Conversely, **full-length bit-faithfulness is physically
impossible for chaotic turbulence**: two bit-identical-start runs diverge after the Lyapunov time, so a
>horizon FP-growth + sign-flipping failure is physics, not a bug. Known horizons: fire ~147, jun25_altocu ~200,
atex_long ~305. The 100-step gate sits within every case's horizon.

**Grid-type dimension.** Most bit-faithful cases are `grid_type=1` (uniform → every zt↔zm weight exactly 0.5, so
stretched-grid paths are NOT exercised). `rico` is `grid_type=2` (formula-stretched), `dycoms2_rf02_so` is
`grid_type=3` (file). A bug that vanishes when a case is switched to `grid_type=1` is a stretched-grid handling bug
— the namelist A/B swap is the key localiser.

**"Entirely in JAX."** The JAX driver references no `clubb_python` anywhere (verified by
`tests/test_standalone_jax.py`, which runs cases with `clubb_python` blocked; machine-guarded by
`test_no_dead_imports.py::test_src_has_no_fortran_runtime_import`). Verify a ported forcing via the STANDALONE
(clubb_python-blocked) test — a plain compare can PASS via a Fortran fallback (false positive). **Variants share a
`runtype`** (rf02 nd/so/do/ds all `'dycoms2_rf02'`; rf01 vs rf01_fixed_sst by `sfctype` 0/1) — key off
`state['runtype']` + the distinguishing flag, not the case-file name.

### Differentiability

Whole-driver `jax.grad` through one `advance_clubb_to_end` step is finite + finite-difference-correct for **all 19
cases** (`compare_grad.py` PASS; mpace_b is BLOCKED — unsupported `microphys_scheme='coamps'`, init-rejected, not a
failure). Achieved via a **tracer-transparent toolkit** in `src/CLUBB_core/tracer_numpy.py` (`_asarray`/`_xp`/
`_iset` route to jnp under a JAX trace and exactly numpy otherwise, so normal runs stay bit-identical;
`_safe_sqrt`/`_safe_pow` for clip-sqrt/fractional-pow), block-level tracer dispatch (`if not _is_tracer_arg([…])`
→ float path, else jnp mirror), and diagnostic-skip / detach-under-trace for post-core diagnostics (radiation,
microphysics — they feed only the next step, dead for a single-step grad).

**Conventions:**
- Hard min/max are differentiable (subgradient); only `while_loop`, `np.asarray`, in-place mutation, and numpy
  ufuncs break tracing — make them tracer-transparent.
- A vanishing-denominator quotient gives 0/0=nan whose VJP `jnp.where` masking does NOT fix (nan·0=nan) — fix AT
  the operation: `_safe_sqrt` for `sqrt(maximum(0,·))`, `_safe_pow` for `maximum(x,0)**p` (p<1), a D_v arg clamp.
  Audit every sqrt/fractional-pow on a possibly-≤0 quantity; clip-sqrt/clip-pow have an inf reverse grad AT the
  clip, so a stable case passes while a convective one fails.
- Never store a tracer in module-global state.
- Some cases show a single-level FD kink at a hard physical threshold (8e-3 `rtm` inversion) — a genuine
  non-smooth point, not a bug.

### Operational gotchas

- **jit-recompilation → unbounded compile-cache → OOM.** An eager `lax.scan` whose body closes over a concrete
  (non-tracer) array bakes that array's VALUES into the jaxpr as constants, so XLA recompiles every timestep →
  cache grows without bound → OOM on long runs. Rule: any per-timestep eager `lax.scan` (or a function containing
  one) should be `jax.jit`-wrapped at a stable entry point. Diagnose with `JAX_LOG_COMPILES=1` +
  `grep -c "Compiling jit(scan)"` (a count that grows each step is this bug). A `Killed`/EXIT=137 with no traceback
  is OOM (not a NaN) — probe `resource.getrusage().ru_maxrss`.
- **Separate open leak:** the per-step `l_sample=True` diagnostic path retains ~85 device buffers/step → a
  per-step-stats Morrison `compare_runs` OOMs ~150-250 steps. Workaround: run with `state['stats_writer']=None`, or
  sample at the case default interval. Low priority.

---

## Correctness standard (tiered: numerical accuracy + differentiability)

The original gate was **bit-faithfulness** to the Fortran oracle (rel 1e-6 / abs 1e-12 on prognostics). That was
the right scaffolding for the incremental port — it caught real bugs (the stretched-grid `weights_zm2zt`
column-swap, the stale `wm_zm`, the KK covar driver) — but it forced the JAX to reproduce the Fortran's
*imprecisions* (single-precision casts, the low-accuracy `expax`), blocked differentiability (hard min/max,
`while_loop`, numpy round-trips), and trajectory bit-agreement is *physically impossible* for chaotic turbulence
past the Lyapunov horizon. The numerical-accuracy refactor relaxed it to a **tiered standard** — a change is
correct if it passes the tiers appropriate to what it touches:

| Tier | Checks | Hardness / tool |
|---|---|---|
| **A. Invariants & conservation** | water/energy/mass conservation, positivity (`rrm,Nrm,rcm≥0`), bounded correlations (`\|corr\|≤1`), finiteness | **strict, oracle-free** — `tests/test_invariants.py`, `run_scripts/invariants.py` |
| **B. Golden-trajectory regression** | vs a stored JAX reference run per case, rel ~1e-9 | **strict-ish** — `run_scripts/golden.py`, `update_golden.py` |
| **C. Physical fidelity vs Fortran** | windowed, field-scaled rel error within the chaos horizon | **relaxed** — `compare_cases.py --tier physical` |
| **D. Climatology / statistics** | time-mean & variance profiles, BL depth, cloud fraction past the chaos horizon | **statistical** (the honest gate for chaos-limited cases) |
| **E. Differentiability** | finite-difference grad checks; whole-driver `jax.grad` | **strict** — `compare_grad.py`, `probe_driver_grad.py` |

**Tier-C field-class tolerances** (point-max `max|Δ|/(max|ref|+floor)`): means (`thlm,rtm,um,vm`) **1e-4**; fluxes
(`wpthlp,wprtp,upwp,vpwp`) **1e-3**; second moments (`wp2,wp3,rtp2,thlp2,em`) **3e-3**; microphysics (`rrm,Nrm`)
**1e-2**; diagnostics + `*_mc` tendencies **report-only** (timing-confounded). Bit-faithful cases pass Tier-C by
construction (rel ~1e-11 ≪ tol).

**Status:** **20/20** DEFAULT_CASES PASS Tier-C (19 strictly bit-faithful + mpace_a within tolerance);
`compare_grad` PASS for all runnable cases. The accuracy-lowering contrivances were removed (`parabolic_expax`
`epss=1e-4`, Morrison `real*4` casts, BUGSrad `sngl`/float32-π), so the JAX is strictly *more* accurate there.
**Preserve:** the Fortran oracle as a reference-within-tolerance (`--tier bit` stays for debugging); golden refs as
the regression net (re-baseline only via `update_golden.py`, deliberate + reviewed); and **Tier A strict** —
relaxed tolerances must never hide a conservation bug.

**Precision rule.** Prefer float64 accuracy; validate within Tier-C rather than reproducing single-precision
artifacts. (Historically the Fortran M2005 interface kept `T_in_K`/`rcm_r4` in single precision, so its `thlm_mc`
carried a ~1e-7 round-trip residual; the JAX once replicated those casts to match bit-for-bit. It no longer does —
`module_mp_graupel.py` computes `thlm_mc` in the algebraically-exact float64 form, so clear-air `thlm_mc≈0`.)
4 persistent diagnostic-only differences remain (not fixable without matching Fortran FP ordering): `rtm_spur_src`
~2e-16, `thlm_spur_src` ~2e-11, `rtp2_pd` ~7e-27, `up2_pd` ~1e-17.

---

## Critical Conventions

**Band ordering:** Both Fortran and JAX use `lhs[0=super, 1=main, 2=sub]`. No flip between diffusion output and
solver input.

**Grid weights (`weights_zm2zt`):** Shape `(ngrdcol, nzt, 2)`. `[:,k,0]` = M_ABOVE (weight for `zm[k]`),
`[:,k,1]` = M_BELOW (weight for `zm[k+1]`). Fortran 1-indexed `m_above=1, m_below=2`. (These were once stored
swapped vs Fortran — invisible on uniform grids; the one real stretched-grid bug found via the grid-type A/B swap.)

**JAX x64 mode:** `jax.config.update("jax_enable_x64", True)` called at module load in
`advance_clubb_core_module.py`. All arrays must stay float64.

**Index mapping (Fortran 1-based → Python 0-based):** Interior loop `k=2..nzm-1` in Fortran → Python `[:,1:-1]` on
zm-level arrays. `clubb_params` is shape `(ngrdcol, 102)`, 0-based — access as `clubb_params[:, iC2rt - 1]`.

**Routine names mirror the Fortran subroutine.** The `_jax` suffix is retired for single-subroutine mirrors. Two
deliberate exceptions remain: **(1)** the jit-alias dual structure — a few leaf modules (`diffusion.py`,
`mean_adv.py`, `turbulent_adv_pdf.py`) define a raw `<name>_jax` AND a `<name> = jit(<name>_jax)` alias carrying
the bare Fortran name; callers/tests import the raw `_jax` version on purpose (it accepts a plain grid object and
stays `jax.grad`-able, whereas the jitted alias rejects a non-pytree `gr`) — do NOT collapse the two. Only drop the
`_jax` suffix in modules with NO `jit()` alias. **(2)** JAX-specific helpers whose bare name is not a Fortran
subroutine because the JAX restructured inline Fortran code into a differently-decomposed function for
differentiability (`calc_pdf_*`, `apply_lhs_band*`, `*_decomp`, `solve_xp2_xpyp`, `adg1_pdf_driver_zt`, …).
**Lesson:** when relocating a routine, delete the original — a left-behind copy silently diverges and confuses the
location mirror; and before a blanket `X_jax`→`X` rename, check the module has no distinct bare-named sibling `X`
(give a reference/standalone sibling a clearly-private name).

---

## What Has Been Built

Per-file Fortran↔JAX mapping with status icons is maintained in **`TRANSLATION_STATUS.md`** (kept current by
`mirror_audit.py`). Summary: every JAX module sits at its Fortran oracle's relative path; all in-scope physics is
ported; the genuinely-unported set is no-oracle/impractical subsystems only — COAMPS microphysics, the GFDL
`aer_ccn_act_wpdf_k` 5-D lookup, the SCM aerosol-activation subsystem, `pdf_hydromet_microphys_wrapper` (0 output
for every gated case), and SILHS RNG (a different RNG can't be bit-matched).

**Subsystem status:**
- **KK microphysics (`khairoutdinov_kogan`)** — COMPLETE and wired per-step (upscaled-mean/covariance analytic
  library + hydrometeor PDF setup + `advance_one_hydrometeor` transport + `calculate_K_hm`).
- **Morrison 2-moment M2005 (`module_mp_graupel.py`)** — COMPLETE (special functions, all process rates incl. the
  full ice block, single-column step, sedimentation, the CLUBB↔M2005 interface, per-step wiring). Faithful case:
  mpace_a; FP-limited: nov11, dycoms2_rf02_morr.
- **BUGSrad correlated-k radiation + `soil_vegetation`** — COMPLETE and wired; `bugs_rad` is jitted (the eager
  dispatch leaked ~700 MB/call). gabls3 is the bit-faithful radiation case. Build is `-Dradoffline -Dnooverlap
  -DCLUBB` (no ghost layer, simple `two_rt` called twice). Pass the constants the Fortran CALLER passes
  (constants_clubb grav/Cp, not BUGSrad's physconst).

`state` is a plain Python dict passed through the call stack — all prognostic arrays (shape `(ngrdcol, nzm)` or
`(ngrdcol, nzt)`), grid object, flags, params, and the `stats_writer`. Arrays enter as NumPy, convert to JAX at
each call site (`jnp.asarray`), and write back as NumPy (`np.asarray`).

---

## Cross-case bit-faithfulness status (vs Fortran, `compare_cases.py`)

"Runs" ≠ "bit-faithful". **20 bit-faithful cases** (all PASS at 30 steps + the 100-step durability gate):
arm, bomex, wangara, gabls2, gabls3, gabls3_night, dycoms2_rf01, dycoms2_rf01_fixed_sst, dycoms2_rf02_nd,
dycoms2_rf02_so, atex, atex_long, fire, neutral, ekman, cobra, jun25_altocu, mpace_a, clex9_nov02, clex9_oct14.
9 are bit-faithful for their entire configured run; mpace_a is the first Morrison case made faithful (stays
clear/sub-saturated, so the only M2005 signal is the clear-air single-precision `thlm_mc`).

**48-case survey:** 20 run bit-faithful, 28 unsupported (each blocked by ONE unported subsystem — BUGSrad
radiation, SILHS, or COAMPS — named by `_check_unsupported_features`), 0 hard crashes.

**Characterized not-bit-faithful cases — do NOT chase as bugs (each is numerically/FP-limited, often because the
JAX is MORE accurate than the low-accuracy Fortran defaults):**
- **rico** (grid_type=2, KK): bit-faithful steps 1-4; from step 5 the near-zero rt-flux clip at the stretched dry
  top amplifies FP-level `rtp2` diffs (the dry-top rtp2 sits at the rt_tol² floor — matching ~0 to rel-1e-6 is
  impossible). Grid verified bit-exact.
- **coriolis_test**: an analytic Foucault-pendulum benchmark that zeroes nearly all closure constants; the undamped
  oscillator accumulates FP noise. Step-1 faithful, no seed.
- **nov11_altocu** (Morrison + ice): bit-faithful through step 5; step 6 is the ice-cloud-edge FP floor (the
  `ice_supersat_frac` erf at near-zero variance, then the `/0.001` Lscale ramp = a 1000× amplifier). Microphysics
  activates at step 60, past the floor, so M2005 transport is unit-tested, not full-run validated.
- **dycoms2_rf02_morr** (warm Morrison): M2005 transport verified ~bit-exact, but a near-singular `rcm_mc` residual
  at the sharp cloud-top CF3D edge plus the M2005 single-precision floor keep it off the gate.
- **dycoms2_rf02_do / _ds** (KK, drizzle): the KK rt/thl covariance cancellation-amplifies the parabolic-cylinder
  `D_v`. The SCM oracle runs `parab` at `epss=1e-4`; the JAX uses the accurate float64 `D_v`, so the bit-gap WAS the
  oracle's deliberate low accuracy. Judged under Tier-C (dynamics) / Tier-D (drizzle), not against the oracle's
  imprecision.

**Durable lessons:**
- **NEVER trust a default-vs-computed value** — a Fortran line `x=default` may be overwritten later; verify the
  actual computed quantity (an unverified `precip_frac=1` cost 12 iterations chasing a 2× K_hm bug).
- **Decouple-the-oracle before blaming a subsystem** — feed the Fortran's own field into the JAX subsystem to
  exonerate it ("steep radiation" was a red herring for jun25; the real seed was a stale per-step `wm_zm`).
- A `*_forcing` stat that disagrees in a microphysics case is often `raw_forcing + lagged *_mc`, not a forcing bug.
  A `-jax` run dying with a Fortran `error stop`/no Python traceback is the unported-case `clubb_api` fallback —
  port the case's tndcy/sfclyr to `prescribe_forcings.py`.
- **An apparent ~1e-8 mismatch is far more often a test-harness defect** (missing x64, wrong hardcoded parameter,
  hidden flag) than a real divergence — check the harness before chasing the physics.

---

## Remaining Work

**The Fortran→JAX port is complete.** Every in-scope `.F90` has a JAX mirror, the driver runs 100% in JAX, the
bit-faithful frontier is at 20 cases, and the name/file mirror is converged. The incremental shadow-comparison
workflow that built the port is retired. Most work now is **refactoring, simplification, differentiability, and
working under the numerical-accuracy standard**.

**★ Performance, GPU, and the `formatting_and_jitting`/`jit-cache-and-f2py-decoupling` work (2026-06-19 → 06-24).**
The branch carried a formatting/Fortran-comment pass + JIT-friendly derived types + a new pure-JAX `src/io/` init
path (`namelist.py`/`sounding.py`/`surface.py`/`grid_file.py` + `derived_types/converters.py`, removing `clubb_api`
usage *except for stats*) and then the performance arc below (persistent cache → whole-step jit → GPU). Verified
state and the queue:

- **Physics (faithfulness gate).** **12/20 DEFAULT_CASES bit-PASS** (arm, bomex, wangara, atex, gabls2,
  gabls3_night, fire, neutral, cobra, dycoms2_rf02_nd, dycoms2_rf01_fixed_sst, jun25_altocu). dycoms2_rf01 runs all
  30 steps fine **standalone** — its `compare_cases` "crash" (rc=1 @ iter 29) was node resource-contention from
  running `compare_cases` (501-level case) and the 165-file unit suite at once, **not** a code bug; never run two
  heavy JAX suites concurrently. **7 cases fail-loud** on features the new init path has not rewired yet:
  Morrison microphysics (mpace_a, clex9_nov02, clex9_oct14), `l_cloud_sed` (atex_long, dycoms2_rf02_so),
  `l_soil_veg` (gabls3), sponge damping (ekman). *Env note:* `clubb_release/install/latest` had gone dangling
  (its scratch target `mpace_dbg` was purged) → repoint to `install/intel_PRECdouble_PYTHON`.

- **Performance — the original finding (the dominant cost; assessed via `JAX_LOG_COMPILES`).** Runs were slow
  because `advance_clubb_core` and the timestep loop ran **eager** over only ~23 *leaf* jits, so each step fired
  hundreds of standalone primitive pjit dispatches with a numpy↔jax round-trip (a device sync) between each, and
  the per-leaf trace/MLIR/XLA cost ran the first step into minutes. **Fix direction (now applied):** persistent
  JIT cache → whole-step jit → GPU.

  **Step 1 (2026-06-19) — persistent JIT cache.** `advance_clubb_core_module.py` enables JAX's on-disk
  compilation cache at import (default `~/.cache/clubb_jax_jit`; opt out `CLUBB_JAX_NO_JIT_CACHE=1`, relocate via
  `JAX_COMPILATION_CACHE_DIR`) → first-step compile becomes a cross-process cache hit (4-col ARM/30-step: cold
  133–147 s → warm 70–78 s, byte-identical so still bit-faithful). Halved only the XLA-backend half; the residual
  was the per-leaf tracing/lowering, removed by step 2.

  **★ Step 2 (2026-06-24) — whole-step JIT (the open lever, now closed).** `advance_clubb_core` is wrapped in a
  single `jax.jit` (`advance_clubb_core_jit`; the driver uses it on every **non-sampled** step, opt out
  `CLUBB_JAX_NO_WHOLE_STEP_JIT=1`). It is gated to `l_sample=False` because fusing the hundreds of per-step
  `stats.update()` writes into the program too balloons the single XLA compile to minutes; sampled steps are
  diagnostic and rare, so they stay eager. `l_sample` gates only diagnostic stats writes, never the prognostic
  physics, so the stats-off jit computes identical state to the proven-bit-faithful stats-on path (and to Fortran).
  The
  whole driver step was *already* reverse-mode `jax.grad`-traceable (the differentiability gate), so the routine
  traces cleanly; the static args are the shape/branch scalars (`nzm/nzt/ngrdcol`, the `*_dim`, `l_implemented`,
  and `clubb_config_flags` — a plain unregistered NamedTuple whose bool/int fields JAX would otherwise trace into
  tracers and break every `if`). Everything else is a traced array or a registered pytree (`gr`, the `JaxStats`
  bridge — `l_sample` lives in its aux_data so the stats branches specialize per compile, `pdf_*`, `err_info`,
  `nu_vert_res_dep`; `sclr_idx` flattens entirely to aux_data so its int indices stay static). XLA now fuses all
  ~23 leaves into one compiled program → **one dispatch/step instead of hundreds.** Numerically transparent:
  ARM stays **bit-faithful** (`compare_runs arm` Result[bit] PASS, 0 prognostic failures, Tier-C PASS) and
  **grad-transparent** (the grad probe is identical eager-vs-jit; the only blocker, reverse-mode through
  `fill_holes_sliding_window`'s dynamic `fori_loop`, pre-exists and is unaffected). Per-step speedup at matched
  stats: **GPU 2727 → 408 ms/step (6.7×)** at ngrdcol=1.

  **★ Step 3 (2026-06-24) — GPU enablement + scaling.** Installed CUDA-12 jaxlib into the jaxenv
  (`jax-cuda12-plugin`/`pjrt` 0.10.2; the cluster's system `cuda12.8` on `LD_LIBRARY_PATH` shadows the bundled
  cuSPARSE → strip `cuda*` from `LD_LIBRARY_PATH`, codified in `run_scripts/jaxenv.sh`). CLUBB is a single-column
  model, so the GPU's data-parallel axis is the column count `ngrdcol` (`run_scm.py -multicol N`). Benchmark:
  `run_scripts/benchmark_backends.py` (env-gated `CLUBB_JAX_BENCH=1` per-step timing in `advance_clubb_to_end`,
  separating step-1 compile from steady state; Fortran timing from its own `CLUBB-TIMER time_total`).

  **Scaling — ARM, steady ms/step, stats off (the production/inference path), whole-step jit, 1× V100S vs intel
  ifx Fortran on the same node:**

  | ngrdcol | JAX-GPU | JAX-CPU | Fortran |
  |--------:|--------:|--------:|--------:|
  |       1 |    66.5 |     8.2 |     0.6 |
  |      64 |    75.7 |    41.5 |    16.8 |
  |     256 |    82.9 |    86.5 |    95.7 |
  |    1024 |    87.3 |   355.2 |   531.4 |

  (Numbers refreshed 2026-06-25 after the thvm dead-code removal in Step 5, which cut the redundant per-step
  compute that scaled with the grid — JAX-GPU dropped ~15–20% at ngrdcol≥256, JAX-CPU ~25–35% at ngrdcol 64–256.)
  **JAX-GPU per-step is nearly flat** (66→87 ms over a 1024× workload increase — launch-latency/fixed-overhead
  bound), so its **throughput scales almost linearly with the column count**, while Fortran (serial column loop)
  and JAX-CPU grow roughly linearly. Crossovers: **JAX-GPU overtakes JAX-CPU at ngrdcol≈256 and Fortran at
  ngrdcol≈256**; at 1024 columns **JAX-GPU is 6.1× faster than Fortran and 4.1× faster than JAX-CPU**
  (11 700 vs 1 930 vs 2 880 columns/s). The first-step compile (~5–35 s on GPU, in the table's `wall`−`steady·n`)
  is a one-time cost amortized by the persistent cache across processes and by any multi-step run; it does NOT
  recur per step. Takeaway: for single-column / few-column short runs Fortran wins on absolute latency; for
  ensemble/batch workloads (many columns — exactly the ML/autodiff use case) **JAX-GPU is the fastest backend and
  the only differentiable one.** Remaining levers (see Step 5 profiling): the GPU core's kernel-launch floor
  (parallel vertical solver) and the per-step eager glue / `lax.scan` over timesteps.

  **Step 4 (2026-06-25) — single/double precision toggle.** Where Fortran picks precision at compile time
  (`-precision single|double`), JAX picks it at process start via the global `jax_enable_x64` flag (must be set
  before the first op). `src/CLUBB_core/clubb_precision.py::configure_jax_precision()` centralizes that decision, gated on
  `CLUBB_JAX_PRECISION` (default `double`); all 54 modules call it instead of hard-coding `jax_enable_x64=True`,
  so the precision is consistent process-wide. `double` is byte-identical to before (arm `compare_runs`
  Result[bit] PASS post-refactor). `single` (float32) runs and stays finite; vs double after 10 arm steps it
  diverges at float32 level — ~1e-7 on means (`thlm`,`rtm`), ~1e-5–1e-4 on second moments/fluxes (`wp2`,`wprtp`) —
  so it is **not** bit-faithful (expected) and is for performance/memory exploration, not the gate.

  **Finding — float32 is a memory win, NOT a speed win for CLUBB (V100S, ARM):**

  | ngrdcol | f64 ms | f32 ms | f64 peak MiB | f32 peak MiB |
  |--------:|-------:|-------:|-------------:|-------------:|
  |       1 |     70 |     92 |          1.5 |          0.5 |
  |      64 |     95 |    120 |           32 |           16 |
  |     256 |     94 |    106 |          126 |           64 |
  |    1024 |     90 |    103 |          503 |          273 |

  float32 is ~10–30 % **slower** across the sweep but uses ~½ the device memory. The V100S has strong fp64 (1:2
  of fp32 peak), and CLUBB's step is dominated by many small ops (tridiagonal solves, elementwise, scans) — it is
  launch/overhead-bound, not FLOP-bound, so fp32's throughput edge never engages and the extra type conversions
  cost a little. The robust f32 benefit is **capacity**: ~2× more columns per GPU. Double remains the default.

  **★ Step 5 (2026-06-25) — per-step phase profiling (where the time actually goes).** Added opt-in phase timing
  (`CLUBB_JAX_BENCH_PHASES=1`; `block_until_ready` at each boundary) splitting the step into pre-core glue
  (forcings) / jitted core / post-core (radiation+micro+stats). ARM, steady ms/step, **post-thvm-cleanup** (below).
  *Caveat:* the boundary syncs serialize work that the plain async benchmark overlaps, so these **totals run higher
  than the Step-3 plain-bench table** (CPU especially) — read this table for the *proportional split*, not absolutes:

  | backend | ngrdcol | total | pre-glue | core | post |
  |---------|--------:|------:|---------:|-----:|-----:|
  | GPU     |       1 |    65 |       11 |   52 |  ~0  |
  | GPU     |     256 |    79 |       15 |   60 |  ~0  |
  | GPU     |    1024 |    81 |       19 |   58 |  ~0  |
  | CPU     |       1 |    10 |        5 |    3 |  ~0  |
  | CPU     |     256 |    87 |       15 |   71 |  ~0  |

  **The jitted core dominates the GPU step (~55 ms) and is nearly flat in ngrdcol** — so it is **kernel-launch
  bound, not FLOP-bound**. The clincher: the *same* core runs in **3 ms on CPU at ngrdcol=1** vs ~52 ms on GPU.
  post-core is negligible (ARM runs no active radiation/micro stats-off); the pre-core **glue** (forcings) is
  host-bound (`prescribe_forcings`'s device↔host round-trips + per-step `zt2zm`/surface-scheme leaf dispatches).

  **What the floor actually is (XLA HLO dump + an nz sweep, 2026-06-25).** The compiled core
  (`jit_advance_clubb_core`, optimized HLO) emits **1332 fusion kernels + 296 while-loops** (the `lax.scan` LU
  sweeps) ≈ **~1600 serially-dependent GPU kernels**, ~30 µs of launch/schedule each → ~50 ms. Critically the core
  is **flat not only in ngrdcol but also in nz** (arm at nzmax 64/128/256/512: core 53/52/49/50 ms) — i.e. it is
  bound by a *fixed kernel COUNT*, independent of problem size, **not** by serial solver *depth*. **This refutes
  the parallel-vertical-solver lever:** the 296 while-loops are a minority and are flat in nz; the cost is the 1332
  elementwise/reduction fusions XLA could not merge (fusion barriers at the reshapes/transposes/scans/gathers
  between the physics steps). Cyclic reduction would not move the dominant term and would cost bit-faithfulness.

  **Realistic frontier:** the GPU small-batch floor is the *fusion-kernel count*, which is inherent to the
  algorithm's many distinct sequential operations — only a major restructuring (fewer, larger fusable ops) would
  cut it, so small-batch GPU will not beat Fortran's single-thread latency. That is **not the GPU's use case**:
  at large ngrdcol the fixed ~50 ms amortizes over all columns and **GPU already wins (6.1× vs Fortran at 1024)**
  — the differentiable, batch/ensemble/ML workload. The remaining *general* lever is **`lax.scan` over timesteps**
  (task #5): it removes the per-step Python/dispatch/host-sync glue (~11–19 ms/step) for long runs and enables
  memory-efficient multi-step `jax.grad`, though the ~50 ms core launches still recur per step. **Blocker (the
  reason it is multi-iteration, not one-shot):** the case forcings reset arrays via numpy *in-place* mutation
  (`state['rtm_forcing'][:] = 0.0`) and the surface scheme (`arm_sfclyr`) is state-dependent. These work today
  only because the arrays are concrete (under one-step `jax.grad` only the perturbed input's dependents are
  tracers); under a full `lax.scan` trace **every carry array is a tracer**, so the in-place resets fail. `lax.scan`
  therefore first requires rewriting the forcings as pure (no in-place mutation) functions and assembling the
  evolving state into a scan carry — a sizeable, must-stay-bit-faithful refactor.

  **Cleanup (dead code).** The driver computed `state['thvm']` every step, but it is read nowhere — the core
  recomputes `thvm` internally from the same inputs and never received the driver's copy (a port vestige; the
  Fortran passes thvm into the core). Removed the per-step `_calculate_thvm` call, the helper, and its import;
  ARM stays bit-faithful (`compare_runs` Result[bit] PASS).

- **Unit-test status (165 files; 118 passing at branch start).** A signature-drift pass updated **35** stale
  tests to the refactor's new JIT-friendly signatures (leading `nzm/nzt/ngrdcol/gr`, reordered args,
  `static_argnums` → arrays in static slots raised `unhashable type`). Two surfaced small **src** fixes (applied):
  `set_sfc_value_of_flux_profiles` now zeros `wpedsclrp` **unconditionally** (matches the Fortran whole-array
  assignment; the refactor had guarded it under `edsclr_dim>0`), and the `fill_holes_sliding_window` grad test
  uses **forward-mode** `jacfwd` (reverse-mode through its dynamic-bound `fori_loop` is unsupported — bounds are
  traced, not static, in the live `fill_holes_vertical` dispatcher). *Caveat:* the 35 are verified individually;
  the aggregate clean-suite re-run was aborted, not yet confirmed green together.

- **12 unit tests still failing — the next-step queue** (none are signature drift):
  - *Structural-audit guards (6) — update allowlists/guards for the legit structural changes:*
    `test_no_dead_imports` + `test_standalone_jax` (src now references `clubb_python` via stats → the "100% JAX"
    guards trip — decide: drop the stats `clubb_python` use, or carve a documented exception), `test_mirror_audit`
    (new `io/` + `converters.py`/`err_info_codes.py` files don't mirror a `.F90` → add to `_JAX_ONLY_FILES`/
    `_RENAMES`), `test_routine_placement` (a routine moved off its mirror file), `test_no_dead_functions`
    (uncalled public fns), `test_param_names` (param-index constants drifted — verify vs the f2py oracle).
  - *Genuine numerical / grad / oracle (6) — need real investigation:* `test_differentiability` (reverse-mode
    through a dynamic-bound `while_loop`/`fori_loop`), `test_kk_rico_oracle` (grad NaN; KK oracle), `test_morrison_rates`
    (`ImportError: morrison_hm_metadata` — tied to the unwired Morrison subsystem), `test_gabls3_night_stability`
    (fm1/fh1 mismatch vs F90, 2.6e-7), `test_transform_pdf_chi_eta_component` (Monte-Carlo variance assertion),
    `test_saturation` (`saturation_formula=2` should-raise decision).

The genuinely-remaining unported `.F90` are all impractical/out-of-scope (no oracle or zero validated payoff):
- **COAMPS microphysics** (`coamps_microphys_driver`) — 7000-line alternative scheme the gated config never uses;
  the Fortran itself fatal-errors on `l_predict_Nc=F` → no oracle.
- **GFDL `aer_ccn_act_wpdf_k`** — the 5-D single-precision aerosol-activation lookup (the CLUBB-side orchestration
  IS ported); part of the `SCM_Activation` subsystem, no case exercises it.
- **`pdf_hydromet_microphys_wrapper`** — would compute `wp2hmp`/`rtphmp`/`thlphmp`, but those are correctly zero for
  all 20 gated cases → zero validated payoff; deferred.
- **`pdf_closure_driver_zm`** — the deferred routine (gated by `l_call_pdf_closure_twice`, no case, no f2py oracle).
  A faithful port needs zm-grid VARIANTS of every closure helper (the JAX decomposed the monolithic Fortran
  `pdf_closure` into zt-specialized pieces) AND has no way to validate them → would be unvalidatable dead code.
- **SILHS** interactive Latin-hypercube sampling — RNG-based, not bit-reproducible vs the Fortran RNG; not a target.

Do NOT chase the genuinely numerically-limited cases (rico, dycoms2_rf02_do/ds) as bugs — they are characterized.
Before declaring a case blocked by an unported subsystem, check whether that subsystem actually *runs* in the gate
window (clex9_nov02/oct14 were "blocked" only by a diagnostic mismatch — their Morrison never activates in 30 steps
— and were bit-faithful all along).

---

## Agent Working Rules

1. **Read `DESIGN.md` in full** at the start of every session. At the end, append one concise entry to
   `CHANGELOG.md`; do not read the full changelog history (it is the append-only work record). Periodically condense
   newer CHANGELOG entries.
2. **Keep the module-naming mirror.** Every `src/CLUBB_core/<name>.py` mirrors
   `clubb_release/src/CLUBB_core/<name>.F90` at the identical relative path; the Fortran stays the algorithm
   reference (now a reference within tolerance, not a per-timestep oracle). Export new public symbols from the
   relevant package `__init__.py`. `mirror_audit.py` enforces the name/file/dir mirror — run it after structural
   changes.
3. **Judge correctness by the tiered standard, not bit-faithfulness:** conservation/invariants (Tier A, strict),
   regression vs the golden JAX trajectory (Tier B), physical fidelity vs Fortran within field-scaled tolerance
   (Tier C), and a `jax.grad`/finite-difference check for any core-physics change (Tier E). `compare_cases.py`
   (Tier C) + `compare_grad.py` (Tier E) are the gates; a NEW "failure" that is a known FP/oracle-precision artifact
   is **characterized, not chased**.
4. **Run the gate after any shared/core change:** `compare_cases.py --max-iters 30` (expect 0 prognostic failures),
   plus a periodic `--max-iters 100` durability pass. Re-baseline golden references only as a deliberate, reviewed
   step.
5. **Prefer the simpler / more-accurate / differentiable form.** When a faithfulness contrivance and a cleaner form
   differ only at the ULP level (smooth vs hard min/max, accurate vs oracle-truncated `D_v`, float64 vs replicated
   `real*4`), take the cleaner one and re-validate under the tiered standard.
6. **Porting a genuinely new subsystem** (COAMPS or SILHS — the only unported pieces): read the Fortran oracle,
   mirror its path under `src/`, and validate with a case-stats oracle (feed the Fortran's own state into the JAX
   routine) or a conservation contract, since the f2py API exposes no microphysics. See "Verification oracles".
