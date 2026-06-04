# CLUBB-JAX Design

Goal: translate the CLUBB turbulence parameterization from Fortran to JAX for differentiable,
composable use in ML and autodiff workflows.

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
│   │   ├── io/             ← stats_writer.py (pure Python NetCDF output)
│   │   ├── clubb_standalone.py  ← thin CLI frontend ↔ clubb_standalone.F90 (entry: python -m clubb_jax.src.clubb_standalone)
│   │   ├── clubb_driver.py      ← run_clubb / init_clubb_case / clean_up_clubb ↔ clubb_driver.F90
│   │   └── advance_clubb_to_end.py  ← timestep loop (advance_clubb_to_end subroutine of clubb_driver.F90)
│   ├── run_scripts/        ← test infrastructure
│   │   ├── compare_runs.py ← Fortran vs JAX comparison
│   │   └── run_scm.py      ← single-case runner
│   └── tests/              ← unit tests
└── clubb_release/          ← git submodule: larson-group/clubb_release (master)
    ├── src/                ← Fortran source oracle
    ├── input/              ← case setups, namelists, sounding files
    ├── clubb_python_api/   ← f2py compiled wrappers (not in submodule git)
    ├── bin/ install/       ← compiled Fortran binaries (not in submodule git)
    └── run_scripts/        ← upstream Fortran test scripts (not modified)
```

`clubb_jax/` and `clubb_release/` are siblings. The JAX package works against upstream
`clubb_release` master — it has no dependency on any custom branch.

---

## How to Test

**Prerequisites:** `clubb_release/bin/clubb_standalone` and `clubb_release/clubb_python_api/*.so`
must be present (compiled artifacts, not in git — copy from a build or compile from source).
The regression tests require `clubb_python_api` to be present (for the Fortran comparison run
and for non-ARM `prescribe_forcings`). The JAX driver itself has zero module-level Fortran imports
since Iter 73 — ARM runs do not need `clubb_python_api` to be importable at all.

```bash
# From CLUBB-JAX/ or any directory:

# Quick smoke test — JAX driver, no Fortran comparison run (~20s):
# (writes clubb_jax/output/arm_stats.nc — NOT the clubb_release/ oracle; see "Output-directory convention")
python clubb_jax/run_scripts/run_scm.py arm -jax -max_iters 3

# Full Fortran-vs-JAX regression for one case (30 timesteps):
python clubb_jax/run_scripts/compare_runs.py --case arm --max-iters 30

# Multi-case regression dashboard (generalises the test beyond ARM):
python clubb_jax/run_scripts/compare_cases.py --max-iters 30
python clubb_jax/run_scripts/compare_cases.py --cases arm,bomex,wangara --max-iters 15

# Whole unit-test suite in ONE command (Iter305) — runs every clubb_jax/tests/test_*.py, reports pass/fail
# per file + a summary; exit 0 iff all green. Tests vs an unavailable oracle SKIP cleanly (so it is portable).
python clubb_jax/run_scripts/run_all_tests.py            # all 21 files (~12 min: bugsrad+standalone are slow)
python clubb_jax/run_scripts/run_all_tests.py -k solver  # only files matching "solver"

# Divergence-onset classifier (Iter311) — after a compare_runs.py run, classify WHEN/HOW each failing
# prognostic diverges: bit-faithful / FP-growth (gradual, undamped-oscillator-style) / JUMP@N (discrete
# onset → rule out a term/threshold bug). Generalises the per-step decomposition done by hand on every case.
python clubb_jax/run_scripts/diagnose_divergence.py rico            # only failing prognostics
python clubb_jax/run_scripts/diagnose_divergence.py rico --var thlp2  # one variable, pass or fail

# Unit tests (pure JAX, no Fortran needed):
python clubb_jax/tests/test_solver.py
python clubb_jax/tests/test_diffusion.py
python clubb_jax/tests/test_penta_solver.py
python clubb_jax/tests/test_penta_faithful.py      # penta_lu solve == Fortran-order numpy replica (0 ULP eager)
python clubb_jax/tests/test_calc_roots.py          # calc_roots cubic/quadratic/cube_root vs polynomial residual + numpy.roots (completeness port)
python clubb_jax/tests/test_pos_definite.py        # pos_definite_adj (Smolarkiewicz limiter) BIT-EXACT vs f2py oracle + conservation (completeness port)
python clubb_jax/tests/test_diagnose_correlations.py # diagnose_correlations (Larson 2011) + PDF helpers — bit-match vs f2py oracle (completeness port)
python clubb_jax/tests/test_kk_local_means.py      # KK grid-mean (local) evap/auto/accr/mvr rates vs independent NumPy + branches (completeness port)
python clubb_jax/tests/test_kk_upscaled_variances.py # variance_KK_mvr vs independent closed-form lognormal moments (rel 0) + 4M-sample Monte-Carlo (rel 1.4e-4) (completeness port)
python clubb_jax/tests/test_ice_dfsn.py            # ice_dfsn vs literal NumPy loop (rel 1.2e-16) + thlm2T_in_K bit-exact vs f2py + cap/branch/grad (completeness port)
python clubb_jax/tests/test_hydromet_pdf_parameter.py # hydromet-PDF parameter containers: zero-init shapes/dims/round-trip (CLUBB_core now fully ported)
python clubb_jax/tests/test_mixed_moment_pdf_integrals.py # mixed_moment_PDF_integrals integrals/covariances vs binomial/tilting closed-forms (<1e-12) + Monte-Carlo (full port)
python clubb_jax/tests/test_hydrometeor_mixed_moments.py # hydrometeor_mixed_moments top driver vs literal Fortran-loop transcription (<1e-12) + grad
python clubb_jax/tests/test_pdf_integrals_all_mm.py # KK all-mixed-moment Dv integrals (trivar+quadrivar families, 8/8) vs analytic base cases + complex-branch MC
python clubb_jax/tests/test_cloud_correlate.py     # BUGSrad cloud-overlap (bugs_ctot + bugs_cloudfit) vs literal Fortran loops (rel 3e-16/1e-14) + invariants
python clubb_jax/tests/test_gfdl_activation.py     # GFDL erff (vs math.erf <1e-6) + updraft_weights (vs literal incl. Fortran quirk) (partial port)
python clubb_jax/tests/test_simple_rad_lba.py      # LBA prescribed radiation (table load + time/vertical interp) vs literal Fortran on real lba_rad.dat
python clubb_jax/tests/test_cloud_feedback_sfclyr.py # CGILS/cloud_feedback drag-law surface fluxes vs literal Fortran (rel 0) + physical invariants
python clubb_jax/tests/test_pressure_coord_forcing.py # Press[Pa]-coordinate time-dependent forcing (interp vs p_in_Pa) + height path byte-identical
python clubb_jax/tests/test_inverse_hydrostatic.py # inverse_hydrostatic (pressure-sounding altitudes) round-trip z→exner→z exact (5.5e-12 m) + literal + analytic
python clubb_jax/tests/test_lba_sfclyr.py          # LBA diurnal surface fluxes + MOST ustar vs literal Fortran + diurnal structure
python clubb_jax/tests/test_mpace_b_lba_tndcy.py   # M-PACE B large-scale subsidence/cooling forcing vs literal + invariants; LBA zero forcing
python clubb_jax/tests/test_silhs_surface_schemes.py # mpace_b/arm_97/twp_ice surface schemes vs literal Fortran (twp_ice == cloud_feedback drag law)
python clubb_jax/tests/test_f2py_advance_xm_wpxp.py # f2py advance_xm_wpxp .so directly callable (oracle unblocked; needs clubb_python_api)
python clubb_jax/tests/test_differentiability.py   # jax.grad through the building blocks (+ mixing-length reverse, REFACTOR B3)
python clubb_jax/tests/test_full_timestep_grad.py  # ★ REFACTOR B4: full-timestep jax.grad through advance_clubb_core (FD-correct)
python clubb_jax/tests/probe_driver_grad.py <case>  # ★ REFACTOR B5: WHOLE-driver jax.grad through advance_clubb_to_end (per case; FD-correct)
python clubb_jax/run_scripts/compare_grad.py        # ★ REFACTOR B5 GATE: whole-driver differentiability dashboard, all cases (grad analogue of compare_cases)
python clubb_jax/tests/test_mono_flux_limiter.py   # REFACTOR B2: JAX lax.scan flux limiter == NumPy (bit-exact) + grad
python clubb_jax/tests/test_invariants.py          # REFACTOR Tier-A: oracle-free conservation/positivity/Cauchy-Schwarz
python clubb_jax/tests/test_pdf_utilities.py       # lognormal<->normal moments — BIT-TO-BIT vs f2py
python clubb_jax/tests/test_kk_autoconversion.py   # KK rate functions vs quadrature/scipy
python clubb_jax/tests/test_kk_rico_oracle.py      # KK autoconv END-TO-END vs Fortran rico rrm_auto
python clubb_jax/tests/test_fill_holes_mean.py     # rtm_cl/thlm_cl mean-field fill (Iter186 fix guard)
python clubb_jax/tests/test_morrison_special.py    # Morrison POLYSVP/DERF1/GAMMA vs scipy (Iter190-192)
python clubb_jax/tests/test_morrison_rates.py      # Morrison warm-rain + ice rates vs nov11_altocu oracle (Iter193-201)
python clubb_jax/tests/test_morrison_differentiable.py  # Morrison rate library is jax.grad-able (Iter203)
python clubb_jax/tests/test_bugsrad.py             # 17 BUGSrad tests: RT machinery (vs Fortran replicas ≤2e-13) + bugs_rad + bugsrad_driver + advance_radiation dispatch (invariants) (Iter255-269)
python clubb_jax/tests/test_soil_vegetation.py     # soil_vegetation force-restore surface BC (gabls3 l_soil_veg): BIT-EXACT vs Fortran replica (Iter270)
python clubb_jax/tests/test_standalone_jax.py      # "entirely in JAX": ARM runs with clubb_python BLOCKED (find_spec import-blocker) (Iter280)
```

**Unit-test sweep convention.** Run each `python clubb_jax/tests/<t>.py` (or `run_all_tests.py`). Any
f2py-oracle import in a test MUST be guarded (try/except ImportError -> SKIP+return) so the JAX-only
assertions still run when `clubb_f2py` is unbuilt -- the suite is then a clean gate in any environment.

**★ Output-directory convention (Iter218; revised).** JAX-produced stats and Fortran-produced (oracle)
stats live in SEPARATE trees, so a JAX run can never clobber an oracle:
- **`run_scm.py <case> -jax`** defaults to **`clubb_jax/output/<case>_stats.nc`**.
- **`run_scm.py <case> -legacy`** (and `-exe`/default Fortran) defaults to **`clubb_release/output/<case>_stats.nc`** -- the oracle home, where the rate/stats tests read it.
- `-out_dir <dir>` overrides either default.
- `compare_runs.py` keeps each side in its own subdir -- Fortran `clubb_release/output/<case>_compare_fort/`, JAX `clubb_jax/output/<case>_compare_jax/`; `diagnose_divergence.py` reads those same paths.
Oracle-generation commands target `clubb_release/output/...` explicitly via `-legacy -out_dir` (e.g.
`rico_fort` = `run_scm.py rico -legacy -max_iters 10 -out_dir clubb_release/output/rico_fort`; `rico_long_fort`
= the same at `-max_iters 250` for developed-rain tests). Regenerate a clobbered oracle with `-legacy`.

**Verification oracles, in order of preference:**
1. **In-loop f2py shadow** -- most CLUBB_core routines are bit-to-bit verifiable via `clubb_f2py`. Caveats: some
   wrappers FPE-trap/core-dump (`f2py_precip_fraction`); the `.so` is also callable DIRECTLY past a stale wrapper
   using the `__doc__`-introspected signature (`run_scripts/debug/cmp_terms_f2py.py`, `debug/compare_xm_wpxp_f2py.py`) -- an
   input-matched per-term comparison the namelist A/B can't do.
2. **Case-stats oracle** (for unported subsystems the f2py API can't reach, e.g. microphysics): a Fortran SCM run
   writes both a rate's PDF-moment INPUTS and its rate OUTPUTS, so the JAX rate is verifiable in isolation by
   feeding the Fortran's own moments in (`test_kk_rico_oracle.py`, `test_morrison_rates.py`). **Timing confound:** a
   routine called mid-step sees inputs differing from the END-of-step stored stats, so rates depending on a field
   created during the step (accr/evap/deposition) match only to a few % -- validate the FORMULA by the median,
   defer bit-faithfulness to a running case; discrete tolerance tests can flip at boundaries, so validate on the
   well-resolved interior.
3. **Conservation contract** (oracle-free, for flux-form transport operators): the column-mass-weighted
   (sum rho_ds*dz*) tendency must equal the net boundary flux (machine precision), immune to the timing confound.
   **Gotcha:** a new microphysics module MUST `jax.config.update("jax_enable_x64", True)` at import -- float32
   silently passes relative-error tests but breaks a conservation (cancellation) contract at ~1e-5.
4. **Invariant** (when the Fortran can't be run for a bit-oracle): e.g. no-cloud -> all-sky fluxes == clear-sky.

**★ Regression gate.** `compare_runs.py --case X` runs Fortran and JAX independently and diffs the stats; all
PROGNOSTIC variables must PASS (rel 1e-6, abs floor 1e-12); diagnostic timing differences are expected.
`compare_cases.py` wraps it over all bit-faithful cases (one pass/fail line each) -- the generalized gate; run it
after any shared/core change. It auto-forces per-step output (stats_tsamp=stats_tout=dt_main) so the comparison
reflects physics, not stats-averaging windows. **Resource note: do NOT run multiple `compare_*` jobs in parallel**
-- concurrent JAX processes OOM-kill each other (looks like a spurious "JAX run failed rc=1"); run sequentially.

**★ Divergence diagnosis.** After a failing `compare_runs.py`, run `diagnose_divergence.py CASE`: per failing
prognostic it reports the onset step and classifies it -- **JUMP@N** (a sudden jump from machine-eps past the floor
= a discrete branch/threshold crossed, e.g. cloud/precip onset -> rule out a term bug) vs **FP-growth** (gradual
accumulation). It also prints a **sign tally** at the gate-cross: balanced/flipping -> FP/chaos; strongly
one-sided & persistent -> a systematic term/coefficient bug. General method that cracks most seeds:
**budget-decompose the prognostic** (`*_bt/_ma/_ta/_cl/_forcing/...` from the per-step stats) to pin the seed to
one term BEFORE chasing the solve, and **check the reference magnitude** (~ the tol floor => FP, not a bug).

**★ Durability & chaos horizon.** The 30-step gate MASKS late-activating events -- run `compare_cases.py
--max-iters 100` periodically. A case whose forcing/microphysics/event activates at a known time (`time >= ...`
gates, `microphys_start_time`, ice onset, diurnal sunset/sunrise where `amu0` crosses 0.01) must be verified PAST
that step. Conversely, **full-length bit-faithfulness is PHYSICALLY IMPOSSIBLE for chaotic turbulence**: two
bit-identical-start runs diverge after the Lyapunov time, so a >horizon FP-growth + sign-flipping failure is
physics, not a bug. Known chaos horizons: fire ~147, jun25_altocu ~200, atex_long ~305. The 100-step gate sits
within every case's horizon, so it is the right practical durability metric.

**★ Grid-type dimension.** Most bit-faithful cases are `grid_type=1` (uniform -> every zt<->zm weight exactly
0.5, so stretched-grid paths are NOT exercised). `rico` is `grid_type=2` (formula-stretched), `dycoms2_rf02_so`
is `grid_type=3` (file). A bug that vanishes when a case is switched to `grid_type=1` is a stretched-grid handling
bug -- the namelist A/B swap is the key localiser. (The one real stretched-grid bug found this way: the
`derived_types` `weights_zm2zt` columns were stored swapped vs Fortran, invisible on uniform grids; Iter151.)

**★ Differentiability / composability status.** Component-level differentiability + composability -- the
practical goal -- is DONE and tested (`tests/test_differentiability.py`): saturation, tridiag + penta solvers,
fill_holes, PDF cloud_frac, Brunt-Vaisala, the ADG1 w/full PDF closure, the KK rate drivers (auto/accr/evap,
full-array, edge-robust to rel ~1e-10), mixing-length forward-mode; the Iter290-291 core jits preserve grad.
Radiation too (`bugs_rad`, `soil_vegetation` are `jax.grad`-able). **Hardening convention:** a vanishing-denominator
quotient gives 0/0=nan whose VJP `jnp.where` masking does NOT fix (nan*0=nan) -- fix AT the operation (custom_jvp
safe-division, double-where safe_sqrt/_pos_pow, a D_v arg clamp), all forward-preserving. **★★ End-to-end
`jax.grad` through the core CLUBB timestep IS NOW AVAILABLE (REFACTOR B4, iter16-21).** `advance_clubb_core`
(the full closure + all prognostic solves + mixing length + flux limiter) is reverse-mode differentiable —
`jax.grad` w.r.t. the mean profile is finite and finite-difference-correct (rel 4.0e-10,
`tests/test_full_timestep_grad.py`). Achieved by **tracer-transparent numpy** (a drop-in shim that is jnp
under a JAX trace and *exactly* numpy otherwise, so the bit-faithful suite is unaffected): `_asarray`,
the `_xp` ufunc/`_like` shim, `_iset` (immutable-safe assignment), removal of dead shadow-comparison
scaffolding, and guarding the `_prev_adg1_j25` module-global under trace. The B2 flux limiter (iter11) and B3
mixing length (iter9) feed into this. **Convention (R6): hard min/max are differentiable (subgradient); only
`while_loop`/`np.asarray`/in-place-mutation/numpy-ufuncs break tracing — make them tracer-transparent, harden
`sqrt(maximum(0,·))` with `_safe_sqrt` and `maximum(x,0)**p` (p<1) with `_safe_pow` (clip-sqrt/clip-pow have
an inf reverse grad AT the clip — they nan only where the quantity actually reaches ≤0, so a stable case
passes while a convective one fails; audit every sqrt/fractional-pow on a possibly-≤0 quantity), and never
store a tracer in module-global state.** Grad uses the
standard differentiable-forward config (`debug_level=0`, `l_sample=False` — diagnostics/stats off). The shim
lives in `src/CLUBB_core/tracer_numpy.py` (`_asarray`/`_xp`/`_iset`/`_safe_sqrt`/`_is_tracer_arg`), shared by
the core and the driver. **★★ WHOLE-DRIVER `jax.grad` is now AVAILABLE for the arm case (REFACTOR B5,
iter25):** `jax.grad` through one full `advance_clubb_to_end` step (thvm + arm surface forcings + the core,
stats off) is finite + FD-correct (`tests/probe_driver_grad.py`; `d(½∑um²)/dum` rel 1.3e-8 exercises the
differentiable surface momentum-flux path). Two B5 patterns, both **bit-identical for concrete runs** (validated
arm Tier-B + Tier-C): **(R7) block-level tracer dispatch** — guard a small branchy block (Python `float()`/
`math`/`max`/fixed-point loops, e.g. the Monin-Obukhov `_diag_ustar`) with `if not _is_tracer_arg([...]):` →
exact original float path, `else:` → `jnp`/`jnp.where` mirror (`_diag_ustar_jax`); guard divisors a `where`
would otherwise leave `nan` in the unused branch (poisons reverse-mode grad even when masked). **(R8)
diagnostic-skip-under-trace** — pure NaN/Inf/stats checks that don't feed the prognostics early-`return`
unchanged when an input is a tracer (`parameterization_check_jax`). **iter26 extended this to the
`generic_forcings` driver** (the path for ~17 cases): the generic surface scheme (incl. the convergence-test
`_mono_cubic_interp` BC and `_compute_ubar`) is now tracer-transparent and `d(½∑um²)/dum` whole-driver grad is
FD-correct for bomex. **iter27–29: bomex whole-driver grad is now COMPLETE (thlm + um both 87/87,
FD-correct rel ≤5.4e-7)** after hardening the inf-grad `sqrt`/`pow` sites that detonate in convective layers
(the binding one: `mixing_length.py:180` `sqrt(maximum(zero_threshold, bv_smth))` with `zero_threshold==0`,
pinned by **stop_gradient bisection**). So both the arm and generic_forcings whole-driver
paths are differentiable. **iters 30–33: whole-driver `jax.grad` is now finite for ~18 of 19 cases** —
all the major subsystems are tracer-transparent: simplified + BUGSrad radiation, soil-veg, KK + Morrison
microphysics (the post-core diagnostics use **detach-under-trace** — they feed only the next step, so are
dead for a single-step gradient; BUGSrad is also reverse-mode memory-prohibitive), the sponge layer
(vectorized to a no-op-outside-sponge form), cloud-droplet sedimentation, and the case surface schemes
(R7 `_diag_ustar_jax` dispatch). Some cases show a single-level FD kink at a hard physical threshold
(8e-3 `rtm` inversion) — a genuine non-smooth point, not a bug. **iter34: the last blocker (gabls3_night
`_landflx_scalar`, a Businger-Dyer land-surface MO scheme) is ported to `_landflx_jax` → ALL 19 cases now have
a finite whole-driver gradient.** `run_scripts/compare_grad.py` is the suite-wide differentiability GATE
(grad analogue of compare_cases); `tests/probe_driver_grad.py <case>` is the per-case validator;
`tests/_nanhunt.py` + stop_gradient bisection locate residual nan; clip-`sqrt`/fractional-`pow` →
`_safe_sqrt`/`_safe_pow`. **The B5 goal ("differentiable, entirely in JAX") is met suite-wide.**

**★ "Entirely in JAX."** The JAX driver imports no `clubb_python` at module level (verified by
`tests/test_standalone_jax.py`, which runs cases with `clubb_python` blocked). 19 cases have entirely-in-JAX
forcings (no Fortran fallback). "Faithful" != "entirely-in-JAX" != "bit-faithful full run" -- verify a ported
forcing via the STANDALONE (clubb_python-blocked) test, since a plain compare can PASS via the Fortran fallback
(false positive) and the fallback-hidden sfclyr often carries a bug. **Variants share a `runtype`** (rf02
nd/so/do/ds all `'dycoms2_rf02'`; rf01 vs rf01_fixed_sst by `sfctype` 0/1) -- key off `state['runtype']` + the
distinguishing flag, NOT the case-file name, and re-run the affected variant after a port/revert.

**★★ jit-recompilation -> unbounded compile-cache -> OOM (Iter290 root-cause + convention).** An eager
`lax.scan` whose body CLOSES OVER a concrete (non-tracer) array bakes that array's VALUES into the jaxpr as
constants, so XLA recompiles every timestep when the values change -> the compile cache grows without bound -> OOM
on long runs. **Rule: any per-timestep eager `lax.scan` (or a function containing one) should be `jax.jit`-wrapped
at a stable entry point** so captured arrays hoist to operands and it compiles once per aval. Diagnose with
`JAX_LOG_COMPILES=1` + `grep -c "Compiling jit(scan)"` -- a count that grows each step is this bug. Fixed for
`parabolic_cylinder.dv_parabolic_cylinder`, the tridiag/penta solvers, and `fill_holes_vertical_jax` (rico
2165->381 total compiles, 137/step -> ~0; OOM gone, ~2x faster); re-verified bit-faithful. A `Killed`/EXIT=137
with no traceback is OOM (not a NaN) -- probe `resource.getrusage().ru_maxrss`. **Separate, still-open leak
(Iter323):** the per-step `l_sample=True` diagnostic path retains ~85 device buffers/step in the XLA backend (NOT
recompilation -- all compiles are at startup) -> a `compare_runs` per-step-stats Morrison run OOMs ~150-250 steps.
Workaround: run `advance_clubb_to_end` with `state['stats_writer']=None` and inspect the state dict, or sample at
the case default interval. Low priority (no current need for long Morrison compare runs).

**★ Precision convention — SUPERSEDED by the REFACTOR (numerical-accuracy standard).** Historically the
Fortran M2005 interface keeps `T_in_K`/`rcm_r4` in SINGLE precision (`real(...)`=REAL(4)), so its `thlm_mc`
carries a ~1e-7 single-precision round-trip residual even with zero microphysics tendencies; the JAX once
*replicated* the `real*4` casts to match it bit-for-bit (the sole reason mpace_a was "bit-faithful"). **Under
the relaxed numerical-accuracy standard (done; see "Correctness standard") we no longer reproduce the oracle's imprecisions.**
`module_mp_graupel.py` now computes `thlm_mc` in the algebraically-exact float64 form `(ten['T']−Lv/Cp·rcm_mc)/
exner` → clear-air `thlm_mc≈0` (correct; was a 2.9e-7 artifact). mpace_a is no longer bit-faithful but PASSES
Tier-C with large margin (means 70× / flux 21× / moment 104× / microphys 40×). **General rule going forward:
prefer float64 accuracy; validate within Tier-C rather than reproducing single-precision artifacts.**

4 persistent diagnostic-only (non-prognostic) differences, not fixable without matching Fortran FP ordering:
`rtm_spur_src` ~2e-16, `thlm_spur_src` ~2e-11, `rtp2_pd` ~7e-27, `up2_pd` ~1e-17.

---

## Correctness standard (relaxed: numerical accuracy + differentiability)

The original gate was **bit-faithfulness** to the Fortran oracle (`compare_runs.py`, rel 1e-6 / abs 1e-12 on
prognostics). That was the right scaffolding for the incremental port — it caught real bugs (the stretched-grid
`weights_zm2zt` column-swap, the stale `wm_zm`, the KK covar driver) — but it outlived its use: it forced the
JAX to reproduce the Fortran's *imprecisions* (single-precision casts, the low-accuracy `expax`), it blocked
differentiability (hard min/max, `while_loop`, numpy round-trips), it produced brittle "failures" that are pure
FP/oracle artifacts at sharp edges, and trajectory-level bit agreement is *physically impossible* for chaotic
turbulence past the Lyapunov horizon anyway. The numerical-accuracy refactor (done on this branch) relaxed it to
a **tiered standard** — a change is correct if it passes the tiers appropriate to what it touches:

| Tier | Checks | Hardness / tool |
|---|---|---|
| **A. Invariants & conservation** | water/energy/mass conservation, positivity (`rrm,Nrm,rcm,…≥0`), bounded correlations (`\|corr\|≤1`), finiteness | **strict, oracle-free** — `tests/test_invariants.py`, `run_scripts/invariants.py` |
| **B. Golden-trajectory regression** | vs a stored **JAX reference run** per case, rel ~1e-9 | **strict-ish** — `run_scripts/golden.py`, `update_golden.py`, `validate_case.py --no-fortran` |
| **C. Physical fidelity vs Fortran** | windowed, field-scaled rel error within the chaos horizon (aggregate, not point bit-match) | **relaxed** — `compare_cases.py --tier physical`, `validation.py` |
| **D. Climatology / statistics** | time-mean & variance profiles, BL depth, cloud fraction past the chaos horizon | **statistical** (the honest gate for chaos-limited cases) |
| **E. Differentiability** | finite-difference grad checks; whole-driver `jax.grad` | **strict** — `compare_grad.py`, `probe_driver_grad.py`, `test_differentiability.py` |

**Tier-C field-class tolerances** (point-max `max|Δ|/(max|ref|+floor)`): means (`thlm,rtm,um,vm`) **1e-4**;
fluxes (`wpthlp,wprtp,upwp,vpwp`) **1e-3**; second moments (`wp2,wp3,rtp2,thlp2,em,…`) **3e-3**; microphysics
(`rrm,Nrm`) **1e-2**; diagnostics + `*_mc` tendencies **report-only** (timing-confounded). Bit-faithful cases
pass Tier-C by construction (rel ~1e-11 ≪ tol); calibrated against rico (near-worst FP case — dynamics PASS
2–10× margin) and arm/bomex (~1e7×).

**Status (this branch):** **20/20** `compare_cases` DEFAULT_CASES PASS Tier-C (19 strictly bit-faithful + mpace_a
within tolerance on its single-precision Morrison residual; clex9_nov02/oct14 added Iter313); **all 19 cases are whole-driver-`jax.grad`-
differentiable** (see "Differentiability status"). The accuracy-lowering contrivances were removed —
`parabolic_expax` (`epss=1e-4`), the Morrison `real*4` casts, BUGSrad `sngl`/float32-π — so the JAX is now
strictly *more* accurate there. **Preserve:** the Fortran oracle as a reference-within-tolerance (`--tier bit`
stays for debugging); golden refs as the regression net (re-baseline only via `update_golden.py`, deliberate +
reviewed); and **Tier A strict** — relaxed tolerances must never hide a conservation bug.

---

## Critical Conventions

**Band ordering:** Both Fortran and JAX use `lhs[0=super, 1=main, 2=sub]`. No flip needed
between diffusion output and solver input.

**Grid weights (`weights_zm2zt`):** Shape `(ngrdcol, nzt, 2)`. `[:,k,0]` = M_ABOVE (weight for
`zm[k]`), `[:,k,1]` = M_BELOW (weight for `zm[k+1]`). Fortran 1-indexed `m_above=1, m_below=2`.

**JAX x64 mode:** `jax.config.update("jax_enable_x64", True)` called at module load in
`advance_clubb_core_module.py`. All arrays must stay float64.

**Index mapping (Fortran 1-based → Python 0-based):**
Interior loop `k=2..nzm-1` in Fortran → Python `[:,1:-1]` on zm-level arrays.

**`clubb_params` indexing:** Shape `(ngrdcol, 102)`, 0-based. Access as `clubb_params[:, iC2rt - 1]`.

---

## What Has Been Built

Each JAX module mirrors its Fortran oracle at the same relative path under `src/CLUBB_core/`.

| JAX Module | Fortran Oracle | Status |
|---|---|---|
| `grid_class.py` | `grid_class.F90` | `zm2zt`, `zt2zm`, `ddzm`, `ddzt`, `zm2zt2zm`, `zt2zm2zt` — unit tests pass |
| `diffusion.py` | `diffusion.F90` | `diffusion_zt/zm_lhs`, `xpyp_term_ta_pdf_lhs/rhs` (centered + upwind) — ≤ machine epsilon |
| `matrix_solver_wrapper.py` | `tridiag_lu_solver.F90` | `tridiag_lu_solve_jax` — bit-exact |
| `calc_roots.py` | `calc_roots.F90` | `cubic_solve` (Cardano, complex128 principal-branch), `quadratic_solve`, `cube_root` — polynomial residual ~4e-16 + numpy.roots set-match; differentiable. Completeness port (the gated ADG1 path doesn't call it; `new_pdf` does) |
| `pos_definite_module.py` | `pos_definite_module.F90` | `pos_definite_adj_jax` — Smolarkiewicz (1989) flux-conservative positive-definite limiter (ascending grid). **Bit-exact vs the f2py oracle (rel 0)** + column-integral conservation; differentiable. Completeness port (gated by `l_pos_def`, off by default — the suite uses `mono_flux_limiter`) |
| `diagnose_correlations_module.py` | `diagnose_correlations_module.F90` | `diagnose_correlations` (Larson 2011 hydromet correlation diagnosis for SILHS: `rearrange_corr_array` + `diagnose_corr`) + PDF helpers `calc_mean`/`calc_varnce`/`calc_w_corr`. **Bit-match vs the f2py oracle (rel 1.6e-15)** across iiPDF_w edge cases; differentiable. Completeness port (gated config uses PRESCRIBED corr; `l_calc_w_corr=True` / approx_w_corr unported) |
| `Microphys/ice_dfsn_module.py` | `Microphys/ice_dfsn_module.F90` | `ice_dfsn` — depletion of cloud water by diffusional growth of ice (Larson 2006; R&Y Eq. 9.4) as a top-to-bottom falling-crystal mass-integration `lax.scan`; `diff_denom` helper. Validated vs a literal NumPy transcription (**rel 1.2e-16**), branch/over-depletion-cap coverage, differentiable. New helper `thlm2T_in_K_jax` (T_in_K_module.py) is **bit-exact vs `f2py_thlm2t_in_k_1d`**. Completeness port (no f2py wrapper for ice_dfsn itself) |
| `Microphys/KK_microphys/KK_upscaled_variances.py` | `KK_microphys/KK_upscaled_variances.F90` | `variance_KK_mvr` — variance of the KK rain mean-volume radius `Var(R_vr)=E[R_vr²]−E[R_vr]²` over the 2-component in-precip bivariate-lognormal PDF (assembled from `bivar_LL_mean_eq` with doubled exponents). Validated against an independent closed-form lognormal-moment computation (**rel 0**) and a 4M-sample Monte-Carlo (**rel 1.4e-4**); differentiable. Completeness port (no f2py wrapper exposed) |
| `advance_xp2_xpyp_module.py` | `advance_xp2_xpyp_module.F90` | Full solve for rtp2/thlp2/rtpthlp/up2/vp2 — machine epsilon |
| `advance_xm_wpxp_module.py` | `advance_xm_wpxp_module.F90` | Full solve for wprtp/rtm/wpthlp/thlm/upwp/um/vpwp/vm — machine epsilon |
| `advance_wp2_wp3_module.py` | `advance_wp2_wp3_module.F90` | Full solve for wp2/wp3/wp2_zt — machine epsilon |
| `advance_windm_edsclrm_module.py` | `advance_windm_edsclrm_module.F90` | No-op for ARM (l_predict_upwp_vpwp=True) — bit-exact |
| `advance_xp3_module.py` | advance_xp3 + Skx_module | rtp3/thlp3/up3/vp3 (ADG1 path) — machine epsilon |
| `advance_helper_module.py` | `advance_helper_module.F90` | Skw, thvm, BV, Ri, Lscale/tau, splat, Cx — machine epsilon |
| `sfc_varnce_module.py` | `sfc_varnce_module.F90` | Surface second-order moments — sub-machine precision |
| `sigma_sqd_w_module.py` | `sigma_sqd_w_module.F90` | σ²_w PDF width parameter — bit-exact |
| `pdf_utilities.py` | `pdf_utilities.F90` | `mean_L2N`/`stdev_L2N` (lognormal->normal moments) **bit-to-bit vs f2py** (rel 0.0); `corr_NL2NN`/`corr_LL2NN` (vs Monte-Carlo); the inverses `corr_NN2NL`/`corr_NN2LL` and `calc_corr_chi_x`/`calc_corr_eta_x` (+ their round-trip inverses) **bit-to-bit vs f2py**. The lognormal-PDF inputs to the KK rate functions |
| `precipitation_fraction.py` | `precipitation_fraction.F90` | `precip_fraction` -- overall (downward cumulative-max) + per-component (`component_precip_frac_specify`) + max_hm limiter. **Bit-exact** vs the rico stats oracle on the well-resolved precip region. f2py wrapper FPE-traps -> stats oracle used |
| `setup_clubb_pdf_params.py` | `setup_clubb_pdf_params.F90` | `calc_comp_mu_sigma_hm` (in-precip component means/stdevs via a mean+variance-preserving quadratic solve, verified by its preservation contract); `compute_mean_stdev` + `norm_transform_mean_stdev` (the `setup_pdf_parameters` orchestration that stacks per-PDF-variable moments into the `(ngrdcol,nzt,pdf_dim)` arrays the rate functions index, iiPDF order [chi,eta,w,Ncn,hydrometeors], and transforms lognormal vars to log space). The KK driver assembles rr/Nr moments through these -- bit-identical + differentiable. `tests/test_calc_comp_mu_sigma_hm.py` |
| `Nc_Ncn_eqns.py` | `Nc_Ncn_eqns.F90` | `Nc_in_cloud_to_Ncnm` (+ `Ncm_to_Ncnm`, `bivar_Ncnm_eqn_comp`) -- cloud-nuclei mean <Ncn> from in-cloud <Nc> and the chi PDF via the erfc integral. **Bit-to-bit vs f2py** (worst rel 2.4e-14); reproduces rico Ncnm exactly |
| `corr_varnce_module.py` | `corr_varnce_module.F90` | `set_corr_arrays_to_default` -- the prescribed in-cloud/below-cloud normal-space correlation arrays from the fixed 12x12 default tables (column-major reshape). The KK driver derives corr(chi,rr)/corr(chi,Nr)/corr(rr,Nr) from it instead of hardcoding (rico oracle bit-identical). `init_pdf_hydromet_arrays` + `HmMetadata` + `kk_hm_metadata` -- the per-hydrometeor PDF metadata (names/tols/flags, in-precip variance ratio, PDF-variable indices) the hydrometeor advance + setup consume. `tests/test_corr_varnce.py` |
| `Microphys/KK_microphys/KK_utilities.py` | `KK_microphys/KK_utilities.F90` + `KK_microphys_module.F90:1177` | `G_T_p` (drop-growth coefficient, Rogers&Yau) + `kk_evap_coef`. Validated via the rico rrm_evap oracle (T_liq=thlm*exner) |
| `fill_holes.py` | `fill_holes.F90` | `fill_holes_vertical`, `fill_holes_wp2_from_horz_tke` — machine epsilon |
| `clip_explicit.py` | `clip_explicit.F90` | `clip_variance`, `clip_skewness`, `clip_covar`, `clip_rcm`, `clip_covars_denom` — bit-exact |
| `adg1_adg2_3d_luhar_pdf.py` | `adg1_adg2_3d_luhar_pdf.F90` + `pdf_closure_module.F90` | Full ADG1 PDF closure — machine epsilon |
| `mixing_length.py` | `mixing_length.F90` | `diagnose_lscale_from_tau` + `compute_mixing_length` (Golaz 2002 nonlocal parcel) — machine epsilon |
| `saturation.py` | `saturation.F90` | `sat_mixrat_liq` (Flatau/Bolton), `rcm_sat_adj` (bisection) — machine epsilon |
| `sponge_layer_damping.py` | `sponge_layer_damping.F90` | `initialize_tau_sponge_damp` + `sponge_damp_xm` (xm fields rtm/thlm/uv) — wired into `advance_clubb_core`; ekman means bit-faithful. xp2/xp3 sponge not yet ported |
| `Microphys/cloud_sed_module.py` | `Microphys/cloud_sed_module.F90` | `cloud_drop_sed` (Stokes-regime cloud-droplet sedimentation, `l_cloud_sed`) — bit-faithful (`sed_rcm` ~1e-11); wired into the driver loop. Unblocked atex_long + dycoms2_rf02_so (Iter100) |
| `Microphys/KK_microphys/kk_microphys_driver.py` | (assembly) + `KK_microphys_module.F90:1196` | The three KK mass-tendency entry points `kk_autoconversion/accretion/evaporation_mean` (vs rico auto 4.7e-7, accr 6.1e-9, evap 3.3e-6); `kk_microphys_adjust` (rates -> rrm_mc/Nrm_mc/rvm_mc/rcm_mc/thlm_mc with source/evap over-depletion limiters); `compute_kk_microphysics` (the full standalone step: hydromet fields + PDF state -> tendencies). Differentiable |
| `Microphys/KK_microphys/KK_Nrm_tendencies.py` | `KK_microphys/KK_Nrm_tendencies.F90` | `KK_Nrm_auto_mean`, `KK_Nrm_evap_local_mean`, `KK_Nrm_evap_upscaled_mean` (vs rico Nrm_evap median 3.2e-6). **All KK rates ported+validated** (rrm auto/accr/evap, Nrm auto/evap, mvr) |
| `kk_microphys_driver.py::kk_sedimentation` | `KK_microphys_module.F90:1542` | KK mean sed velocities Vrr/VNr from the mean volume radius (KK00 Eq.37), clipped <=0, top zero-flux BC. **Bit-exact vs the rico oracle** (\|d\|max 1.1e-16 on rain points, via the bit-faithful `zt2zm`); differentiable. The V_hm input `advance_hydrometeor` needs |
| `Microphys/advance_microphys_module.py` | `advance_microphys_module.F90` | The full hydrometeor transport solve: `sed_centered_diff_lhs` + `term_turb_sed_lhs/rhs` (mean + turbulent sedimentation, flux-form), `microphys_lhs`/`microphys_rhs` (the implicit Crank-Nicholson tridiagonal: 1/dt + 1/2 diffusion_zt + term_ma_zt + sed + turb-sed), `advance_one_hydrometeor` (assemble + `tridiag_lu_solve`), and `calculate_K_hm` (the hydrometeor eddy diffusivity, capped at \|corr(w,hm)\|<=1). Verified by the conservation contract (~5e-15) + the rico `rrm_ma`/`rrm_ts` budgets; K_hm bit-validated vs the oracle's stored `K_hm_<hm>`. `tests/test_kk_rico_oracle.py` |
| `Microphys/KK_microphys/KK_upscaled_turbulent_sed.py` | `KK_microphys/KK_upscaled_turbulent_sed.F90` | `kk_sed_vel_covars` -- the rain sed-velocity covariances <V'r'>/<V'N'> (bivariate-lognormal, impc/expc semi-implicit split). **Bit-faithful-to-the-gate vs the rico oracle** (rel 4.5e-11, no timing confound); differentiable. Feeds `term_turb_sed_lhs` |
| `Microphys/KK_microphys/{parabolic_cylinder,PDF_integrals_means,KK_upscaled_means}.py` | `Microphys/KK_microphys/{KK_utilities,PDF_integrals_means,KK_upscaled_means}.F90` + `parameters_KK.F90` | **Complete upscaled-KK analytic means library** -- all 4 means (auto/accr via bivar_NL, evap via trivar_NLL over the chi<0 half, mvr via bivar_LL), built on the parabolic-cylinder D_v (1F1 series + optimally-truncated asymptotic, accurate float64 everywhere — the do/ds `epss=1e-4` `expax` reproduction was removed in the REFACTOR, A1). Verified vs scipy.pbdv + brute-force quadrature (<=3e-11); all jitted + differentiable |
| `Microphys/Morrison_microphys/module_mp_graupel.py` | `Morrison_microphys/module_mp_graupel.F90` | **Complete Morrison 2-moment (M2005) port.** Special functions `polysvp_jax`/`derf1_jax`/`gamma_jax` (bit-exact / vs scipy ~1e-15). All process rates -- warm-rain (KK bulk PRC/PRA + evap PRE) + the full ice block (deposition/sublimation, snow/graupel collection, aggregation, nucleation, freezing, self-collection, melting), oracle-validated to a few % (timing confound) or bit-exact. The single-column step (`compute_m2005_rates` -> `m2005_step_tendencies`, cold/warm branches selected per level by T>=273.15 + the PCC saturation adjustment) verified by the **water-conservation contract** (sum mass tendencies ~1e-21). Sedimentation (`morrison_sedimentation`: rain/ice/snow/graupel/cloud, shared-NSTEP CFL, conservation-verified; vertical=LAST axis; the CLUBB<->M2005 grid index FLIP) + pre-rate slope clamps + the `morrison_microphys_driver` CLUBB interface (`hydromet_mc=(field_final-field_initial)/dt`; `thlm_mc` via the `real*4` round-trip, see the precision lesson). Wired via `morrison_microphys_step.py` (gated on `microphys_scheme=='morrison'`). Runs float64 except the deliberate single-precision interface casts. `tests/test_morrison_{special,rates,differentiable}.py` |
| `T_in_K_module.py` | `T_in_K_module.F90` | `calculate_thvm` — bit-exact |
| `calc_pressure.py` | `calc_pressure.F90` + `hydrostatic_module.F90` | `hydrostatic`, `init_pressure` via `jax.lax.scan` |
| `parameters_tunable.py` | `parameters_tunable.F90` | `init_clubb_params`, `calc_derrived_params` — bit-exact |
| `model_flags.py` | `model_flags.F90` | `get_default_config_flags` — all 88 flags |
| `numerical_check.py` | `numerical_check.F90` | `parameterization_check`, `check_clubb_settings`, `check_parameters` |
| `Benchmark_cases/arm.py` | `arm.F90`, `prescribe_forcings.F90`, `time_dependent_input.F90`, `sfc_flux.F90`, `diag_ustar_module.F90` | Full ARM forcing (Monin-Obukhov, time-interpolated) |
| `io/stats_writer.py` | `stats_netcdf.F90` | Pure Python NetCDF stats output (StatsWriter) — bit-exact |
| `advance_clubb_core_module.py` | `advance_clubb_core_module.F90` | Full ARM timestep — **zero Fortran calls** |
| `src/clubb_standalone.py` | `clubb_standalone.F90` | Thin CLI frontend (argv → `run_clubb`) — mirrors the 88-line Fortran program. Entry point for `-jax` |
| `src/clubb_driver.py` | `clubb_driver.F90` | `run_clubb` (init → advance → cleanup) + `init_clubb_case` + `clean_up_clubb` — **zero Fortran API imports** |
| `src/derived_types/` | `clubb_python_api/clubb_python/derived_types/` | Pure-Python mirrors: ConfigFlags, ErrInfo, SclrIdx, Grid, pdf_parameter, implicit_coefs_terms |
| `src/Radiation/radiation.py` | `radiation_module.F90`, `cos_solar_zen_module.F90`, `rad_lwsw_module.F90` | `cos_solar_zen`, `sunray_sw`, `simple_rad` — **zero Fortran imports** |
| `advance_clubb_to_end.py` | `clubb_driver.F90` (`advance_clubb_to_end` subroutine) | Timestep loop: stats + forcing → advance → stats. Kept in its own submodule for size; imported by `run_clubb`. **Zero module-level Fortran imports**; `prescribe_forcings` uses lazy import for non-ARM |

**ARM per-timestep Fortran calls: ZERO.** All prognostic state, diagnostics, forcings, and
stats output are pure JAX/Python.

**Module-level Fortran dependency status:**
- `clubb_standalone.py` / `clubb_driver.py`: zero (`derived_types` now local)
- `advance_clubb_to_end.py`: zero (lazy import only in non-ARM `_prescribe_forcings`)
- `radiation.py`: zero (`cos_solar_zen` and `sunray_sw` ported to pure Python)

---

## Cross-case bit-faithfulness status (vs Fortran, `compare_cases.py`)

"Runs" ≠ "bit-faithful". A case can run end-to-end in JAX yet diverge from the Fortran
oracle. Verified status (prognostic, rel tol 1e-6):

| Case | Status | Notes |
|---|---|---|
| arm | ✅ PASS (225 steps) | bit-faithful reference |
| wangara | ✅ PASS (30) | land case; uses `l_modify_bc_for_cnvg_test` 25 m BC |
| gabls2 | ✅ PASS (30) | land case |
| bomex | ✅ PASS (30) | fully bit-faithful (Iter79 BC + Iter80 thv + Iter81 rc_coef_zm) |
| dycoms2_rf01 | ✅ PASS (30) | fully bit-faithful (Iter80 thv + Iter81 rc_coef_zm) |
| atex | ✅ PASS (30) | bit-faithful since Iter84 (xm monotonic flux limiter wired in) |
| gabls2 | ✅ PASS (30) | bit-faithful (instantaneous output) |
| gabls3_night | ✅ PASS (30) | bit-faithful since Iter86 (um_f/ug time-dependent wind forcing) |
| fire | ✅ PASS (30) | bit-faithful since Iter87 (bulk surface scheme, sfctype=1) |
| neutral | ✅ PASS (30/55) | bit-faithful since Iter91 (neutral_case_sfclyr: ustar=0.5 + momentum flux) |
| ekman | ✅ PASS (30) | bit-faithful since Iter94 (sponge damping Iter93 + `ice_supersat_frac` at the cold 10 km top, Iter94) |
| cobra | ✅ PASS (40) | bit-faithful since Iter94 ice fix (Iter87-89 surface/wind/subsidence + the cold-cloud `ice_supersat_frac`); confirmed Iter96. Its step-14 cloud onset (T=266-270 K) was the SAME ice bug, not FP-boundary |
| dycoms2_rf02_nd | ✅ PASS (30) | bit-faithful (Iter96). "_nd" = **no drizzle** — a standard stratocumulus case, NOT the drizzle variant; was mislabeled blocked |
| dycoms2_rf01_fixed_sst | ✅ PASS (30) | bit-faithful (Iter98). dycoms2_rf01 variant with fixed SST surface |
| atex_long | ✅ PASS (30) | bit-faithful (Iter100). Unblocked by the `cloud_drop_sed` port (`l_cloud_sed`) |
| dycoms2_rf02_so | ✅ PASS (30) | bit-faithful (Iter100). Unblocked by `cloud_drop_sed`; do/ds variants still need drizzle microphysics |
| jun25_altocu | ✅ PASS (30) | bit-faithful (Iter188). Cold-cloud altocumulus + "simplified" radiation; unblocked by the per-step `wm_zm` (subsidence) recompute fix |
| gabls3 | ✅ PASS (30) | **bit-faithful (Iter273-274, 17th case)** — 0 prognostic failures at the full 30-step gate. Full BUGSrad correlated-k radiation + interactive soil_vegetation + gabls3 surface flux + omega subsidence. **bugs_rad is jitted (Iter274)** — fixes the eager-dispatch ~700 MB/call OOM-after-6-steps + ~2.4× faster (~6 s/step, JAX 30-step run 194 s); configured run is 1440 steps (24 h) |
| mpace_a | ✅ PASS (30) | bit-faithful (Iter299, 18th case). Morrison (l_ice_microphys) but clear/sub-saturated; the only Morrison signal is the clear-air single-precision thlm_mc round-trip residual |
| clex9_nov02 | ✅ PASS (30) | **bit-faithful (Iter313, 19th case)** — CLEX-9 cold-cloud altocumulus. Morrison configured but `microphys_start_time` (51411 s) is beyond the 30-step window → never activates; prognostically bit-exact + Tier-C clean once the pre-activation Ncm/Nc_in_cloud diagnostic was fixed to match `advance_microphys`'s early return |
| clex9_oct14 | ✅ PASS (30) | **bit-faithful (Iter313, 20th case)** — sibling of clex9_nov02 (same campaign, same Morrison pre-activation window) |

**Bit-faithful: 20 cases** (the table above; all pass `compare_cases.py` at 30 steps and the durability gate at
100). 9 are bit-faithful for their ENTIRE configured run (dycoms2_rf01, cobra, bomex, neutral, dycoms2_rf02_nd,
dycoms2_rf02_so, wangara, atex, dycoms2_rf01_fixed_sst). mpace_a (Iter299) is the first Morrison case made
faithful -- it stays clear/sub-saturated, so the only M2005 signal is the clear-air single-precision `thlm_mc`.

**48-case coverage (Iter153 survey): 20 run, 28 unsupported, 0 hard crashes.** Each of the 28 is blocked by one
unported SUBSYSTEM (the `_check_unsupported_features` message names it): **morrison microphysics -- ~19 cases**
(the dominant lever; M2005 is now ported, see Remaining Work), **bugsrad radiation** (mostly ALSO morrison; the
one clean win was gabls3, now faithful), **COAMPS microphysics** (arm_0003), **SILHS interactive sampling** (not
bit-reproducible vs the Fortran RNG -- not a target).

**Characterized not-bit-faithful cases (do NOT chase as bugs -- each is numerically/FP-limited, often because the
JAX is MORE accurate than the low-accuracy Fortran defaults):**
- **rico** (grid_type=2, KK): bit-faithful steps 1-4; from step 5 the near-zero rt-flux clip at the stretched dry
  top amplifies FP-level `rtp2` diffs (the dry-top rtp2 sits at the rt_tol^2 floor -- matching ~0 to rel-1e-6 is
  impossible). Grid verified bit-exact (Iter306); the step-1 seed (missing rtm/thlm `fill_holes`) was fixed Iter186.
- **coriolis_test**: an analytic Foucault-pendulum benchmark that zeroes nearly all closure constants and needs the
  nontraditional-Coriolis flag; the undamped oscillator accumulates FP noise (no decay). Step-1 faithful, no seed.
- **nov11_altocu** (Morrison + ice + SW radiation): bit-faithful through step 5; step 6 is the ice-cloud-edge FP
  floor (the `ice_supersat_frac` erf at near-zero scalar variance, then the `/0.001` Lscale ramp = a 1000x
  amplifier) -- every link verified faithful. Microphysics activates at step 60 (gated by `microphys_start_time`),
  past the FP floor, so the M2005 transport is validated by unit tests, not the full run.
- **dycoms2_rf02_morr** (warm Morrison, active from step 1): the M2005 hydrometeor transport (K_hm + sedimentation)
  is verified ~bit-exact (K_hm matches the oracle's stored `K_hm_<hm>`), but a tiny near-singular `rcm_mc` residual
  at the sharp cloud-top CF3D edge (the in-cloud /CF3D <-> grid-mean *CF3D conversion where cloud_frac->0) plus the
  M2005 single-precision floor keep it off the gate.
- **dycoms2_rf02_do / _ds** (KK, drizzle): the KK rt/thl covariance is physically correct but cancellation-amplifies
  the parabolic-cylinder `D_v`. The SCM oracle runs `parab` at `epss=1e-4`; the JAX uses the accurate float64 `D_v`,
  so the bit-gap WAS the oracle's deliberate low accuracy (proven with oracle numbers, Iter310). **REFACTOR A1
  (iter7): the `expax` reproduction of the oracle's epss=1e-4 artifact was DELETED** (`parabolic_expax.py` removed) —
  the JAX is now simply more accurate than the low-accuracy oracle. do/ds are not bit-faithful by design and are
  judged under Tier-C (dynamics) / Tier-D (drizzle hydrometeors), not against the oracle's imprecision.

**Durable lessons (the conventions these investigations produced):**
- **NEVER trust a default-vs-computed value** -- a Fortran line `x=default` may be overwritten later; verify the
  actual computed quantity (an unverified `precip_frac=1` cost 12 iterations chasing a 2x K_hm bug).
- **Decouple-the-oracle before blaming a subsystem** -- feed the Fortran's own field into the JAX subsystem (e.g.
  Fortran cloud into `_simple_rad_lw`) to exonerate it; "steep radiation" was a red herring for jun25 (the real
  seed was a stale per-step `wm_zm` -- a grid-staggered partner not recomputed when its `_zt` counterpart updated).
- A `*_forcing` stat that disagrees in a microphysics case is often `raw_forcing + lagged *_mc`, not a forcing bug;
  an isolated oracle debug build (`compile.py -install <scratch> -debug`) is the decisive tool when static analysis
  stalls. A `-jax` run dying with a Fortran `error stop`/no Python traceback is the unported-case `clubb_api`
  fallback -- port the case's tndcy/sfclyr to `generic_forcings.py`.

---

**★★ Numerical-accuracy refactor — COMPLETE (both criteria met).** **(b) Faithful:** all 20 `compare_cases`
DEFAULT_CASES PASS Tier-C (`--tier physical --max-iters 30`) — 19 stay strictly bit-faithful (0 prognostic
failures), mpace_a passes within tolerance (the intended A2 reclassification: float64 `thlm_mc` is more accurate
than the Fortran single-precision artifact). The accuracy-lowering contrivances (A1 expax, A2 Morrison real*4,
A3 BUGSrad sngl/float32-π) were removed and the differentiability work (B2–B5) was all forward-identical, so the
suite has ZERO faithfulness regression; Tier-B goldens baselined for all 18. **(a) Differentiable:** whole-driver
`jax.grad` through one `advance_clubb_to_end` step is finite + finite-difference-correct for **all 19 cases**
(`compare_grad.py`); see "Differentiability status". The sole case outside the faithful suite is **rico**, whose
KK rain-microphysics *transport+feedback* is a deliberately staged, gated-off port (`l_kk_micro_apply` default
off) — a pre-existing incomplete subsystem, not touched by this refactor.

---
## Remaining Work

**★ Achievable-state assessment -- read before picking the next piece.** The non-subsystem bit-faithful
frontier is nearly saturated (20 cases as of Iter313). Most remaining gains need a LARGE subsystem port with
poor ROI, because the cases they unblock are themselves numerically-limited (see the characterized cases above).
**BUT (Iter313) the frontier was NOT fully saturated:** clex9_nov02/oct14 were "unported" only because their
Morrison scheme never activates in the gate window — they are pure closure physics and were bit-faithful all
along, blocked only by a diagnostic-output mismatch (pre-activation Ncm). Lesson: before declaring a case
blocked by an unported subsystem, check whether that subsystem actually *runs* in the gate window. **Do NOT
chase the genuinely numerically-limited microphysics cases (rico, dycoms2_rf02_do/ds) as "bugs" -- they are
characterized.** Full 48-case completion is gated by Fortran numerical limits plus impractical ports.

**★ Completeness loop — final state (Iters 1–33).** A 33-iteration sweep ported every remaining **in-scope,
oracle-validatable, self-contained** routine and unit-tested each (differentiable; oracle = f2py bit-shadow
where exposed, else closed-form / Monte-Carlo / round-trip / literal-transcription). Highlights: the entire KK
PDF-integral mixed-moment machinery (`mixed_moment_PDF_integrals` + `PDF_integrals_all_MM`, both ✅), the BUGSrad
cloud-overlap (`cloud_correlate`, both subroutines → Radiation 100% ported), `ice_dfsn`, the GFDL droplet-activation
CLUBB-side (erff/updraft_weights/aer_act_clubb_ndrop), `inverse_hydrostatic`, the CGILS pressure-coordinate /
`T_f` / `um_ref` forcing-reader capability (guarded, gated cases byte-identical), and **all benchmark-case
surface/forcing schemes** (lba, mpace_b, arm_97, twp_ice, arm_3year, arm_0003, cloud_feedback). Verified
regression-free: 10/20 gated cases re-confirmed across every type (forcing-pipeline, sounding, analytic, cloud-sed,
Morrison); a test-infrastructure shadowing bug was found+fixed (iter 31).
The **genuinely remaining unported `.F90` (3 files, all impractical/out-of-scope)**: `coamps_microphys_driver`
(7000-line alternative microphysics the gated config never uses; the Fortran itself fatal-errors on `l_predict_Nc=F`
→ **no oracle**), `gfdl_activation`'s `aer_ccn_act_wpdf_k` lookup core (the ➖ `SCM_Activation` subsystem —
Gauss-Hermite + Köhler + 5-D single-precision lookup, no case exercises it), and `pdf_hydromet_microphys_wrapper`
(would wire `hydrometeor_mixed_moments` to compute `wp2hmp`/`rtphmp`/`thlphmp` — but those are **correctly zero for
all 20 gated cases** (no active hydrometeors), so the wiring has **zero validated payoff** and needs a
setup_pdf_parameters correlation-processing port; deferred). SILHS sampling is ➖ (a different RNG can't be
bit-matched). The differentiable+faithful JAX port is **complete for all tractable/in-scope code.**

**★ Post-loop completeness extensions (iters 81–96).** A further sweep closed the last in-scope, oracle-
validatable routines and extended faithfulness to the CGILS family:
- **All alternative PDF closures** end-to-end f2py-validated: ADG2, LY93, 3-D Luhar, new-TSDADG, new-pdf, and the
  full **new-hybrid driver** (`new_hybrid_pdf_main.py`, 1.15e-14). Plus `mirror_lower_triangular_matrix`, the
  Godunov-upwind `xpyp_term_ta`, and `sponge_damp_xp2/xp3` (all f2py bit-exact).
- **`remapping_module.F90` fully ported** (both methods): Ullrich-linear (eq. 30) + the E3SM **PPM** (map1_ppm/
  ppm2m/steepz/kmppm); f2py same-grid bit-exact + mass-conservation-rel-0 on a refined grid (`remapping_module.py`).
- **CGILS/cloud_feedback init+radiation fixed** → **cgils_s11 reaches Tier-C PASS** (was rel ~1e3): the Press[Pa]
  sounding→altitude conversion (`convert_pressure_sounding_to_z`), the absolute-temperature `T[K]`→θ init
  (clubb_driver.F90:5499-5524), and the case-specific radiation extended atmosphere from the deep sounding + ozone
  sounding (`convert_snd2extended_atm` → `build_case_extended_atmosphere`, gated on `l_use_default_std_atmosphere=
  .false.`). thlm is now bit-exact at init/step1; the residual is FP-limited (cloud-topped-BL chaos). Added to the
  `compare_cases.py --cases tier_c` physical-fidelity suite. All gated cases byte-untouched (the new paths are
  gated on Press[Pa]/T[K]/the std-atm flag). **Iter97** then fixed a systematic forcing-reader bug affecting the
  whole family: `_parse_forcings_file` edge-extrapolated the forcing outside its vertical range, but the Fortran's
  `zlinterp_fnc` (via read_to_grid) **zero-fills** — so cloud_feedback's out-of-range bottom levels got a spurious
  ≈−1.6e-5 thlm forcing. With `left=right=0` the cloud_feedback means → Tier-C PASS (moments now FP-limited at cloud
  onset); gated file-forcing cases (gabls3_night/…) byte-identical (their forcings cover the model range).
- **Last validation checks** ported (`assert_corr_symmetric`, `sfc_varnce_check`) — these have no observable f2py
  oracle (err_code not exposed), validated by transcription/behavior.
The genuinely-remaining unported `.F90` are unchanged (COAMPS, GFDL lookup core, pdf_hydromet_microphys_wrapper,
SILHS RNG) — all no-oracle/zero-payoff. **No in-scope, oracle-validatable Fortran routine remains.**

**★ The strategic pivot (done) — the bit-faithful ceiling was an artifact of the *gate*, not the physics.**
Several "numerically-limited" cases were limited only because the JAX is MORE accurate than the low-accuracy
Fortran oracle, and a few modules existed solely to reproduce the oracle's imprecision (`parabolic_expax` at
`epss=1e-4`; the Morrison `real*4` casts; BUGSrad's `sngl`/float32). The **numerical-accuracy refactor** (this
branch) relaxed the gate to the tiered standard (see "Correctness standard"), which simultaneously simplified
the code, improved accuracy, and unlocked whole-driver `jax.grad`. **New work should judge correctness by the
tiered standard, not the bit-faithful frontier.**

**Subsystem status:**
- **KK microphysics (`khairoutdinov_kogan`)** -- COMPLETE and wired per-step: the full upscaled-mean/covariance
  analytic library + hydrometeor PDF setup + the `advance_one_hydrometeor` transport solve + `calculate_K_hm`.
  Unblocks rico, dycoms2_rf02_do/ds -- all three numerically-limited (above). The covar `expax` port closed the
  epss artifact.
- **Morrison 2-moment M2005 (`module_mp_graupel.py`)** -- COMPLETE: the special-function layer, all process rates
  (warm-rain + full ice block, oracle-validated), the single-column step assembly (water-conservation contract),
  sedimentation, the CLUBB<->M2005 interface, and the per-step wiring + hydrometeor transport. Runs float64 except
  the deliberate `real*4` interface casts. Faithful case: mpace_a; FP-limited: nov11, dycoms2_rf02_morr.
- **BUGSrad correlated-k radiation + `soil_vegetation`** -- COMPLETE and wired (`Radiation/BUGSrad/`,
  `bugsrad_driver.py`); `bugs_rad` is jitted (the eager dispatch leaked ~700 MB/call). gabls3 was the one
  clean radiation-only win (bit-faithful before the REFACTOR; now Tier-C). Notes: pass the constants the
  Fortran CALLER passes (constants_clubb grav/Cp, not BUGSrad's physconst); the build is `-Dradoffline
  -Dnooverlap -DCLUBB` (no ghost layer, simple two_rt called twice, `newexp` unused). **REFACTOR A3 (iter8):
  cloudg's deliberate float32 `sngl` truncation + float32-π were dropped (now float64) — ~1e-7 more accurate,
  within Tier-C; the JAX no longer reproduces those single-precision artifacts.**
- **COAMPS microphysics** (arm_0003) -- unported. **SILHS** interactive Latin-Hypercube sampling
  (rico_silhs/mpace_b/lba) -- random, not bit-reproducible; not a bit-faithfulness target.

**Microphysics-port roadmap (the pattern, for any future subsystem):** hydrometeor infrastructure first
(`hydromet_dim`, rrm/Nrm init, the `hydromet`/`wphydrometp`/`K_hm` transport) -- a rate can't be tested in-context
until the case RUNS (working rule 6) -- then the process rates (validatable via a case-stats oracle), then the
`advance_clubb_core` thv buoyancy coupling (the `{wpthvp,wp2thvp,thlpthvp,rtpthvp} -= thv_ds*...` hydrometeor terms,
a no-op for hydromet_dim=0). The f2py API exposes ZERO microphysics, so these are verifiable only by full-case
comparison once the whole subsystem is complete.

**Differentiability (secondary goal):** component-level DONE; full-timestep grad is blocked by the three coupled
items in the Differentiability status above (the all-or-nothing orchestration-numpy refactor, the numpy
`mono_flux_limiter`, the `mixing_length` while_loop).

**Minor unported pieces** (none of the current target cases need them): `sponge_damp_xp2`/`xp3` (xp2/xp3 sponge
damping for wp2/wp3/up2_vp2); the `pdf_closure_driver` `ipdf_pre_advance_fields` path.

---

## Agent Working Rules

**The Fortran→JAX port is complete.** Every `clubb_release/src/CLUBB_core/*.F90` has a JAX mirror (CLUBB_core
is now 100% ported, Iter312), the driver runs 100% in JAX, and the bit-faithful frontier is at 20 cases
(Iter313). The incremental
**shadow-comparison** workflow that built the port (run JAX beside the Fortran oracle in-loop, match to
machine epsilon, remove the Fortran call) is **retired** — there is nothing left to port that way. Most work
now is **refactoring, simplification, differentiability, and working under the numerical-accuracy
standard** (see "Correctness standard" above).

1. **Read `DESIGN.md` in full** at the start of every session. At the end, append one concise entry to
   `CHANGELOG.md`; do not read the full changelog history (it is the append-only work record).
2. **Keep the module-naming mirror.** Every `src/CLUBB_core/<name>.py` mirrors
   `clubb_release/src/CLUBB_core/<name>.F90` at the identical relative path; the Fortran stays the algorithm
   reference (now a *reference within tolerance*, not a per-timestep oracle). Export new public symbols from
   the relevant package `__init__.py`.
3. **Judge correctness by the tiered standard, not bit-faithfulness** (see "Correctness standard"):
   conservation / invariants (strict, Tier A), regression vs the golden JAX trajectory (Tier B),
   physical-fidelity vs Fortran within the field-scaled tolerance (Tier C), and — for any change to the core
   physics glue — a `jax.grad` / finite-difference differentiability check (Tier E). `compare_cases.py`
   (Tier C) + `compare_grad.py` (Tier E) are the gates; a NEW "failure" that is a known FP / oracle-precision
   artifact (sharp-edge sedimentation, covariance cancellation, single-precision residual) is
   **characterized, not chased**.
4. **Run the gate after any shared/core change**: `python clubb_jax/run_scripts/compare_cases.py
   --max-iters 30` (expect 0 prognostic failures across the bit-faithful cases), plus a periodic
   `--max-iters 100` durability pass. Re-baseline golden references only as a deliberate, reviewed step.
5. **Prefer the simpler / more-accurate / differentiable form.** When a faithfulness contrivance and a
   cleaner form differ only at the ULP level (smooth vs hard `min/max`, accurate vs oracle-truncated `D_v`,
   float64 vs replicated `real*4`, smooth vs NumPy flux limiter), take the cleaner one and re-validate under
   the tiered standard — that is the whole point of the numerical-accuracy refactor.
6. **Porting a genuinely new subsystem** (COAMPS or SILHS — the only unported pieces): the historical
   technique still applies — read the Fortran oracle, mirror its path under `src/`, and validate with a
   case-stats oracle (feed the Fortran's own state into the JAX routine) or a conservation contract, since
   the f2py API exposes no microphysics. See DESIGN.md "Verification oracles."
