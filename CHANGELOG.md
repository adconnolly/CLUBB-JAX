# CLUBB-JAX Changelog

Condensed work record. The full per-iteration history is recoverable from git; this file keeps the human-readable
arc. For design, the correctness standard, and conventions see `DESIGN.md`; for the per-file port map see
`TRANSLATION_STATUS.md`. Append one concise entry per session at the bottom of "Recent work".

---

## Numerical-accuracy refactor — final status

The refactor relaxed the **bit-faithfulness** gate to a **tiered numerical-accuracy standard** (DESIGN.md
"Correctness standard") to favor differentiability, simplicity, and accuracy. Outcome:

- **Differentiable (entirely in JAX): all 19 cases.** Whole-driver reverse-mode `jax.grad` through one
  `advance_clubb_to_end` step is finite + finite-difference-correct — gated by `compare_grad.py`. Achieved by a
  tracer-transparent toolkit (`CLUBB_core/tracer_numpy.py`: `_asarray`/`_xp`/`_iset` route to jnp under a trace,
  exactly numpy otherwise → normal runs bit-identical; `_safe_sqrt`/`_safe_pow` for clip-sqrt/fractional-pow), plus
  block-level tracer dispatch and diagnostic-skip / detach-under-trace for post-core diagnostics.
- **Faithful (vs Fortran): 20/20 DEFAULT_CASES PASS Tier-C** (19 strictly bit-faithful + mpace_a within Tier-C).
  Every differentiability change was forward-identical → zero faithfulness regression. The accuracy-lowering
  contrivances were removed (`parabolic_expax`, Morrison `real*4`, BUGSrad `sngl`/float32-π) → strictly more
  accurate there.
- **Tooling added:** tiered `validation.py`/`validate_case.py`, Tier-A `invariants.py`/`test_invariants.py`,
  golden-trajectory regression (`golden.py`/`update_golden.py`), `compare_grad.py`/`probe_driver_grad.py`,
  `mirror_audit.py`, `diagnose_divergence.py`.

---

## Condensed history (chronological)

**Phase 1 — Incremental port (2026-05-27 → 05-28, Iters 1–65).** Built the port routine-by-routine with in-loop
shadow comparison (run JAX beside the Fortran oracle, match to machine epsilon, remove the Fortran call): core grid
operators + LHS/RHS terms, the four advance functions (xm_wpxp / xp2_xpyp / wp2_wp3 / windm_edsclrm), the ADG1 PDF
closure, pre-advance diagnostics, and the initialization ports (hydrostatic, rcm_sat_adj, calculate_thvm). End
state: **zero Fortran calls in the advance loop**.

**Phase 2 — Cross-case faithfulness (→ 20 bit-faithful cases).** Extended beyond ARM by porting each case's
forcing/surface scheme and fixing per-case seeds (BC/thv/rc_coef, time-dependent wind forcing, bulk surface
schemes, cloud-droplet sedimentation, the cold-cloud `ice_supersat_frac`, the per-step `wm_zm` recompute). Added
gabls3 (BUGSrad correlated-k radiation + interactive soil_vegetation), the Morrison cases (mpace_a), and the
clex9 altocumulus pair. Characterized the FP-/oracle-limited cases (rico, coriolis_test, nov11_altocu,
dycoms2_rf02_morr/do/ds) as not-bugs.

**Phase 3 — Completeness loop (in-scope routine coverage).** Ported every remaining in-scope, oracle-validatable,
self-contained routine and unit-tested each: the KK PDF-integral mixed-moment machinery, BUGSrad cloud-overlap,
`ice_dfsn`, the GFDL droplet-activation CLUBB side, `inverse_hydrostatic`, all alternative PDF closures
(ADG2/LY93/3-D Luhar/new/TSDADG/new-hybrid, f2py end-to-end), `remapping_module` (both methods), and all benchmark
surface/forcing schemes. Brought in the CGILS/cloud_feedback family — Press[Pa]-sounding→altitude +
absolute-`T[K]`→θ init + the case-specific BUGSrad extended atmosphere + the forcing out-of-range zero-fill fix —
so cgils_s11/s12 reach Tier-C PASS and are differentiable.

**Phase 4 — Numerical-accuracy refactor.** Relaxed the gate to the tiered standard, removed the accuracy-lowering
contrivances, and made the whole driver `jax.grad`-differentiable (the tracer-transparent toolkit + conventions).
See "final status" above.

**Phase 5 — Mirror-refactor loop (2026-05-29 → 06-08, iters to ~1354).** Made the JAX *file and routine names*
mirror the `.F90` oracle without changing any numbers (every step byte-identical): relocated inlined subroutines to
their Fortran-home modules, un-inlined the pdf_closure / advance_xp2_xpyp / advance_xm_wpxp drivers, retired the
`_jax` suffix for single-subroutine mirrors and all shadow-comparison/iteration-tag scaffolding, deduplicated
constants against `constants_clubb`/`parameters_KK`, and aligned routine casing to the exact Fortran. Built
`mirror_audit.py` (a reproducible 7-dimension name/file/dir diff → PASS) with standing test guards
(`test_mirror_audit`, dead-import/function/config/param-roundtrip, source-grounded excusal tripwires). Ran a
behavioral-validation campaign: ~100 named mirrors now carry direct f2py-oracle bit-shadow unit tests, and every
load-bearing data structure (the 102 tunable params, the 67 config flags, the physical constants, the derived-type
field layouts) is directly pinned. Converged to a single deliberately-deferred routine, `pdf_closure_driver_zm`.

---

## Recent work

### 2026-06-19 — JAX-vs-Fortran speed assessment + persistent JIT cache (`main`)
- **Environment reconstruction.** This checkout arrived with no jax, an empty `clubb_release` submodule, and no
  compiled oracle. Rebuilt all of it: jax venv (jax 0.10.2 + numpy + netCDF4 + tabulate) at
  `/burg/home/ac5006/scratch/jaxenv`; `git submodule update --init clubb_release`; compiled the intel `ifx` +
  f2py oracle (`compile.py -python -precision double` → `install/intel_PRECdouble_PYTHON`). Two build blockers
  fixed: (1) cmake 3.28 module had a GLIBCXX mismatch → used a pip-wheel cmake; (2) every `ifx` Fortran link
  failed on `for_alloc_allocatable_handle` because a stale `parallel_studio_xe_2020` ifort runtime shadowed the
  HPCKit `libifcore` on `LIBRARY_PATH` → prepend `HPCKit/.../compiler/lib/intel64_lin`. The module's
  netcdf-fortran is ifort-built (ifx can't read its `.mod`), so CMake correctly FetchContent-builds netcdf-fortran
  with ifx. (`run_bindiff_all.py` needs `tabulate`.)
- **Comparison runs.** `run_jax_vs_fortran_cases.py` + `run_scm.py -jax`: ARM/30 steps is **bit-faithful** (PASS,
  0 vars over 1e-7). Speed: **Fortran 2.0 s vs JAX cold 133–147 s** — confirmed compile-dominated (244 XLA
  compiles via `JAX_LOG_COMPILES`), matching the DESIGN performance finding.
- **Fix applied (DESIGN Next Steps, step 1).** Enabled JAX's **persistent on-disk compilation cache** at import in
  `advance_clubb_core_module.py` (default `~/.cache/clubb_jax_jit`; `CLUBB_JAX_NO_JIT_CACHE=1` to disable,
  `JAX_COMPILATION_CACHE_DIR` to relocate). Cross-process cache hit → **warm 70–78 s (~47% faster)**; numerically
  transparent (ARM still bit-faithful, grad unaffected — cached executable is byte-identical). The residual ~70 s
  is tracing+MLIR-lowering of the 244 leaves; the whole-step-jit / `lax.scan` restructuring remains the open lever.

### 2026-06-19 — `formatting_and_jitting` branch (WIP)
- **Branch scope:** a formatting / Fortran-comment-copying pass; **JIT-friendly derived types** (`derived_types/`
  ConfigFlags/Grid/pdf_parameter made jit-static); a new **pure-JAX `src/io/` init path**
  (`namelist.py`/`sounding.py`/`surface.py`/`grid_file.py` + `derived_types/converters.py`); reintroduced the JAX
  `prescribe_forcings`; **removed `clubb_api` usage except for stats**; shrank stats buffers (memory/runtime win).
- **Status:** 12/20 DEFAULT_CASES bit-PASS; 7 cases fail-loud on features the new init path hasn't rewired
  (Morrison microphysics, `l_cloud_sed`, `l_soil_veg`, sponge); dycoms2_rf01 runs fine standalone (its
  `compare_cases` "crash" was node contention, not a bug). **Whole-step jit is NOT done** — only ~23 leaf
  functions are jitted while `advance_clubb_core`/the timestep loop stay eager, so first-call JIT compilation
  dominates runtime (full analysis + fix direction in DESIGN.md "Remaining Work → `formatting_and_jitting`").
- **This session:** fixed **35** signature-drift unit tests (refactor added leading `nzm/nzt/ngrdcol/gr` args,
  reordered args, `static_argnums`) + 2 src fixes (`set_sfc_value_of_flux_profiles` zeros `wpedsclrp`
  unconditionally; `Input_fields/sounding.py` stale `calculate_thvm` call). **12 tests still failing** — 6
  structural-audit guards (new `io/` files + the stats-`clubb_python` keep trip the mirror/100%-JAX guards) and 6
  genuine numerical/grad/oracle (see DESIGN.md next-step queue).

### 2026-06-08 — Documentation condensation
- Reviewed + corrected `TRANSLATION_STATUS.md` against the live repo and condensed it (392 → 218 lines): dropped
  the per-iteration narrative, replaced the stale "Total 202 / ported 128" headline with the current
  `mirror_audit.py` framing (PASS, 1 DEFERRED, 296 source files scoped out by subsystem), kept the Fortran↔JAX
  mapping tables with one-line notes.
- Condensed `DESIGN.md` (1214 → ~280 lines): cut ~600 lines of iter-NNN narrative blockquotes; preserved the
  operational core (repo structure, test instructions, verification oracles, gates, divergence/durability/grid-type
  guidance, differentiability conventions, the tiered correctness standard, critical conventions, remaining work,
  agent rules); pointed "What Has Been Built" at `TRANSLATION_STATUS.md`.
- Condensed this changelog (5547 → ~110 lines): collapsed 256 dated iteration entries into the phase summaries
  above (full detail remains in git).
- Proposed a condensed `CLAUDE.md` in `CLAUDE_PROPOSAL.md` for review (fixes stale refs: `generic_forcings.py` →
  `prescribe_forcings.py`, `radiation.py` → `radiation_module.py`, test-file count).
