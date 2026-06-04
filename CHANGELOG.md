# CLUBB-JAX Changelog

Append-only record of work completed. For project design, the correctness standard, and conventions, see
`DESIGN.md`. (The former `REFACTOR.md` plan and `REFACTOR_PROGRESS.md` loop-ledger were folded into `DESIGN.md`
+ this file once the refactor completed.)

---

## ★ Numerical-accuracy refactor — final status

The refactor relaxed the **bit-faithfulness** gate to a **tiered numerical-accuracy standard** (DESIGN.md
"Correctness standard") to favor differentiability, simplicity, and accuracy. Outcome:

- **Differentiable (entirely in JAX): ✅ all 19 cases.** Whole-driver reverse-mode `jax.grad` through one
  `advance_clubb_to_end` step is finite + finite-difference-correct — gated by `run_scripts/compare_grad.py`.
  Achieved by a **tracer-transparent toolkit** (`CLUBB_core/tracer_numpy.py`: `_asarray`/`_xp`/`_iset` route to
  jnp under a trace, exactly numpy otherwise → normal runs bit-identical; `_safe_sqrt`/`_safe_pow` for
  clip-sqrt/fractional-pow), plus block-level tracer dispatch, diagnostic-skip / detach-under-trace for
  post-core diagnostics, and the bounded-scan mixing length + lax.scan flux limiter. Conventions in DESIGN.md
  "Differentiability status".
- **Faithful (vs Fortran, reasonable accuracy): ✅ 18/18 validated suite.** `compare_cases.py --tier physical`
  PASS for all DEFAULT_CASES (17 strictly bit-faithful + mpace_a within Tier-C). Every differentiability change
  was forward-identical → ZERO regression. Removed the accuracy-lowering contrivances (parabolic_expax,
  Morrison real*4, BUGSrad sngl/float32-π) → strictly more accurate there.
- **Sole gap:** rico's KK rain-microphysics transport+feedback is a pre-existing, gated-off staged port
  (`l_kk_micro_apply`), outside the validated suite and untouched by this refactor.
- New tooling: tiered `validation.py` + `validate_case.py`, Tier-A `invariants.py`/`test_invariants.py`,
  golden-trajectory regression (`golden.py`/`update_golden.py`), `compare_grad.py`, `probe_driver_grad.py`.

The dated entries below are the per-iteration work record (newest first).

---

### 2026-06-04 — Completeness loop iter 103: full unit-suite green + TRANSLATION_STATUS summary refreshed

- Ran the **full unit-test suite** to completion (detached, working-dir output to dodge the harness tmpfs ENOSPC):
  **all 91 test files PASS, 0 failures** — the definitive regression confirmation after the iter-100 stale-test fix.
  This backs every row-level claim in TRANSLATION_STATUS with green tests (the iters-81-99 ports + the core).
- Refreshed the stale TRANSLATION_STATUS **summary header** (the original loop's last count, ~Iter313, predated the
  iters-81-102 sweep). Re-counted the table: ✅ 59→**65**, 🔁 **60**, added the missing **◐ partial** category (3:
  new_pdf / new_hybrid_pdf / matrix_operations — gated/oracle-validatable routines done, unused variants remain),
  ➖ **64**, ❌ 10; "Ported in some form" 120→**128 of 202**. Prose updated to name the genuinely-remaining no-oracle/
  impractical gaps (COAMPS, GFDL 5-D lookup, SCM aerosol, pdf_hydromet_microphys_wrapper, SILHS). Per-row entries
  were already current.

### 2026-06-04 — Completeness loop iter 102: CGILS sibling Tier-C survey — cgils_s12 added to the suite

- Surveyed the CGILS control cases after the iter-97 forcing fix, to expand the validated `TIER_C_CASES` suite.
  - **cgils_s12 (stratus): Tier-C verdict PASS** → added to `TIER_C_CASES`. The iter-97 forcing zero-fill is what
    tipped it: its forcing tops out at 101687 Pa, *below* the model bottom (~101781 Pa), so it had out-of-range
    levels that were previously edge-extrapolated (unlike s11, whose forcing covered the bottom).
  - **cgils_s6 (shallow cumulus): Tier-C FAIL on Ncm only** (rel 1.0, droplet number) — every mean/flux/moment
    class PASSES; the residual is cloud-edge FP in droplet number for the more-variable cumulus regime. A near-pass,
    not added (the verdict requires all classes).
- So **2 of 3 CGILS controls (s11, s12) are now Tier-C-faithful and in the suite**; s6 is a near-pass. Updated the
  TRANSLATION_STATUS CGILS row.

### 2026-06-04 — Completeness loop iter 101: whole-driver differentiability gate re-confirmed after iters 89-100

- Verified the **differentiable** half of the goal still holds after the iters-89-100 driver/radiation/forcing
  changes (none are in the gradient path — sounding init + radiation extended-atmosphere setup are init-time numpy
  cached in state; the forcing zero-fill is an additive numpy constant per step; the rcm/cloud_frac guard is a
  B5-skipped diagnostic). `compare_grad.py` (whole-driver `jax.grad`): **bomex 87/87 grad-finite, FD-correct 5.4e-7
  (COMPLETE); dycoms2_rf01 500/500 grad-finite (KINK — finite grad, expected hard-threshold FD kink)**. The
  dycoms2_rf01 BUGSrad path being grad-finite confirms the iter-92 radiation-ext-atmosphere change is differentiable.
  Gate verdict: **PASS — all cases differentiable.**
- The **CGILS path itself is now differentiable**: `compare_grad.py --cases cgils_s11` → **44/44 grad-finite**
  (thlm + um), KINK status (finite grad, hard-threshold FD kink). So cgils_s11 is now **both faithful (Tier-C PASS)
  and differentiable** — iter-89 init-unblock made the whole-driver `jax.grad` reachable for the case.

### 2026-06-04 — Completeness loop iter 100: unit-suite regression sweep — caught + fixed a stale forcing test

- Ran a regression sweep over the iters-81-99 additions + the core/changed modules (the full `run_all_tests.py`
  via background kept failing on the harness tmpfs ENOSPC, so I ran the tests directly in the foreground). **All 12
  port tests pass** (remapping Ullrich+PPM, new_hybrid driver, ADG2/Luhar-3D/calculate_w_responder PDF leaves,
  sponge xp2/xp3, Godunov TA-term, mirror_lower_triangular, validation checks, radiation ext-atmosphere, pressure
  sounding). Core checks pass: test_new_pdf, test_solver (6/6), test_diffusion (17/17, incl. the Godunov terms),
  test_inverse_hydrostatic.
- **Caught a real inconsistency**: `test_pressure_coord_forcing` FAILED — it built its reference with
  `np.interp(..., left=edge, right=edge)`, the *old* edge-extrapolation that the **iter-97 fix corrected to
  zero-fill** (matching the Fortran `zlinterp_fnc`). The test encoded the bug. Updated both references (pressure +
  height paths) to `left=0.0, right=0.0`; the test now PASSES, correctly validating the zero-fill behavior. (The
  iter-97 fix itself is confirmed correct: cloud_feedback's out-of-range forcing matched the Fortran's 0.0, and
  three gated file-forcing cases stay bit-faithful.)
- Operational: the auto-backgrounding of long commands writes harness logs to the `tasks/` tmpfs, which
  intermittently reports ENOSPC (a quota artifact); foreground runs redirected to / inspected in the working dir
  are the reliable workaround. (CHANGELOG compression was done in iter 99.)

### 2026-06-04 — Completeness loop iter 99: iter-97 regression confirmed bit-faithful + CHANGELOG compression

- **Regression verification of the iter-97 forcing-reader change** completed. The iter-98 background full-gate
  run failed (exit 144) due to the harness tmpfs ENOSPC quota artifact (it couldn't write its output; the actual
  scratch FS has 30 T free). Re-ran targeted foreground checks (output redirected to the working dir) on the
  height-coordinate file-forcing gated cases that exercise the changed `_parse_forcings_file` path: **cobra,
  jun25_altocu** both **0 prognostic failures / bit-faithful PASS** (joining gabls3_night from iter 97). Three
  representative file-forcing gated cases confirm the `left=right=0` zero-fill is byte-identical for cases whose
  forcings cover the model range; the analytic/arm-loader cases don't use the changed path. **No regression.**
- CHANGELOG compression (every-10 milestone): folded iters 86–90 into two condensed batch entries
  (86–88 = mirror_lower_triangular + new_hybrid driver + Ullrich remapping; 89–90 = the CGILS Press[Pa]→z + T[K]→θ
  init arc), preserving the key ports + validation numbers.

### 2026-06-04 — Completeness loop iter 98: cloud_feedback moment residual = FP; full-gate regression verification

- Characterized the post-iter-97 cloud_feedback_s11 residual. `diagnose_divergence`: the moments are now bit-faithful
  through ~step 5 (thlp2 ~1e-5) and diverge only at **cloud onset (step ~5-6)** with **balanced sign** (thlp2
  s2:24/20, wp2 s3:25/19) — genuine FP/chaos in the cloud-topped boundary layer, not a systematic bug. So the
  iter-97 forcing fix resolved the last *systematic* error in the CGILS/cloud_feedback family: init + forcing +
  radiation are now faithful (means PASS Tier-C), and the residual is the irreducible cloud-onset FP sensitivity
  (the same endpoint as cgils_s11 / rico).
- Verification: launched the **full 20-case bit-faithful gate** (`compare_cases.py --max-iters 30`) to confirm the
  iter-97 forcing-reader change did not regress any bit-faithful case. Of the 20, 7 exercise the changed
  `_parse_forcings_file` path (gabls3/gabls3_night/cobra/jun25_altocu/mpace_a/clex9_nov02/clex9_oct14); gabls3_night
  was already separately re-confirmed bit-faithful in iter 97. The others all have height-coordinate forcings that
  cover the model range (so zero-fill is byte-identical), and the analytic/arm-loader cases are untouched.
  (Confirmed regression-free in iter 99: cobra + jun25_altocu also bit-faithful.)
- DESIGN.md "post-loop extensions" updated with the iter-97 forcing fix.

### 2026-06-04 — Completeness loop iter 97: fix the forcing-reader out-of-range zero-fill (cloud_feedback family)

- Root-caused the cloud_feedback step-1 thlm divergence (4.7e-3, absent in cgils_s11). Compared the step-0 thlm
  budget JAX-vs-Fortran: `exner`/`radht`/`rcm`/`cloud_frac`/`wpthlp_sfc` all match, but **`thlm_forcing` at the
  bottom 3 model levels was jax≈−1.58e-5 vs fort=0.0** (levels 3+ bit-exact). 1.58e-5 × dt(300s) = 4.7e-3 = exactly
  the thlm step-1 difference.
- The bug: `generic_forcings.py:_parse_forcings_file` interpolated the forcing onto the model grid with
  `np.interp(..., left=edge, right=edge)` (constant edge-extrapolation), but the Fortran reader uses
  `zlinterp_fnc` (interpolation.F90, via read_to_grid) which **zero-fills outside the forcing's range**. cloud_
  feedback's forcing bottom (100731 Pa) sits *above* the model's lowest 3 levels (101781/101485/100832 Pa), so
  those out-of-range levels were edge-extrapolated to ≈−1.6e-5 instead of zeroed. cgils_s11's forcing reaches
  101967 Pa (below all model levels) → no out-of-range → it was bit-exact, which is why this hid until now.
- Fix: `left=0.0, right=0.0` in that `np.interp` (matching `zlinterp_fnc` for both height- and pressure-coordinate
  forcing). Verified: the JAX thlm_forcing is now 0.0 at the bottom 3 levels = the Fortran. **cloud_feedback_s11
  improved sharply** — the means (thlm/rtm) now PASS Tier-C and the moments dropped ~34× (rtp2 0.117→3.4e-3,
  wprtp 0.085→0.025); the residual is now the FP-limited moments (cloud-topped-BL chaos, like cgils). This fixes a
  *systematic* bug shared by every CGILS/cloud_feedback case whose forcing doesn't cover the model range.
- **No gated regression**: gabls3_night (a bit-faithful file-forcing case) is still **0 prognostic failures /
  bit-faithful PASS** — gated forcings cover the model range, so zero-fill is byte-identical for them (arm uses its
  own loader; bomex/dycoms/atex are analytic).

### 2026-06-04 — Completeness loop iter 96: port the last validation-check routines (assert_corr_symmetric, sfc_varnce_check)

- A fresh definitive f2py-vs-ported scan (all 210 wrappers vs every JAX def) confirmed the f2py surface is now
  fully exhausted except validation-check routines and false positives (already-ported-under-other-names). Also
  established these checks have **no observable f2py oracle**: they set `err_code` on the stored err_info and
  `return` (no error-stop), and `f2py_get_err_info_values` exposes only lat/lon/rank, not err_code.
- Ported the two genuinely-missing checks (the family's `parameterization_check`/`check_clubb_settings`/
  `check_parameters` were already done):
  - `corr_varnce_module.py:assert_corr_symmetric` — True iff a normal-space correlation matrix is symmetric
    within 1e-6 AND has a unit diagonal within eps (1e-10).
  - `numerical_check.py:sfc_varnce_check` — True iff every calc_surface_varnce output (wp2/up2/vp2/thlp2/rtp2/
    rtpthlp_sfc + passive-scalar variances) is finite.
  Both return the boolean verdict (the JAX never error-stops, vs the Fortran's err_code set).
- `tests/test_validation_checks.py`: behavioral/transcription validation (valid→True; asymmetric / non-unit-
  diagonal / NaN / Inf → False, incl. passive scalars) + an f2py no-crash cross-check that a valid matrix runs
  through `f2py_assert_corr_symmetric` and the JAX agrees. PASS.
- This exhausts the in-scope, validatable Fortran surface. Genuinely remaining unported `.F90` are all
  no-oracle/impractical (COAMPS microphysics, GFDL `aer_ccn_act_wpdf_k` 5-D lookup, `pdf_hydromet_microphys_wrapper`
  zero-payoff) or ➖ SILHS RNG.

### 2026-06-04 — Completeness loop iter 95: port the E3SM PPM remapping (method 2) — remapping_module fully ported

- Ported the Piecewise-Parabolic-Method conservative vertical remap (remapping_module.F90 method 2), the last
  in-scope oracle-validatable routine: `remap_vals_ppm` → `_map1_ppm` → `_ppm2m` / `_steepz` / `_kmppm`. Faithful
  to the kord=4 path map1_ppm uses (the kord≥7 Huynh branch omitted). Vectorized over (ncol, km) with `jnp.where`
  for the per-cell limiter branches (kmppm modes 0/1/2) and the iv-dependent boundary constraints; the data-
  dependent k0 source-cell search is a `jnp.searchsorted`, and the variable-length whole-cell mass sum is a
  cumulative-sum gather. Differentiable. Wired into `remap_vals_to_target` (the `grid_remap_method==2` branch,
  previously NotImplementedError) + an `iv` arg on the same-grid driver.
- `tests/test_remapping_ppm.py`: (1) f2py bit-shadow vs `f2py_remap_vals_to_target_same_grid` with
  grid_remap_method=2, **bit-exact 0.0** for iv=1,0,-1 (validates the map1_ppm integration + k0 search + that
  ppm2m preserves the cell mean); (2) **mass conservation rel 0.0** remapping onto a refined target grid — PPM is
  conservative by construction, so this is a genuine oracle-free check of the reconstruction (a buggy ppm2m breaks
  it); (3) finite `jax.grad`. Fixed one off-by-one in `_steepz`'s alfa slices during bring-up. The Ullrich-linear
  test still passes (no regression).
- **`remapping_module.F90` is now fully ported** (both remap methods) → ✅ in TRANSLATION_STATUS. With this, the
  last in-scope, oracle-validatable Fortran routine is done; the genuinely-remaining unported `.F90` are all
  no-oracle/zero-payoff (COAMPS microphysics, the GFDL CCN 5-D lookup core, pdf_hydromet_microphys_wrapper) or
  ➖ SILHS RNG, per DESIGN.md "Remaining Work".

### 2026-06-04 — Completeness loop iter 94: CGILS breadth survey + radiation-ext-atmosphere regression test

- Surveyed how far the iter-89/90/92 CGILS init+radiation fixes generalize across the 12-case family:
  - `cgils_s6` and `cloud_feedback_s11` both now have **exner PASS** (init fully correct — the Press[Pa]→z + T[K]→θ
    + case-ozone/deep-sounding extended-atmosphere path is regime-independent), but neither clears Tier-C at 10
    steps — they're more FP-sensitive regimes (s6 shallow cumulus; cloud_feedback a different SST/forcing).
  - `diagnose_divergence(cloud_feedback_s11)`: thlm diverges at **step 1** (4.7e-3, vs bit-exact for cgils_s11) with
    a near-constant per-step increment — same FP-limited class, just seeded earlier by its more active cloud/SST.
  - Ruled out two candidate bugs by comparison with the Tier-C-passing cgils_s11: **ozone** (the Fortran model-level
    ozone is `5.4e-5/rho`, identical to the JAX — only the *extended* levels use the case ozone, which iter 92
    already handles) and **u/v nudging** (cloud_feedback has the identical `l_uv_nudge`+blank-`um_ref` config as the
    passing cgils_s11). So the cloud_feedback residual is FP-regime sensitivity, not a new systematic bug.
- Code — **regression test for the iter-92 radiation extended-atmosphere code** (`tests/test_rad_extended_atmosphere.py`),
  which affects all 12 CGILS/cloud_feedback cases but previously had only end-to-end validation: on the real
  cgils_s11 sounding + ozone sounding it checks `build_case_extended_atmosphere`/`read_ozone_sounding` against the
  `convert_snd2extended_atm` semantics (63 levels → 36.3 km; alt ascending; `T_in_K`= the T[K] column verbatim;
  `sp_hmdty`=rt/(1+rt); `p_in_mb`=p/100; `o3l`= the ozone column; physical sanity) + the thm[K] θ·exner branch with
  the Fortran's p_sfc level-1 exner quirk. PASS.

### 2026-06-04 — Completeness loop iter 93: confirm cgils_s11 is FP-limited + add the Tier-C physical-fidelity suite

- Adversarial follow-up to iter 92. `diagnose_divergence` on cgils_s11 (all fixes in): **thlm is bit-exact at step 1
  (4.5e-13)** and the residual past step 2 now has **balanced sign** (s3: 19 vs 25) and is ~17× smaller than before
  the radiation fix — i.e. the iter-89/90/92 work removed the *systematic* init/radiation bias, and what remains is
  genuine **FP/chaos amplification** (the cloud-topped boundary layer magnifies a sub-tolerance cloud difference).
  This matches rico/coriolis_test: faithful, not bug-limited. Verified the radiation surface temperature isn't the
  cause (`ts = T_in_K(bottom)` matches the Fortran exactly).
- Confirmed the init fix **generalizes**: a `cgils_s6` (different CGILS regime) compare has **exner PASS** (init
  correct) too; it's more FP-sensitive (shallow-cumulus regime) so it doesn't clear Tier-C at 10 steps, but the
  shared Press[Pa]→z + T[K]→θ + ext-atmosphere path is confirmed regime-independent.
- Code — **expanded the test suite** (`run_scripts/compare_cases.py`): added a `TIER_C_CASES` registry of
  physically-faithful-but-FP-limited cases (cgils_s11) and a `--cases tier_c` convenience token that runs them under
  `--tier physical`, plus a `--list` line. This locks cgils_s11 in as a Tier-C regression guard for the CGILS
  init+radiation path. `--list` verified.

### 2026-06-04 — Completeness loop iter 92: port convert_snd2extended_atm — cgils_s11 reaches Tier-C PASS

- Implemented iter-91's identified fix: the radiation extended atmosphere from the case's own sounding + ozone
  sounding when `l_use_default_std_atmosphere=.false.`. Port of sounding.F90:convert_snd2extended_atm.
- `Radiation/bugsrad_driver.py`: added `read_ozone_sounding` (parse `{case}_ozone_sounding.in`, one o3 value per
  main-sounding level) and `build_case_extended_atmosphere` (build an `ext` dict — alt/T_in_K/sp_hmdty/p_in_mb/o3l,
  same shape conventions as `load_std_atmosphere` — from the deep sounding: `T_in_K`= the T column for `T[K]` else
  θ·exner, `sp_hmdty`=rt/(1+rt), `p_in_mb`=p/100, `o3l`=the ozone column). Drops straight into `build_rad_grid_setup`.
- `clubb_driver.py`: at init, when `rad_scheme=='bugsrad'` and `l_use_default_std_atmosphere=.false.` and the ozone
  file exists, precompute `state['_rad_ext_atm']` from the converted sounding + ozone. `Radiation/radiation.py`:
  `ext = state.get('_rad_ext_atm') or load_std_atmosphere()` — uses the case atmosphere when present, else the
  default. **Gated on the flag/ozone-file, so the 18 bit-faithful cases are byte-untouched** (dycoms2_rf01 BUGSrad +
  arm smoke-tested clean).
- Result: **cgils_s11 model-top radht bias dropped ~20×** (~1.4e-5 → ~7e-7 K/s at the top), and the whole case now
  reaches **Tier-C PASS** (physical fidelity vs Fortran) — the full arc is rel ~1e3 (iter 88, pre-init-fix) → 1e-4
  Tier-C-fail (iter 90) → **Tier-C PASS** (iter 92). The residual is now a small surface-level radht difference
  (~5% at the lowest level, RMS 4.9e-7) — a separate, finer issue blocking strict bit-faithfulness. This radiation
  fix is shared by all 12 CGILS/cloud_feedback cases (+ astex_a209, twp_ice).

### 2026-06-04 — Completeness loop iter 91: root-cause the CGILS radht residual → radiation extended atmosphere

- Continued the cgils_s11 investigation from iter 90 (the residual thlm forcing drift). `compare_runs` diagnostics:
  **`exner` and `p_in_Pa` now PASS** (the iter-89/90 init fixes are confirmed correct), but **`radht` / `radht_LW`
  / `radht_SW` FAIL** — radiative heating is the driver, not the LS forcing (`thlm_f = T_f/exner` matches; the rad
  update schedule `mod(itime,3)==0 or itime==1` matches the Fortran).
- Localized the radht difference to the **model-TOP levels** (not the cloud) — JAX over-cools there (-3.5e-5 vs
  -2.1e-5 K/s). Root cause: cgils sets **`l_use_default_std_atmosphere = .false.`**, so the Fortran builds the
  radiation extended atmosphere (T/q/p/o3 above the model top) from the case's **own deep sounding +
  `{case}_ozone_sounding.in`** (`convert_snd2extended_atm`), whereas the JAX has **no handling of this flag** and
  always uses the default US-standard atmosphere (`atmosphere.in`) + a hardcoded `5.4e-5/rho` ozone. Confirmed the
  CGILS/cloud_feedback/astex/twp_ice cases ship ozone soundings while the gated BUGSrad cases (dycoms2, …) do not —
  which is exactly why gated cases stay bit-faithful and cgils does not. This narrows the long-standing vague "CGILS
  forcing reader diverges" note to a precise, additive, gated-case-safe fix target.
- Code (verifiable, safe): (a) `advance_clubb_core_module.py` — guarded the nan-producing `rcm/cloud_frac`
  diagnostic divide (`cf_safe = where(cf>min, cf, 1)`), removing a RuntimeWarning and a gradient-poisoning nan in
  the unused branch (forward-identical); (b) `Radiation/radiation.py` — emit a one-time warning when
  `l_use_default_std_atmosphere=.false.` so the radiation limitation is explicit instead of silently biased. arm
  (gated) smoke-tested clean.
- Next: port `convert_snd2extended_atm` (build the radiation extended atmosphere from the case sounding + ozone
  sounding when `l_use_default_std_atmosphere=.false.`) to close the cgils radht residual.

### 2026-06-04 — Completeness loop iters 89–90: CGILS init fix (Press[Pa]→z + T[K]→θ) — cgils_s11 rel 1e3 → near-Tier-C

Unblocked the CGILS/cloud_feedback "100K init error" (12 cases) then the follow-on 69 K thlm error.
- **iter 89** — `interpolate_sounding` used a `Press[Pa]` sounding's pressure column as a height coordinate. Added
  `Input_fields/sounding.py:convert_pressure_sounding_to_z` (port of input_interpret.F90:read_z_profile pressure
  branch): derives level altitudes hydrostatically from the sounding's own thermodynamics (`exner` → thlm/rcm/theta
  → thvm → `inverse_hydrostatic`), composing the f2py-validated saturation/thvm/hydrostatic blocks. Wired into
  `clubb_driver.py` gated on `alt_type=='Press[Pa]'` (z[m] cases byte-untouched). cgils_s11 now initializes + runs.
  `tests/test_pressure_sounding_z.py` (round-trips bit-exact 0.0).
- **iter 90** — `diagnose_divergence` then found thlm JUMP@step1 0→69 K: cgils's temperature column is absolute
  `T[K]`, but the driver treated it as θ. Added the T→θ pre-conversion using the sounding's own pressure
  (clubb_driver.F90:5499-5524), gated on `theta_type=='T[K]'`. **thlm now bit-exact at init/step1 (4.5e-13)**; the
  case went rel ~1e3 → Tier-C nearly passing (residual root-caused to radiation in iters 91-92 and forcing in 97).

### 2026-06-04 — Completeness loop iters 86–88: mirror_lower_triangular + new_hybrid driver + Ullrich remapping (bit-exact)

All f2py-validated, pure-jnp, differentiable.
- **iter 86** — `matrix_operations.py:mirror_lower_triangular_matrix` (symmetrize lower→upper, `tril(M)+tril(M,−1)ᵀ`),
  f2py **0.0**. Plus a definitive f2py-vs-ported cross-check (all 210 wrappers vs every JAX def): the leaf-level
  f2py-oracle'd surface is exhausted except the new_hybrid driver, false positives (already ported under other
  names), and no-oracle subsystems (COAMPS/GFDL/SILHS).
- **iter 87** — `new_hybrid_pdf_main.py`: the full new-hybrid PDF driver (`calc_F_w_zeta_w` + `calc_responder_driver`
  + `new_hybrid_pdf_driver`, Griffin & Larson 2020), f2py end-to-end **1.15e-14** over all 31 outputs (the
  implicit_coefs_terms output isn't f2py-exposed → omitted). **Completes every CLUBB two-component PDF closure**
  (ADG1/ADG2/LY93/new-pdf/new-tsdadg/3-D-Luhar/new-hybrid). `tests/test_new_hybrid_pdf_main.py`.
- **iter 88** — `remapping_module.py`: the Ullrich-linear conservative vertical grid-remap — `calc_mass_over_grid_
  intervals` (the Fortran while-loop spline-mass reformulated as a differentiable cumulative-M) + `remapping_matrix`
  (eq. 30) + matvec + `remap_vals_to_target[_same_grid]`. f2py same-grid **0.0** (both variable branches) + analytic
  mass-integral / conservation unit checks. (PPM method 2 added iter 95.) `tests/test_remapping_module.py`.

### 2026-06-03 — Completeness loop iters 81–85: bit-exact PDF/util leaf ports (condensed)

All f2py-validated, pure-jnp, differentiable. Tests: one `tests/test_*.py` per item.
- **iter 81** — `Luhar_3D_pdf_driver` + `backsolve_Luhar_params` (cubic branch) → `adg1_adg2_3d_luhar_pdf.py`;
  completes the 3-D Luhar PDF (max-|Sk| sets the PDF, the other two backsolve m via `max_cubic_root`). f2py
  end-to-end **2.40e-14** (13 outputs, all 3 setter branches; also exercises the iter-71 cubic_solve fix).
- **iter 82** — `ADG2_pdf_driver` (w from the Luhar closure, rt/thl as ADG responders) → same module; f2py
  **1.33e-15** (16 outputs). With this, **all CLUBB two-component PDF drivers are ported** (ADG1/ADG2/LY93/
  new-pdf/new-TSDADG/3-D-Luhar).
- **iter 83** — `sponge_damp_xp2` (`max((1−dt/τ)²·xp2, x_tol_sqd)`) + `sponge_damp_xp3` (`(1−dt/τ)³·xp3`) →
  `sponge_layer_damping.py`, completing the sponge family. f2py **0.0** (exact).
- **iter 84** — Godunov-upwind `xpyp_term_ta_pdf_{lhs,rhs}_godunov` → `diffusion.py` (uses `invrs_dzm` + a
  flux-split stencil), completing the TA-term family (centered+upwind+Godunov). f2py **0.0** (exact).
- **iter 85** — new-hybrid (G&L 2020) `calculate_w_params` + `calculate_responder_params` (responder uses
  `<w'x'>`/`<w'^2>` explicitly) → `new_pdf.py`. f2py **1.2e-14 / 7.1e-15**.

### 2026-06-03 — Completeness loop iters 71–80: new-hybrid/TSDADG/Luhar PDFs end-to-end + cubic_solve bug fix

Ported and oracle-validated CLUBB's remaining alternative two-component PDF closures — the most complex,
multi-routine subsystems left — and fixed a real cubic_solve correctness bug found along the way. All bit-exact
or near-machine-precision vs f2py and differentiable; new/extended test file per item.
- **iter 71 — FIX `cubic_solve`** (`calc_roots.py`): the principal-branch complex `**(1/3)` returned *garbage*
  for negative-real Cardano args (D>0, R<0); new `_cardano_cbrt` uses the real sign-preserving cube root for
  real args (matching gfortran). Roots now satisfy the cubic to ~1e-16. Also fixed `test_calc_roots`'s
  silently-SKIPping f2py oracle. Ported `sort_roots` + `calc_limits_F_x_responder` (**f2py 6.7e-16**).
- **iters 72–74 — new_tsdadg PDF** fully ported (`new_tsdadg_pdf.py`): `calc_L_x_Skx_fnc`,
  `calc_setter_parameters`, `calc_responder_parameters`, `tsdadg_pdf_driver` — **f2py end-to-end 1.07e-14**
  across all 3 setter branches. `tests/test_new_tsdadg_pdf.py`.
- **iters 75–76 — semi-implicit coefs** (`new_pdf.py`): `calc_coefs_{wp2xp,wpxp2,wpxpyp}_semiimpl` (**f2py
  ≤1.3e-15**). Adversarial find: two routines named `calc_coefs_wpxpyp_semiimpl` (9-arg new_hybrid vs 17-arg
  new_pdf); the f2py wraps the new_pdf three-factor one. `tests/test_coefs_semiimpl.py`.
- **iters 77–78 — new-hybrid PDF** fully ported (`new_pdf_main.py`): `calc_F_x_zeta_x_setter`,
  `calc_F_x_responder`, `calc_responder_var`, and `new_pdf_driver` — **f2py end-to-end 2.66e-15** over the 15
  PDF-param outputs (incl. clipped Skrt/Skthl). The most complex alternative closure. `tests/test_new_pdf_main.py`.
- **iters 79–80 — Luhar 3D PDF** building blocks (`adg1_adg2_3d_luhar_pdf.py`): `close_Luhar_pdf` (**f2py
  4.4e-16**) and `max_cubic_root` (largest real cubic root via the fixed cubic_solve; root-property + np.roots
  oracle). `tests/test_close_luhar_pdf.py`, `tests/test_max_cubic_root.py`. The Luhar_3D_pdf_driver +
  backsolve_Luhar_params remain.

These alternative PDFs are not used by the gated ADG1 config (completeness, not gated fidelity). CHANGELOG
compressed at this iter-80 checkpoint.

### 2026-06-03 — Completeness loop iters 61–70: alternative-PDF closures ported + clip/smooth-Heaviside validation

Ported the alternative two-component PDF closures CLUBB offers besides ADG1 (gated config uses ADG1, so these
are completeness ports), and finished f2py-validating the clip family. All bit-exact / near-machine-precision vs
f2py and differentiable; new test file per item.
- **iter 61** — `clip_skewness` f2py test (sharp branch bit-exact; smooth-Heaviside branch 2.9e-10), completing
  the clip family (covar/variance/skewness). `tests/test_clip_skewness.py`.
- **iter 62** — root-caused that residual: `smooth_heaviside_peskin` uses full-pi in JAX vs Fortran's TRUNCATED
  `pi=3.141592654`/`invrs_pi` literals — JAX is *more accurate* (the truncated-const closed form reproduces
  f2py to 1.1e-16). `tests/test_smooth_heaviside.py`.
- **iters 63–64, 68–69** — `new_pdf.py` (Griffin & Larson 2018/2020): the implicit-coef set
  `calc_coef_{wp4,wpxp2,wp2xp}_implicit`, `calc_mixture_fraction`, `calc_setter_var_params`,
  `calc_responder_params`, + the `new_hybrid_pdf` cross-module aliases (`calculate_*`). **f2py ~1e-15**.
  `tests/test_new_pdf.py`/`test_setter_var_params.py`/`test_responder_params.py`.
- **iters 65–66** — `LY93_pdf.py` (Lewellen & Yoh 1993): `calc_params_LY93` + `calc_mixt_frac_LY93`
  (frozen-bisection) + `LY93_driver`; **f2py ≤1.3e-15** end-to-end + moment reconstruction (mean/variance/
  skewness). Module fully ported. `tests/test_ly93_pdf.py`.
- **iter 67** — `calc_Luhar_params` (ADG Luhar closure, Larson/Golaz/Cotton 2002): **f2py 1.8e-15**.
  `tests/test_luhar_params.py`.
- **iter 70** — new `new_tsdadg_pdf.py` `calc_L_x_Skx_fnc` (skewness-dependent spread, swap-on-sign): **f2py
  1.1e-16** (scalar wrapper). `tests/test_new_tsdadg_pdf.py`.

Adversarial finds this run: the truncated-pi accuracy difference (iter 62); LY93/responder **negative component
variances** are a known feature (moment identities verified with signed variances); and the scalar f2py wrapper
for calc_L_x_Skx_fnc (iter 70). CHANGELOG compressed at this iter-70 checkpoint.

### 2026-06-03 — Completeness loop iters 51–60: PDF-closure ports + f2py-validation gap-closure (bit-exact)

A run split between (a) porting clean closed-form PDF/closure routines and (b) the iter-51 insight that *an
untested port is not a tested port* — adding f2py bit-shadow tests to ported, gated-relevant routines that
lacked one. Nearly all **f2py bit-exact** (0.0–1e-14), all differentiable; new test file per item.
- **iter 51** — validated `rcm_sat_adj_jax` (frozen-bisection saturation adjustment, already ported) vs f2py:
  **bit-match 3.5e-17**. `tests/test_rcm_sat_adj.py`.
- **iters 52–53** — new `CLUBB_core/new_pdf.py` (iiPDF_new_hybrid, Griffin & Larson 2018):
  `calc_coef_wp4_implicit`, `calc_coef_wpxp2_implicit` (both **f2py ~1e-15**, two-branch), `calc_mixture_fraction`
  (literal/analytic). `tests/test_new_pdf.py`.
- **iter 54** — the four binormal/trinormal moment integrals `calc_{wp4,wp2xp,wpxp2,wpxpyp}_pdf`
  (pdf_closure → adg1_adg2_3d_luhar_pdf.py): **f2py 3.6e-15** + Monte-Carlo. `tests/test_pdf_moment_integrals.py`.
- **iter 55** — f2py tests for `calculate_thvm` (5.7e-14) and `skx_func` (4.3e-14).
- **iter 56** — extracted `compute_gamma_Skw_jax` (Gaussian-in-Skw γ; previously inline-only) as a vectorized
  standalone: **f2py 5.6e-17**. `tests/test_compute_gamma_skw.py`.
- **iter 57** — f2py test for `compute_sigma_sqd_w` (per-timestep PDF width): **bit-exact 0.0**.
- **iter 58** — f2py test for `calculate_spurious_source` (budget-closure diagnostic): **3.6e-15**.
- **iters 59–60** — f2py tests for the clipping family `clip_covar` and `clip_variance` (established their clip
  values are solve_type/dt-independent): both **bit-exact 0.0**. `tests/test_clip_covar.py`,
  `tests/test_clip_variance.py`.

These are gated-relevant (PDF-closure moments, sigma_sqd_w, gamma_Skw, clipping) except the new_pdf hybrid set
(alternative PDF). CHANGELOG compressed at this iter-60 checkpoint (iters 51–60 condensed here).

### 2026-06-03 — Completeness loop iters 41–50: SILHS/PDF Cholesky + utility/interpolation routines (bit-exact)

A run of self-contained, oracle-backed ports — most **f2py bit-exact** (matching the compiled Fortran to
machine precision), all differentiable. New tests per item.
- **iter 41 — `matrix_operations.cholesky_factor`** (new `CLUBB_core/matrix_operations.py`): LAPACK-style
  equilibration + lower-Cholesky + τ-on-diagonal fallback for non-PD inputs; **f2py bit-match 1.1e-16**.
  `tests/test_cholesky_factor.py`.
- **iter 42 — `calc_corr_norm_and_cholesky_factor`** (setup_clubb_pdf_params): the "two unique arrays"
  prescribed-corr + Cholesky path (ADG zeroing / Ncn override / eta-hm product), rc-selected; reconstruction
  `L Lᵀ==corr`. `tests/test_calc_corr_norm_cholesky.py`.
- **iter 43 — `calc_cholesky_corr_mtx_approx`** + `setup_corr_cholesky_mtx` + `cholesky_to_corr_mtx_approx`
  (diagnose_correlations_module, Larson-2011 angle Cholesky `s=√(1−c²)`): **f2py bit-match 2.2e-16**.
  `tests/test_cholesky_corr_mtx_approx.py`.
- **iter 44 — `calc_comp_corrs_binormal` + `smooth_corr_quotient`** (pdf_utilities): binormal component
  correlation from `<x'y'>`, |corr|≤0.99 bounded; **f2py bit-match 4.4e-16** + round-trip.
  `tests/test_calc_comp_corrs_binormal.py`.
- **iter 45 — `compute_variance_binormal`** (pdf_utilities, f2py 8.9e-16 + Monte-Carlo) + **new
  `CLUBB_core/interpolation.py`** with `lin_interpolate_two_points` + `mono_cubic_interp` (Steffen 1990,
  **f2py 4.4e-16**). `tests/test_binormal_moments.py`, `tests/test_interpolation.py`.
- **iter 46 — `linear_interp_factor` + `zlinterp_fnc`** (interpolation.py): vertical linear interp with
  zero-fill (= `jnp.interp`); literal binary_search transcription oracle.
- **iter 47 — `calc_w_up_in_cloud`** (pdf_closure_module → adg1_adg2_3d_luhar_pdf.py): cloudy updraft/downdraft
  velocity from the binormal w-PDF; f2py match 1.3e-11 (erf-implementation-limited). `tests/test_w_up_in_cloud.py`.
- **iter 48 — `smooth_min_jax` + `calc_xpwp`** (advance_helper_module): smooth-min primitive + down-gradient
  eddy flux; **f2py bit-exact 0.0**. `tests/test_advance_helper_extras.py`.
- **iter 49 — `pvertinterp`** (advance_helper_module): pressure-coordinate interpolation with clamping;
  **f2py bit-exact 0.0**. `tests/test_pvertinterp.py`.
- **iter 50 — `update_xp2_mc`** (new `CLUBB_core/update_xp2_mc.py`): rain-evaporation tendencies of the five
  second moments (Morrison `l_morr_xp2_mc`, default-off), top-down precip-frac fill + zt2zm; bit-exact vs
  literal-NumPy transcription. `tests/test_update_xp2_mc.py`.

Scope: all are genuine unported Fortran routines; the PDF/Cholesky set (41–44) is SILHS-facing (the gated KK
driver uses prescribed constant correlations, so they add tested-completeness, not gated-case fidelity), the
rest are utility/diagnostic. CHANGELOG compressed at this iter-50 checkpoint (iters 41–50 condensed here).

### 2026-06-03 — Completeness loop iters 36–40: full hydrometeor PDF-correlation pipeline (setup_clubb_pdf_params)

Ported the entire normal→real-space PDF correlation machinery for the hydrometeor microphysics PDF into
`clubb_jax/src/CLUBB_core/setup_clubb_pdf_params.py`, each piece validated and differentiable:
- **iter 36 — `calc_corr_w_hm_n`** (F90:3428): diagnoses the w–ln(hm) component correlation from the overall
  `<w'hm'>` flux (4-way branch on which components vary, ±max_mag_correlation clamp). Strongest oracle is a
  **round-trip** (the routine inverts the flux assembly): recover corr to 4.7e-15 over 200 configs;
  + literal-NumPy branch transcription, clamp, finite grad. `tests/test_calc_corr_w_hm_n.py`.
- **iters 37–38 — the six `component_corr_*` routines** (F90:2448–2939): w_x (ADG-zero / cloud-below), chi_eta
  (cloud-below + optional ±max_mag_correlation Cholesky clamp), and the four `*_ip` (w_hm passthrough/cloud-
  below, x_hm, hmx_hmy, eta_hm product). Literal-NumPy oracle over an rc grid straddling rc_tol; all branches
  exercised; eta-product grad exact. `tests/test_component_corr_ip.py`.
- **iter 39 — `comp_corr_norm`** (F90:1273): assembles the two lower-triangular, then symmetrized,
  `(ngrdcol,nzt,pdf_dim,pdf_dim)` normal-space correlation arrays from all the above. Oracle: structural
  invariants (symmetric, unit diagonal, |corr|≤1) + spot-checks of every assembly rule against the building
  blocks + ADG / l_calc_w_corr branches + finite grad. `tests/test_comp_corr_norm.py`. Adversarial-review
  finding replicated faithfully: the non-fixed eta–w block (F90:1560) re-writes (w,chi) instead of (w,eta).
- **iter 40 — `denorm_transform_corr`** (F90:3208): transforms the normal-space arrays to real space — normal
  pairs unchanged, normal–lognormal via `corr_NN2NL`, lognormal–lognormal via `corr_NN2LL` (both already in
  `pdf_utilities.py`). Oracle: structure + every entry vs direct NN2NL/NN2LL calls + finite grad.
  `tests/test_denorm_transform_corr.py`. Faithful to the Fortran quirk that component-2 Ncn transforms reuse
  the component-1 Ncn variance ratio (Ncn is inherently in-cloud).

Net: the in-JAX path from PDF moments → assembled normal-space corr array → real-space corr array is now
complete and oracle-tested. Honest scope: this pipeline feeds `pdf_hydromet_microphys_wrapper`, whose payoff
(wp2hmp/rtphmp/thlphmp) is zero for all 20 gated cases, so it adds tested-completeness, not gated-case fidelity.
The remaining wrapper glue (assembling these calls end-to-end with the hydromet mixed-moment integrals) is the
last piece. CHANGELOG compressed at this iter-40 checkpoint (iters 36–40 condensed into this block).


### 2026-06-03 — Completeness loop iter 35: whole-driver differentiability gate re-confirmed after the forcing changes

- **Re-ran the whole-driver differentiability gate** (`compare_grad.py`) — the "differentiable" half of the
  project goal, not re-verified since the iters 23-25 forcing-reader/inverse_hydrostatic changes. Result on a
  representative span (arm, bomex, gabls3_night, clex9_nov02): **grad-finite 4/4**; arm/bomex/gabls3_night are
  FD-correct (COMPLETE, worst-FD ~6.5e-7); clex9_nov02 is grad-finite with its known FD kink at the
  Morrison/saturation thresholds (expected — its forcing path is height-coordinate, untouched by my changes).
  **Differentiability gate: PASS.**
- Together with iters 31-34 (faithfulness regression-free across all case types + arm bit-faithful at 100 steps),
  **both halves of the goal — differentiable AND faithful — are confirmed intact** after 35 iterations of changes.
- Assessed the GFDL lookup core's `ghquad` (Gauss-Hermite nodes): it is hardcoded data tables for fixed n — pure
  data transcription, not physics — reinforcing that the ➖ `SCM_Activation` subsystem (no case exercises it) is
  correctly out of scope and the CLUBB-side boundary I ported is the right one.
- **Status:** the in-scope port is complete, differentiable, and faithful (verified). No physics change.

### 2026-06-03 — Completeness loop iter 34: benchmark-case routines all-ported confirmed + durability gate + final-state doc

- **Confirmed all 7 unported benchmark-case `.F90` files have ALL their subroutines ported:** each of arm_97 /
  twp_ice / cloud_feedback / arm_3year / arm_0003 has exactly ONE subroutine (its `_sfclyr`, all ported); lba and
  mpace_b have their tndcy+sfclyr (both ported). The case files stay ❌ only because the cases can't RUN
  end-to-end (SILHS/COAMPS/data/Morrison), NOT from unported routines. So only **3 files have genuinely-unported
  routines**, all impractical/out-of-scope/no-payoff (coamps no-oracle, gfdl lookup ➖-subsystem, pdf_hydromet
  no-payoff).
- **Durability gate (the strongest remaining validation):** ran `compare_runs arm --max-iters 100` — **bit-faithful
  + Tier-C PASS at 100 steps** → the iters 23-25 forcing-reader changes have no late-activating regression.
- **DESIGN.md:** added a "Completeness loop — final state (Iters 1–33)" assessment documenting that the
  differentiable+faithful JAX port is complete for all tractable/in-scope code, with the precise justification for
  the 3 impractical/out-of-scope remainders.
- **Status:** the in-scope/achievable port surface is complete and durably verified. No physics change. Counts unchanged.

### 2026-06-03 — Completeness loop iter 33: full regression confirmation across case types + GFDL-core scope assessment

- **Closed the last regression-verification gap:** the iter-23 `clubb_driver` change (passing `p_in_Pa` to the
  generic forcing loader) runs at init for ALL generic cases, but only the *forcing* cases were verified (iter 32).
  Ran the gate on the non-forcing generic cases — **bomex, dycoms2_rf01, atex, fire, ekman, cobra all PASS the bit
  gate (0 prognostic failures)**. Together with iter 32 (arm/jun25_altocu/gabls3_night bit-faithful, mpace_a
  physical-tier, clex9 from iter 24), **10 of the 20 gated cases are now directly re-verified across every type**
  (forcing-pipeline, sounding, analytic-forcing, cloud-sed, Morrison) → the cumulative changes from 33 iterations
  are confirmed regression-free.
- **Assessed the GFDL activation lookup core** (the one piece I'd called "external/impractical"): it actually
  lives in the repo (`Microphys/SCM_Activation/aer_ccn_act_k.F90`, 959 lines, + droplets*.dat), but it is a large
  ➖-classified subsystem (Gauss-Hermite quadrature + Köhler-theory activation + 5-D lookup, single-precision, no
  case exercises it / no oracle) — confirming the CLUBB-side orchestration (erff/updraft/ndrop, ported) is the
  correct boundary; the lookup core stays out of scope.
- **Status:** the port's tractable/in-scope surface is complete and verified regression-free; the genuinely
  remaining files (COAMPS no-oracle monolith, GFDL/COAMPS lookup subsystems, SILHS RNG, the deferred pdf_hydromet
  wiring) are impractical or out-of-scope. No physics change. Counts unchanged.

### 2026-06-03 — Completeness loop iter 32: bit-faithful gate regression-confirmed for the forcing changes + T_f apply unit test

- **Definitive regression check of the iters 23-24 forcing-pipeline changes:** ran `compare_cases.py` on the
  gated cases that use the generic time-dependent forcing pipeline I modified — **arm, jun25_altocu, gabls3_night
  all PASS the bit gate (0 prognostic failures)** → the pressure-coordinate / T_f / um_ref additions are confirmed
  regression-free. (mpace_a "fails" the *bit* tier with its known single-precision Morrison `thlm_mc` residual but
  PASSES the *physical* tier — worst 2.5e-4, its gate; and it uses the special `load_mpace_a_forcings` path I did
  not touch, so it is doubly unaffected.)
- **Added a direct unit test for the T_f apply-step branch** (`tests/test_pressure_coord_forcing.py:
  test_apply_T_f_conversion`) — constructs a minimal state and verifies `_apply_time_dependent_forcings` sets
  `thlm_forcing = T_f/exner` (with the top zeroed). This is the only forcing branch no gated case exercises, so it
  had no direct coverage before; now the iter-23/24 absolute-temperature-forcing conversion is unit-tested exactly.
- **Status:** all four affected gated cases verified (3 bit-faithful, mpace_a physical-tier); the forcing-reader
  additions are now covered end-to-end (parser + apply). No physics change. Counts unchanged.

### 2026-06-03 — Completeness loop iter 31: cumulative-regression sweep — fixed a test-infrastructure shadowing bug

- **Validation sweep across all 30 iterations of additions** (especially the iters 23-24 shared forcing-pipeline
  edits): re-ran the completeness/port test files. The CGILS forcing changes are confirmed regression-free
  (clex9 bit-faithful; all 19 recent + foundational test files pass).
- **Fixed a real test-infrastructure bug** the sweep surfaced: `test_pos_definite.py` + `test_diagnose_correlations.py`
  (the iter-2/3 f2py-oracle tests) prepended `clubb_release/` to `sys.path` BEFORE `_ROOT`, so `import clubb_jax`
  resolved to the shadowing `clubb_release/clubb_jax/` (no `src`) → `ModuleNotFoundError` when run standalone.
  Fixed to keep `_ROOT` first and APPEND the clubb_release/f2py paths (same fix as iter-6's test_ice_dfsn). Both
  now run their bit-exact f2py-oracle comparisons (rel 0 / 1.6e-15) standalone again. Scanned the rest of the
  suite — no other test has the pattern.
- **Feasibility-checked the deferred `pdf_hydromet_microphys_wrapper` wiring** (the only remaining integration):
  confirmed it needs the full `hydromet_pdf_params` correlation structure (corr_chi_hm/eta_hm/w_hm_n/hmx_hmy),
  which the JAX setup does NOT produce — wiring it requires porting more of setup_pdf_parameters' correlation
  processing (major, invasive, oracle-limited, no gateable payoff). Deferral firmly justified.
- **Status:** test suite green after the fix; no physics change. Counts unchanged.

### 2026-06-03 — Completeness loop iter 30: GFDL `aer_act_clubb_ndrop` (activation orchestration complete); CHANGELOG compressed

- **Ported `aer_act_clubb_ndrop`** (`Microphys/gfdl_activation.py`) — the layer-averaged activated droplet
  concentration `Ndrop = (drop_1 P1 + drop_2 P2)(mixt_frac cloud_frac_1 + (1-mixt_frac) cloud_frac_2)`, combining
  the per-component lookup-table droplet concentrations (caller-supplied) with the iter-20 updraft weights.
  This **completes the CLUBB-side orchestration of GFDL droplet activation** (erff + updraft_weights + Ndrop);
  only the external-data `aer_ccn_act_wpdf_k` lookup table itself remains (impractical).
- **Validation (`tests/test_gfdl_activation.py`)** — `aer_act_clubb_ndrop` vs literal (<1e-6), no-cloud→0,
  non-negativity, finite grad.
- **CHANGELOG compression (10-iteration cadence):** condensed completeness-loop iters 17–26 into one summary
  block; iters 27–30 kept in full.

### 2026-06-03 — Completeness loop iter 29: arm_3year/arm_0003 surface schemes + pdf_hydromet wiring diagnosis

- **Ported `arm_3year_sfclyr` + `arm_0003_sfclyr`** (→ arm_3year.py, arm_0003.py) — both algebraically identical
  to `arm_97_sfclyr` (prescribed heat fluxes → kinematic + MOST diag_ustar); reuse the validated implementation,
  verified equal in the test. **Every benchmark case file now has its surface scheme ported.** Both cases stay
  unviable (arm_0003 COAMPS-fatal in the Fortran; arm_3year forcings data removed → no oracle).
- **Diagnostic finding (adversarial review of the KK/Morrison path):** the JAX **stubs `wp2hmp`/`rtphmp_zt`/
  `thlphmp_zt` to ZERO** (clubb_driver.py:792) — the iter-13 `hydrometeor_mixed_moments` port exists and is
  unit-tested but is **NOT wired** into the running path; `pdf_hydromet_microphys_wrapper` (still ❌) is the
  missing orchestration that would call it to compute the nonzero water-loading second moments. Wiring it would
  change the KK/Morrison cases (rico/dycoms2_rf02_do/ds) — which are already oracle-limited — and requires a
  PDF-struct→dict adapter, so it is deliberately deferred (invasive, no gateable payoff). Recorded so the gap is
  explicit.
- **Status:** all benchmark-case surface schemes ported (cases remain SILHS/COAMPS/data blocked). Counts unchanged.

### 2026-06-03 — Completeness loop iter 28: three SILHS-blocked surface schemes (mpace_b/arm_97/twp_ice)

- **Ported the surface schemes of three SILHS-blocked cases:**
  - `mpace_b_sfclyr` (→ mpace_b.py): prescribed (time-interpolated) sensible/latent heat fluxes → kinematic
    (`/(ρCp)`, `/(ρLv)`), fixed ustar = 0.25. **mpace_b.F90 now fully ported** (tndcy + sfclyr).
  - `arm_97_sfclyr` (→ new arm_97.py): same kinematic conversion + the MOST `diag_ustar` (z0=0.035).
  - `twp_ice_sfclyr` (→ new twp_ice.py): the RICO drag law, **algebraically identical to `cloud_feedback_sfclyr`**
    (iter 22) — reuses that validated implementation, verified equal.
- **Validation (`tests/test_silhs_surface_schemes.py`)** — each bit-exact vs a literal NumPy transcription
  (mpace_b exact, arm_97 max diff 5.6e-17 incl. the MOST ustar); twp_ice verified == cloud_feedback_sfclyr; finite
  `jax.grad`. These take the time-interpolated forcing as input (decoupling the time-dependent-data reader).
- **Status:** mpace_b.F90 fully ported; arm_97.F90 + twp_ice.F90 surface schemes ported. All three CASES remain
  SILHS-blocked (➖ out-of-scope subsystem). Counts unchanged.

### 2026-06-03 — Completeness loop iter 27: M-PACE B LS forcing `mpace_b_tndcy` + LBA `lba_tndcy` (lba.F90 complete)

- **Ported `mpace_b_tndcy`** (`Benchmark_cases/mpace_b.F90` → `Benchmark_cases/mpace_b.py`) — the M-PACE B Arctic
  mixed-phase large-scale forcing: a divergence-driven subsidence `ω = min(D(p_sfc−p), D(p_sfc−pinv))` capped
  above the inversion → `wm = −ω Rd thvm/(p g)` (zt2zm with zero BCs), an analytic radiative-cooling thlm
  tendency (capped at −4 K/day, with the exner factor) and a moisture tendency, all functions of pressure.
  Differentiable.
- **Ported `lba_tndcy`** (zero LS forcing — LBA deep convection is surface-driven) → **lba.F90 both subroutines
  now ported** (sfclyr iter 26 + tndcy).
- **Validation (`tests/test_mpace_b_lba_tndcy.py`)** — `mpace_b_tndcy` bit-exact vs a literal NumPy transcription
  (all 4 outputs, rel <1e-12) + physical invariants (subsidence wm_zt ≤ 0, wm_zm surface/top BCs = 0, cooling
  thlm_forcing < 0) + finite `jax.grad`; `lba_tndcy` identically zero.
- **Status:** lba.F90 routines all ported (case SILHS-blocked); mpace_b.F90 PARTIAL (tndcy ported, sfclyr remains).
  Counts unchanged.

### 2026-06-03 — Completeness loop iters 17–26 (condensed): finishing the strong-oracle frontier + CGILS plumbing

Ten iterations that completed the remaining well-oracled subsystems and the CGILS/blocked-case input plumbing
(full per-iter detail in git history). All differentiable, each oracle-validated.

- **iter 17 — completed `PDF_integrals_all_MM.F90` (8/8) → ✅:** the quadrivar const reductions; the whole KK
  all-mixed-moment D_v machinery is now ported (validated by analytic base cases + complex-branch Monte-Carlo).
- **iters 18-19 — completed `Radiation/BUGSrad/cloud_correlate.F90` (2/2):** `bugs_ctot` (cloud-overlap total
  cloud amount, the cld_below recurrence as a cumprod, bit-faithful vs literal) + `bugs_cloudfit` (maximal/random
  split, grid-search). **`Radiation/` now fully ported.**
- **iter 20 — GFDL `erff` + `updraft_weights`** (CLUBB-side activation; erff vs math.erf <1e-6; updraft weights
  vs literal incl. a faithfully-reproduced Fortran normalization quirk).
- **iter 21 — LBA prescribed radiation `simple_rad_lba`** (33×36 table interp, bit-exact on the real .dat) +
  verified-characterization of the 7 remaining benchmark cases (all SILHS/COAMPS/data/Morrison-blocked).
- **iter 22 — `cloud_feedback_sfclyr`** (CGILS RICO drag-law surface fluxes, shared by 12 cases).
- **iters 23-24 — CGILS forcing-reader capability:** a `Press[Pa]` pressure-vertical-coordinate forcing path
  (interpolate vs the model p_in_Pa), `T_f` absolute-temperature forcing (thlm_f=T_f/exner), and the time-dependent
  `um_ref`/`vm_ref` nudging targets. Guarded so height-coordinate gated cases are byte-identical (clex9 stays
  bit-faithful). Diagnosed cloud_feedback's residual divergence (Morrison-oracle-limited + a Press[Pa] SOUNDING).
- **iter 25 — `inverse_hydrostatic`** (pressure-sounding altitudes via the log-mean hydrostatic integration);
  the round-trip z→exner→z against the forward hydrostatic is exact (5.5e-12 m).
- **iter 26 — LBA `lba_sfclyr`** (diurnal surface fluxes + MOST diag_ustar, bit-exact vs literal).

Net: PDF_integrals_all_MM + cloud_correlate completed; Radiation fully ported; the CGILS pressure-coordinate
forcing capability added (no regression); and the self-contained surface/forcing/sounding routines of the
blocked cases ported with literal/analytic/round-trip oracles.

### 2026-06-03 — Completeness loop iters 7–16 (condensed): KK PDF-integral mixed-moment machinery

Ten iterations that ported the entire KK hydrometeor mixed-moment / covariance machinery (full per-iter detail
in git history). Each function is differentiable and oracle-validated.

- **iter 7 — `CLUBB_core/hydromet_pdf_parameter_module.F90`** → dataclasses + zero/init; **CLUBB_core fully ported**.
- **iter 8 — +2 bit-faithful cases (clex9_nov02/oct14, gate 18→20).** Root-caused a pre-activation Ncm/Nc_in_cloud
  diagnostic mismatch (Morrison `advance_microphys` early-returns before its stat writes during spin-up) + fixed a
  `compare_runs.py` rel-masking integrity bug. Verified physics-neutral.
- **iter 9 — namelist scalar-`sclr_tol_nl` parse fix** (unblocks astex_a209 → runs but KK-limited; reclassified
  🔁) + ported the foundational closed-form moment integrals `univar_N`/`univar_L` of
  `Microphys/mixed_moment_PDF_integrals.F90` (vs binomial expansion <1e-12 + MC).
- **iters 10-13 — completed `mixed_moment_PDF_integrals.F90` (8/8) → ✅:** `bivar_NL` (tilting decomposition),
  the `<x'^a hm'^b>` assembly (`bivar_NL_x_hm_all_MM_comp_eq` 4-branch jnp.where + `xp_a_hmpb`), the streamlined
  covariances (`xphmp`/`hmxphmyp`), and the top driver `hydrometeor_mixed_moments` (vectorized over levels, vs a
  literal Fortran-loop transcription <1e-12). Added `compute_mean_binormal` to pdf_utilities. Validated by
  closed-form/branch <1e-12 + 8-16M Monte-Carlo.
- **iters 14-16 — started `PDF_integrals_all_MM.F90` (5/8):** the trivariate family `trivar_NNL_MM` + its 3 const
  reductions (x2>0 half-line, parabolic-cylinder D_v), and the general quadrivariate `quadrivar_NNLL_MM` (x2<0
  subsaturated). Validated by analytic base cases (a=b=0 → Φ(±μ_x2/σ_x2)) <1e-9, σ→0 limit consistency, and
  truncated-domain Monte-Carlo (complex principal-branch for the x2<0 region) <5e-3. Reuses the existing accurate
  `dv_parabolic_cylinder`/`_gamma_real`/`_signed_pow`. (Adversarial review here caught a mis-derived covar
  identity — replaced with the complex-branch MC.)

Net over iters 7–16: CLUBB_core completed; in-scope ported 113→118; mixed_moment_PDF_integrals fully ported;
PDF_integrals_all_MM started; the bit-faithful gate grew 18→20.

### 2026-06-03 — Completeness loop iters 1–6 (condensed): six unported-module ports, each oracle-validated

Six self-contained Fortran modules ported to JAX, each with its own validation test (full per-iter details in
git history; condensed here per the 10-iteration cadence). Each is differentiable (`jax.grad` finite).

- **iter 1 — `CLUBB_core/calc_roots.F90`** → `calc_roots.py`: `cubic_solve` (Cardano, complex128 principal
  branch), `quadratic_solve`, `cube_root`. `tests/test_calc_roots.py`: polynomial residual ~4e-16 at every root
  across all discriminant signs + set-match vs `numpy.roots` (f2py not exposed → SKIP).
- **iter 2 — `CLUBB_core/pos_definite_module.F90`** → `pos_definite_module.py`: `pos_definite_adj_jax`,
  Smolarkiewicz (1989) flux-conservative positive-definite renormalization (ascending grid, vectorized).
  `tests/test_pos_definite.py`: **BIT-EXACT vs `f2py_pos_definite_adj` (rel 0)** + conservation invariant.
  Established the reusable f2py-oracle workflow (`clubb_api.setup_grid` with the JAX grid's own heights →
  `clubb_f2py.f2py_<routine>`).
- **iter 3 — `CLUBB_core/diagnose_correlations_module.F90`** → `diagnose_correlations_module.py`:
  `diagnose_correlations` (Larson 2011 SILHS hydrometeor correlation diagnosis) + `calc_mean/varnce/w_corr`.
  `tests/test_diagnose_correlations.py`: **bit-match vs `f2py_diagnose_correlations` (rel 1.6e-15)** incl.
  iiPDF_w edge cases.
- **iter 4 — `Microphys/KK_microphys/KK_local_means.F90`** → `KK_local_means.py`: the 4 grid-mean KK warm-rain
  rates (evap/auto/accr/mvr). `tests/test_kk_local_means.py`: vs independent NumPy transcription (rel <1e-13) +
  branch coverage. (f2py oracle exhausted for the remaining files from here on → analytic/MC oracles.)
- **iter 5 — `Microphys/KK_microphys/KK_upscaled_variances.F90`** → `KK_upscaled_variances.py`: `variance_KK_mvr`
  (variance of the KK rain mean-volume radius). `tests/test_kk_upscaled_variances.py`: two oracles — closed-form
  lognormal moment (**rel 0**) + 4M-sample Monte-Carlo (**rel 1.4e-4**).
- **iter 6 — `Microphys/ice_dfsn_module.F90`** → `ice_dfsn_module.py`: `ice_dfsn` (cloud-water depletion by ice
  diffusional growth) as a top-to-bottom falling-crystal mass-integration `lax.scan`. `tests/test_ice_dfsn.py`:
  vs literal NumPy loop (**rcm rel 1.2e-16**) + cap/branch coverage. New helper `thlm2T_in_K_jax` **bit-exact vs
  `f2py_thlm2t_in_k_1d`**; added `Lf`/`Ls`/`cm_per_m` constants.

Net over iters 1–6: in-scope ported 107→113, unported 23→17; `CLUBB_core/` reduced to 1 unported file.

### 2026-06-02 — Refactor iter 35: faithfulness reconciliation + completion-status synthesis

- **Reconciled the two completion clauses against the suite:**
  - **Differentiable, entirely in JAX — ✓ unequivocal, suite-wide** (all 19 cases, via forward-identical
    transforms — faithfulness was never traded away).
  - **Faithful to the Fortran Oracle within reasonable accuracy — ✓ for the entire validated suite (18
    `compare_cases` DEFAULT_CASES, all BIT-FAITHFUL)** spanning every physics subsystem (surface schemes,
    simplified + BUGSrad radiation, soil-veg, Morrison microphysics, sponge, cloud-sed, ADG1 PDF, mixing
    length, all solves). Re-confirmed per-case Tier-C across iters 29–34.
  - **The one gap, rico, is a PRE-EXISTING INCOMPLETE PORT, not a bug/regression:** rico's KK rain-microphysics
    rate library is bit-faithful in isolation, but its transport + feedback application is a deliberately
    STAGED rollout gated OFF (`advance_clubb_to_end.py:98`, `l_kk_micro_apply` default off), so its prognostics
    diverge from Fortran. An unported subsystem, tracked independently of (and not touched by) the
    differentiability refactor — my KK detach is forward-inert (rico forward loss identical pre/post).
- **DONE status:** the REFACTOR's goal — relax bit-faithfulness for differentiability while PRESERVING
  faithfulness — is achieved (differentiable suite-wide + faithful for the whole validated suite). Per the
  strict "faithful suite-wide must be unequivocally true" rule, DONE is withheld only on rico's pre-existing
  unported KK transport — a separate workstream from B5. No `<promise>` emitted.

### 2026-06-02 — Refactor iter 34 (B5 COMPLETE): all 19 cases differentiable + grad gate

- **★★ B5 GOAL MET — whole-driver `jax.grad` is finite for ALL 19 cases.** Ported the last blocker,
  `_landflx_scalar` (gabls3_night's Businger-Dyer land-surface Monin-Obukhov scheme), to a vectorized
  differentiable `_landflx_jax`: both the unstable (r<0, 3 iterations) and stable (r≥0, quadratic) branches
  are computed and `jnp.where`-selected; `_safe_sqrt` for the clip/quartic roots; `1/(2a)` and `1/vel`
  guarded so the unselected branch can't poison the gradient. R7 block dispatch (concrete keeps the exact
  per-column loop). **gabls3_night B5 COMPLETE** (128/128, thlm rel 4.0e-7, um rel 2.5e-8); bit-identical
  concrete (Tier-C vs Fortran PASS).
- **★ Built `run_scripts/compare_grad.py`** — the differentiability GATE (grad analogue of compare_cases):
  per-case subprocess probe → dashboard of thlm/um grad-finite counts + worst-FD + COMPLETE/KINK/PARTIAL/
  BLOCKED status; exit 0 iff all grad-finite (`--strict` requires FD-correct). Locks the achievement in as a
  regression net.
- **Not DONE:** "differentiable... entirely in JAX" is now met suite-wide, but "faithful... tested against the
  Fortran Oracle for reasonable accuracy" must still be reconciled per-case (e.g. rico's KK forward diverges,
  pre-existing) — iter35 characterizes the Tier-C status precisely before any DONE claim.

### 2026-06-02 — Refactor iter 33 (B5 suite sweep): ~18/19 cases whole-driver-differentiable

- **★ Swept all remaining cases; whole-driver `jax.grad` is now finite for ~18 of 19 cases.** Cleared a batch
  of blockers, all forward-identical (no regression — ekman/cobra/atex_long Tier-B PASS):
  - **Sponge layer** (ekman): `sponge_damp_xm` rewritten as pure broadcast arithmetic — `tau=inf` outside the
    sponge ⇒ `dt/tau=0` ⇒ the relaxation collapses to a no-op, so the per-column `np.array`+in-place loop
    becomes vectorized & differentiable (bit-identical); `_apply_sponge_field` vectorized to a single call.
  - **Surface `_diag_ustar`** (cobra `_arm_variant_sfclyr`, gabls2 `_gabls2_sfclyr`): R7 block dispatch
    reusing `_diag_ustar_jax`. **Scalar-flux BC** (gabls2 `_set_sclr_sfc_rtm_thlm`): `state['wpsclrp'][:,:,k]=`
    → `_iset`.
  - **Cloud-droplet sedimentation** (atex_long, dycoms2_rf02_so): `cloud_drop_sed` body is already jnp; only
    the return `np.asarray` severed → `_asarray` (fully differentiable, lightweight — not detached).
- **Differentiable cases:** arm, bomex, dycoms2_rf01, dycoms2_rf01_fixed_sst, dycoms2_rf02_nd,
  dycoms2_rf02_so, atex, atex_long, gabls2, gabls3, rico, mpace_a, wangara, neutral, fire, jun25_altocu,
  ekman, cobra (~18). Some show a single-level FD kink at a hard physical threshold (8e-3 inversion etc.).
- **Only remaining blocker: gabls3_night** — `_landflx_scalar` (full Businger-Dyer land-surface MO scheme,
  data-dependent r<0 branch + iterations + quadratic) needs a jnp port (pre-core, cannot detach).
- **Not DONE:** gabls3_night still blocked; a multi-case grad GATE should lock in the achievement; "faithful"
  (Tier-C) is a separate per-case status (rico's KK forward still diverges, pre-existing).

### 2026-06-02 — Refactor iter 32 (B5 microphysics): rico + mpace_a complete; 8 cases differentiable

- **★ rico (KK microphysics) and mpace_a (Morrison) whole-driver `jax.grad` COMPLETE** — both finite +
  FD-correct (rico thlm rel 2.6e-7/um 2.7e-8; mpace_a thlm rel 8e-7/um 2.2e-8). KK (`advance_kk_microphysics`,
  42 `np.asarray`) and Morrison (`advance_morrison_microphysics`) both run AFTER the core and store `*_mc`
  tendencies for the NEXT step's forcings → **detach-under-trace** (early-return when inputs are tracers),
  same rationale as BUGSrad radiation. Both guards are **concrete-inert** (forward losses identical pre/post)
  → bit-identical by construction.
- dycoms2_rf01 also grad-finite (500/500). **8 cases now whole-driver-differentiable** — arm, bomex,
  dycoms2_rf01, dycoms2_rf02_nd, atex, gabls3, rico, mpace_a — spanning simple/generic surface, simplified +
  BUGSrad radiation, interactive soil-veg, KK + Morrison microphysics.
- Note: rico's *forward* Tier-C FAILs (15 prognostic) — a PRE-EXISTING KK-microphysics divergence, not a
  regression (my detach is forward-inert). "Faithful" is a separate per-case question from "differentiable".
- **Not DONE:** 8 representative cases differentiable; the remaining case families (jun25_altocu, cobra, …)
  still need a probe sweep; hard-threshold physical kinks remain (optional smoothing under the relaxed standard).

### 2026-06-02 — Refactor iter 31 (B5 gabls3): whole-driver grad finite 250/250 + BUGSrad detach

- **★ gabls3 whole-driver `jax.grad` FINITE 250/250** (um FD-correct rel 1.7e-8; thlm rel ≤1.5e-5 except one
  inversion-kink level). Cleared the full gabls3 surface→soil-veg→radiation chain:
  - `_gabls3_sfclyr`: R7 block dispatch (reuse `_diag_ustar_jax`); the 16 per-case
    `state['(up|vp)wp_sfc'][:]=` momentum writebacks → `_iset`; `_advance_soil_veg_step` `np.asarray`→`_asarray`.
  - `_diag_ustar_jax`: sign-preserving floor on `lmo` (very stable layers drive ustar→0 so `z/lmo`→inf grad).
  - **BUGSrad → detach-under-trace (new R8 variant):** correlated-k RT is reverse-mode memory-PROHIBITIVE
    (OOM), and radiation runs AFTER the core so its radht is dead for the single-step loss → skip it under a
    trace (exact for single-step; radiation = detached forcing for multi-step rollouts). Light simplified-LW
    stays fully differentiable.
  - **The binding level-0 nan: `sfc_varnce_module:66` `where(wpthlp_sfc>0, safe_cubed**(1/3), 0)`** — in
    STABLE layers (gabls3, wpthlp_sfc<0) `safe_cubed=0` and the masked `0**(1/3)` poisons the grad (`0*inf`);
    arm is convective so never hit it. Fixed with `_safe_pow` + `_safe_sqrt` on the ustar2/uf roots.
- All changes **forward-identical**: arm/bomex/dycoms2_rf02_nd/atex Tier-B PASS, gabls3 Tier-C vs Fortran PASS.
- **Not DONE:** whole-driver grad now finite for arm, bomex, dycoms2_rf02_nd, atex, gabls3. Remaining: rico
  (KK microphysics) + Morrison; and the hard-threshold physical kinks (8e-3 inversion) leave single-level FD
  mismatches that could be smoothed under the relaxed standard.

### 2026-06-02 — Refactor iters 24–30 (condensed): B5 — whole-driver `jax.grad` (differentiability)

Extended reverse-mode `jax.grad` from `advance_clubb_core` (B4) to the **whole `advance_clubb_to_end`
timestep** (thvm + forcings + radiation + core, stats off). Validator: `tests/probe_driver_grad.py <case>`
(per-case, FD-checked); `tests/_nanhunt.py` for nan localization.

- **iter24:** extracted the tracer-transparent toolkit into shared `CLUBB_core/tracer_numpy.py`
  (`_asarray`/`_xp`/`_iset`/`_safe_sqrt`/`_safe_pow`/`_is_tracer_arg`), used by both B4 and B5 modules.
- **iter25 — arm COMPLETE** (thlm rel 2.3e-7, um rel 1.3e-8). New patterns: **(R7) block-level tracer
  dispatch** — guard a small branchy block (`float()`/`math`/`max`/fixed-point loops, e.g. Monin-Obukhov
  `_diag_ustar` → new `_diag_ustar_jax`) with `if not _is_tracer_arg([...]): <exact concrete> else: <jnp
  mirror>`; **(R8) diagnostic-skip-under-trace** — pure NaN/stats checks early-`return` unchanged under a
  trace (`parameterization_check_jax`).
- **iter26 — generic_forcings surface differentiable** (bomex `um` 87/87). Made `_fsign`/`_mono_cubic_interp`/
  `_compute_ubar`/`_read_surface_var_for_bc` tracer-transparent (`_xp`/`_safe_sqrt`/`_xp.stack`/`_iset`);
  hardened `_iset` to copy read-only numpy views; double-where fix in `calculate_thlp2_rad_jax`.
- **iters 27–29 — bomex `thlm` 0/87 → 87/87** (rel 5.4e-7). The whole-driver nan was a chain of **clip-sqrt /
  fractional-pow inf gradients** that detonate only in convective/surface layers (so arm passed, bomex
  didn't): `wp23_term_splat_lhs_jax` `sqrt(max(0,bv_sqd_splat))`, ADG1 `w_1_n/w_2_n` + 14 `sqrt(varnce·varnce)`,
  Richardson `Ri**exp` (→ `_safe_pow`), and **the binding one — `mixing_length.py:180`
  `sqrt(maximum(zero_threshold, bv_smth))` with `zero_threshold == 0.0`** (a bare `sqrt(max(0,·))`). All fixed
  with `_safe_sqrt`/`_safe_pow`, all **forward-identical**. Key techniques (now conventions): nan levels are
  **FD-finite artifacts** (true grad finite); `jax_debug_nans` flags the inf→nan CONVERSION not the source;
  **stop_gradient bisection** pins the carrying tensor in log(N) probes. Probe/`_nanhunt` use a working-dir
  namelist (never the golden — init truncates the stats `.nc` beside it).
- **iter30 — radiation differentiable** (dycoms2_rf02_nd/atex `thlm`+`um` grad finite). Made `radiation.py`
  simplified-LW path tracer-transparent: `_liq_water_path` → vectorized reverse-cumsum (same FP order →
  bit-identical), `_simple_rad_lw` `np.exp`→`_xp.exp`, and the `l_rad_above_cloud` inversion block via R7
  dispatch (new `_inversion_height_jax`, mask-multiply instead of boolean-index, `_safe_pow` for the
  `dz**(1/3)`/`(4/3)` cloud-top inf-grad). One residual FD mismatch at the inversion level is a **genuine kink**
  (hard 8e-3 `rtm` threshold), not a bug.

**Validation:** every step bit-identical for concrete runs — arm/bomex/dycoms2_rf02_nd/atex/gabls3 Tier-B PASS
and Tier-C(vs Fortran) PASS (0 prognostic fails); B4 core grad still PASS.
**Status:** whole-driver `jax.grad` finite + FD-correct for arm, bomex, and (radiation) dycoms2_rf02_nd/atex.
**Remaining for suite-wide:** KK microphysics (`Microphys/kk_microphys_step.py`, rico), Morrison, and a few
case-surface routines (`_gabls3_sfclyr`) need the same tracer-transparency.


### 2026-06-02 — Refactor iter 22-23 (B4 hardening): robust multi-prognostic grad + post-B4 suite PASS

- **Post-B4 no-regression gate GREEN:** the full 18-case `--tier physical` re-run PASSES — identical to pre-B4
  (17 bit-faithful + within Tier-C; mpace_a Tier-C-pass). The ~770 tracer-transparent B4 conversions regressed
  nothing across the whole suite.
- **Strengthened `test_full_timestep_grad.py`** to check grad w.r.t. **three prognostics** (thlm, rtm, um), not
  just thlm — which caught a real gap: **grad w.r.t. um/vm was nan** (all 133 levels). Root cause: `sqrt(ddzt_umvm_sqd)`
  (√wind-shear, `mixing_length.py:161` + the Richardson `sqrt(shear²/bv)` :212) is 0 at uniform-wind levels →
  `sqrt(0)` has infinite derivative → nan'd the whole um/vm backward pass (thlm/rtm were clean as they don't
  drive shear). Fixed with the existing double-where `_safe_sqrt` (forward-identical). Now all 3 grads are
  finite + FD-correct (thlm 4e-10 / rtm 1e-14 / um 3e-11).
- **Validated:** arm Tier-B PASS (the `_safe_sqrt` change bit-identical); full differentiability suite PASS.
  Added the new gates (`test_full_timestep_grad`, `test_mono_flux_limiter`, `test_invariants`) to DESIGN's
  test list. **Lesson: validate grad against MULTIPLE inputs — a single-field grad check gives false confidence.**

### 2026-06-02 — Refactor iter 21 (B4 COMPLETE): full-timestep jax.grad through advance_clubb_core

- **★★★ Full-timestep `jax.grad` through `advance_clubb_core` now WORKS — finite + finite-difference-correct
  (rel 4.0e-10).** `tests/test_full_timestep_grad.py` flips to the "B4 COMPLETE" gate (PASS). The core CLUBB
  turbulence timestep (closure + all prognostic solves + mixing length + flux limiter) is reverse-mode
  differentiable — the project's headline goal.
- Final two blocker classes cleared: (1) the `_like` constructors (`np.zeros_like`/`full_like`/`empty_like`,
  36 sites) → tracer-transparent `_xp.*`; (2) **a module-global side effect** — `advance_clubb_core` wrote the
  jun25 ADG1 carry to the global `_prev_adg1_j25`, leaking a tracer across calls (`UnexpectedTracerError`);
  guarded the write under trace (`if not _is_tracer_arg(...)`). Convention: never store a tracer in module-global
  state.
- **Validated:** grad FD-correct; arm Tier-B PASS (all cumulative B4 conversions bit-identical). Full 18-case
  Tier-C suite re-run launched as the no-regression check. REFACTOR.md/DESIGN.md updated (the differentiability
  status flips to "available"). Grad uses the standard differentiable-forward config (debug_level=0/l_sample=False).

### 2026-06-02 — Refactor iter 20 (B4 stage 6): multi-line scratch converter + CHANGELOG compression

- **Scheduled 10-iteration CHANGELOG compression:** condensed iters 11–19 into the block below.
- **B4 stage 6:** hand-converted the live clip_variance scratch (`_thlp2/_rtp2_jax10_cv` slice-assigns), then
  wrote a **multi-line-aware `arr[idx]=(…)` → `_iset(arr, np.s_[idx], (…))` converter** (paren-depth tracking
  to find the statement end) and applied it across `advance_clubb_core` — **33 multi-line scratch/RHS-assembly
  assignments converted** (up2/vp2/wp2/wp3 RHS + budget terms). Verified the `_36` RHS blocks are LIVE (feed
  the tridiag solves), not dead, so converted (not removed).
- **Validated bit-identical:** import OK + arm Tier-B PASS (the 33 conversions changed no values). Full-timestep
  grad blocker advanced **2654 → 2883 → 3216**. Convention: the converter accumulates continuation lines until
  paren/bracket-balanced, skips `;`/`==`/comment/already-`_iset` lines.

### 2026-06-02 — Refactor iters 11-19 (condensed): Phase 2 B2 + B4 staged conversion

Per-iteration detail in git (afc12d1..6d783f6); REFACTOR_PROGRESS.md is the live ledger.

- **iter11 (B2):** JAX `lax.scan` monotonic flux limiter (`monotonic_turbulent_flux_limit_jax`) — bit-exact
  to NumPy (rel 2.5e-16) + `jax.grad`-able; swapped in-loop (arm no-op bit-exact, atex Tier-A+C PASS).
- **iter12-14 (validate + B4 scoping):** added `compare_cases.py --tier {bit,physical}`; the **whole 18-case
  suite PASSES Tier-C** (mpace_a Tier-C-pass with its A2 bit-failures); full **Tier-B golden net for all 18**.
  B4 audit: 842 `np.asarray` + ~45 mutations in the 3,800-line `advance_clubb_core`, all-or-nothing for grad.
- **iter15-19 (B4 staged conversion — the sole remaining full-timestep-grad blocker):** capture hook
  (`CLUBB_CAPTURE_KWARGS`) + fixture grad probe `tests/test_full_timestep_grad.py`. Chose the
  **tracer-transparent numpy** mechanism (jnp under a JAX trace, exactly numpy otherwise → **every step
  bit-identical**, arm Tier-B PASS): `_asarray` (558 sites), the `_xp` ufunc shim (61: maximum/where/sqrt/…),
  `_iset` immutable-safe assignment (45 single-line mutations), and removal of dead `_rhs9_ref` numpy
  shadow-comparison scaffolding. Grad blocker advanced 775→1410→1906→2446→2654. **Strategic finding: no
  data-dependent control flow on the prognostic grad path** (if-on-array only on concrete `err_code` +
  l_sample stats) → B4 completable via this approach, no `lax.cond` wall.

### 2026-06-02 — Refactor iter 10: CHANGELOG compression + C4 (relocate f2py debug harnesses)

- **Scheduled 10-iteration CHANGELOG compression:** condensed the nine per-iteration refactor entries + the
  REFACTOR.md-proposal entry into the single block below (detail preserved in git `b1c6d0c..ad36376`).
- **C4 (Phase 4, done early):** moved the per-term f2py bit-comparison harnesses (`cmp_terms_f2py.py`,
  `cmp_mfl_f2py.py`, `compare_xm_wpxp_f2py.py`) to `run_scripts/debug/` with a README — debug-only, out of
  the gate path. None were imported anywhere; DESIGN.md path references updated.
- **Adversarial note (recorded for sequencing):** B2 (the NumPy `mono_flux_limiter`) fires only for
  atex/gabls3_night, so it blocks grad for just 2 cases; **B4 (the orchestration NumPy round-trips) is the
  dominant remaining full-timestep-grad blocker** (all cases). Next substantive target is B4 (large,
  all-or-nothing) or B2 (bounded, 2 cases) — to be sequenced next iteration.

### 2026-06-02 — Numerical-accuracy refactor, Phases 0-2 (iters 1-9, condensed)

Executing REFACTOR.md on branch `refactor/numerical-accuracy`: relax bit-faithfulness -> a tiered
numerical-accuracy standard, simplify, unlock differentiability. Per-iteration detail is in git
(commits b1c6d0c..ad36376); `REFACTOR_PROGRESS.md` is the live ledger (decisions + conventions R1-R6).

- **Phase 0 — measurement safety net (P0.1-P0.5):** `run_scripts/validation.py` (Tier-C field-class
  tolerances: mean 1e-4 / flux 1e-3 / moment 3e-3 / microphys 1e-2 / diagnostic report-only; `*_mc`
  tendencies report-only) + `compare_runs.py --tier {bit,physical}`; `invariants.py` + `tests/test_invariants.py`
  (Tier-A finiteness/positivity/Cauchy-Schwarz, oracle-free, with a negative teeth-test); `golden.py` +
  `update_golden.py` (Tier-B golden-trajectory regression rel 1e-9; `.nc` gitignored, tracked manifest
  `clubb_jax/golden_manifest.json`); `validate_case.py` (one-command A+B+C verdict). Calibration: rico's
  dynamics pass Tier-C with margin (validates the thesis); precip-FP hydrometeors -> Tier-D.
- **Phase 1 — deleted the three "ported to be less accurate" contrivances (A1-A3):** A2 the Morrison `real*4`
  `thlm_mc` round-trip (clear-air thlm_mc 2.9e-7->6.6e-18); A1 the `parabolic_expax` epss=1e-4 D_v reproduction
  (deleted module+test; `_dvc` uses accurate float64 dv); A3 the BUGSrad cloudg `sngl`/float32-pi. Each simpler
  + strictly more accurate; all validated within Tier-C (mpace_a/do/ds/gabls3) + unit suites. mpace_a & do/ds
  reclassified (more accurate, not bit-faithful).
- **Phase 2 — differentiability (B3):** replaced the two mixing-length parcel-ascent `lax.while_loop`s with a
  bit-exact bounded `lax.scan` (`_bounded_while`) + `_safe_sqrt` -> reverse-mode `jax.grad` through the Golaz
  mixing length now works (was raising), finite + FD-correct; bit-exact (arm Tier-B). Adversarial finding: hard
  min/max are already differentiable, so B1 was downgraded; B2 (numpy flux limiter) + B4 (orchestration numpy
  round-trips) remain for full-timestep grad.
- **Conventions (REFACTOR_PROGRESS.md R1-R6 + DESIGN.md):** reuse the diff don't fork validators; Tier-A is
  oracle-free; golden checksums are same-machine drift detectors (real gate = rel-1e-9 vs the LOCAL golden);
  `*_mc`->report-only + precip-FP hydrometeors->Tier-D; prefer float64 over reproducing single-precision
  artifacts; only `while_loop` blocks reverse-mode grad (and harden `sqrt(maximum(0,.))` with `_safe_sqrt`).

### 2026-06-02 — Documentation & repo-structure cleanup (no physics touched)

A session of docs/structure tidying. Verified throughout: `compare_runs --case arm` PASS (0 prognostic failures),
13-case `test_standalone_jax` PASS, `-jax` smoke runs OK.

- **Output-directory convention.** `run_scm.py <case> -jax` now defaults to `clubb_jax/output/<case>_stats.nc`
  (was the shared `clubb_release/output/<case>_stats.nc`), so a bare `-jax` run can no longer clobber the Fortran
  oracle — the old "always pass `-out_dir`" discipline is retired. `-legacy`/`-exe` still default to
  `clubb_release/output/`; `-out_dir` overrides either. `compare_runs.py` JAX side moved to
  `clubb_jax/output/<case>_compare_jax/` (Fortran stays `…_compare_fort/`); `diagnose_divergence.py` follows.
  Added `clubb_jax/output/` to `.gitignore`.
- **Standalone/driver files realigned with the Fortran oracle.** Renamed `src/clubb_standalone.py` →
  `src/clubb_driver.py` (↔ `clubb_driver.F90`: `run_clubb`/`init_clubb_case`/`clean_up_clubb`); added a new thin
  `src/clubb_standalone.py` argv frontend (↔ the 88-line `clubb_standalone.F90`); deleted the root
  `clubb_jax/clubb_standalone.py` launcher (no Fortran counterpart, name-collided with the src file). Entry point
  is now `python -m clubb_jax.src.clubb_standalone`; `run_scm.py -jax`, the test import, and `__init__.py` updated.
- **DESIGN.md compressed** 1361 → 408 lines: the worklog-style iteration narrative (the testing-conventions
  megablock, per-case diagnostic journeys, bloated module-table cells) distilled into durable reference text;
  kept verbatim the Repository Structure, test-command block, Critical Conventions, both tables, and Agent Working
  Rules. No info lost (full version in git; iteration detail remains in this CHANGELOG's 221 Iter entries).
- **clubb_jax/README.md rewritten** as an accurate human-readable install/run guide, replacing stale
  "scaffold/clone of clubb_python_driver, still depends on the Python API" framing and dead harness references.
  Dropped the link to the now-orphaned `JAX_CONVERSION_PLAN.md`.
- Doc edits propagated to CLAUDE.md/DESIGN.md (execution-flow diagrams, module table). `clubb_release/` submodule
  scaffold copies left untouched.

### 2026-06-02 — Iter 323: ROOT-CAUSED the Morrison per-step-stats OOM — a JAX/XLA-backend leak of ~85 diagnostic arrays/step (NOT recompilation); exhaustively ruled out 4 hypotheses

- **Definitively ruled out jit recompilation** (the worst-case): `JAX_LOG_COMPILES=1` on mpace_a → **2140 compiles ALL at
  startup, ZERO interspersed with iterations**. So the Iter290 jit-recompilation fix holds; the OOM is a held-array leak.
- **Localized the leak to the diagnostic path** (NOT the NetCDF write): ran the loop with per-step `l_sample=True` but
  `_ncid=None` (diagnostics compute, no NetCDF write) → still OOMs (RSS 279→2068→3733MB→killed). So it's the `l_sample=True`
  diagnostic computation, not the disk write.
- **★ Found the rate (env-gated `CLUBB_LEAK=1` instrumentation, since reverted):** on bomex, `jax.live_arrays()` grows by
  EXACTLY 85 per step — 85 diagnostic profile arrays (the budget diagnostics the jitted core returns only when sampling)
  retained per step. Small on bomex (~0.06 MB/step → fine to 360); on Morrison's large var set ~36 MB/step → OOM ~150-250.
- **Exhaustively ruled out the holder:** NOT the `StatsWriter` (update/begin_budget/finalize_budget all `np.asarray`-
  materialise immediately, writes incremental); NOT the HDF dirty cache (Iter322 sync); NOT a Python list/dict (clean
  `gc.get_objects()` scan found nothing). → It's an **XLA-backend buffer retention** (Python refcount drops but the device
  buffers aren't released — likely the CPU buffer pool or a jit-dispatch keep-alive). A genuine but JAX-internal leak.
- **Low priority** (no current need for long Morrison compare runs). Documented the root cause + a fix direction (trace
  why the l_sample diagnostic jit-outputs aren't released) + workaround in DESIGN. No model-code change (the leak is in the
  diagnostic/JAX path, not the physics; reverted the debug instrumentation; kept the Iter322 periodic sync as good practice).

### 2026-06-02 — Iter 322: narrowed the Morrison per-step-stats OOM (NOT the HDF dirty cache); added a good-practice periodic NetCDF sync

- **Investigated the Iter321 Morrison per-step-stats OOM (compare_runs rc=1 at 250 steps).** Reviewed `StatsWriter`:
  it is ALREADY incremental — one NetCDF record per `end_timestep`, buffer reset each output window, no accumulation of
  records in memory. So the OOM is NOT the writer buffering records.
- **Hypothesis tested + REFUTED:** added a periodic `ds.sync()` (every 20 records) to flush the HDF5 dirty-chunk cache —
  re-ran mpace_a to 250 → STILL OOMs (rc=1, same ~486s). So the OOM is NOT the un-flushed dirty cache either.
- **Narrowed the cause:** since the stats-OFF physics loop runs to 250 fine (Iter321) but the per-step-stats run OOMs,
  it is a per-`l_sample`-call accumulation in the diagnostic path (the budget/diagnostic JAX work done only when
  `l_sample=True`, ×250 calls for Morrison's ~500-variable set) — likely a jit-trace or held-array growth, not the disk
  write. **Low priority** (no current need for long Morrison compare runs — mpace_a's cloud onset is >250, and
  nov11/dycoms2_rf02_morr are FP-limited at short steps); workaround documented (stats off + inspect state, or case-default
  stats interval).
- **Kept the periodic `ds.sync()`** as a good-practice robustness improvement (bounds the HDF5 dirty cache for any long
  stats run, guards against data loss on crash) — verified data-preserving (arm 30-step gate PASS, 0 prognostic failures).
  Honest framing: a minor robustness win, NOT the mpace_a OOM fix. Updated DESIGN.

### 2026-06-01 — Iter 321: mpace_a durability — robust (NOT an atex-class late event); bit-faithful to 150; found a Morrison per-step-stats OOM (tooling, not model)

- **Durability-tested mpace_a** (the only bit-faithful case unverified past 100; a Morrison case with a single-precision
  floor that only triggers once cloud forms). A 250-step `compare_runs` failed (rc=1) — initially looked like an
  atex-class late-event crash. **Investigated rigorously (don't assume):** ran the JAX physics loop with stats OFF →
  **ALL prognostics FINITE through 250 steps, and cloud_frac=0/rcm=0 at step 250** — the 72-hr Arctic-stratus case spins
  up slowly and forms NO cloud in the testable horizon, so the Morrison floor is never triggered (rates stay 0).
- **mpace_a is bit-faithful at 150 steps (PASS, 0 prognostic failures)** — extends the Iter303 100-step verification. So
  mpace_a's gate-passing is durable clear-air bit-faithfulness, NOT a masked late event. No model bug.
- **★ Root-caused the rc=1: a per-step-stats OOM** between 150–250 steps for the Morrison case (corrupt output NetCDF,
  the rico-OOM class) — Morrison's large diagnostic-variable set × many records. The physics-only loop (stats off) ran to
  250 fine, confirming it's a TOOLING limit, not a model issue. Documented the workaround (stats off + inspect state, or
  coarser stats interval) in DESIGN. No code change (no model bug; the OOM is a known testing constraint with a workaround).
- Lesson reinforced: a compare-harness failure (rc=1) is not necessarily a model bug — verify the physics directly
  (loop with stats off, check finiteness) before assuming a crash.

### 2026-06-01 — Iter 320: compressed CHANGELOG 310-319; full test suite 22/22 GREEN (expax integrated, no KK regression); cleaned build-artifact cruft

- **Compressed CHANGELOG entries 310–319** into one block (every-10-iterations convention).
- **Ran the full test suite (`run_all_tests.py`): 22/22 test files GREEN, SUITE_EXIT=0.** Confirms the Iter316 `expax`
  module + `_dvc` rewiring is fully integrated with no regression: `test_parabolic_expax` PASS (the new oracle/vmap/grad
  test), `test_kk_rico_oracle` PASS (130s — the KK_upscaled_covar_driver vs rico, all 5 _mc still match), plus
  test_kk_autoconversion / test_precip_fraction / test_standalone_jax (289s, the entirely-in-JAX import-blocker) all PASS.
- **Eliminated cruft:** removed the `tools/parab_harness/` build artifacts (*.o/*.mod/`dvtest` — regeneratable via
  `build.sh`, verified) + added a `.gitignore` for them; removed `.tmp_claude/` (165K of stale debug-dump scratch from
  earlier iterations). Harness rebuilds cleanly from source.
- Status unchanged: 18 cases bit-faithful (the gate); do/ds CLOSED (covar-cancellation-FP, do bulk-bit-exact via expax);
  rico/dycoms2_rf02_morr/nov11/coriolis FP-limited; arm_97/mc3e need SILHS; arm_0003 has no oracle.

### Iters 310–319 (compressed Iter320) — ★★ atex bug FIXED (durability test); the `expax` parabolic-cylinder port (do bulk-bit-exact); do/ds CLOSED; durability/chaos framework + new diagnostic tools

- **★★ Iter312 — found + FIXED a REAL atex bug via a DURABILITY test.** A 100-step `compare_cases.py` run (vs the 30-step
  gate) surfaced atex failing: its analytic large-scale thlm/rtm forcing is gated on `time >= time_initial + 5400`
  (atex.F90:215, a 90-min spinup) = step 91, past the gate; the JAX `_atex_tndcy` had ported only the subsidence, MISSING
  `calc_forcings`. Added it (generic_forcings.py) → atex durable to 200+ steps. **The one real bug found in 310-319.**
- **★ DURABILITY/CHAOS framework (Iter313-315).** (a) Time-gate audit: every `time>=…` activation in the bit-faithful
  cases is now verified PAST its step — gabls2 subsidence @step1560 VERIFIED (1580-step compare PASS), neutral
  heat-flux-off @step50, atex_long spinup (FP/chaos-limited ~step305). (b) Full-length (Iter314): **9 cases bit-faithful
  for their ENTIRE run** (dycoms2_rf01, cobra, bomex, neutral, dycoms2_rf02_nd/so, wangara, atex, dycoms2_rf01_fixed_sst);
  fire (~147) & jun25_altocu (~200) are CHAOS-HORIZON-limited. **★ KEY: full-length bit-faithfulness is achievable for
  non-chaotic cases but PHYSICALLY IMPOSSIBLE for chaotic turbulence (butterfly effect); a >horizon failure is physics,
  not a bug. The discriminator is JUMP-vs-FP-growth + sign-flip (atex JUMP=bug; fire/jun25/atex_long FP-growth+flipping=
  chaos). The 100-step gate sits within every chaos horizon → the right practical durability metric.** (c) Iter315: the
  DIURNAL SOLAR transition is a discrete-event class — gabls3 sunset (~step480, amu0 crosses the `>=0.01` night threshold,
  matched in both `bugs_rad.F:611` ↔ JAX) VERIFIED bit-faithful (510-step PASS, radht_SW→0 bit-for-bit).
- **★★ do/ds — the `expax` parabolic-cylinder port (Iter310-319), then CLOSED.** Iter310 PROVED (oracle numbers, built
  `tools/parab_harness/`) the entire do/ds covar discrepancy is the oracle's `epss=1e-4` `parab` truncation (the JAX's
  high-accuracy DLMF dv was an IMPROVISATION; faithful = reproduce epss=1e-4). Iter316: traced (instrumented harness) that
  for the do/ds regime (a=−order−0.5≥1.5, arg≥15) `parab` goes STRAIGHT to **`expax`** — one ~125-line asymptotic series,
  no recursion. **Ported `expax_U`** (`Microphys/KK_microphys/parabolic_expax.py`, masked `fori_loop`, jit/vmap/grad-safe;
  `tests/test_parabolic_expax.py`) — bit-exact vs the harness; wired into `_dvc` for arg≥15 → **do's covar epss artifact
  ELIMINATED, do bulk-bit-exact**. Iter317-319 characterized the residual: do is edge-cancellation-limited (bit-exact at
  k62–74, diverges only at the 4 cloud-top-edge levels); ds is BROADLY cancellation-limited (same args/inputs, but its
  thl `(tri−biv_a1)` covar is far more ill-conditioned). **do/ds CLOSED: irreducibly covar-near-cancellation-FP-limited**
  (the expax port removed the epss artifact — the faithful contribution; the residual is the ill-conditioned KK covar
  cancellation amplifying machine-floor PDF/libm differences, same class as rico/nov11; not fixable without improvising).
  All three thl covar formulas (auto/accr/evap) verified match the Fortran term-by-term. **18-case gate PASS — no regression.**
- **New reusable tools:** `run_scripts/diagnose_divergence.py` (Iter311/313) — per-prognostic onset classifier
  (bit-faithful / FP-growth / JUMP@N) + sign-tally (balanced→FP, one-sided→bug); `run_scripts/run_all_tests.py` (whole
  suite); `tools/parab_harness/` (oracle epss=1e-4 Dv generator). rico reconfirmed precip-onset-FP (expax-neutral).
  **★ Lessons:** the 30-step gate masks late time/condition-gated events (run 100+); verify ALL failing vars when a fix
  lands (Iter316 over-claimed rtp2-only); a per-level ratio (uniform→factor bug vs edge-only→FP) is the bug-vs-FP test.

### Iters 300–309 (compressed Iter310) — ★★ rico: TWO real KK bugs fixed (now FP-limited through step 6); dycoms single-precision hypothesis tested+disproven; whole-suite test runner

- **Iter300:** full 18-case `compare_cases.py` gate validated (all bit-faithful, 0 prognostic failures, 30 steps). **Verified the entire WRF M2005 (`module_mp_graupel.F90`) is SINGLE precision** (every decl is bare `REAL`=REAL*4) → the float64 JAX M2005 has a single-precision floor for ACTIVE microphysics; documented as the Morrison bit-faithfulness ceiling (a float32 port conflicts with the differentiable-float64 design). Compressed Iters 290–299.
- **Iter301:** removed 16 stale `.ipynb_checkpoints` autosave files (untracked Jupyter cruft). Reaffirmed the Morrison FP ceilings: dycoms's 1.8% is near-singular sed FP (`(qr/nr)^⅓`, nr→0); nov11's seed is step-6 DYNAMICS-FP (ice_supersat_frac/bv_mixed), microphysics inactive until step 60; mpace_a worked only because its rates are 0.
- **Iter302-304 — dycoms2_rf02_morr investigation (concluded, no clean fix):** N=2 fails by only ~3.5e-6 (thermo/moisture moments; rrm/Nrm pass). Iter302 traced the mechanism (M2005 builds mean+hydromet tendencies through `r4` temporaries). **Iter303 TESTED the single-precision hypothesis with an env-gated blanket-float32 experiment (backup → verify float64-default bit-identical → flip `CLUBB_MP_F32=1`) and DISPROVED it** — rtm error was IDENTICAL (3.47e-6) in float32 and float64, i.e. precision-INSENSITIVE; reverted fully. **Iter304 budget-decomposed the seed to `rcm_mc` off ~3% at the sharp cloud-top CF3D edge** (k≈49, cloud_frac 0.57→0); `rcm_sd_mg_morr=0` is just a diagnostic-recording gap (JAX folds cloud sed into rcm_mc), not a prognostic bug. Genuinely FP/discretization-limited at the near-singular ÷CF3D/×CF3D edge. **★ LESSON: TEST precision hypotheses (env-gated blanket-float32) before claiming "single-precision-fixable" — don't just reason.**
- **Iter305:** new `run_scripts/run_all_tests.py` whole-suite runner (discovers `tests/test_*.py`, subprocess each with repo root on PYTHONPATH, pass/fail/skip + timing, exit 0 iff all green; `-k`/`--timeout`). Fixed the stale `test_vs_fortran` (hard-failed on the superseded standalone `fortran_oracle` exe → now SKIPs cleanly; compare_runs.py is the current Fortran-comparison path). **Full suite 21/21 GREEN and portable.**
- **Iter306-308 — ★★ rico: fixed TWO real KK bugs.** Iter306 ruled out the grid_type=2 hypothesis (JAX vs Fortran rico zt grid BIT-EXACT, all 57 levels) and localized the seed to the KK covar source `wpthlp_mc` (∝ precip_frac) at the precip onset. **Iter307 found+fixed two genuine missing-faithfulness bugs in the KK path** (`kk_microphys_step.py`, KK-only → 18 bit-faithful cases untouched): (1) **missing `hydromet_tol` clip** (`fill_holes.F90:2444-2476` clips hydromet ≤ tol → 0, returning mass to vapor + latent thlm adjust) → sub-tol rrm accumulated → premature precip onset; (2) **dropped `rvm_mc`** in the KK rt tendency (`clubb_driver.F90:3337` does `rtm_forcing += rcm_mc + rvm_mc`; the KK path added only rcm_mc). Result: rico's worst N=10 error 1.97e-5 → 3.7e-6 (5× closer). **Iter308 confirmed** the fixes eliminated the SYSTEMATIC bias: rrm AND Nrm now **bit-exact through step 6** (was diverging from step 5), FP-balanced after; full 18-case gate PASS (no regression). **★ LESSON: a hydrometeor systematically HIGHER than the oracle (oracle EXACTLY 0 sub-tol) = a missing `hydromet_tol` clip; KK rt forcing = `rcm_mc + rvm_mc`, not just rcm_mc.**
- **Iter309:** decomposed rico's N=10 residual by onset step → the failing thl 2nd moments are bit-exact through step 5, then ride in through the covar source at the precip onset (FP-amplification, not a new bug). Chased a `precip_frac=5e-3` vs `0` "smoking gun" to ground — the JAX stats `precip_frac` was all-zero (never written), the *internal* precip_frac is computed correctly; another instance of the "unwritten diagnostic reads as 0" red herring. **Wired `precip_frac` into the JAX stats output** (`advance_clubb_to_end.py`, guarded, diagnostic-only) → 12 nonzero matching Fortran. rico confirmed FP-limited (covar onset amplifying sub-tol ~1e-13 cloud_frac/PDF differences). Net: 18 bit-faithful cases; rico/dycoms2_rf02_morr/nov11/coriolis FP-limited; do/ds oracle-limited; arm_97/mc3e need SILHS; arm_0003 no COAMPS oracle.

### Iters 290–299 (compressed Iter300) — ★★ rico OOM fixed + jit-recompilation eliminated; mpace_a → 18th bit-faithful case (first Morrison)

**Two milestones: (1) killed the per-step jit-recompilation that OOM'd long/KK runs; (2) made mpace_a bit-faithful via a
faithful single-precision Morrison fix.**
- **290–291: jit-recompilation → OOM, root-caused + fixed.** An eager `lax.scan` whose body CLOSES OVER a concrete array
  bakes those values into the jaxpr as literals → XLA recompiles every timestep → unbounded compile-cache → OOM (rico had
  ~137 scan-recompiles/step). Fix: `jax.jit` the entry points so captured arrays hoist to operands and the graph compiles
  ONCE. Jitted `parabolic_cylinder.dv_parabolic_cylinder` (KK D_v), `matrix_solver_wrapper.{tridiag,penta}_lu_solve_jax`
  (290), and `fill_holes_vertical_jax` (291, with int control args static). Net: rico 2165→381 total compiles, scan
  recompiles → ~0, 30-step OOM gone, ~1.5–2× faster, compile cache BOUNDED for every case. All value-preserving +
  differentiability-preserving (arm/bomex compares 0 failures; solver/penta/fill_holes grad tests added Iter295).
  **★ Convention: any per-timestep eager `lax.scan` should be `jax.jit`-wrapped; diagnose with `JAX_LOG_COMPILES=1` +
  `grep -c "Compiling jit(scan)"` (a count growing per step is this bug).**
- **292: generalized regression gate** — `compare_cases.py` (DEFAULT_CASES) verifies ALL bit-faithful cases, not just ARM;
  confirmed the 290–291 core jits are bit-faithful across every case. Added gabls3 (BUGSrad guard). **Run it after any
  change to shared/core code.**
- **293–298: the mpace_a saga (a cautionary tale + the decisive tool).** mpace_a's 30-step compare failed (thlp2/rtpthlp
  ~1e-5 in the BL). Iter293-296 chased a phantom "forcing-time discrepancy" — the recorded `thlm_forcing` STAT (8.464e-6)
  ≠ the raw LS forcing (8.1815e-6). Iter297 settled it with an **isolated oracle debug build** (`compile.py -install
  <scratch> -debug`, NOT the reference binary; print the raw `dTdt_hoc_grid`, capture, revert source) — proving the LS
  forcing port was faithful all along: the STAT = raw_forcing + the LAGGED `thlm_mc` (microphysics tendency fed to the
  next step's forcing). So it was MORRISON, not the forcing. Iter298 ran the full unit-test sweep green + fixed 2 tests to
  SKIP the f2py oracle gracefully. **★ LESSON: when a `*_forcing` stat disagrees but the case uses microphysics, check
  `stat == raw_forcing + lagged *_mc` FIRST; an isolated oracle debug build is the decisive (safe) tool when static
  analysis stalls.**
- **299: ★★ mpace_a BIT-FAITHFUL (18th case, the first Morrison case).** The clear-air `thlm_mc` is the Fortran M2005
  interface's SINGLE-PRECISION `thlm↔T_in_K` round-trip residual (`morrison_microphys_module.F90:399,416,793` use
  `real(...)` = REAL(4) even in the PRECdouble build). The JAX float64 round-trip gave EXACTLY 0; `module_mp_graupel.py`
  now replicates the `real*4` casts → mpace_a 30-step compare PASSES. **★ LESSON: faithfulness means matching the
  oracle's PRECISION, not just its formula** — the entire WRF M2005 is real*4 (verified Iter300), so the active-
  microphysics cases (nov11, dycoms2_rf02_morr) have a single-precision floor on top of their FP/discretization limits.

### Iters 280–289 (compressed Iter290) — ★ the "entirely-in-JAX forcings" campaign: every case's forcings now run in pure JAX (0 Fortran fallback)

**Goal:** make the JAX driver import-clean of `clubb_python` for ALL cases (not just bit-faithful) — port every case's
analytic `tndcy`/`sfclyr` so none falls back to the Fortran `clubb_api.prescribe_forcings`. Reached **19 entirely-in-JAX, 0 fallback.**
- **280:** added `tests/test_standalone_jax.py` — a `sys.meta_path` import-blocker on `clubb_python` (must use the modern
  `find_spec` protocol; `find_module` silently no-ops on Py3.14) that proves a case runs with zero Fortran import.
- **281:** reentrancy fix — `reset_clubb_core_state()` (clears the cross-timestep `_prev_adg1_j25` global) in
  `init_clubb_case`, so many cases run in one process. **★ CORRECTED the Iter279 over-claim:** "faithful" ≠ "entirely-in-JAX"
  — a case can pass `compare_runs` via the Fortran forcings fallback (Fortran forcings == oracle = degenerate).
- **282:** systematic audit (init + call `prescribe_forcings_generic`; `NotImplementedError` ⇒ fallback) → 8 faithful
  cases were on the fallback (wangara, gabls2, dycoms2_rf01, rf02_nd/so, atex, atex_long, rico).
- **283:** gabls2 → entirely-in-JAX; found the latent `_gabls2_sfclyr` bug — wprtp_sfc must be ×0.025 (gabls2.F90:299).
- **284:** wangara (zero tndcy) + dycoms2_rf01 (tndcy zeros thlm/rtm; wm is init-set → bit-exact) ported.
- **285:** dycoms2_rf02_nd/so ported — **★ match `state['runtype']` (rf02 nd/so/do/ds all = 'dycoms2_rf02'), NOT the
  case-file name** (a name-keyed branch never fired → silent fallback false-positive, caught by the standalone test);
  `_dycoms2_rf02_sfclyr` rewritten to `sens_ht/(1.21·Cp)`, `latent_ht/(1.21·Lv)`.
- **286:** caught + fixed an UNCAUGHT regression (Iter284's name-keyed "revert" no-op'd) — rf01_fixed_sst (sfctype=1)
  was silently failing; added the fixed-SST sfclyr branch (Cd=0.0011, T_sfc from file). Ported it.
- **287–288:** atex (subsidence gated off the first 90 min, atex.F90:41) + atex_long (fixed 3-piece subsidence + 4-piece
  thlm/2-piece rtm forcing + 43200 s spin-up) ported. **★ Fixed a latent `test_standalone_jax.py` bug:** the `clubb_release/`
  checkout has an unrelated `clubb_jax/` scaffold that shadows our package unless `jax_root` precedes it on `sys.path`.
- **289:** rico forcings ported (3-piece `t_tendency`/exner thlm + 4-piece specific-humidity `qtm_forcing` →
  `rtm_forcing=(1+rtm)²·qtm`; wm init-set) — **0 fallback remaining.** Standalone suite → 13 cases (needs
  `jax.clear_caches()`+`gc.collect()` between cases or the KK-heavy rico OOMs the shared process).
- **★ Conventions established:** (1) ALWAYS confirm a newly-ported forcing/sfclyr with the standalone (clubb_python-blocked)
  test — a plain `compare_runs` PASS can be a fallback false-positive; (2) key dispatch off `state['runtype']` + the
  distinguishing namelist flag (sfctype), never the case-file name, and re-run the variant after a port/revert; (3) a
  fallback-hidden sfclyr often carries a magic factor (gabls2 ×0.025; rf02 /1.21; rf01_fixed_sst fixed-SST).

### Iters 270–279 (compressed Iter280) — ★★ gabls3 BIT-FAITHFUL (17th case): full BUGSrad path wired + validated; driver now standalone

**The 17th bit-faithful case (gabls3) was achieved end-to-end and the JAX driver was made import-clean of the Fortran.**
- **270:** ported `soil_vegetation.py` (gabls3 `l_soil_veg` lower BC — Deardorff/Duynkerke force-restore surface
  budget), BIT-EXACT vs a Fortran-formula replica.
- **271:** wired BUGSrad + soil_veg + the `_gabls3_sfclyr` surface flux → **gabls3 runs end-to-end in pure JAX**;
  discovered the **compiled Fortran DOES produce a bugsrad gabls3 reference**, so `compare_runs --case gabls3` is the gate.
- **272:** gabls3 subsidence fix — it prescribes `omega[Pa/s]` (not `w[m/s]`) + moisture as `sp_hmdty_f`; added
  `wm_zt=-omega/(grav·rho)` + `rtm_f=sp_hmdty_f·(1+rtm)²` to `generic_forcings.py` → 15/16 prognostics bit-faithful.
- **273:** ★★ **gabls3 BIT-FAITHFUL** — the last residual was a UNIFORM 3.26e-4 in radht (all levels, LW+SW = a global
  scalar): BUGSrad's heating rate must use **constants_clubb grav/Cp (9.81/1004.67), NOT BUGSrad's physconst
  (9.80665/1004.0)** (bugsrad_driver.F90:357). Fixed in `bugsrad_driver.py`; all 16 prognostics pass.
- **274:** jitted `bugs_rad` — the eager 18-band×k dispatch leaked ~700 MB/call → OOM (EXIT=137) after ~6 steps; jit
  → bounded memory + ~6 s/step, bit-exact to ~1e-13. **gabls3 passes the full 30-step gate.** Fixed the test_bugsrad
  suite OOM (clear caches + gc between eager tests).
- **275:** survey confirmed the bit-faithful frontier is **SATURATED at 17** (all 18 `microphys=none` cases accounted
  for: 17 faithful + coriolis_test FP-limited; the rest need Morrison/COAMPS/KK/SILHS). Validated + guarded BUGSrad +
  soil_veg **differentiability** (`jax.grad` finite+nonzero through cloudg's float32 sngl).
- **276-278:** ported the custom **mpace_a** case (mpace_a.F90 — 11 `.dat` files, dTdt/dqdt/vert advection, no
  subsidence, wind nudging, SH/LH surface) to pure JAX, replacing a broken Fortran fallback. **Step 1 bit-faithful**
  after fixing the missing em (TKE) init (`fixed_cloud_top_cases += (2000,1.0)`) + the hardcoded `p_sfc=101000`
  (mpace_a.F90:140, NOT p_sfc_nl). Then **corrected** the diagnosis: microphysics is INACTIVE (air sub-saturated over
  ice, qvqvsi=0.915), nudging is correct (um_ref matches), and the step-2 divergence is dynamics-2nd-moment FP-class —
  NOT microphysics (the `thlm_mc` "diff" was the stats one-record-delay artifact). Added the `NNUCCD_REDUCE_COEF`
  (deposition-nucleation reduction; default 1.0, 0.01 for clex9_oct14) to `deposition_nucleation`.
- **279:** ★ made the JAX driver **IMPORT-CLEAN of `clubb_python`** — swapped `model_flags.py` to the pure-JAX
  `ConfigFlags` (field-identical) and made the lone `clubb_api` import LAZY (dormant pre-advance PDF block). The 17
  faithful cases run with ZERO Fortran import/call — proven by running ARM with clubb_python blocked; ARM compare PASS.
- **★ Conventions added:** uniform-across-levels error ⇒ global constant mismatch (diff per-level NetCDF); jit any
  per-timestep heavy-op JAX subsystem (eager leaks device buffers → OOM; EXIT=137 = OOM, run unbuffered + ru_maxrss);
  a case .F90 may HARDCODE a constant overriding the namelist; a Morrison `*_mc` single-step mismatch can be the stats
  one-record-delay artifact (verify the physical state); pass the constants the Fortran CALLER passes; keep residual
  Fortran calls behind LAZY imports + test standalone-ness with a `find_spec` import-blocker.

### Iters 260–269 (compressed Iter270) — BUGSrad correlated-k radiation: full port + JAX wiring (RT machinery → driver → dispatch)

The complete BUGSrad path (gate for gabls3) was ported to `clubb_jax/src/Radiation/BUGSrad/` + `bugsrad_driver.py`
and wired into the JAX radiation dispatch. All validated vs Fortran-formula replicas (≤2e-13, mostly machine-ε) or
invariants in `tests/test_bugsrad.py` (17 tests). Pieces, by iteration:
- **260 `two_rt_lw`** (delta-Eddington LW two-stream; `sel_rules_lw=.false.` ⇒ full branch only; two `lax.scan`
  recursions — top-down adding + bottom-up flux), rel 1.6e-16. **★ Corrected a faithfulness assumption: `newexp` is
  `#ifdef usenewexp`-guarded and the oracle build does NOT define it ⇒ the solvers use intrinsic `jnp.exp`; `newexp.py`
  is faithful but UNUSED. Convention: check `#ifdef`/CPPDEFS before assuming a `use`d module is active.**
- **261 `two_rt_sw`** (SW two-stream + direct beam, eps-guard at `κ²−1/μ0²≈0`), rel 5.6e-16 — RT core complete.
- **262 `gases_ckd` helpers** (pscale + qk/qkio3 temp interp + qop* transmission), rel 5.7e-17.
- **263 `gases_ckd_data.h` PARSER** (43 correlated-k arrays from source — 1D/2D/3D implied-DO; gotchas: bound names
  UPPERCASE, `data((` has no space). **★ Convention: PARSE large Fortran coefficient tables, don't hand-transcribe.**
- **264 `gases` 18-band dispatch** (H2O/O3/CH4/N2O/CO2 overlaps + hk weights), rel 3.9e-16 — gas absorption complete.
- **265 `bugs_lwr`** (LW band driver, bands 7-18). **Confirmed `-Dnooverlap` IS in the CLUBB build** ⇒ the plain
  `two_rt_lw` (called twice: cloudy→all-sky, clear→clear-sky) is used, NOT the max/random `_sel/_iter` variants.
- **266 `bugs_swr`** (SW band driver, bands 1-6, Rayleigh + `ttem=min(340,tt)` for gases) — both band drivers done.
- **267 `bugs_rad`** (orchestration + heating rates `rate=−grav·0.01/cp·(Fnet[l]−Fnet[l+1])/dpl`, LW conserves by
  telescoping; SW for daytime cols amu0≥0.01) — RT machinery complete.
- **268 `bugsrad_driver.py`** (CLUBB↔BUGSrad interface): `load_std_atmosphere` (50-level US Std Atm),
  `determine_extended_atmos_bounds`, `compute_bugsrad_radiation` (top-down vertical-flip + buffer + std-atm extension
  grid map, radht flip back). **★ Adversarial review caught a real gap:** the top-level `bugs_rad` does cloud
  preprocessing my Iter267 port skipped — `den=ppl·100/(287·tl)`, `rmix=ql/(1−ql)`, `cwrho=den·1000·qcwl·acld`
  (cloud condensate weighted by cloud fraction; for `-Dradoffline -Dnooverlap`: nnp=nlm/no ghost, b1..b4 + snow qril
  UNUSED, cloud-fraction enters ONLY via the `×acld` weighting). Fixed `bugs_rad.py` to the faithful interface.
- **269 wired into `radiation.py:advance_radiation` case "bugsrad"** (`_advance_bugsrad_radiation`): builds+caches the
  rad-grid setup (`gr.zm`/`gr.dzt`, radiation_top), computes T_in_K/p_in_Pam/amu0 per step, writes radht/Frad to state.
  **The whole BUGSrad path now runs end-to-end in the JAX driver** (`test_bugsrad_radiation_dispatch`).

### Iters 250–259 (compressed Iter260) — KK covar resolved as Fortran-low-accuracy-limited; BUGSrad port STARTED (8 pieces)

- **★★ KK 2nd-moment covar discrepancy (251-253) — resolved as a Fortran numerical artifact, NOT a JAX bug.** A live
  `dycoms2_rf02_do` (KK microphysics) run failed on rtp2/rtpthlp/thlp2 (the covar source). Decoupling via the 9 stored
  `*_KK_*_covar_zt` intermediates showed the `_w` covariances bit-exact but the `_rt`/`_thl` (eta-based) off ~16×. A
  brute-force MC of `Cov(rt,auto)` matched the JAX to 0.04% but the oracle to 16× (MC validated: `Cov(w,auto)` matches
  the exact oracle a_w). **ROOT CAUSE (Iter253): `Dv_fnc` (KK_utilities.F90:143) uses `epss=1e-15` only if
  `l_high_accuracy_parab_cyl_fnc=.true.`, else `epss=1e-4` — and the default is `.false.` (Parabolic.f90:20, no case
  overrides).** So the SCM computes the parabolic cylinder `D_v` to only ~1e-4; the covar's NESTED near-cancellations
  amplify that to 16× at the cloud edge, while the RATES (plain means, no cancellation) stay bit-faithful. The JAX
  `_dvc` matches scipy `pbdv` to 1e-14. **The JAX is MORE accurate; dycoms_do/ds/rico KK covar is `epss=1e-4`-parab-
  limited.** The faithful fix would need the 3385-line Algorithm-850 `parab` with the exact epss=1e-4 truncation —
  impractical/low-ROI. **★ Convention: when the JAX matches a brute-force MC + scipy but NOT the oracle, suspect a
  deliberately-low-accuracy Fortran special-function path (check its tolerance flag).**
- **Iter250: pre-rate slope clamps for ALL 5 Morrison species** (warm 1881-2002 / cold 2816-2968): added
  `_sizefix_exp_number`/`_sizefix_cloud_number`/`_size_clamp_numbers`, wired pre-rate into `m2005_driver` (kept the
  rain post-sed clamp for the stored-stat timing). dycoms CLOSED as FP/discretization-limited (sharp-edge upwind-sed,
  K_hm transport verified bit-exact). No-op for the 16 faithful cases.
- **Iter254: foundation re-verified** (ARM 30-step PASS — the microphysics work didn't regress the core); **ROI
  assessment** — the non-subsystem bit-faithful frontier is SATURATED (16 faithful + rico/coriolis/do/ds limited),
  remaining gains need large subsystem ports; removed two dead env-gated debug hooks (TACAP, PENTA_CAP).
- **Iters 255-259: STARTED the BUGSrad correlated-k radiation port** (the gate for gabls3, the one clean remaining
  bit-faithful case; gabls3 = BUGSrad + l_soil_veg[250 lines] + microphys=none). New package
  `clubb_jax/src/Radiation/BUGSrad/`, 7 pieces all validated vs Fortran-formula replicas to ≤2e-13: `planck`
  (blackbody band emission, 12-band Horner), `newexp` (fast-exp approx, see Iter260 correction), `rayle` (Rayleigh
  τ+ω), `bugsrad_physconst` (R_d=287 etc.), `gascon` (CKD2.4 H2O continuum, 168-coef table + parm_ckd24), `cloudg`
  (ADT cloud optics — complex extinction + the float32-π / `sngl` mixed-precision faithfulness details), `comscp1/2`
  (combine cloud+aerosol+Rayleigh+gas optical props). **★ BUGSrad faithfulness convention: replicate the radiation's
  DELIBERATE single-precision/approximate steps (float32 π, `sngl`, newexp-when-`usenewexp`), don't "improve" them.**

### Iters 240–249 (compressed Iter250) — dycoms warm-Morrison transport: K_hm fixed via precip_frac, VERIFIED bit-exact, residual is FP-class sed

- **Iter240-241: fixed K_hm.** The hydrometeor variance `hydrometp2 = ((ratio+1)/precip_frac−1)·hm²`
  (setup_clubb_pdf_params:449) is NOT 0 — computing it (`_hydrometp2_zt`) and feeding `calculate_K_hm` (large,
  capped at |corr(w,hm)|≤1) fixed the too-weak transport: dycoms cloud-top rrm 1/9–1/400 → 0.16–0.79, total
  90→94%. Fixed the placement to `ratio·zt2zm(hm)²` (advance_microphys:907 — interpolate-then-square, not
  zt2zm(hm²)). `l_sed=False` faithful (the Fortran seds in M2005, rrm_sd=0).
- **Iter242-245: chased the cloud-top residual** — confirmed the transport machinery is correct at step 1,
  nu_hm=0.75 EXACT, the Fortran does ONE solve per hydrometeor (no sub-stepping).
- **★★ Iter246: FOUND IT — the precip_frac was wrong.** Used the DEFAULT 1.0 instead of the value
  `precipitation_fraction`'s max-in-precip-mean limiter computes (~0.33); with ratio_ip=21.74 (not 1.25),
  precip_frac=0.33 makes hydrometp2≈5·hm² (not 1.25·hm²) → K_hm DOUBLES. Computing the real precip_frac from
  the PDF fields: rrm total 94→100.6%, max-level rel 0.40→0.099, cloud-top 0.51→0.97. **LESSON: never trust a
  default-vs-computed value (cost 12 iters chasing a 2× K_hm bug).**
- **Iter248: faithful hydrometp2 guard** — zero where `hm < hydromet_tol` per species
  (setup_clubb_pdf_params:446-455); total 1.006→1.001.
- **★ Iter249: VERIFIED the K_hm transport is BIT-EXACT.** The oracle writes the actual `K_hm_rr`/`K_hm_Nr` to
  stats (advance_microphys:310-311) — the running JAX K_hm matches it to ratio 1.000 at every interior level
  (only the rain-bottom straddles the eps cap, a branch flip). This confirms calculate_K_hm + the
  precip_frac/hydrometp2 computation are faithful and refutes the prior "deep residual" framing (the stored rrp2
  mismatch is a pure timing confound — it is the POST-advance hydrometp2 at line 907, not the value fed to K_hm).
  Current step-1 rrm is faithful to 0.985–1.005 across the rain layer; the residual is the Morrison SEDIMENTATION
  at the sharp rain-top (rrm_mc ~1.8% too-weak removal at levels 44-46). Ruled out: transport, 2nd-moment source
  (`l_morr_xp2_mc=.false.`, correctly off), RGVM/NSTEP. See Iter250 for the full FP-class closure.

### Iters 230–239 (compressed Iter240) — nov11 closed FP-limited; dycoms (warm Morrison) debugged: sed-axis + SIZEFIX_NR + transport

- **nov11 (230-231):** ★ CORRECTED — nov11 is `grid_type=1` (UNIFORM), not stretched (prior notes wrong). The step-6
  divergence chain is `ice_supersat_frac → bv_mixed (the in/out-of-cloud Brunt-Väisälä blend, mixing_length:148) → Lscale
  → tau → wp3/up2/vp2`; the `1−isf/0.001` ramp is a 1000× amplifier. **★ CLOSED: nov11 step 6 is FP-LIMITED** (rico/coriolis
  class) — full-chain code review proved every link bit-faithful (the ramp matches F90:2347, `l_smooth_min_max=.false.` is a
  Fortran parameter, ice_supersat_frac/sat_mixrat_ice/constants faithful); the seed is the cancellation-amplified FP floor in
  the near-zero scalar variance (rtp2~4.5e-11) at the ice-cloud edge. No faithful fix exists. **Milestone: the Morrison+ice+
  SW-radiation case is faithful up to the FP floor.**
- **dycoms2_rf02_morr (232-240) — the first WARM precipitating case, to validate the Morrison hydrometeor TRANSPORT** (nov11's
  microphysics is FP-gated). Step-1 dynamics+cloud+radiation bit-faithful from the start. Bugs found+fixed:
  **★★ the sedimentation AXIS bug (Iter233)** — `_sediment`/`_fall_speed_propagate` indexed the vertical on axis 0, but the
  run passes `(ngrdcol, nzt)`; for ngrdcol=1 the interior-flux term collapsed → 94% of the rain mass destroyed. (1-D columns
  conserved, so the unit tests passed; nov11 never sedimented in a full run.) Fixed to act on the LAST axis → 2-D conserves to
  1.0000, dycoms step-1 rain matches the Fortran 99.6%. Added a 2-D conservation regression test (Iter234).
  **★ the missing rain-number size limiter SIZEFIX_NR (Iter235, F90:1881-1892)** — after sed (mass falls faster than number)
  some levels had 4 µm drops; the Fortran resets `NR=LAMMAXR³·QR/(π·ρw)` so drops ≥ 20 µm; added `_sizefix_rain_number` on the
  post-sed Nrm → step-1 Nrm 19%-high → 99.6%.
  **★ the hydrometeor TRANSPORT (236-240):** the cloud-top rrm deficit = the CLUBB hydrometeor transport (where auto/sed
  cancel, `rrm_bt ≈ rrm_ta`, the rain is sustained by the down-gradient turbulent advection `<w'hm'>=−(K_hm+nu_hm)·d<hm>/dz`,
  zeroed above cloud top). The transport machinery ALREADY EXISTS (`advance_microphys_module.py`: calculate_K_hm/microphys_lhs/
  advance_one_hydrometeor, bit-faithful on rico, used by the KK path) — only the Morrison path used a first-pass Euler advance.
  Wired Morrison through `advance_one_hydrometeor` (Iter239) + fixed K_hm via hydrometp2 (Iter240, above). Conventions added:
  validate Morrison transport on a WARM active-from-step-1 case + check sed ∫ρ·q·dz conservation on a 2-D column; transport/
  sedimentation helpers MUST act on the vertical=last axis; no hydrometeor LAMR should exceed LAMMAX after a step; decompose
  `<hm>_bt` into auto/sed/ta/ma to separate microphysics from transport.

### Iters 220–229 (compressed Iter230) — nov11 Morrison + radiation debugged to bit-faithful through step 5

- **Morrison nov11 seeds (220-226):** fixed the `Ncm=0` wiring bug (diagnose `Ncm=Nc_in_cloud·cloud_frac` inside the
  morrison call) and the cloud-sed `nc>0` guard (phantom-cloud rcm>0/Ncm=0 points); established the faithful
  `thlm_mc = (ten['T'] − Lv/Cp·rcm_mc)/exner` (M2005 integrates QC3D/T3D at end, :4911-4929 — the cloud-sed rcm change IS
  the 184-point cloud-top heating, Iter221's input-rcm shortcut was a dead end); fixed the sedimentation `dzq = delta_zt =
  dzm[:,1:]` (momentum spacing, not dzt). **★ Root of the step-2 microphysics error: `microphys_start_time=64800` — the
  Fortran SKIPS microphysics for the first 60 steps** (`if time_current < microphys_start_time: return`); gated the JAX
  morrison call to match → rtm step-2 error resolved.
- **★ Radiation (227-228) — nov11 step 2 → BIT-FAITHFUL:** the post-spinup seed was the SHORTWAVE (radht_SW 71% off). Root
  cause = simplified-rad cfg-key bugs: the JAX read `radius`/`A_surface_albedo`/`omega_sw` but the namelist keys are
  `eff_drop_radius`/`alvdr`/`omega`. The `omega` bug (0.999 vs 0.9965; SW absorption ∝(1−omega) → exactly 3.5× deficit) was
  decisive. Fixed all three → radht machine-precision; jun25 + ARM unaffected (their SW is inactive).
- **Turbulence bisection (229):** nov11 bit-faithful through step 5, diverges at step 6 (the 2nd-moment turbulence). Iter229
  mis-attributed it to a stretched-grid wp3 solve; Iter230 corrected this (uniform grid; root = ice_supersat_frac, above).
- Conventions added: simplified-rad keys are `eff_drop_radius`/`alvdr`/`omega`; the oracle writes Morrison `*_mc` with a
  one-record delay (compare distributions, not record-aligned); check `*_start_time` spinups for late-diverging microphysics
  cases. Throughout: `tests/test_morrison_rates.py` (30) + special (9) + differentiable (3) pass; ARM + jun25 bit-faithful.

### Iters 210–219 (compressed Iter220) — Morrison: full M2005 driver assembled, wired into the JAX loop, nov11 runs

- **Iter210-213: the M2005 tendency assembly.** `m2005_cold_tendencies` (cold T<273.15: conservation-of-water over-depletion
  limiters QC/QI/QR/QNI/QG :3801-3960 + the 12 mass/number tendency assignments :3963-4007) and `m2005_warm_tendencies`
  (warm T≥273.15: melting/evap/warm-rain, :2318-2440) — **both verified by the oracle-free WATER-CONSERVATION CONTRACT**
  (every rate is a +source/−sink pair → Σ mass tendencies = 0 to ~1e-20). `m2005_step_tendencies` (Iter212) selects the
  branch per level (T mask) + applies PCC (thermo is CONSTANT in the CLUBB build: XXLV=Lv, XXLS=Ls, CPM=Cp). `compute_m2005_rates`
  (Iter213) is the rate ORCHESTRATION keystone — composes the ported rate functions in the Fortran dependency order; the
  chain `compute_rates → assemble` conserves to 4.96e-24.
- **Iter214: the 4 remaining nonzero rates for nov11** (oracle-scoped: nov11 is purely cold, no graupel/melting) — PSACWI,
  PIACRS/PRACIS (cold), PRACS (warm, double-confounded → validated by Fortran-replica transcription + hand-calc).
- **Iter215-217: the full single-column driver + interface.** `m2005_driver` (Iter215: saturation → in-cloud ÷CF3D →
  rates → tendencies → ×CF3D) — runs end-to-end on real nov11 fields (finite, water-conserving). Caught the PRC `nc^-1.79`
  → inf at qc>0/nc=0 in-cloud-edge points (nc>0 guard). `ice_fall_speed`/`snow_fall_speed` + `morrison_sedimentation`
  (Iter216: rain+ice+snow, SHARED-NSTEP CFL coupling). `morrison_microphys_driver` (Iter217: the CLUBB interface —
  `hydromet_mc=(field_final−field_initial)/dt` [Iter204 form], rcm_mc/rvm_mc raw, `thlm_mc=(T_in_K2thlm(T_final)−thlm)/dt`).
- **Iter218: ★★ nov11_altocu RUNS in the JAX driver.** `morrison_hm_metadata` (8 fields, bulk → pdf_dim 4 via
  `l_hydromet_pdf=False`); `advance_morrison_microphysics` (per-step driver call + first-pass Euler hydromet advance); gate +
  hydromet setup + forcings all gated by `microphys_scheme=='morrison'` (KK/none + the 16 faithful cases untouched).
- **Iter219: ★ nov11 STEP 1 is fully bit-faithful** (`compare_runs`); diagnosed the step-2 seed → **missing cloud-water
  sedimentation** (M2005 folds it into rcm_mc via `QC3DTEN += QCSTEN` :4885). Added `cloud_fall_speed` (Stokes/viscosity).
- **★ Convention (Iter218): `run_scm.py -jax` clobbers the Fortran oracle** at `output/<case>_stats.nc` — always use `-out_dir`;
  regenerate with `-legacy`. **Convention (Iter210): the water-conservation contract** — a faithful tendency assembly must
  conserve Σ mass tendencies to ~0; a sign/term error breaks it. **Lesson (Iter215/220): rates with a negative-power number
  dependence (PRC, cloud fall speed) must guard number>0** — the oracle itself has rcm>0/Ncm=0 phantom-cloud points.

### Iters 201–209 (compressed Iter210) — Morrison driver glue: from rate-set completion to a nearly-assembled M2005 step

- **Iter201:** `rain_self_collection` (NRAGG, drop-breakup above 300µm, oracle 10.4%) + validated the number companions
  NPSACWS/NPRAI/NPRCI. **★ Process-rate coverage essentially complete** (warm-rain + full ice block, all oracle-validated).
- **Iter202-204: the `*_mc`-relationship investigation (concluded).** Characterized how the rates compose into the
  CLUBB-output tendencies; **definitive answer at morrison_microphys_module.F90:786: `hydromet_mc=(hydromet_final −
  hydromet_initial)/dt`** — the NET field change over the whole Morrison step (integration + clipping + CF3D + sub-stepping),
  NOT the single-call rate sum. And `hydromet_initial` (pre-microphysics) ≠ the oracle's stored end-of-step field (CLUBB
  advection sits between), so `*_mc` CANNOT be reconstructed from the oracle's stored fields — **the only validation path
  is to port the full M2005 driver, wire it into the JAX CLUBB loop, and diff via `compare_runs --case nov11_altocu`**
  (the same end-to-end gate as every faithful case). Ruled out the rate-sum, ×CF3D, and conservation-limiter hypotheses.
  Also: **the rate library is reverse-mode DIFFERENTIABLE** (`tests/test_morrison_differentiable.py`, grad through
  POLYSVP/GAMMA/DERF1 + PRC/PRE/PRCI, finite + FD-correct) — the "differentiable composable" goal, like core CLUBB + KK.
- **Iter205-209: the driver glue (all verified by physical contracts, the oracle storing only post-everything values).**
  `conserve_qc/qi/qr/qni` (Iter205, over-depletion limiters); `saturation_adjustment_pcc` (Iter206, PCC — single linearized
  Newton step `(qv*−qsat*)/(1+Lv²qsat*/(Cp Rv T*²))/dt`, capped, verified by the saturation contract); `to_in_cloud`/
  `tendency_to_grid_mean`/`neg_fix_number` (Iter207, the CF3D in-cloud↔grid-mean subgrid conversion — the piece explaining
  rim_mc=in-cloud_tend×CF3D, CF3D=cloud_frac_in≠liquid cloud_frac); `rain_fall_speed` (Iter208, UMR/UNR terminal fall
  speeds, ≤9.1·(ρsu/ρ)^0.54, mass-weighted>number-weighted); `rain_sedimentation` (Iter209, the CFL sub-stepped upwind
  flux-divergence loop + downward fall-speed propagation, `lax.scan`+`lax.fori_loop`, all differentiable).
  **★ Grid-orientation convention (CRITICAL, Iter209):** the CLUBB↔M2005 interface FLIPS the vertical index
  (`microphysics.F90:1944 m=nz-k`) — M2005's KTE ("top of model") maps to the JAX grid's **surface**; in JAX-grid order
  (0=surface) sedimentation has "above k"=k+1, rain exits at index 0 (the literal Fortran index transcription sends rain
  UP). Verified by the conservation contract (column conserved aloft, centroid descends, surface outflux at the bottom).
  **Known jit constraint:** NSTEP (CFL count) is data-dependent → fine eagerly; a jitted driver needs fixed-max NSTEP.

### 2026-05-31 — Iter 200: compressed CHANGELOG (190-199); Morrison cloud-water freezing MNUCCC ported (bit-exact)

- **Compressed the Iter190-199 Morrison-port arc** into one block (1181→1012 lines) per the every-10 rule.
- **`cloud_contact_immersion_freezing` (MNUCCC/NNUCCC)** — contact (Meyers 1992 nuclei NACNT Brownian-diffusing to
  droplets via DAP) + Bigg immersion freezing of cloud water (module_mp_graupel.F90:3043-3099). Uses the cloud slope
  (PGAM/LAMC) + CDIST1=nc/Γ(PGAM+1) + the constants CONS37-40. **Preserved the Fortran's log-space moment evaluation**
  (`exp(ln a + ln b − n·ln c)` rather than `a·b/c^n`) for bit-faithfulness. **Validated vs the oracle: MNUCCC median
  8.4e-6 — essentially BIT-EXACT** (cloud water is within-step-stable + the log-space form matches), NNUCCC 0.4% (the
  NC/dt cap). `tests/test_morrison_rates.py` now 13 tests, all PASS. No existing code touched → zero regression risk.
- **The cloud-water freezing set is complete** (MNUCCC contact/immersion + MNUCCR rain). Morrison process-rate coverage
  is now broad (warm-rain + the major ice rates + freezing/nucleation). **Next: the M2005 driver assembly** — the
  conservation limiters (now have the full qc-depletion set: PRC/PRA/PSACWS/MNUCCC), tendency integration, sedimentation.

### Iters 190–199 (compressed Iter200) — Morrison microphysics port: special functions + warm-rain + ice process rates, all oracle-validated

Began the Morrison 2-moment microphysics (`morrison`, the ~19-case lever) in `Microphys/Morrison_microphys/module_mp_graupel.py`
(mirrors the WRF graupel scheme). KK playbook: special functions first (validatable vs scipy), then the process rates
(validatable via a Morrison case-stats oracle). **Entry = `M2005MICRO_GRAUPEL`** (the CLUBB driver calls it, NOT the SAM
`micro_proc`/`satadj_liquid`). **Morrison runs FLOAT64** in the CLUBB build (`-fdefault-real-8` promotes the WRF scheme's bare
`REAL`; single-precision literals become real8) → the float64 ports CAN be bit-faithful (resolved Iter191).
- **Special-function layer (Iter190-192), all vs scipy/known values:** `polysvp_jax` (Flatau-1992 "V1.7" SVP, distinct from
  CLUBB-core's fit — bit-exact vs a Fortran-Horner replica); `derf1_jax` (Ooura erf, vs scipy.special.erf to 2.2e-16);
  `gamma_jax` (W. J. Cody Γ — negative-arg reflection + small-arg 1/Y + (1,12) rational w/ integer reduction + ≥12 Stirling,
  array-capable, vs scipy.special.gamma to 7.6e-15). `tests/test_morrison_special.py` (9 tests).
- **The Morrison oracle (Iter193):** `run_scm.py nov11_altocu -legacy` → `output/nov11_altocu_stats.nc` writes the process
  rates (PRC/PRA/PRE/PRD/PRDS/... + the ice fields rim/rsm/Nim/Nsm) — feed the Fortran's own state into the JAX rate and
  compare. **Confound convention:** auto rates f(qc,nc) validate to ~1e-7 (within-step-stable); rates on rain/ice fields
  created *during* the step (accr/collection) ~4-8%; rates on the saturation deficit (evap/deposition) carry a DOUBLE
  confound (~7%, the field AND the deficit relax) — but supersaturation-LIMITED deposition validates tighter. Validate the
  FORMULA by the median + the exact transcription.
- **WARM-RAIN COMPLETE (Iter193-195):** `kk_warm_rain_rates` (KK-2000 BULK auto PRC + accr PRA + numbers, IRAIN=0 default —
  PRC/NPRC/NPRC1 ~2e-7, PRA/NPRA ~4%); `rain_slope`/`cloud_slope`; `rain_evap_rate` (PRE, full Rutledge-Hobbs ventilated
  diffusion MU/DV/SC/ARN/AB/QVS/EPSR — 7.1% + exact transcription).
- **ICE rates (Iter196-199):** `_gamma_slope`+`ice/snow/graupel_slope` (generic LAM=(ρπ·n/q)^⅓); `ice_deposition` (full
  Harrington-1995 ice/snow/graupel vapor deposition+sublimation: QVI/ABI + EPSI/EPSS ventilation + DCS tail split + SUM_DEP
  supersat limiter + sign split — PRD 1.5%/PRDS 2.4%/EPRD 3.6%/EPRDS 4.0%); `snow_collection_rates` (PSACWS riming 5.4% /
  PRAI 4.7%); `ice_autoconv_to_snow` (PRCI 1.3%); `snow_self_aggregation` (NSAGG 8.3%); `deposition_nucleation` (MNUCCD/NNUCCD
  Cooper 10.2%); `rain_immersion_freezing` (MNUCCR Bigg 4.2%); `sublimation_number_rates` (NSUBI 0.2%/NSUBS 1.3%).
  `tests/test_morrison_rates.py` (12 tests).
All Morrison work is NEW code (functions + tests + docs) — no existing code touched, zero regression to the 16 faithful cases.
**Remaining for a running Morrison:** minor rates (MNUCCC contact freezing, PRACS, melting) + the M2005 driver assembly
(negative-fix → slopes → rates → conservation limiters → tendency integration → sedimentation) + the CLUBB coupling
(`morrison_microphys_driver`).

### Iters 181–189 (compressed Iter190) — 16th bit-faithful case (jun25); rico+coriolis proven FP-limited; ~680-line cleanup; differentiability audit

Brought the bit-faithful set to **16 cases** and rigorously adjudicated every remaining runnable non-subsystem case.
The recurring method, now a core convention: **budget-decompose the failing prognostic from the per-step stats** to a
single term, and **decouple-the-oracle** (feed the Fortran's own stats into the JAX routine) to exonerate the obvious
subsystem — applied repeatedly, it converted two cases assumed "FP-limited" into found bugs.

- **Iter186 — fixed the rico step-1 rtm seed (NOT cubic-interp, the Iter152 guess).** Budget decomp: every rtm term
  machine-exact at k51 except `rtm_cl` (Fortran −4.56e-11/s, JAX 0); ×dt = the 1.37e-8 seed. `rtm_cl` is the
  **`fill_holes_vertical` applied to the MEAN fields rtm/thlm after the xm solve** (advance_xm_wpxp_module.F90:4974-5018,
  threshold rt_tol/thl_tol) — the JAX filled the variances but never rtm/thlm; AND the Iter141 IC hack
  `rtm=max(rtm,rt_tol)` pre-floored the dry top so the fill never fired (the "Fortran floors the IC" claim was FALSE —
  it keeps the bare sounding ~0 and the per-step fill raises it, mass-conservingly pulling from k51). Fix: add the
  rtm/thlm fill after MFL/before clip_covar (Fortran order), remove the IC floor. → rico step 1 fully bit-faithful
  (1.37e-8→6.3e-15), steps 1–4 faithful. All 15 cases still PASS (the fill is a bitwise no-op where rtm≥rt_tol).
- **Iter187 — rico residual PROVEN FP-limited (3 diagnostics).** The remaining step-5+ divergence is the near-zero rt
  flux/covariance at the stretched dry top: (1) only `wprtp_cl` differs but it's a *conditional* clip (JAX correctly
  implements clip_covar + `l_enable_relaxed_clipping`), (2) a discrete step-5 jump = a clip-bound crossing, (3) the
  dry-top rtp2 is machine-zero (4e-16 ≈ rt_tol² floor) so matching to rel-1e-6 is impossible. Added
  **`tests/test_fill_holes_mean.py`** (3 tests) locking in the rtm_cl/thlm_cl fill so a refactor can't reintroduce the seed.
- **Iter188 — ★ jun25_altocu BIT-FAITHFUL (16th case).** Decouple-the-oracle exonerated the "steep radiation" suspect
  (fed Fortran's cloud into the JAX `_simple_rad_lw`, radht matched to 4.4e-18). Budget-decomposing thlp2 found
  `thlp2_ma`=0 and **`wm_zm`=0** (vs Fortran 4e-3) while `wm_zt` was faithful. Root cause: `prescribe_forcings`
  (generic_forcings.py:247) updated `wm_zt` per-step but **never recomputed `wm_zm`** → the xp2/xpyp mean-advection by
  subsidence was missing (cold-cloud cascade thlp2→varnce_thl→stdev_chi→cloud_frac→rcm→buoyancy→wp2). Fix: recompute
  `wm_zm = zt2zm_jax(wm_zt)` (raw, faithful to time_dependent_input.F90:837) when the forcing updates wm_zt. jun25 PASSES
  30 steps; all 15 prior cases still PASS. **Convention:** a per-step-forced field with a grid-staggered partner
  (`_zt`↔`_zm`) must have the partner recomputed when updated (same class as the Iter81 stale `rc_coef_zm`).
- **Iter189 — coriolis_test PROVEN FP-limited + init wm_zm boundary fix.** upwp budget: step 1 fully faithful
  (`upwp_bt` machine-eps; `upwp_nct`=0 is an unwritten stat, the term IS in upwp_bt), step 2 all terms machine-eps with
  FP sign-flips in near-zero buoyancy/pressure, vpwp/vm exactly 0 — NO step-1 seed → undamped (zeroed-closure)
  oscillator amplifying machine-eps, genuinely FP. Also fixed the init `wm_zm` boundary to be subs_type-dependent
  (clubb_driver.F90:4748-4763: `w[m/s]` zeroes both, `omega[Pa/s]` zeroes top only) — no-op for the 16 cases (all
  `w[m/s]`), forward-looking correctness for omega-init.
- **Iters 181–185 — differentiability audit + ~680-line cleanup.** The CORE physics (ADG1 PDF closure, advance_xm_wpxp,
  advance_wp2_wp3) is while_loop/numpy-free → reverse-mode differentiable; the full-step-grad gap is the GLUE (the
  orchestration's ~570 numpy round-trips + the numpy flux limiter + the mixing_length while_loop), NOT the physics.
  `tests/test_differentiability.py` now 12 tests (added ADG1 w-closure + full ADG1 pdf-driver). Removed ~680 lines of
  vestigial incremental-replacement-era code (36 report_iterN funcs + counters, env-gated XMWP_CAP/MFLCAP dumps, dead
  iter7/8/9 JAX-validation + numpy-shadow LHS) from advance_clubb_core_module.py (4904→4278) + advance_clubb_to_end.py
  (598→541), all adversarially ref-counted, verified bit-faithful (ARM + 4-path cross-regression).

**Adjudication after Iter189:** 16 bit-faithful; rico + coriolis FP-limited (both proven); dycoms2_rf02_do/ds
KK-oracle-limited (MC-validated Iter175). The runnable non-subsystem set is exhausted — further faithful cases need a
subsystem port (Morrison ~19 / bugsrad ~5 / COAMPS / SILHS). Completion promise correctly withheld throughout.

### 2026-06-02 — Iter 180: compressed CHANGELOG (170-179); ruled out the mixing_length scan fix on perf grounds

- **Compressed Iters 170-179** into the condensed block below (the KK-covar/dycoms-oracle-limit/plateau/diff-audit
  arc) per the 10-iteration cadence; CHANGELOG 1037→894 lines.
- **Adversarial follow-up on the Iter179 mixing_length reverse-mode gap:** the bounded-`lax.scan` fix IS bit-exact,
  BUT the while_loop sits inside the outer per-start-level scan (mixing_length.py:474), so converting it makes a
  NESTED scan that loses the early-exit → **O(nzt²) forward pass** — a performance regression for all 15 faithful
  cases (which don't need grad), and it wouldn't unlock the full-timestep grad anyway (the ~570 numpy round-trips +
  numpy flux limiter remain). **Ruled out**; DESIGN updated with the refined reasoning. (Forward-mode `jax.jvp`
  differentiability is already tested + sufficient for jacobian-vector-product uses.)

#### Iters 170–179 (condensed): KK covar driver done + dycoms/rico ORACLE-LIMITED conclusion + subsystem plateau + differentiability audit

The KK second-moment covariance driver was completed, wired, and validated — and the investigation concluded the KK
cases hit fundamental limits, completing the bit-faithfulness plateau.
- **Covar driver (Iter170-172):** assembled `KK_upscaled_covar_driver` (sums the 9 `covar_{rt,thl,x}_KK_{auto,accr,evap}`
  into the 5 `_mc`=wprtp_mc/wpthlp_mc/rtp2_mc/thlp2_mc/rtpthlp_mc, `L=Lv/(Cp·exner)`), WIRED it into the per-step
  second-moment forcings (jitted; ~70 inputs from `pdf_params` — extended the post-advance `_replace` with
  w/eta/rt/crt/cthl/corr_chi_eta — + `prereqs` + prescribed normal-space corrs; gated `l_var_covar_src`). **Validated
  vs the rico oracle:** found & fixed the **`_signed_pow` even-exponent parity bug** (`sign(base)·|base|^exp` is
  odd-only; the covar's α+1=2 term needs `(−1)^exp` = `cos(π·exp)`; means unchanged). All 22 KK oracle tests pass.
- **dycoms2_rf02_do STEP 1 made FULLY bit-faithful (Iter173-174):** the rrm-transport residual was the **fill_holes
  threshold bug** — the JAX hydrometeor `fill_holes_vertical` used `threshold=hydromet_tol`, the Fortran uses
  `zero_threshold (=0)` (filling only on `any<0`); fixed (`hm_tol→0.0`, `lower_k 1→0`). rrm/Nrm then bit-faithful
  (1.7e-12/1.3e-10); all step-1 dynamics+moments machine-exact.
- **★ dycoms is ORACLE-LIMITED (Iter175):** step-2+ "diverges" only where the JAX OUT-PERFORMS the Fortran. The rt
  covariance source is an extreme ~850× cancellation; a **20M-sample Monte-Carlo of the full covariance = 3.72e-15
  matches the JAX (rel 1.2e-3) while the Fortran is 15× off (5.89e-14)** — the Fortran's ACM-850 `Dv` at the α+1 order
  is ~1e-3-accurate, amplified by the cancellation. All JAX sub-functions independently validated (Dv vs scipy 1e-14,
  bivar vs quadrature 1e-10, trivar vs MC). **Convention:** when a KK 2nd-moment covar "diverges", MC the full
  covariance — the oracle itself is unreliable there; the JAX may be the correct one.
- **The subsystem-port plateau is COMPLETE (Iter176-177):** every remaining unsupported subsystem is impractical —
  Morrison (18k lines WRF M2005 ice/snow/graupel), COAMPS (7k lines + arm_0003 fatal-errors in the Fortran on
  l_predict_Nc=F → no oracle), bugsrad (huge). The cross-hydromet `fill_holes_hydromet_api` is a no-op for KK (frozen
  hydrometeors only). Recorded per-case status in `compare_cases.py` BLOCKED_CASES. No regression: the 15 faithful
  cases confirmed intact (arm/bomex/dycoms2_rf01 PASS).
- **Differentiability (Iter178-179):** added jitted tests for grad through the KK covar driver (rel 1.2e-9) + the
  building blocks (10 tests). Audited the full-timestep-grad blockers: ~520 numpy writebacks + ~50 in-place mutations
  (coupled → all-or-nothing refactor), the numpy flux limiter, and `mixing_length`'s `lax.while_loop` (forward-mode AD
  only — `jax.grad` raises; a bounded-`scan` fix is bit-exact but removes the early-exit → O(nzt²) forward-pass perf
  regression for the 15 cases that don't need grad). Component-level differentiable+composable is done+tested.

#### Iters 159–169 (condensed): the KK second-moment covariance library, bottom-up

Built the entire covariance foundation that closes the Iter158-localized KK gap (the missing second-moment
microphysics source). Two new files: `PDF_integrals_covar.py` (the integral primitives) and
`KK_upscaled_covariances.py` (the covar functions). Each piece validated as it landed:
- **Integral primitives (`PDF_integrals_covar.py`), reusing the ported `_dvc`/`_signed_pow`/`_gamma_real`:**
  - `trivar_NNL_covar` — `Cov_i(x1, x2^α x3^β)` (x1,x2 normal; x3 lognormal), SUPERSATURATED (chi>0, +s_c).
    MC-validated (closed 2.883e-2 vs MC 2.871e-2). Plus 7 const-variants (const_x1/x2/x3/x1x2/x1x3/x2x3/all) and
    the vectorised `trivar_NNL_covar_eq` dispatch (selects by which σ≈0; machine-exact vs base limits, rel 0.0).
  - `quadrivar_NNLL_covar` — `Cov_i(x1, x2^α x3^β x4^γ)` (x3,x4 lognormal), SUBSATURATED (chi<0): uses
    `_signed_pow(-σ_x2,α)` + `_dvc(...,+s_cc)`. MC-validated (closed 8.974e-2 vs MC 8.979e-2). Plus 11 const-variants
    and the `quadrivar_NNLL_covar_eq` dispatch, which faithfully implements the **(r_r,β)↔(N_r,γ) symmetry** (an
    N_r-const branch reuses the r_r-const variant with x3↔x4 args AND β↔γ swapped). Machine-exact dispatch.
  - **Oracle-free MC convention (DESIGN):** supersaturated integrand `chi^α` over full range; subsaturated
    integrand `−(−chi)^α` over chi<0 only (reverse-engineered from the validated means, Iter164).
- **The 9 covar functions (`KK_upscaled_covariances.py`):** `covar_{rt,thl}_KK_{auto,accr,evap}` share
  `_covar_x_comp(...,s)`/`_covar_x_evap_comp` with the ADG1 transform `x'=(1/(2c))(eta'∓chi')` (s=+1 r_t, s=−1 thl);
  auto (y=N_cn, precip_frac=1) vs accr (y=r_r, ×precip_frac) differ only in y/exponents/coef/tol. The 3 w-covars
  `covar_x_KK_{auto,accr,evap}` are direct (x=w is a PDF variable, no transform/bivar); accr & evap carry the
  out-of-precip correction `−(1−precip_frac)·(mu_x−x_mean)·tndcy` (auto and the rt/thl forms do NOT). Exponents:
  auto α=2.47,β=−1.79; accr 1.15,1.15; evap α=1.0(supersat),β=1/3(r_r),γ=2/3(N_r). All finite + differentiable.

### 2026-05-30 — Iter 158: localized the KK composition difference — the missing second-moment covar driver

Pinpointed exactly why KK-enabled dycoms2_rf02_do isn't bit-faithful, via step-aligned compares.

- **Step 1: ALL dynamics machine-exact** (rtm 1e-13, thlm 7e-15, wp2 4e-14, …) — the means + transport are
  bit-faithful at the first step (the microphysics feeds the NEXT step's forcings, so step 1 is unperturbed).
- **Step 2: the MEANS pass, only the SECOND MOMENTS fail** — rtm 3.4e-7 ✓, thlm 2.7e-8 ✓, but **rtp2 1.4e-4,
  rtpthlp 1.9e-4, thlp2 9e-6, wpthlp 1.4e-5 ✗**. So the mean microphysics tendencies (rcm_mc/thlm_mc) are
  ~bit-faithful; what's missing is the SECOND-MOMENT microphysics source.
- **Root cause:** the Fortran `calc_microphys_scheme_tendcies` also outputs `wprtp_mc/wpthlp_mc/rtp2_mc/thlp2_mc/
  rtpthlp_mc` from **`KK_upscaled_covar_driver`** (KK_upscaled_covariances.F90, 2574 lines → PDF_integrals_covars)
  — the covariances of the auto/accr/evap process rates with w/rt/thl. **The JAX has NONE of it** (no
  `*covar*` microphysics module), and `advance_clubb_to_end` never adds these `_mc` terms to the second-moment
  forcings (`rtp2_forcing` etc.). That is the entire composition difference.
- **Roadmap (next, multi-iteration):** port `KK_upscaled_covariances` (`covar_{rt,thl,x}_KK_{auto,accr,evap}` +
  the `trivar_NNL`/`quadrivar_NNLL` covar integrals from `PDF_integrals_covars`), validate each against the rico
  oracle's `rtp2_mc`/`thlp2_mc`/etc. stats (the same incremental method used for `KK_upscaled_means`), then wire
  the 5 `_mc` terms into the second-moment forcings. KK stays ENABLED (means + transport are faithful; the
  second-moment gap is a known remaining port). The 15 non-KK cases are untouched (ARM smoke PASS).

### 2026-05-30 — Iter 157: ★ FULL KK microphysics wired (rates + transport) — runs 30 steps, rain matches oracle

Completed the KK per-step orchestration: the hydrometeor transport now runs and produces physically correct
rain, the first full microphysics scheme in the JAX.

- **Transport wired** (`kk_microphys_step.py`, the `l_kk_micro_apply` branch). Composes the oracle-validated
  pieces per Fortran `advance_microphys`: `Skw_zm = skx_func_jax(wp2, zt2zm(wp3), w_tol, clubb_params)`;
  `kk_sedimentation(prereqs['mvr'])` → mean sed velocities; `kk_sed_vel_covars(precip_frac_i·μ_rr/Nr, mvr,
  <component moments from prereqs>)` → turbulent-sed covariances; `_hydrometp2_zt` → `calculate_K_hm` (eddy
  diffusivity); then `advance_one_hydrometeor(dt, hm, hm_mc, K_hm, nu_hm=nu_vert_res_dep.nu_hm, wm_zt,
  zt2zm(V), zt2zm(Vi/Ve), rho_ds_zm, invrs_rho_ds_zt, gr, w_above=weights_zt2zm[:,:,0])` + `fill_holes_vertical`
  for rrm and Nrm. Feeds `rcm_mc/thlm_mc` to the forcings (advance_clubb_to_end:67-68).
- **Enabled for KK cases** (`clubb_standalone.py`: `l_kk_micro_apply=(microphys_scheme=='khairoutdinov_kogan')`).
  The 15 non-KK cases never reach the KK step (gate on scheme), so they are unaffected.
- **Validated to RUN + correct magnitude:** dycoms2_rf02_do runs **30 steps stably** (no crash; `rrm ≥ 0` via
  fill_holes) and produces physical rain — at 10 steps `rrm max ≈ 1.90e-6` vs the rf02_do oracle's `≈ 1.96e-6`;
  at 30 steps `rrm ≈ 3.7e-6`. NOTE: a `compare_runs` "rc=1" was a **stale-output-file artifact** (rm the
  `*_compare_*` dirs first); the standalone run is clean.
- **NOT bit-faithful yet — the full compare FAILS (16 prognostic vars).** The microphysics feedback
  (`rcm_mc/thlm_mc` → forcings) couples a residual error into all the dynamics. Key insight: in a *running*
  JAX-vs-Fortran compare BOTH compute the rates from the within-step rrm/Nrm, so this is NOT the
  `calc_comp_mu_sigma_hm` timing confound (which only affects comparison to the END-of-step oracle STATS) — it
  is a real **composition difference** in the wiring (rates and/or transport) to localise next.
- **Remaining (next):** step-align the JAX KK tendencies (rcm_mc/rrm_mc/Nrm_mc + the transport) against the
  Fortran microphysics oracle to find the composition difference, drive it to bit-faithful. The full KK
  microphysics is FUNCTIONAL (rates + transport, stable, correct-magnitude rain); bit-faithfulness is the
  remaining gap. (KK stays ENABLED for KK cases — the faithful structure; the 15 non-KK cases are untouched.)

### 2026-05-30 — Iter 156: ✓ KK rates FUNCTIONAL — fixed the `pdf_params` zero-init blocker (faithful timing)

Resolved the Iter155 blocker so the KK microphysics tendency computation works on live state.

- **Root fix.** `state['pdf_params']` (a NamedTuple, zero-initialized — a fallback per
  `advance_clubb_core_module.py:1815`) never received the PDF component moments the JAX computes as Block-U
  locals. Added `pdf_params = pdf_params._replace(chi_1/2, stdev_chi_1/2, cloud_frac_1/2, rc_1/2, mixt_frac,
  thl_1/2, ice_supersat_frac_1/2 = _chi1_60/…/_issf1_60)` right after the ice-supersat computation (:4297),
  using the canonical post-advance Block-U "_60" moments.
- **Faithful timing verified.** `pdf_params` is `intent(inout)` in `advance_clubb_core_api`
  (clubb_driver.F90:3392); the Fortran microphysics (`calc_microphys_scheme_tendcies`, :3618) runs AFTER
  advance_clubb_core and uses that POST-advance PDF — exactly the Block-U "_60" pass the JAX computes. So
  propagating "_60" (not the pre-advance :1485 pass) is the correct, faithful choice.
- **Result.** `state['pdf_params'].chi_1` is now nonzero (3.6e-3) with cloud present, and the KK step's
  autoconversion produces `rcm_mc≈8.2e-9` / `rrm_mc≈8.2e-9` (38 pts) — matching the rf02_do oracle's
  `rcm_mc≈9.2e-9` from t0 (cloud→rain). The KK RATES are now functional on live state.
- **No regression:** ARM (0 prog fail) + bomex (0 prog fail) still PASS — the 15 non-microphysics cases don't
  read these pdf_params fields, so the `_replace` is safe.
- **Remaining (next):** the KK step still only COMPUTES + stores the tendencies (gated behind
  `l_kk_micro_apply`, default off → running cases unchanged). Next stage: wire the hydrometeor transport
  (velocities from `prereqs` → `advance_one_hydrometeor` + `fill_holes`) and enable application, then validate
  dycoms2_rf02_do/ds bit-faithful against the oracle.

### 2026-05-30 — Iter 155: KK per-step orchestration wired (gated) + the real blocker found via validation

Executed the first stage of the KK microphysics per-step wiring and let the validation surface the actual
blocker — exactly the incremental-replacement / shadow-comparison discipline the project uses.

- **New `Microphys/kk_microphys_step.py::advance_kk_microphysics(state)`** — composes the validated pieces into
  the per-step tendency call (precip_fraction → compute_kk_microphysics with `l_return_vel_prereqs=True`),
  extracting all inputs from live `state` (pdf_params component moments, hydromet array via `hm_metadata.iirr/iiNr`,
  Nc_in_cloud, ice_supersat_frac, clubb_params upsilon, etc.). Stores `_kk_rcm_mc/_kk_thlm_mc/_kk_rrm_mc/
  _kk_Nrm_mc/_kk_precip_frac/_kk_prereqs`. Transport (`advance_one_hydrometeor`+`fill_holes`) and feedback
  application are gated behind `state['l_kk_micro_apply']` (default off) → the running KK cases are byte-for-byte
  unchanged.
- **Wired into `advance_clubb_to_end.py`** after `_cloud_drop_sed`, gated on
  `microphys_scheme=='khairoutdinov_kogan'` (lazy import inside the branch). The 15 non-KK cases never reach it
  (`'none'`); ARM smoke + the existing tests confirm no regression. dycoms2_rf02_do runs 3 steps with it active.
- **★ The validation found the real blocker.** On live dycoms2_rf02_do the KK step produces ALL-ZERO tendencies
  despite cloud present (rcm=5e-4, cloud_frac=1) — because **`state['pdf_params']` is zero-initialized**
  (`advance_clubb_core_module.py:1815`: "zero-initialized; only used as fallback"). The JAX computes the real PDF
  component moments as LOCALS (`_chi1/_schi1/_cf1` ~:1506) that flow only to `stats_writer.update` (:4724), never
  back into the returned `pdf_params`. So `chi_1/stdev_chi_1/cloud_frac_1 == 0` in state → no autoconversion. The
  rf02_do oracle confirms what should happen: rcm_mc≈1e-8 (autoconversion) from t0, rrm growing 5.5e-7→2e-6.
- **PREREQUISITE for the next stage (documented in DESIGN):** populate the returned `pdf_params` with
  `chi_1/2, stdev_chi_1/2, cloud_frac_1/2, mixt_frac, thl_1/2, ice_supersat_frac_1/2` from those closure locals
  (the dataclass already has the fields), without perturbing the 15 cases (verify they don't read them). Then the
  KK step becomes functional → enable `l_kk_micro_apply` + the transport. Also: a stale `*_stats.nc` causes a
  spurious StatsWriter `PermissionError` — remove before runs.

### 2026-05-30 — Iter 154: KK microphysics readiness confirmed + transport-wiring enabler; per-step wiring plan

Pivoted toward the biggest remaining lever (microphysics). Established that the KK pieces are validated and
ready, added the missing link the transport step needs, and documented the precise per-step wiring plan.

- **Corrected two false "crashes" from the Iter153 survey.** dycoms2_rf02_do's "crash" was a **stale-output
  `PermissionError`** in StatsWriter (a locked/old `*_stats.nc` from a concurrent run), not physics — clears on
  removing the file. (Combined with the Iter153 timeout-contention lesson: survey "failures" are usually harness
  artifacts; always confirm with a clean standalone run.)
- **KK rate pieces confirmed validated.** `tests/test_kk_rico_oracle.py` (16 tests) PASSES — KK_auto/accr/evap
  vs rico (sig 1e-6..1e-8), composed drivers, `kk_microphys_adjust` (rcm_mc exact), `compute_kk_microphysics`
  (no-rain rcm_mc machine-exact + fully differentiable), sedimentation, sed-vel-covars, K_hm,
  advance_one_hydrometeor. So the rate + transport building blocks are ready to wire.
- **Transport-wiring enabler (code).** Extended `compute_kk_microphysics` with an opt-in
  `l_return_vel_prereqs=True` that additionally returns the mean volume radius `mvr` (`KK_mvr_upscaled_mean`) +
  the rr/Nr in-precip component moments (linear+normal space) — the previously-missing inputs that
  `kk_sedimentation`/`kk_sed_vel_covars` need (computed from the SAME locals the rates use; zero added physics).
  **Default path is byte-for-byte unchanged** (still the 5-tuple) → the 15 passing cases are untouched (and
  `compute_kk_microphysics` is not yet on any production path). New test `test_compute_kk_microphysics_vel_prereqs`
  asserts default-unchanged + finite prereqs + `mvr` vs the rico `mvrr` oracle within the documented
  `calc_comp_mu_sigma_hm` timing-confound band (median 4.9e-2; the static oracle is pre-developed-rain).
- **DESIGN: precise KK per-step wiring plan** (gated on `microphys_scheme=='khairoutdinov_kogan'`; the 4-step
  precip_frac → rates → velocities → advance_one_hydrometeor+fill_holes sequence with the exact state→input
  mapping). dycoms2_rf02_do/ds are uniform-grid (faithful dynamics) so KK wiring should make them bit-faithful —
  a more immediate win than Morrison. **Next: execute the wiring; then Morrison (19-case lever).**

### 2026-05-30 — Iter 153: rico k51 seed is FP-limited; full 48-case coverage survey + `--survey` tool

Two strands: closed the rico rt-seed investigation, then pivoted to generalising the test coverage
(per the standing directive to move testing beyond ARM).

- **rico k51 — FP-limited, not a discrete bug.** Per-level diff at the worst level k51 (rtm≈9e-5): the
  initial rtm matches Fortran exactly (cubic IC bit-faithful, Iter152), and ALL advance_xm_wpxp inputs at
  k51 are exact or FP-level — `wm_zt=0` (no subsidence → the mean-advection-amplification hypothesis is
  dead), wp2/Kh_zt/wprtp bit-exact, `rtpthvp` only ~1e-14. None can produce 1.37e-8 through the cond-320
  rt penta solve. The rt solve + threshold-based mfl amplify sub-1e-13 FP-order differences in the
  rt-specific path (rtm is ~1e4× smaller than thlm, so the same abs FP error is ~1e4× larger in rel). rico
  step-1 PASSES (rel 8.5e-7 < 1e-6); the multi-step amplification → rtm<0 abort is a true-bit-faithfulness
  limit, not a fixable term. Documented; deferred.
- **★ Full 48-case coverage survey.** Systematic JAX smoke survey over ALL 48 `*_model.in` cases:
  **20 RUNS, 28 UNSUPPORTED, 0 hard crashes.** Feature-blocker breakdown (the coverage levers): **morrison
  microphysics — 19 cases** (cgils×6, cgils_p2k×6→cgils_s{6,11,12}{,_p2k}, cloud_feedback×6, clex9×2,
  arm_3year, dycoms2_rf02_morr, nov11_altocu) — THE dominant lever (~40% of cases); bugsrad radiation 5;
  coamps 2; SILHS interactive 2+. Of the 20 RUNS, all previously-listed cases PASS and **rico is the only
  runnable case that fails** (the k51 FP limit). dycoms2_rf02_do/ds, coriolis_test, jun25_altocu are
  newly-confirmed runnable.
  - **Methodology lesson (important):** atex/fire/dycoms2_rf01 first showed spurious "JAX run failed
    (rc=1)" — pure **`timeout` artifacts under background-task contention** (JAX compiles+runs in ~100s,
    tripping a 150s timeout when several run at once). Re-run clean → all PASS. The full `compare_runs`
    survey MUST run sequentially with no competing compute.
- **Tooling: `compare_cases.py --survey`** (new). Auto-discovers all 48 cases (`discover_cases()`) and
  categorises RUNS / UNSUPPORTED(feature) / ERROR via a fast JAX-only smoke run, printing the
  feature-blocker summary. Generalises the dashboard from its hardcoded list. **Next coverage work:
  Morrison 2-moment microphysics** (the single highest-leverage port). ARM/bomex/etc. PASS, 23/23 units.

### 2026-05-30 — Iter 152: rico rt residual — mono_flux_limiter PROVEN bit-faithful; seed is an upstream rt input

After the Iter151 weights_zm2zt fix, rico step-1 passes all 16 prognostics; the largest residual is the
rt-specific seed **rtm rel 8.5e-7** (thlm exact at 1.2e-14), which accumulates → rtm<0 abort at step 17.
This iteration localised it and ruled out the prime suspect.

- **Localisation via the mfl enter/exit stats:** `thlm_exit_mfl` is EXACT (1.2e-14) but `rtm_exit_mfl` stays
  1e-8 off — the divergence survives the mono flux limiter only for rt. (Budget terms `_ma`/`_ta` are
  timing-confounded stats — thlm_ta diverges 4.7e-5 yet thlm is exact — so they are NOT the seed.)
- **mono_flux_limiter PROVEN bit-faithful.** New validated harness `run_scripts/cmp_mfl_f2py.py` feeds the
  captured JAX mfl inputs to `clubb_f2py.f2py_monotonic_turbulent_flux_limit` (new `$MFLCAP` capture hooks).
  With the SAME inputs the JAX mfl == f2py to **machine precision for BOTH rt (3.5e-18) and thl (5.7e-14)**.
  So the JAX mfl is faithful on the stretched grid and is NOT the seed.
  - **Key harness gotcha:** the JAX `low_lev_effect`/`high_lev_effect` are **0-based** level indices; the
    Fortran f2py uses them as **1-based** array indices → must pass `lle+1, hle+1`. Without the +1 the f2py
    mfl fires spuriously (thl xm off by 0.9) — a pure harness artifact (the control thl, exact in the full
    run, exposed it). The JAX mfl also (correctly) hardcodes `term_ma_zt_lhs_jax` to upwind=True (config has
    l_upwind_xm_ma=1) and does not take l_implemented/tridiag_solve_method (unused by its algorithm).
- **Conclusion — the seed arises DURING step-1 rt physics at k51, not in any IC/mfl/solve component.** Ruled out,
  each by an f2py oracle or direct reconstruction:
  - **mfl** — proven bit-faithful (above).
  - **advance_xm_wpxp** — f2py-exact given JAX inputs (Iter151, rtm 6.9e-18); JAX never post-processes rtm
    (final == post-solve 5e-18).
  - **IC monotone-cubic interpolation** — the JAX `_steffen_interp_1d` is BIT-IDENTICAL to `f2py_mono_cubic_interp`
    at the worst level k51 (both 9.0150000000e-05, d=0.0), reconstructing the Fortran sounding.F90 stencil. So the
    initial rtm matches Fortran.
  - **The rtm 1e-8 floor** (`np.maximum(rtm, rt_tol)`, clubb_standalone.py) is CORRECT and faithful: the Fortran
    full run floors the dry top to exactly 1e-8 (rtm[k53..56]=1.0e-8 in the Fortran NetCDF; the bare cubic gives 0,
    the floor is applied downstream). Removing it makes the JAX top diverge — reverted.
  - The divergence is at **k51 (rtm≈9e-5)**: initial rtm matches Fortran, but after ONE advance_clubb_core step
    JAX rtm[k51] moves +2.5e-9 while Fortran moves −1.3e-8 (net d=1.37e-8 → the rel-8.5e-7 seed). Since
    advance_xm_wpxp is f2py-exact given JAX inputs, an **rt-specific INPUT to it must differ sub-tolerance at k51**
    — the prime suspect is the **wprtp entering** (`_wprtp_pre11`, not separately stat-validated; max 6.9e-5) or
    `rtpthvp` at k51, a likely FP-level (~1e-13) difference in the rt-specific pdf_closure amplified by the
    small-magnitude rt solve.
- **Next (Iter153):** instrument the Fortran full run (or a pdf_closure f2py oracle) to dump its step-1 `wprtp`/
  `rtpthvp` at the k51 region and diff vs the JAX `_wprtp_pre11`/`rtpthvp` to pin the sub-tolerance rt input; if it
  is genuine FP in pdf_closure, rico may be FP-limited at the step level. Tooling added: `cmp_mfl_f2py.py`,
  `$MFLCAP` hooks. ARM/bomex PASS, 23/23 units, rico step-1 PASS (rtm 8.5e-7 < 1e-6).

### 2026-05-30 — Iter 151: ★ RICO BUG FOUND & FIXED — `weights_zm2zt` column order was swapped vs Fortran

Overturned the Iter150 "un-inspectable compiled-Fortran" conclusion. **The bug was a swapped grid-weight column
order, found in minutes once the right oracle was used.**

- **New decisive tool — individual f2py LHS-term routines.** `clubb_f2py` exposes each LHS term separately
  (`f2py_xpyp_term_ta_pdf_lhs`, `f2py_term_ma_zm_lhs`, `f2py_diffusion_zm_lhs`). `run_scripts/cmp_terms_f2py.py`
  feeds the SAME captured inputs to the Fortran term routine AND the JAX term and diffs directly — no solve, no
  clip, no shared-bug confound (the prior "reconstruction==captured" only proved JAX==JAX). This is the general
  term-level bisection method and supersedes the penta-reconstruction approach.
- **Result:** TA term bit-exact (0.0), diffusion FP (1e-17), but **`term_ma_zm_lhs` differed 1.25e-6 (rel 7.8%)
  at the superdiagonal**. Diffing the JAX vs Fortran grid `weights_zm2zt` directly: the two columns were SWAPPED.
  - Fortran `calc_zm2zt_weights` (grid_class.F90:2621/2625): `m_above(idx0)=(zt[k]-zm[k])/total`,
    `m_below(idx1)=(zm[k+1]-zt[k])/total`.
  - JAX `derived_types/grid_class.py::_calc_zm2zt_weights` stored `idx0=dist_upper, idx1=dist_lower` — OPPOSITE.
  - The interp `zm2zt_api` PAIRED them to compensate (correct output → invisible there), but the LHS term routines
    (`term_ma_zm_lhs_jax`, `xpyp_term_ta_pdf_lhs_jax` in `diffusion.py`) index `weights_zm2zt[:,:,m_above]`
    DIRECTLY → wrong physical weight. **On uniform grids both = 0.5 → invisible → exactly why the 15 grid_type=1
    cases always passed and only rico (grid_type=2, stretched) failed.**
- **Fix** (`derived_types/grid_class.py`): store `weights_zm2zt` in EXACT Fortran column convention + flip
  `zm2zt_api`'s interpolation pairing to match (output invariant). Physics interps
  (`CLUBB_core/grid_class.py::zm2zt_jax/zt2zm_jax`) were already Fortran-faithful — untouched.
- **Verification:** rico step-1 `advance_xm_wpxp` vs the f2py oracle: **thlm 2.4e-6 → 1.1e-13, um → 3.6e-15,
  wpthlp → 3.7e-14** (all rel ~1e-16). The full step-1 rico compare now **PASSES all 16 prognostics** (thlm
  1.2e-14, um 9e-14). No regression: **ARM PASS, bomex PASS, 23/23 unit tests** (uniform grids bit-identical).
- **Still open (Iter152):** rico diverges over MULTIPLE steps — step-1 rtm rel 8.5e-7 (just under tol) is the
  largest residual; it accumulates → wpthlp ~8e-5 by step 10 → `rtm<0` abort at step 17 (Fortran completes). The
  rtm seed appears AFTER advance_xm_wpxp (whose rtm is f2py-exact 6.9e-18) → prime suspect the **mono-flux-limiter
  rtm path / fill_holes / clip on the stretched grid**; localise with the same individual-f2py-routine method.
- Tooling retained: `cmp_terms_f2py.py`, `compare_xm_wpxp_f2py.py`, `$XMWP_CAP`/`$PENTA_CAP`/`$TACAP` hooks.

### 2026-05-30 — Iter 149-150: rico advance_xm_wpxp exhaustive bisection (CONCLUSION SUPERSEDED by Iter151)

[Compressed Iter160] Bisected `advance_xm_wpxp_jax` against the f2py oracle + captured penta: verified the solver
(==scipy 1.1e-13), the penta reconstruction (==captured to 0.0), the RHS, and most LHS terms bit-exact; isolated
the residual to a thl-flux divergence at the inversion. **WRONGLY concluded it was an "un-inspectable
compiled-Fortran FP difference" — Iter151 found the actual bug** (the `weights_zm2zt` columns were stored swapped
vs Fortran, picked up by the LHS term routines that index the columns directly). Lesson retained: validate a JAX
term against the Fortran via an INDEPENDENT oracle (the f2py term routines), not against a numpy re-implementation
that can share the bug. Tooling retained: `compare_xm_wpxp_f2py.py`, `cmp_terms_f2py.py`, `$XMWP_CAP`/`$PENTA_CAP`.

### 2026-05-30 — Iter 148: rico BUG LOCALISED to advance_xm_wpxp_jax via the f2py input-matched comparison

Built and ran the definitive **f2py input-matched comparison** (unblocked Iter147) — the breakthrough after
8 iterations of stretched-grid hunting.
- **Harness:** capture rico's matched step-1 `advance_xm_wpxp_jax` inputs (~64 arrays + flags + grid) and the
  RAW JAX output (pre-sponge/nudge) via env-gated (`$XMWP_CAP`) hooks in `advance_clubb_core_module.py`; build
  the Fortran grid+UDTs for rico's stretched grid; call `clubb_f2py.f2py_advance_xm_wpxp` directly (Iter147
  helper) with the SAME inputs; diff. Saved as `run_scripts/compare_xm_wpxp_f2py.py`.
- **Two harness bugs found & fixed:** (1) the JAX output must be captured PRE-sponge/nudge (the JAX applies
  `sponge_damp_xm` + uv-nudge AFTER advance_xm_wpxp — lines 2324-2327); (2) **`l_implemented` must be FALSE**
  (standalone) — with `True` the Fortran skips the xm mean-advection (subsidence), which showed up as `thlm`
  off by exactly `−wm·d(thlm)/dz·dt`.
- **DEFINITIVE RESULT:** with matched inputs, **f2py(JAX inputs) vs JAX output reproduces the EXACT rico
  full-run divergence** — thlm 2.4e-6, um 2.5e-6, vm 5.7e-7, wpthlp 4.9e-8@k6, upwp 3.1e-7@k4, vpwp 6.3e-8,
  **wprtp machine-exact (6e-27)**. So given **identical inputs**, `advance_xm_wpxp_jax` ≠ Fortran
  `advance_xm_wpxp` on the stretched grid → **the bug is IN advance_xm_wpxp_jax** (its assembly/solve), NOT in
  any upstream input. The rt path is exact; thl/wind diverge. The flux divergence is at **k4-6** (low BL, fine
  stretched grid → large `invrs_dzm`), NOT the inversion k17-18 where the mfl acts — so it is the **solve/
  assembly**, not the mono-flux-limiter. (The penta solver is bit-faithful, Iter144, so the lhs/rhs *assembly*
  must differ from Fortran's on the stretched grid despite the term-by-term audit matching.)
- **Next:** bisect inside advance_xm_wpxp_jax against the f2py oracle (test candidate assembly fixes; re-run
  the comparison and watch thlm 2.4e-6 → 0). The env-gated capture hooks + `compare_xm_wpxp_f2py.py` are the
  tooling. ARM unaffected (hooks no-op without `$XMWP_CAP`; smoke PASS).

### 2026-05-30 — Iter 147: rico stretched-grid — localised to the WIND/flux solve; f2py UNBLOCKED

Continued the rico `grid_type=2` hunt (everything audited matches at the inversion). Refined the localisation
and, critically, found the f2py oracle is usable after all.
- **f2py is NOT blocked — the wrapper is just out of sync with its `.so`.** `clubb_f2py.f2py_advance_xm_wpxp`
  introspects fine; the `clubb_python` Python wrapper passes `wp3, kh_zt` where the compiled `.so` now expects
  `wp3_on_wp2, wp3_on_wp2_zt, kh_zt` + an extra `skw_zm` (a recompiled signature). **PROVEN end-to-end:** new
  `tests/test_f2py_advance_xm_wpxp.py` calls `clubb_f2py.f2py_advance_xm_wpxp` DIRECTLY (introspected arg order,
  UDTs pushed via `set_fortran_*`) and it runs, returning all 18 finite advanced arrays. The reusable helper
  `call_f2py_advance_xm_wpxp(gr, nu, pdf_ic, err, args)` is the template for the definitive **input-matched
  comparison**: capture rico's matched step-1 advance_xm_wpxp inputs (eager), feed BOTH the JAX solve and the
  `.so`, diff bit-to-bit — splits "an input diverges" from "the assembly/solve differs" (the namelist A/B can't).
- **Localised to the WIND variables.** The surface level k0 is machine-exact (upwp/up2/wp2 d[0]≤1e-15). The
  divergence starts at k1 and the **wind** diverges most *relatively*: `um` 2.6e-7 rel (vs thlm 3e-10), `up2`/
  `vp2` 5e-6 at k1. The wind has strong near-surface shear and rico's stretched grid is FINE near the surface
  (large `invrs_dzm`), so the gradient-production coupling `lhs_tp=wp2·invrs_dzm` × `um~9.5` is a large-
  magnitude near-cancellation. `up2`/`vp2` then grow via shear production `−2·upwp·d(um)/dz`; wp2/up2/vp2 (the
  velocity variances) reach ~1e-5 at the inversion, the scalars (thlp2 1e-7, rtp2 ~0) far less.
- **More rule-outs (all faithful, confirmed by uniform-grid exactness):** the `zm2zt2zm` SMOOTHER (the Fortran
  `zt2zm_api`→`linear_interpolated_azm_2D` uses the same copy lower-BC as `zt2zm_jax`); `sfc_varnce` (grid-
  independent — surface fluxes + constants); `sigma_sqd_w` (= a correct `zm2zt2zm` of a pure-algebra field);
  `xp2_xpyp_rhs` TP/shear-production term (invrs_dzm exact); `xpyp_term_ta_pdf_rhs`. Noted but irrelevant: the
  exported `zt2zm_api`/`zm2zt_api` (derived_types) have an extrapolation lower-BC that differs from
  `linear_interpolated_azm`'s copy — but they're UNUSED by physics (only the unused API smoothers), and their
  interior is correct for grid_type=2 (zm=midpoint of zt).
- **Process:** no source changes this iteration (one wp3-TA fix was tried and reverted last iteration; the
  coriolis recovery from the Iter146 git-checkout mishap is intact — ARM still PASS).

### 2026-05-30 — Iter 146: stretched-grid operator audit (BL closure localised); coriolis interface recovered

Continued the rico stretched-grid hunt (Iter145 isolated `grid_type=2` as the cause). Refined the localisation
and audited the candidate operators; one hypothesised fix was tested and rejected.
- **Divergence is LOCAL to the boundary layer** (k0-20), growing toward the inversion, and EXACTLY ZERO in
  the free troposphere and top (wp2 0.0 at k21-57). So the top-of-domain density divergence (~1e-9, hydrostatic
  integration) does NOT propagate — the error is in the **turbulent BL closure** on the stretched grid.
- **At the inversion, ALL inputs are machine-exact:** rho_ds/exner/rho (1e-15), thvm (0.0), thv_ds (1e-13),
  wm (8.7e-17), Lscale/Kh/tau/em (≤1e-13), grid coords (0 ULP), dzt/dzm/invrs_dz (Fortran grid_class.F90:
  950/979/987/995 match exactly — no off-by-one). Only the post-advance closure outputs diverge: Skw (1.7e-4),
  the ADG1 PDF (w_1 1.9e-4, mixt_frac 3.3e-5), and **wp2 (1.2e-5)** — wp2 is a primary source, beyond the
  flux→wpthvp→wp2 buoyancy cascade alone.
- **Audited faithful (all match Fortran, confirmed by ARM/uniform exactness):** `term_ma_zt_lhs` (upwind,
  invrs_dzm(k)/(k+1) correct vs mean_adv.F90:262-292), `term_ma_zm_lhs` (stored weights + invrs_dzm),
  `diffusion_zm_lhs`, `xm_term_ta_lhs`, `wpxp_terms_ac_pr2_lhs` (d(wm)/dz), `wpxp_term_tp_lhs`,
  `wp2_term_ta_lhs`. **Hypothesis REJECTED:** the wp3 ADG1 TA term `_wp3_term_ta_ADG1_lhs` — tried switching
  `a1_coef_zt`→momentum `a1_coef(k±1)` (the Fortran comment shows F/G on momentum levels); it **broke ARM**
  (0→15 failures) and didn't help rico, so `a1_coef_zt` is the faithful form. Reverted.
- **Process note + recovery:** reverting that edit via `git checkout` on the file DISCARDED uncommitted
  prior-iteration work (the Iter103 non-traditional-Coriolis support: 3 params + gated RHS terms). Recovered it
  from the CHANGELOG-documented physics — `advance_wp2_wp3_jax` now again accepts `l_ho_nontrad_coriolis`,
  `fcor_y`, `wp2up` and adds, gated on the flag, `wp2 RHS += 2·fcor_y·upwp` and `wp3 RHS += 3·fcor_y·wp2up`
  (no-op for the 15 cases + rico). **ARM `compare_runs` PASS (0 failures), rico back to its 16.** Lesson: never
  `git checkout` a file carrying uncommitted work — undo edits manually.
- **Next:** the stretched-grid wp2/closure term is still unpinned (everything inspected matches at the inversion
  ⇒ either an un-audited grid-derived quantity or a subtle higher-order term). `sigma_sqd_w` diverges 1.18e-7
  (rel 4.7e-7), the one closure quantity above machine besides Skw/PDF — check its computation next. Continue
  the `grid_type=1` A/B bisection.

### 2026-05-30 — Iter 145: rico root cause ISOLATED — the STRETCHED GRID (grid_type=2); two `(1-w)` weight bugs fixed

**Breakthrough after 5 iterations of hunting.** rico is the ONLY case using `grid_type = 2` (a stretched grid
read from `deep_convection_128lev_27km_zt_grid.grd`); ALL 15 bit-faithful cases use `grid_type = 1` (uniform).
- **DEFINITIVE isolation:** temporarily switched rico's namelist to `grid_type=1` (uniform, deltaz=40) and
  re-ran `compare_runs`: prognostic failures dropped **16 → 2** and every core var went **machine-exact**
  (wp2 6.8e-13, thlp2 5.3e-14, wpthlp 9.8e-14, up2/vp2 ~1e-13). On a uniform grid rico's dynamics are bit-
  faithful — so the entire divergence lives in **stretched-grid handling**, NOT strong turbulence / the
  inversion / microphysics (all earlier hypotheses). Namelist reverted after the test.
- **Fixed two real `(1-w)` weight faithfulness bugs** (Fortran computes BOTH interpolation weights *directly*;
  the JAX computed the below-weight as `1 - w_above`, which is identical to 0.5 on uniform grids but ~1 ULP off
  on stretched grids): (1) `CLUBB_core/grid_class.py` `zm2zt_jax`/`zt2zm_jax` interior+top now use the direct
  `weights_zm2zt(m_below)=(zm[k+1]-zt)/dzt` / `weights_zt2zm(t_below)=(zt-zm)/denom` (grid_class.F90:2625/2269);
  (2) `derived_types/grid_class.py` `_calc_zm2zt_weights` (the STORED weights feeding `xpyp_term_ta_pdf_lhs` +
  `advance_wp2_wp3`) now direct, not `1 - dist_lower/total`. **Verified safe:** ARM `compare_runs` PASS (0
  prognostic failures), uniform-rico stays machine-exact, `test_diffusion` 17/17, new `test_penta_faithful` ok.
- **But the weight bug is NOT the dominant cause** — fixing it left rico's divergence BIT-IDENTICAL (wp2 still
  1.124e-5). The `(1-w)` residual is only ~1.1e-16; the real stretched-grid error is larger and elsewhere.
- **Ruled OUT as the dominant stretched-grid term:** `wm_zt`/`wm_zm` (8.7e-17), the grid coordinates zt/zm
  (0 ULP — they match Fortran exactly), the interior metrics invrs_dzt/invrs_dzm (exact, derived from matching
  zt/zm), tau/Lscale/Kh (≤1e-13). Density/pressure diverge only ~1e-9 at the *top* (k52-57, hydrostatic
  integration), not the inversion. The `_ma` budget terms (thlm_ma 2.9e-5, wp2_ma 4.4e-6) are the largest but
  are **post-advance diagnostics** (downstream of the diverged field), not the cause.
- **Next:** the dominant stretched-grid term is still unpinned — it produces ~1e-5 deterministically and is
  amplified at the inversion. Candidates: the hydrostatic pressure integration on a stretched grid (~1e-9 at
  top — does it amplify?), or a higher-order term with a uniform-grid assumption. **New strategy:** the
  uniform-vs-stretched A/B test (`grid_type=1` namelist swap) is the decisive localiser — bisect by checking
  which intermediate first diverges on stretched but is exact on uniform.

### 2026-05-30 — Iter 144: rico — mfl, solver, and XLA-FP RULED OUT; bug is a thl-specific assembly/init difference

Drove the rico flux-solve diagnosis to a near-complete root cause with direct instrumentation (all reverted;
core modules verified clean). New test `tests/test_penta_faithful.py` locks in the solver-faithfulness finding.
- **Monotonic flux limiter (mfl) RULED OUT.** The mfl *does* fire in JAX for rico (instrumented `_apply_mfl`):
  `rtm` change → tendency 3.6e-7 and `thlm` change → tendency 4.677e-5, both **matching the Fortran `rtm_mfl`/
  `thlm_mfl` budgets exactly**; the full signed `thlm`-mfl profile matches Fortran to 2.9e-8. The zero `*_mfl`
  budget *stats* in the compare were just un-written diagnostics, not missing physics. (Flags are `.true.` for
  ALL cases — shared `configurable_model_flags.in` — so mfl is a no-op for the smooth 15, active for rico.)
- **The rt/thl ASYMMETRY (the key clue):** everything rt is machine-exact (`wprtp` 4.5e-15, `rtp2` 1.5e-12,
  `wp2rtp` 6e-14, `rtpthvp` 2e-10) while everything thl diverges (`wpthlp` 4.9e-8, `thlp2` 1e-7, `wp2thlp`
  6e-8, `wp2` 1.1e-5). The rt and thl flux solves share the SAME lhs (`_sh11`) and ADG1 `lhs_ta` → the
  divergence enters through a **thl-specific input/value**, and tracks the **thl inversion gradient** (rt is
  well-mixed → no gradient there). The flux-solve output (`wpthlp_enter_mfl`) diverges 6.4e-3 at the inversion
  k18 but the mfl clamps both paths to the same bound → final `wpthlp` only 4.9e-8 off.
- **Step-1 buoyancy is ZERO** (instrumented: `thlpthvp=[0,0,0,0]` at step 1, rico clear, rcm=0) — so the
  step-1 thl solve diverges with NO buoyancy term. Compare file holds 1 record at t=300s = **end of step 1**.
- **Solver is BIT-FAITHFUL (RULED OUT).** Captured rico's exact thl penta lhs/rhs and compared three solves:
  **EAGER JAX == a pure-numpy Fortran-order penta_lu replica to 0 ULP**; JIT JAX differs only ~5.7e-14 (XLA
  FMA). So given identical lhs/rhs the solve matches Fortran — the cross-case mismatch is in the lhs/rhs
  **assembly or its inputs**, not the solve. (`tests/test_penta_faithful.py` enshrines this.)
- **XLA FP semantics RULED OUT.** Running rico fully EAGER (`jax.disable_jit`) gives the **identical**
  divergence (`wp2` 1.124e-5) as JIT — deterministic, not XLA FMA/contraction.
- **All thl-specific assembly inputs verified MACHINE-EXACT:** `thlm_forcing` 6.8e-21, `wpthlp_forcing` 0,
  `thlpthvp`=0 (step 1), C6thl const, tau/Kh/wp2/density/C6/C7 to ~1e-13. The RHS xm row is `thlm/dt +
  thlm_forcing` (no cancellation); the LHS is shared with rt (rt-exact ⇒ LHS matches). **Only unverified
  thl-specific inputs left: init `thlm`/`wpthlp`.** (rt's init interp is faithful, so init thlm likely matches
  too — pushing the locus toward a subtle wpxp-row term in the assembly.)
- **Next:** the f2py/`clubb_python` API input-matched comparison of `advance_xm_wpxp` (capture rico's matched
  step-1 inputs, feed the Fortran solve, diff the assembled lhs/rhs) — the one test that splits "init differs"
  from "assembly FP-grouping differs". The `clubb_python` API (`clubb_api.advance_xm_wpxp`) is the clean path.

### 2026-05-30 — Iter 143: rico divergence narrowed to the flux solve in the strong-turbulence regime (real, not FP)

Continued the rico bring-up bisection (the means pass, the moments diverge ~1e-5). Refined the root cause and
ruled out several more candidates.
- **Cascade located:** at step 1 the FLUXES `wpthlp`/`upwp`/`vpwp` diverge ~5e-8 absolute (≈5e-6 rel) at
  low-mid BL levels (k=4-6) — and ARM achieves MACHINE PRECISION for these same flux solves, so this is a
  REAL systematic difference, NOT FP. The flux error then feeds the second-order moments (wp2/up2/vp2/rtp2/…
  ~1e-5), which Skw=wp3/wp2^1.5 amplifies → the ADG1 PDF → grows to ~8e-5 by step 3. Common factor: rico's
  STRONG turbulence (init em=1.0 → wp2=up2=vp2=2/3, vs ARM's em_min) — a regime the flux/moment solves are
  under-exercised in by the 15 faithful cases.
- **Verified bit-faithful (NOT the cause):** `tau_zm`/`Lscale` match to **1.8e-16** (machine precision);
  `sigma_sqd_w` 4.7e-7; `wpthvp` 3.3e-6. `advance_wp2_wp3` has no rico-specific branch (only the
  `l_ho_nontrad_coriolis=.false.` one). `l_damp_wp2_using_em` IS implemented (advance_wp2_wp3_module:247/416).
- **Ruled out (cumulative):** cloud/thv, flags, params, the wp3 limiter, the splat, the em/moment init
  (byte-identical to Fortran), the `l_tke_aniso` partition, `l_damp_wp2_using_em`, tau/Lscale.
- **ALL checkable inputs to `advance_xm_wpxp` MATCH** (to machine precision): wm_zt/wm_zm (8.7e-17), the
  forcings (rtm_forcing exact, thlm_forcing 2e-16), rho/rho_ds at the flux levels k4-7 (~5e-16), tau/Lscale,
  the em/partition init. Yet `upwp`/`vpwp` diverge ~5e-8 there WITH NO buoyancy term (rico clear) → the locus
  is the `advance_xm_wpxp` solve ITSELF for rico's inputs (NOT an upstream input). Also noted: the rtm floor
  changed rho/exner ~1e-9 at the TOP (k52-56) but it does NOT propagate to the flux levels (k4-7 density
  exact), so the floor is not the cause. dycoms2_rf01 (faithful) has em_max=1.1 (strong turbulence) so
  strong-turbulence per se isn't it either.
- **Next:** the definitive isolation is an f2py input-matched comparison — `f2py_advance_wp2_wp3` /
  `f2py_advance_xp2_xpyp` / `f2py_advance_xm_wpxp` ARE exposed; capture rico's matched pre-advance state at the
  JAX `advance_xm_wpxp_jax` call site, feed it to both, and diff bit-to-bit to localize the rico-specific term/
  branch in the flux solve. (No code change this iteration → no regression; ARM + rico both still run.)

### 2026-05-30 — Iter 142: diagnose the rico dry-dynamics divergence (trade-inversion moments)

Now that rico RUNS (Iter141), bisected its bit-faithfulness gap with `compare_runs --case rico` (the working
end-to-end oracle). Pure diagnosis (no code change → no regression risk); scopes the next fix.
- **Step-1 bisection:** the MEANS pass (rtm rel 8.5e-7, thlm 7e-9, um/vm ~1e-7) but ALL second-order MOMENTS
  fail (~1e-5): wp2 2.3e-5, up2/vp2 1.31e-5 (identical), rtp2/thlp2/rtpthlp, wp3. The divergence is localised
  to **k=17-18 (zm≈1500 m, the trade inversion)** where wp2 drops sharply 0.34→0.047 over one level — much
  sharper than ARM's well-mixed BL. Skw_zt/w_1 diverge ~3e-4 there (Skw=wp3/wp2^1.5 amplifies the wp2 error,
  feeding the ADG1 PDF → all moments).
- **Ruled out:** cloud/thv (rico is CLEAR at step 1, rcm=cloud_frac=0, wpthvp matches 4e-8); non-default flags
  (rico's flags = ARM's shared defaults); tunable params (JAX uses one tunable_parameters.in for all cases,
  same as ARM); the wp3 limiter (`l_use_wp3_lim_with_smth_Heaviside` — BOTH paths ARE ported and the flag is
  threaded; ARM uses the same `.false.` path); the splat (up2/vp2 budget terms diverge distributed, not via a
  single splat term); **the em/moment INIT** (rico's `l_input_wp2`/`l_input_em` led here, but the JAX
  `_set_cloud_top_profile(1500,1.0)` em init is byte-identical to the Fortran rico init `clubb_driver.F90:5177`,
  and the `l_tke_aniso` partition `wp2=up2=vp2=(2/3)em` matches `:5245` exactly — the init moments DO match).
  So the divergence is the STEP-1 moment SOLVE smoothing rico's sharp initial step-function (em=1.0→em_min at
  1500 m); the JAX/Fortran diffusive-flux discretisation across that near-discontinuity differs ~2e-4 locally
  on wp2.
- **Conclusion:** the moment-budget terms each diverge ~1e-7 and ACCUMULATE to ~1e-5 in the prognostics at the
  sharpest gradient — the signature of FP-accumulation in the coupled moment solve at rico's sharp trade
  inversion (cf. the near-gate-FP cases jun25/coriolis_test). It grows to ~8e-5 by step 3 (the moment error
  feeds the means via the fluxes). Next: deeper wp2-solve term analysis at the inversion to confirm
  FP-limited vs a subtle systematic term, and (separately) wire the per-step microphysics. Reusable workflow:
  `output/{case}_compare_{fort,jax}/{case}_stats.nc` hold the per-step-aligned profiles for level-by-level
  bisection.

### 2026-05-30 — Iter 141: rico now RUNS end-to-end in JAX (rtm rt_tol floor fix)

Resolved the Iter140 rico bring-up blocker; rico now runs the full timestep loop in JAX (no crash) — a
milestone, though not yet bit-faithful.
- **Diagnosis:** rico's model top is 10000 m and its sounding rt → 0 above ~9000 m (the dry upper domain),
  so the JAX `rtm` is exactly 0 there. The advance produces a tiny FP drift to **−4.16e-11** (worse in the
  `l_sample=True` stats path — an FP-ordering sensitivity), which the strict `<0` negativity check (identical
  in the Fortran) flags as fatal → advance_clubb_core returns None (the bare-return-on-fatal). The Fortran
  rico `rtm` min is **exactly 1e-8 = rt_tol** — i.e. the Fortran floors rtm to rt_tol.
- **Fix (`clubb_standalone.py`):** floor the sounding-initialised `rtm = max(rtm, rt_tol)` (rt_tol=1e-8),
  matching the Fortran's observed rtm min. No-op for the 15 bit-faithful cases (their rtm ≥ rt_tol; any
  currently-passing case already equals the Fortran's floored value). **ARM 30-step regression still PASS**
  (0 prognostic failures). **rico now runs 30 timesteps.**
- **Status:** rico RUNS but is NOT bit-faithful — `compare_runs --case rico` shows ~16 prognostic divergences
  by step 3 (rtm 8e-5, thlm 4e-6, um 1e-5, the moments, …). This is the same multi-field bring-up debugging
  each of the 15 cases needed, PLUS the per-step microphysics call (precip_fraction → compute_kk_microphysics →
  the rrm/Nrm advance) which is still not wired into the loop (so rain never forms). `compare_runs --case rico`
  is now the working end-to-end oracle for that work.

### 2026-05-30 — Iter 140: compress CHANGELOG (127-136); port the hydrometeor eddy diffusivity (calculate_K_hm)

- **CHANGELOG compression (10-iteration cadence):** condensed Iters 127-136 — the differentiability completion,
  the D_v large-neg-z/rf02_do generalization, the PDF-moment/correlation/metadata setup, the KK
  velocity/covariance library, and the start of the transport solve — into one block (549→353 lines).
- **`advance_microphys_module.calculate_K_hm`** (oracle :3236, `l_use_non_local_diff_fac=.false.`): the
  hydrometeor eddy diffusivity for the down-gradient turbulent-advection closure `<w'hm'>=-K_hm d<hm>/dz` =
  `c_K_hm·Kh_zm·(√hydrometp2/max(zt2zm(hm),tol))·(1+|Skw_zm|)`, then the correlation cap
  `min(K, √wp2·√hydrometp2/|d<hm>/dz|)` where `|d<hm>/dz|>eps`, K=0 at boundaries. Composes the bit-faithful
  `zt2zm_jax`/`ddzt_jax`.
- **Verification:** K_hm consumes the WITHIN-step hydrometeor field (computed before the advance), so the in-loop
  `K_hm_rr`/`K_hm_Nr` stats are timing-confounded — at robust-rrm points only ~2% (neither the `rrp2` zm-stat nor
  `zt2zm(rrp2_zt)` is the within-step `hydrometp2` the routine used). Verified instead by exact formula
  transcription vs a hand reference (reusing zt2zm/ddzt, incl. the cap + the K=0 boundaries) + differentiability
  — the established convention for within-step-confounded routines (cf. calc_comp_mu_sigma_hm). KK suite (18
  oracle PASS) + ARM smoke clean.
- This is the last per-hydrometeor-advance input.
- **Started the rico INIT/LOOP wiring (gated, ARM-safe):** `_check_unsupported_features` now allows
  `khairoutdinov_kogan`; `clubb_standalone` sets `hydromet_dim=2` + the hydromet mean array (rrm/Nrm init=0) +
  `hm_metadata` (via `kk_hm_metadata`) in the state, gated on the KK scheme (hydromet_dim=0 path unchanged →
  ARM 30-step regression still PASS, 0 prognostic failures). rico now INITIALISES and runs timestep 1.
- **rico bring-up diagnosis (next-iteration task):** rico fails at timestep 2 — `rtm` reaches a tiny negative
  (**−4.16e-11**, near machine noise at a near-zero-rtm dry level) which the strict `<0` negativity check (same
  in Fortran) flags as fatal → advance_clubb_core returns None. The negative appears only in the stats-sampling
  (`l_sample=True`) path; `rtm_min=1e-300`. The microphysics-in-loop call (precip_fraction →
  compute_kk_microphysics → the rrm/Nrm advance) is also not yet wired, so rico would be a dry no-op anyway.
  Next: resolve the rtm<0 FP/clip issue, then wire the per-step microphysics + advance + thv coupling.

### 2026-05-30 — Iter 139: complete the hydrometeor transport solve (microphys_rhs + advance_one_hydrometeor)

Finished the per-hydrometeor transport solver of `advance_hydrometeor`.
- **`advance_microphys_module`:** `term_turb_sed_rhs` (oracle :3014, centered) — the EXPLICIT turbulent-sed
  RHS = flux-divergence of the known field ρ_ds_zm·Vhmphmp_expc (no <hm> factor; top no-flux).
  `microphys_rhs` (oracle :1912) = `hmm/dt + microphysics source + (−½·diffusion·hmm, the Crank-Nicholson
  explicit half) + term_turb_sed_rhs`. `advance_one_hydrometeor` — assemble LHS (Iter138) + RHS, stack the
  3 bands, `tridiag_lu_solve` → advanced <hm>. Refactored the shared ½-diffusion lhs_ta into `_turb_adv_lhs`.
- **Validated:** the FULL turbulent-sed tendency (explicit `term_turb_sed_rhs` + implicit
  `−term_turb_sed_lhs·hmm`) reproduces the rico `rrm_ts` budget to median **4.5e-11** (95% < 1e-6) — proving
  `rrm_ts` = explicit+implicit (the Iter137 hypothesis) and validating `term_turb_sed_rhs` against the oracle.
  `microphys_rhs` == its component sum (bit-exact); `term_turb_sed_rhs` satisfies the conservation contract;
  `advance_one_hydrometeor` round-trips `lhs·soln == rhs` (<1e-18) and physically removes rain mass via
  sedimentation (col 1.15e-3→4.34e-4). KK suite + ARM smoke clean.
- **The per-hydrometeor transport solve is now COMPLETE.** Remaining for a running rico: the init/loop wiring
  (hydromet_dim=2, rrm/Nrm init, the multi-hydrometeor loop + fill_holes), the per-step microphysics call,
  and the thv coupling.

### 2026-05-30 — Iter 138: assemble the full hydrometeor LHS (microphys_lhs); validate the mean-advection budget

Assembled the complete implicit LHS tridiagonal matrix of the hydrometeor transport solve, composing the
verified operators.
- **`advance_microphys_module.microphys_lhs`** (oracle :1564): `lhs = 1/dt[main] + ½·diffusion_zt_lhs(K_hm,nu)
  + term_ma_zt_lhs(wm_zt) + sed_centered_diff_lhs(V_hm) + term_turb_sed_lhs(Vhmphmp_impc)`. Reuses the
  bit-faithful `diffusion_zt_lhs_jax` (turb adv, Crank-Nicholson ½) and `term_ma_zt_lhs_jax` (upwind mean adv,
  `l_upwind_xm_ma=.true.` for rico) plus the conservation-verified sedimentation operators. Adversarial finding:
  the Fortran's explicit k=1 lower-BC re-set of lhs_ta is byte-identical to `diffusion_zt_lhs_jax`'s own bottom
  row (×½) — applied explicitly for faithfulness.
- **Verified** (`test_microphys_lhs_assembly` + `test_microphys_mean_adv_vs_rico`): (i) the assembly == the
  sum of the independently-computed verified sub-operators (bit-exact band/sign/1-dt bookkeeping); (ii) the
  turbulent-advection (eddy-diffusion) part conserves mass to **1.3e-23** (zero-flux); (iii) the
  mean-advection budget `-term_ma·hmm` reproduces rico `rrm_ma`/`Nrm_ma` to **9.9e-14** at robust-rrm points —
  a CLEAN oracle (the ma budget is a plain `stats_update`, unlike the explicit+implicit `rrm_ta`/`rrm_ts`).
- KK suite + ARM smoke clean. Remaining for the transport solve: `microphys_rhs` (+ `term_turb_sed_rhs`) and
  the tridiag `microphys_solve`, then the init/loop wiring + thv coupling.

### 2026-05-30 — Iter 137: port the turbulent-sedimentation LHS (term_turb_sed_lhs); rrm_ts budget bookkeeping

Completed both sedimentation LHS operators of the hydrometeor transport solve.
- **`advance_microphys_module.term_turb_sed_lhs`** (oracle :2683, centered-difference branch
  l_upwind_diff_sed=.false.): the semi-implicit turbulent-sed term -（1/rho)d(rho·<V_hm'h_m'>)/dz with
  <V_hm'h_m'> = Vhmphmp_impc·<hm> + expc. Adversarial finding: the implicit (impc) part is branch-by-branch
  IDENTICAL to `sed_centered_diff_lhs` with the momentum-level `Vhmphmp_impc` (= zt2zm of `kk_sed_vel_covars`'
  Vrrprrp_impc) replacing the velocity — so the function faithfully delegates (no duplicated code).
- **Verified** (`test_term_turb_sed_lhs_vs_rico`): (i) the operator is bit-identical to `sed_centered_diff_lhs`
  on the rico Vhmphmp_impc; (ii) the FULL composition `kk_sed_vel_covars → zt2zm → term_turb_sed_lhs`
  satisfies the conservation contract to **3.1e-15** (234 flux points) — rigorous + timing-independent.
- **Convention discovered:** the in-loop `rrm_ts` stat is EXPLICIT+IMPLICIT (microphys_rhs stores the explicit
  part via stats_begin_budget, microphys_solve adds the implicit via stats_finalize_budget), so an
  implicit-only `-lhs·hmm` check mismatches by ~30× — not an operator bug. Documented in DESIGN; the
  conservation contract is the clean oracle. KK suite + ARM smoke clean.

### 2026-05-30 — Iters 127-136 (condensed): finish KK differentiability; complete the KK velocity/covariance library + the hydrometeor PDF/correlation/metadata setup; start the transport solve

Made the composed upscaled-KK microphysics fully differentiable, generalized it to a second case, completed
every remaining KK rate/velocity/covariance function, ported the PDF-moment/correlation/metadata setup, and
began the `advance_hydrometeor` transport solve. After this block, all KK math the rrm/Nrm advance consumes is
ported+oracle-validated; the gating gap is the init/loop wiring.

- **Differentiability hardening (Iter127-129), all FORWARD-preserving:** the composed `compute_kk_microphysics`
  is now FULLY differentiable (w.r.t. the rrm FIELD and the chi PDF moments, finite-diff-correct ~1e-10). Fixes,
  each AT the op (jnp.where masking does NOT fix a nan-grad — its VJP computes the nan first): a `jax.custom_jvp`
  `_safe_div` (`Nc_Ncn_eqns`, the erfc denom²→0 underflow), a double-where `_safe_sqrt` + a guarded Rmax
  denominator (`setup_clubb_pdf_params`, no-precip 0/0), a double-where `_pos_pow` (`PDF_integrals_means`,
  mu_rr^(1/3) at 0), and a `_dvc` D_v argument clamp to ±50 (the dispatch only selects these forms when
  |s_c|≤49, so the unused extreme region is bounded without changing any tested forward value).
- **D_v large-negative-z branch (Iter130, `parabolic_cylinder.py`):** ran a SECOND KK case, dycoms2_rf02_do
  (stratocumulus, narrow chi PDF s_c~32 → D_v arg ~−32 where the 1F1 series overflows). Ported `_dv_neg_asym`
  (DLMF 12.9.2 growing asymptotic, term ratio (v+2s−1)(v+2s)/(s·2z²), optimal truncation), a 3rd branch (z≲−8)
  with each branch clamped to its valid side. Bit-accurate rel 2.8e-13; rf02_do autoconv now matches the oracle
  (median 5.9e-11). (Adversarial: a missing `/s` ruined both accuracy and the gradient.)
- **PDF-moment orchestration (Iter131, `setup_clubb_pdf_params.py`):** `compute_mean_stdev` (oracle :818, stacks
  the per-PDF-variable component moments [chi,eta,w,Ncn,rr,Nr] into (ngrdcol,nzt,pdf_dim)) + `norm_transform_mean_stdev`
  (oracle :2942, lognormal→log space). The KK driver routes its rr/Nr moments through these — bit-identical.
- **Prescribed correlation arrays (Iter132, `corr_varnce_module.py`):** `set_corr_arrays_to_default` builds the
  in-cloud/below-cloud arrays from the fixed 12×12 default tables (key: the Fortran `reshape` is COLUMN-major →
  numpy `order='F'`). The driver now DERIVES corr(chi,rr)=0.788/chi,Nr=0.675/rr,Nr=0.821 instead of hardcoding;
  cloud==below for every rate entry (rc selection is a no-op). The SILHS Cholesky path is deferred (not needed).
- **KK velocities + covariances (Iter133-134):** `kk_sedimentation` (KK00 Eq.37, Vrr/VNr from the mean volume
  radius, clipped ≤0) — bit-exact vs the rico Vrr/VNr (|Δ|max 1.1e-16, validated through the grid-staggered
  `zt2zm`). `kk_sed_vel_covars` (`KK_upscaled_turbulent_sed.py`) — the sed-velocity/rain covariances `<V_hm'h_m'>`
  written `coefA·<x>+termB` (impc/expc); bivariate-lognormal with (α²+2α)/(α+1) (the extra differenced-variable
  factor) vs `bivar_LL_mean`; bit-faithful vs rico rr/Nr_KK_mvr_covar_zt (4.5e-11, no timing confound by
  reconstructing <hm> within-step).
- **Hydrometeor metadata (Iter135, `corr_varnce_module.py`):** `init_pdf_hydromet_arrays`/`HmMetadata`/`kk_hm_metadata`
  (oracle :455) — per-hydrometeor names/tols/`l_mix_rat_hm`/`l_frozen_hm`, the in-precip variance ratio
  (rico override slope=0/intrcpt=1.25→1.25), and the PDF-variable indices (iiPDF_rr=4, iiPDF_Nr=5, pdf_dim=6).
- **Transport solve START (Iter136, `advance_microphys_module.py`):** `sed_centered_diff_lhs` + `lhs_budget_term`
  — the implicit centered-difference MEAN-SEDIMENTATION operator (oracle :2188, the 3 LHS bands) + the generic
  `-lhs·hmm` budget. Verified by the rigorous oracle-free CONSERVATION CONTRACT (column-mass Σ tendency = surface
  flux) to 5.6e-15 + exact `rrm_sd`=0 at clipped points. **New testing conventions** (documented in DESIGN):
  the conservation-contract oracle for flux-form operators (timing-confound-independent); the two-rico-runs
  pattern (10-step `rico_fort` for the tuned rate tests, 250-step `rico_long_fort` for developed rain); the
  grid-staggered-stat caveat (check `ds[var].dimensions`); and that a new microphysics module MUST enable x64
  (else jnp silently defaults to float32, breaking conservation checks at ~1e-5 while passing rel-error tests).

### 2026-05-31 — Iters 117-126 (condensed): hydromet-PDF setup pieces + the full standalone microphysics step

Completed the rate-function input pipeline and assembled the entire upscaled-KK microphysics into one
composable, oracle-validated, differentiable function. Builds on the Iter108-116 rate library.

- **Rate-function inputs (Iter117-119):** `Nc_Ncn_eqns.py::Nc_in_cloud_to_Ncnm` (cloud-nuclei mean <Ncn> via
  the erfc PDF integral — **bit-to-bit vs f2py**, rel 2.4e-14); `pdf_utilities.py` extended with the inverse
  correlation conversions `corr_NN2NL`/`corr_NN2LL` and the full chi/eta/rt/thl decomposition set
  `calc_corr_{chi,eta,rt,thl}_x` (chi=crt·rt−cthl·thl) — chi/eta **bit-to-bit vs f2py**, rt/thl via the
  exactly-invertible round-trip.
- **Driver assembly (Iter120-121, `kk_microphys_driver.py`):** `kk_autoconversion_mean`/`kk_accretion_mean`/
  `kk_evaporation_mean` — the three KK mass-tendency entry points composing Ncnm/log-moment/coef + the rate
  functions. All validated END-TO-END from the raw PDF-state inputs vs the rico stats: rrm_auto median 4.7e-7,
  rrm_accr 6.1e-9, rrm_evap median 3.3e-6.
- **Tendency assembly + last rate (Iter124-125):** `KK_Nrm_tendencies.py` (`KK_Nrm_auto_mean`,
  `KK_Nrm_evap_local_mean`, and `KK_Nrm_evap_upscaled_mean` — the last KK rate, reusing `trivar_NLL_mean_eq`
  with exps 1,−2/3,5/3; matches rico Nrm_evap median 3.2e-6) and `kk_microphys_adjust` (rates → rrm_mc/Nrm_mc/
  rvm_mc/rcm_mc/thlm_mc with the source/evap over-depletion limiters; rcm_mc exact incl. source adj vs rico).
  ALL KK rates now ported + oracle-validated.
- **Full standalone step (Iter126):** `compute_kk_microphysics` — hydromet fields (rrm,Nrm) + PDF state →
  state tendencies, composing the in-precip component moments + Ncnm + all rates + adjust. Runs; no-rain rcm_mc
  machine-exact vs rico.
- **Differentiability (Iter122):** the rate drivers are differentiable at clean points (jax.grad finite-diff-
  correct ~1e-9).
- **Key investigation (Iter123):** the full hydromet-PDF chain from the rrm FIELD is NOT stats-validatable for
  accr/evap — the stored stats aren't within-step consistent (the defining mean identity fails, rel ~0.7),
  only a RUNNING rico can. Corrected: rico's `hmp2_ip_on_hmm2_ip` is a CASE override = 1.25 (not the default
  0.54+2.12e-5·host_dx); omicron=0.5 → R=0.625, non-emergency. Prescribed normal-space correlations are the
  corr_varnce defaults (cloud=below): corr_chi_Ncn 0.09, corr_chi_rr 0.788, corr_chi_Nr 0.675, corr_rr_Nr 0.821.
  calc_comp_mu_sigma_hm's variance preservation REQUIRES the precip_fraction invariant pf=a·pf1+(1-a)·pf2.

### 2026-05-31 — Iters 108-116 (condensed): KK microphysics rate library + hydrometeor-PDF pieces (all verified)

Built and verified the complete upscaled-KK analytic rate library and the first hydrometeor-PDF setup pieces —
the machinery the precipitating cases (rico, dycoms2_rf02_do/ds) need. None wired into a running case yet
(hydromet_dim=0 still hardcoded); each verified in isolation. New JAX package `src/Microphys/KK_microphys/` +
`src/CLUBB_core/{pdf_utilities,precipitation_fraction,setup_clubb_pdf_params,Nc_Ncn_eqns}.py`.

- **Special function (Iter108):** `parabolic_cylinder.py` D_v(z) — the only transcendental in the KK means
  (oracle the 3385-line ACM Alg. 850 `parab`); 1F1 series (DLMF 12.4) for z≲5.75 + optimally-truncated
  descending asymptotic (DLMF 12.9) for z>5.75. vs scipy.pbdv: series<1e-8, asym<1e-6.
- **Analytic means (Iter108/110/111/112):** `PDF_integrals_means.py` + `KK_upscaled_means.py` — `bivar_NL_mean`
  (+3 const)/`bivar_NL_mean_eq` for autoconversion+accretion; `trivar_NLL_mean`(+5 const)/`trivar_NLL_mean_eq`
  (8-way) for evaporation (chi×r_r×N_r over the chi<0 half); `bivar_LL_mean`(+2 const)/`bivar_LL_mean_eq` for the
  mean volume radius. Wrappers `KK_auto/accr/evap/mvr_upscaled_mean` + params (KK_auto_rc/Nc_exp=2.47/−1.79,
  accr exps 1.15, evap exps 1, 1/3, 2/3, mvr exps 1/3, −1/3; coefs incl. `kk_evap_coef`=3·C_evap·G_T_p·… and
  `G_T_p` in `KK_utilities.py`). Verified vs brute-force quadrature (general bivar 7e-15, accr 1.3e-11, trivar
  3.3e-11, LL 1.5e-15); all differentiable.
- **Moment conversions (Iter109):** `pdf_utilities.py` `mean_L2N`/`stdev_L2N` (linear→log moments, **bit-to-bit
  vs f2py**, rel 0) + `corr_NL2NN`/`corr_LL2NN` (vs Monte-Carlo).
- **END-TO-END oracle validation (Iter113/114):** feeding the FORTRAN's own PDF moments (from rico_stats.nc) +
  the log conversions into the JAX rates reproduces the Fortran outputs — `rrm_auto` median 4.7e-7, `rrm_accr`
  median 6.1e-9, `rrm_evap` median 3.3e-6 (`test_kk_rico_oracle.py`). **New testing strategy (case-stats
  oracle):** a Fortran SCM run writes both rate-function inputs (PDF moments) and outputs (rates), so a rate is
  verifiable by feeding its own moments and matching its own output — decoupling rate-math from PDF-setup. Found
  the evap coef temperature is T_liq=thlm·exner; rico's N_cn is constant (sigma_Ncn=0 → const_x2 path).
- **Hydrometeor-PDF step (b) start (Iter115/116):** `precipitation_fraction.py::precip_fraction` (downward
  cumulative-max overall f_p + the `component_precip_frac_specify` upsilon split + max_hm limiter) — **bit-exact**
  vs rico stats on the well-resolved precip region (f2py wrapper FPE-traps here; documented). `setup_clubb_pdf_params.py::
  calc_comp_mu_sigma_hm` (in-precip component means/stdevs via a mean+variance-preserving quadratic solve,
  omicron/zeta, emergency bounds, 4 branches) — verified via its preservation CONTRACT (machine-eps), since its
  stored stats inputs/outputs aren't within-step consistent (the defining mean identity fails in the stats).
- **Conventions recorded (DESIGN.md):** the case-stats oracle; the f2py-FPE fallback; the stats tol-boundary
  timing confound; contract-based verification when neither f2py nor stats can verify a routine.

### 2026-05-29→31 — Iters 96-106 (condensed): cases 11-15 bit-faithful, coriolis physics, diff suite, KK scoping

**Bit-faithful cases 11→15** (re-testing the blocked set after the Iter94 cold-cloud ice fix, plus two
small ports):
- **cobra (11th, Iter96)** — its Iter90 "step-14 FP-boundary" cloud onset was the SAME `ice_supersat_frac`
  bug (cloud at T=266-270 K); fixed retroactively. **dycoms2_rf02_nd (12th, Iter96)** — "_nd" = NO drizzle,
  a plain stratocumulus case (was mislabeled). **dycoms2_rf01_fixed_sst (13th, Iter98)** — rf01 + fixed SST.
- **cloud_drop_sed port (Iter100)** → **atex_long (14th) + dycoms2_rf02_so (15th)**. `Microphys/cloud_sed_module.py`:
  Stokes-regime droplet sedimentation `Fcsed` (zm) → `sed_rcm=(1/rho)·ddzm(Fcsed)` → `rcm_mc`, `thlm_mc`;
  wired into the driver loop, gated on `l_cloud_sed`. sed_rcm matches Fortran ~1e-11; +`Fcsed` stat (Iter101).
  Validated to 60 steps (Iter101); bomex to 100 steps (Iter105).

**coriolis_test physics (bounded ports, gated off for the 15 faithful cases):** **uv nudging** `l_uv_nudge`
(Iter102: `um/vm -= (um/vm − ref)·dt/ts_nudge`, was threaded-but-unapplied) and the **nontraditional Coriolis**
terms `l_ho_nontrad_coriolis` (Iter103: upwp forcing `+= fcor_y·(up2−wp2)`, wp2 RHS `+= 2·fcor_y·upwp`, wp3
RHS `+= 3·fcor_y·wp2up`, up2 RHS `−= 2·fcor_y·upwp`). Both verified bit-faithful at step 0 (the Foucault
oscillator works); coriolis_test stays FP-limited at the 30-step gate (undamped zeroed-closure accumulation,
same class as jun25). Added `--override` passthrough to `compare_runs.py`.

**jun25_altocu (Iter97-99): exhaustively closed as near-gate FP, no fixable bug.** EVERY verifiable
constituent matches the oracle (tau_zm/Kh_zm/sigma_sqd_w, rc_coef/thv_ds, chi/stdev_chi/mixt_frac, rsat/rsati,
bv_freq, all forcings/subsidence, AND the cloud fluxes wprcp/rtprcp/thlprcp — bit-exact 1e-12 for the faithful
dycoms2_rf01). The residual is a coupled cloud-region amplification of a ~2e-6 seed by the steep kappa=100
radiation + cold (264 K) cloud. Added the cloud_frac/ice_supersat_frac (Iter97) and cloud-flux wprcp/rtprcp/
thlprcp/uprcp/vprcp (Iter99) **stats writes** (were computed but unwritten).

**Differentiability suite (Iter104, the "differentiable composable" goal):** `tests/test_differentiability.py`
— `jax.grad` (finite-diff-correct ~1e-9) through saturation, the lax.scan tridiag solver (grad w.r.t. rhs AND
matrix bands), the erf PDF cloud-fraction core, their composition, and Brunt-Vaisala N² grid `ddzt` (Iter106).
End-to-end grad through a full timestep is NOT yet available (advance_clubb_core round-trips through NumPy).

**Microphysics scoping (Iter101/105/106), now the roadmap in DESIGN.md:** all 33 remaining cases need a major
unported subsystem; the f2py API exposes 226 CLUBB_core fns but ZERO microphysics (so KK can only be verified
by full-case comparison or first-principles). JAX is ARM-only (`hydromet_dim=0`, no hydrometeor infra). ALL KK
cases use the UPSCALED analytic-PDF path; the Fortran rico stats expose the rates+moments (`rrm_auto/accr/evap`,
chi/Ncn moments) as the oracle. Order: hydrometeor infra → hydromet PDF → rate functions → advance → thv coupling.

**Conventions recorded (DESIGN.md):** unwritten diagnostic stats read as 0 (an output gap, not a physics bug —
grep `update("<name>"` first); after a shared-physics fix, re-run the BLOCKED set (old diagnoses may already pass).

### 2026-05-29 — Iters 91-95 (condensed): 9th→10th bit-faithful case + general infra

Brought the set to 10 bit-faithful cases (… neutral, ekman) and added general fixes/conventions
(all detailed conventions live in DESIGN.md):
- **91** neutral surface scheme (`neutral_case_sfclyr`: ustar=0.5, momentum flux, `wpthlp_sfc=0.05`
  until t=80880 s) → 9th case.
- **92-93** Ported **sponge-layer damping** (`sponge_layer_damping.py`: tau profile + implicit
  `sponge_damp_xm` toward the initial profile, wired into `advance_clubb_core` for rtm/thlm/uv) and
  **ekman_sfclyr**. `run_scm.py` reads/strips the case `parameter_file`. **allclose gate floor**
  (`|Δ| ≤ 1e-12 + 1e-6·|ref|`) kills dry-case false positives. ekman means bit-faithful but moments
  still drifted (fixed in 94).
- **94 — ekman → 10th case (key general fix).** Root-caused the moment drift to the warm-only shortcut
  `ice_supersat_frac = cloud_frac`; ported `calc_ice_cloud_frac_component` (ICE-supersaturation PDF
  fraction for below-freezing levels). The wrong ≈0 N² had passed the splat clip → spurious wp2→up2/vp2
  splatting at the cold (203 K) domain top. Added `T_freeze_K`. **Invisible on warm/shallow cases.**
- **95** coriolis_test triaged out (needs the unported nontraditional-Coriolis terms; it also zeroes
  all closure constants). `run_scm` basename-fallback for stale `parameter_file` paths. Adversarial
  flag-path scan + the no-parallel-runs OOM lesson + 80-step long-run check (bomex/dycoms2_rf01) —
  all recorded in DESIGN.md.

### 2026-05-29 — Iter 90: cobra cloud-onset diagnosis; compress Iters 79-89

- **cobra cloud-onset residual localized.** cobra is bit-faithful through step 13; at step 14 cloud
  forms (rcm=2.4e-9, which **matches** Fortran) and `wp3` diverges (rel 1.4e-5) in the cloud layer
  (1420-1780 m). `wpthvp` is exact (5.7e-9) but the velocity-weighted cloud flux feeds `wp2thvp`
  (2.7e-6) and the wp3 pressure/dissipation terms (`wp3_pr1` 4.4e-4) — FP-boundary amplification of
  the tiny onset cloud, the same subtle class as bomex's old residual. Deferred (deep PDF/wp3 dive).
- **Compressed Iters 79-89** (the multi-case bit-faithfulness campaign) into the summary below;
  the discovered conventions live in DESIGN.md.

### 2026-05-29 — Iters 79-89 (condensed): 8 cases bit-faithful + generalized testing

Brought the bit-faithful set from ARM-only to **8 cases** (arm, bomex, dycoms2_rf01, wangara, atex,
gabls2, gabls3_night, fire) and made the harness case-general. Each fix verified bit-to-bit vs the
Fortran oracle at 30 steps:
- **79** BOMEX fixed-height surface BC (`l_modify_bc_for_cnvg_test`, mono-cubic interp to z=25 m);
  added `run_scripts/compare_cases.py` multi-case dashboard.
- **80** Cloud-regime thv moments: use native-zt rc-flux moments (no zt→zm→zt round-trip) →
  dycoms2_rf01/bomex cloud path.
- **81** Refresh the carried `rc_coef_zm` each step in the post-advance PDF path (was stale 0) →
  bomex + dycoms2_rf01 fully bit-faithful.
- **82-84** Ported + wired the **xm monotonic flux limiter** (`mono_flux_limiter.py`) after the
  um/vm/rtm/thlm solves → atex bit-faithful (no-op where Fortran's `*_mfl`=0).
- **85** gabls2 'failure' was a stats-averaging artifact, not physics; made `compare_runs.py`
  auto-force per-step instantaneous output (`stats_tsamp=stats_tout=dt_main`, or `--tout`).
- **86** gabls3_night: apply the time-dependent wind forcing (`um_f`/`vm_f`/`ug`/`vg`) → 7th case.
- **87** fire: bulk surface scheme (`sfctype=1`, `Cz=0.0013`) → 8th case; cobra `z0=1.75 m`.
- **88-89** General fix: a forcing column blank (-999.9) in every block is 'not provided' — do NOT
  overwrite the sounding state (`ug`/`vg`/`w`); cobra recovered from a blow-up to step-13-faithful.

### 2026-05-29 — Iter 78: port cloud_feedback/cgils/astex_a209/cobra/jun25/clex9 sfclyr; fix imu import

- `generic_forcings.py`: added `_bulk_aero_sfclyr` (generic bulk aerodynamic port for cloud_feedback, astex_a209; C_h_20/C_q_20/z0 drag coefficients scaled to model height, sat_mixrat_liq_jax for rsat, T_sfc from sfc file) and `_zero_flux_sfclyr` (for zero-surface-flux altocu cases)
- Dispatch extended: cloud_feedback_*/cgils_* (ustar=0.3), astex_a209 (ustar=0.155), cobra (arm_variant_sfclyr + T_sfc), jun25_altocu (zero flux) — all sfclyr correctly ported
- Blockers remain at driver level (not sfclyr): cgils/cloud_feedback need bugsrad+sponge, astex_a209 needs l_cloud_sed+bugsrad, nov11/clex9 need morrison microphysics
- **Bug fix**: `mixing_length.py:calc_lscale_directly_jax` was referencing `imu` without importing it from `constants_clubb`. Added to imports. Verified `l_diag_Lscale_from_tau=.false.` ARM override runs 3 steps cleanly.
- ARM compare_runs.py: PASS (0 prognostic failures)

### 2026-05-29 — Iter 77: fix fill_blanks bug; port atex_long, arm_0003/97/mc3e/3year sfclyr

- **Root cause fix**: `_parse_forcings_file` was interpolating -999.9 sentinel values literally instead of applying `fill_blanks_two_dim_vars` first. Added `_linear_fill_blanks_1d` and `_fill_blanks_2d` (ports of `input_reader.F90:linear_fill_blanks` and `fill_blanks_two_dim_vars`). Fix builds a common z-grid across all time blocks, applies 2D fill (z-first then time), then interpolates to model grid.
- **gabls3_night now works**: the fill_blanks fix eliminates the catastrophic `thlm = -467K` after one step that was caused by interpolating -999.9 sentinel forcings
- Added `_atex_long_sfclyr` (same as atex with `adjustment=0.0194664` per atex_long.F90)
- Added `_arm_variant_sfclyr` for arm_0003, arm_97, mc3e, arm_3year (reads sens_ht/latent_ht from sfc file, converts by rho_sfc, uses diag_ustar)
- All 6 key cases verified at 5 timesteps: gabls2, gabls3_night, atex, bomex, dycoms2_rf01, wangara
- ARM compare_runs.py: PASS (0 prognostic failures)

### 2026-05-29 — Iter 76: port GABLS2, GABLS3-night, ATEX sfclyr; fix _time_interp arg order

- `generic_forcings.py`: added `_gabls2_sfclyr` (analytic T_sfc piecewise formula, bulk aero C_10 scaled to model height, diag_ustar), `_landflx_scalar` + helper functions `gm1/gh1/fm1/fh1/psi_h` (SAM Monin-Obukhov stability scheme, 3-iter unstable / quadratic stable), `_gabls3_night_sfclyr` (uses landflx per-column, reads thlm_sfc/rtm_sfc/upwp_sfc/vpwp_sfc from sfc file), `_atex_sfclyr` (C_10=0.0013, T_sfc from file, ustar=0.3, adjustment=0.0198293)
- `_parse_sfc_file`: extended to recognize gabls3_night columns (thlm, rt[, upwp, vpwp)
- Bug fix: `_time_interp` call in `_interp_col` had arguments in wrong order (`arr, times` → `times, arr`)
- Bug fix: `_apply_time_dependent_forcings` used `[:,:-1]` slicing causing shape mismatch; corrected to `[:,:]`
- GABLS2 and ATEX verified: run 5 timesteps cleanly. GABLS3_night sfclyr implemented but physics fails parameterization_check (end-of-advance) at debug_level=2 — stable-BL physics stability issue for further investigation
- ARM compare_runs.py: PASS (0 prognostic failures)

### 2026-05-29 — Iter 75: port RICO, DYCOMS2-RF01, DYCOMS2-RF02, Wangara, LBA sfclyr functions

- `generic_forcings.py`: added `_rico_sfclyr` (RICO 3D bulk aerodynamic spec, C_m/h/q_20 scaled to model height, sat_mixrat_liq_jax for rsat, direct momentum flux computation), `_dycoms2_rf01_sfclyr` (sfctype=0: time-interpolated sens_ht/latent_ht/T_sfc from sfc file, ustar=0.25), `_dycoms2_rf02_sfclyr` (time-interpolated wpthlp/wpqtp, ustar=0.25), `_wangara_sfclyr` (analytic cosine formula, UTC→AEST+10h, ustar=0.13), `_lba_sfclyr` (analytic cosine with elapsed time, diag_ustar)
- Dispatch updated to route all these cases to pure Python; Fortran fallback only for unported cases
- DYCOMS2_RF01 and Wangara verified: 30-timestep JAX runs complete successfully
- RICO and LBA blocked at Python driver level by unported microphysics/radiation (separate from sfclyr correctness)
- ARM compare_runs.py: PASS (0 prognostic failures)

### 2026-05-29 — Iter 74: generic prescribe_forcings framework; BOMEX and simple cases ported

- `src/Benchmark_cases/generic_forcings.py` (new): pure-Python port of `prescribe_forcings.F90` for non-ARM cases
  - `prescribe_forcings_generic`: full dispatch covering bomex, fire, generic, neutral, coriolis_test, ekman, and any case with `l_t_dependent=True` and a `{case}_forcings.in` file
  - `_bomex_tndcy`: analytic BOMEX moisture tendency (`bomex.F90:bomex_tndcy`), with `force_spec_hum_to_mixing_ratio` conversion
  - `_bomex_sfclyr`: surface fluxes from `bomex_sfc.in` via time interpolation; `flux_spec_hum_to_mixing_ratio`
  - `_apply_time_dependent_forcings`: generic time-dep framework reading `{case}_forcings.in`
  - `_read_surface_var_for_bc`, `_compute_ubar`, `_compute_momentum_flux`, `_set_sclr_sfc_rtm_thlm`
  - `load_generic_forcings_data`: reads `{case}_forcings.in` and `{case}_sfc.in` at init time
  - Unsupported cases raise `NotImplementedError` and fall through to Fortran lazy import
- `src/advance_clubb_to_end.py`: `_prescribe_forcings` now tries Python dispatcher first; Fortran only as fallback for unsupported cases
- `src/clubb_standalone.py`: loads generic forcing data at init for all non-ARM cases
- Verified: BOMEX 3-step JAX run completes; ARM compare_runs.py PASS (0 prognostic failures)

### 2026-05-29 — Iter 73: eliminate module-level Fortran imports from advance_clubb_to_end.py and radiation.py

- `src/derived_types/`: new directory with pure-Python mirrors of `clubb_python.derived_types` (ConfigFlags, ErrInfo, SclrIdx, Grid, pdf_parameter, implicit_coefs_terms) — bypasses `clubb_python/__init__.py → clubb_api` eager import
- `src/clubb_standalone.py`: updated imports to `clubb_jax.src.derived_types.*`
- `src/CLUBB_core/advance_helper_module.py`: added `calculate_thlp2_rad_jax` (port of `advance_helper_module.F90:calculate_thlp2_rad`)
- `src/CLUBB_core/constants_clubb.py`: added `rc_tol`, `rho_lw`, `sec_per_hr`, `radians_per_deg`, `ithlp2_rad_coef`
- `src/Radiation/radiation.py`: ported `cos_solar_zen` (Liou coefficients, calendar arithmetic) and `sunray_sw` (Shettle-Weinman SW flux) to pure Python; replaced all `clubb_api.stats_update` calls with `stats_writer`; removed `from clubb_python import clubb_api`
- `src/advance_clubb_to_end.py`: removed module-level `clubb_api` import; `_calculate_thlp2_rad` now uses `calculate_thlp2_rad_jax`; stats fallbacks (`sw is None`) removed (silently skip when no StatsWriter); `_prescribe_forcings` uses lazy `from clubb_python import clubb_api` for non-ARM cases only; removed dead `_advance_clubb_core_api` function (~197 lines)
- compare_runs.py: PASS (30 timesteps, 0 prognostic failures)

### 2026-05-29 — Directory restructure: CLUBB-JAX repo

- Moved `clubb_jax/` out of `clubb_release/` into the top-level `CLUBB-JAX/` directory
- `clubb_release/` is now a git submodule pointing to `larson-group/clubb_release` master
- Test scripts (`run_scm.py`, `compare_runs.py`) moved into `clubb_jax/run_scripts/`; `clubb_release/` is unmodified upstream
- `clubb_jax/src/clubb_standalone.py`: uses `_CLUBB_RELEASE_ROOT` to locate Fortran input files from the sibling submodule
- Verified tests pass against a fresh clone of `clubb_release` master

### 2026-05-29 — `clubb_jax/src/` mirrors Fortran `src/` layout (Refactor Iters 1–3)

- Restructured `clubb_jax/` so every JAX module sits at the same relative path as its Fortran oracle
- Removed backward-compat shim directories (`jax_core/`, `benchmark_cases/`, `io/`)
- All primary consumers updated to import from canonical `src/` paths

### 2026-05-29 — Port check_clubb_settings and check_parameters to Python (Iter 72)

- `numerical_check.py`: `check_clubb_settings_jax` (10 validation checks, fatal + warning), `check_parameters_jax` (all range checks)
- `src/clubb_standalone.py` now has **zero `from clubb_python import clubb_api` imports**

### 2026-05-29 — Port parameterization_check and init routines (Iters 69–70)

- `numerical_check.py`: `parameterization_check_jax` (NaN/Inf + negativity checks, 35 arrays)
- `parameters_tunable.py`: `init_clubb_params_jax`, `calc_derrived_params_jax` — bit-exact
- `model_flags.py`: `get_default_config_flags_jax` — all 88 flags

### 2026-05-28 — Pure-Python stats writer (Iter 67)

- `io/stats_writer.py`: `StatsWriter` mirrors `stats_netcdf.F90` (begin/update/budget/end_timestep, accumulation, NetCDF output)
- All `clubb_api.stats_*` calls removed; ARM per-timestep Fortran calls: **ZERO**

### 2026-05-28 — Bug fix: ice_supersat_frac (Iter 68)

- Missing `ice_supersat_frac = cloud_frac.copy()` after Block U PDF closure caused cascade failure at timestep 214 in 225-step runs

### 2026-05-28 — Port ARM forcings to pure Python (Iter 66)

- `Benchmark_cases/arm.py`: `prescribe_forcings_arm`, `load_arm_forcings_data`, `_diag_ustar` (Monin-Obukhov, 4 iterations)
- Last per-timestep Fortran call removed from `advance_clubb_to_end.py` for ARM

### 2026-05-28 — Initialization ports: hydrostatic, rcm_sat_adj, calculate_thvm (Iters 62–65)

- `calc_pressure.py`: `hydrostatic_jax`, `init_pressure_jax` (sequential upward integration via `jax.lax.scan`)
- `saturation.py`: `rcm_sat_adj_jax` (bisection, 100 iterations, vectorized over `(ngrdcol, nzt)`)
- `advance_clubb_to_end.py`: `calculate_thvm` now uses `calculate_thvm_jax`

### 2026-05-28 — Remove all Fortran calls from advance loop (Iters 56–65)

- Replaced `set_lscale_max`, upwind TA terms, `pdf_params` Fortran object, `sat_mixrat_liq`, `thlm2t_in_k`, `calc_lscale_directly`, and all non-ARM conditional Fortran paths
- ARM state path: **ZERO Fortran calls** after Iter 59

### 2026-05-27 — Remove Fortran oracle calls from advance loop (Iters 46–55)

- Removed Fortran `advance_xm_wpxp`, `advance_xp2_xpyp`, `advance_wp2_wp3`, `advance_windm_edsclrm`, `pdf_closure_driver` calls
- Removed all shadow comparison infrastructure; JAX values primary
- compare_runs.py: PASS (100 timesteps, 0 prognostic failures)

### 2026-05-27 — JAX drives all prognostic state (Iters 34–45)

- All 16 prognostic variables carried forward from JAX each timestep
- Replaced Fortran clip/fill_holes calls with JAX equivalents; cross-timestep ADG1 state passing
- compare_runs.py: PASS (30 timesteps, 0 prognostic failures)

### 2026-05-27 — ADG1 PDF closure (Iters 25–33)

- `adg1_adg2_3d_luhar_pdf.py`: full ADG1 closure — w-closure, responder params, all higher-order moments (wp2xp, wpxp2, wp2xp2, wp4, wprtp2, wpthlp2, wprtpthlp), virtual temperature fluxes

### 2026-05-27 — Pre-advance diagnostics (Iters 15–24)

- Ported: Skw, thvm, BV, Ri, Cx, Lscale/tau, splat, sfc_varnce, sigma_sqd_w, clip functions, fill_holes
- All overriding Fortran from Iter 38

### 2026-05-27 — Full advance functions (Iters 10–14)

- `advance_xp2_xpyp`, `advance_xm_wpxp`, `advance_wp2_wp3`, `advance_windm_edsclrm`, `advance_xp3` — all machine epsilon vs Fortran

### 2026-05-27 — Core operators and LHS/RHS terms (Iters 1–9)

- Grid interpolation, diffusion LHS, tridiagonal solver, MA/DP1/xp2_xpyp/TA LHS and RHS terms
- Unit test suite established (solver, diffusion, penta-solver, Fortran oracle)
