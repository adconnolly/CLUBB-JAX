# COAMPS_PORT.md — runbook: port COAMPS microphysics to JAX

**One-line goal:** port the COAMPS (NRL Rutledge & Hobbs bulk mixed-phase)
microphysics scheme from fixed-form F77 to JAX so the cloudy MPACE-B case can run its
native microphysics in the JAX driver — faithful to the Fortran source (there is **no
f2py oracle**: COAMPS is compiled out by default and fatal-errors on `l_predict_Nc=F`),
and eventually differentiable (jnp + the tracer-transparent toolkit, same as Morrison).

This is a **loop-until-done / multi-session** task. This document is the bootstrap: the
mirror structure, the driver top-level flow, the leaf utilities, the CLUBB wiring, and a
runnable MPACE-B are DONE. The bulk of the physics (`adjtq` + ~40 `eqa*.F`) remains.

---

## Environment (every session)
```bash
PY=/burg/home/ac5006/scratch/jaxenv/bin/python
export PYTHONPATH=/burg-archive/glab/users/ac5006/CLUBB-JAX:/burg-archive/glab/users/ac5006/CLUBB-JAX/clubb_release
export JAX_PLATFORMS=cpu
export HDF5_USE_FILE_LOCKING=FALSE
```
Read `DESIGN.md` (correctness standard, tracer-transparent toolkit, mirror convention)
and `MORRISON_PORT.md` (the porting methodology this mirrors). Toolkit:
`clubb_jax/src/CLUBB_core/tracer_numpy.py` (`_xp`, `_is_tracer_arg`, `_safe_sqrt`, `_safe_pow`, `_iset`).

**Run MPACE-B via the driver path** (the standalone `run_scm.py` path only supports
`microphys_scheme='none'`; COAMPS/Morrison need `clubb_driver.init_clubb_case` +
`advance_clubb_to_end`, exactly like Morrison — see `MEMORY: two-jax-driver-paths`):
```bash
# 1. build the aggregate namelist (run_scm writes it before the standalone init fails):
$PY clubb_jax/run_scripts/run_scm.py mpace_b -max_iters 3 \
    -out_dir clubb_jax/output/mpace_b_compare_jax   # writes .../mpace_b.in (init errors — OK)
# 2. run through the driver path (a 12-line script; see scratchpad/run_mpace_b.py):
$PY - <<'EOF'
from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end
s = init_clubb_case("clubb_jax/output/mpace_b_compare_jax/mpace_b.in")
s['ifinal'] = 3
advance_clubb_to_end(s, l_stdout=True)
EOF
```
Mirror check after structural changes: `$PY clubb_jax/run_scripts/mirror_audit.py`.

---

## COAMPS call map (what calls what)

CLUBB dispatch (`microphys_driver.F90:calc_microphys_scheme_tendcies`, `case "coamps"`):

```
coamps_microphys_driver           coamps_microphys_driver_module.F90  (964 lines) — the CLUBB<->COAMPS wrapper
  ├─ rvm = rtm - rcm ; thm ; T_in_K = thlm2T_in_K_api                 (CLUBB, ported)
  ├─ sat_mixrat_liq_api / sat_mixrat_ice  -> qsatv3d / qsati3d         (CLUBB saturation, ported)
  ├─ gamma(3..9, bsnow/bgrp magic numbers)  -> gm3..gmbov2g            (COAMPS gamma.F, PORTED)
  ├─ slope intercepts sloper/slopes/slopeg
  ├─ in-cloud/saturated-level detection (sat>0 or any hydromet>=pcut)  -> len, kcomp
  ├─ build k-FLIPPED COAMPS arrays (COAMPS assumes k=1 = domain top)
  ├─ if len>0:  call adjtq(...)   <-- THE MASTER PROCESS ROUTINE       (adjtq.F 1609 lines, STUB)
  ├─ un-flip; clip nc3/nr3/ncn3 >= 0 ; Nccnm/Nsm bookkeeping
  └─ tendencies = (field_after - field_before)/dt  -> {ri,rr,rg,rs,nr,nc,ni}tend, rvm_mc, rcm_mc, thlm_mc
                  + fall speeds Vrr/VNr/Vrs/Vri/Vrg

adjtq  (adjtq.F) — advances qc/qi/qr/qg/qs + nc/nr/ni over one dt in a two-pass structure
                   (warm/collection pass, then ice deposition/nucleation, then autoconv/accretion),
                   with an internal saturation adjustment. Subroutines it calls:
  leaves (PORTED):     slope, esatv, esati, qsatvi        (+ esat_new = Goff-Gratch, PORTED)
  fall speeds:         tgqr, tgqs, tgqg, tgqi
  ice/adjust/number:   frzh, conice, adjmlt, qtadj, nrmcol, nrmtqw, nrmtqi
  process rates eqa*:  eqa6(pcond) eqa7(praut) eqa9(pracw) eqa12(prevp) eqa15(pint)
                       eqa18(pdepi) eqa19(pconv) eqa21(psaci) eqa22(psacw) eqa25(psmlt)
                       eqa26(psdep) eqa27(pmltse) eqa27r(piacw) eqa28(psmlti)
                       + graupel variants (l_graupel): eqa5g eqa7g eqa8g eqa9g eqa10g
                         eqa11g eqa12g eqa13g eqa14g eqa17g eqa18g eqa19g eqa20g eqa21g eqa22g
```

MPACE-B config (`mpace_b_model.in`): `microphys_scheme="coamps"`, `l_ice_microphys=.true.`,
`l_graupel=.false.`, `l_predict_Nc=.true.`, `l_ignore_forcings=.true.` (no LS forcing;
surface-flux + microphysics driven), `sfctype=0` (fixed sens/latent heat from `mpace_b_sfc.in`),
`rad_scheme="simplified"`, evenly-spaced grid `nzmax=56`. `ldrizzle=.false.` (rain/drizzle off;
must be false when `l_ice_microphys=.true.`). `icase=75` mapping is per-case in the driver.

---

## Files created / modified (this bootstrap)

**Created — JAX mirror of the Fortran (all under `clubb_jax/src/Microphys/`):**
- `COAMPS_microphys/__init__.py` — package marker.
- `COAMPS_microphys/gamma.py` — COAMPS gamma approx (gamma.F). PORTED, verified.
- `COAMPS_microphys/slope.py` — rain/snow/graupel slope factors (slope.F). PORTED.
- `COAMPS_microphys/esat_new.py` — Goff-Gratch sat vapour pressure (esat_new.F). PORTED
  (uses the closed-form "exact" branch, not the 0.1 K lookup table — differentiable).
- `COAMPS_microphys/esatv.py` / `esati.py` — over-water / over-ice wrappers (esatv.F/esati.F). PORTED.
- `COAMPS_microphys/qsatvi.py` — sat mixing ratios (qsatvi.F). PORTED.
- `COAMPS_microphys/adjtq.py` — master routine. **STUB** (no-op passthrough + full call-graph
  docstring). This is the remaining bulk of the port.
- `coamps_microphys_driver_module.py` — CLUBB<->COAMPS wrapper (coamps_microphys_driver_module.F90).
  Top-level flow PORTED and runs; produces 0 tendencies while adjtq is a stub.
- `coamps_microphys_step.py` — JAX-only per-step wiring (analogue of morrison_microphys_step.py).

**Modified — wiring (no Morrison/KK code touched):**
- `Microphys/microphys_driver.py` — added `elif scheme == 'coamps':` dispatch.
- `clubb_driver.py` — allow `microphys_scheme='coamps'`; give it Morrison's 8-field
  hydrometeor metadata (rr/Nr/ri/Ni/rs/Ns/rg/Ng — COAMPS predicts a subset, Ns/Ng stay 0).
- `Benchmark_cases/prescribe_forcings.py` — mpace_b LS forcing = zero (`l_ignore_forcings`);
  tightened the generic `l_t_dependent` surface branch to ignore a degenerate scalar
  `wpthlp_sfc` key so the sens_ht/latent_ht path is taken (mpace_b + any sens_ht-only file).
- `run_scripts/mirror_audit.py` — allowlist `coamps_microphys_step` as JAX-only glue.

---

## How far MPACE-B gets

- **Inits**: YES — `init_clubb_case` returns cleanly (nzm=76, nzt=75, ngrdcol=1,
  hydromet_dim=8, microphys_scheme='coamps').
- **Runs**: YES — 3 and 20 steps via the driver path, stable, no NaNs. The COAMPS driver
  executes its full top-level flow every step (gamma constants, saturation, in-cloud
  detection, k-flip, adjtq stub, tendency loop). A one-time `RuntimeWarning` announces the
  adjtq stub.
- **Physics**: all `*_mc` tendencies are **0** (adjtq is a no-op), so COAMPS currently
  applies no microphysics. NOTE — unlike clear mpace_a, **mpace_b is genuinely cloudy**
  (rcm_max ≈ 5.4e-4 kg/kg from the CLUBB PDF closure alone) — the right target for a
  cloudy-case microphysics/tuning demo once adjtq is ported.

---

## File-by-file port checklist

| Fortran | JAX | status |
|---|---|---|
| coamps_microphys_driver_module.F90 | coamps_microphys_driver_module.py | ✅ top-level flow (0-tendency until adjtq) |
| gamma.F | gamma.py | ✅ ported + verified |
| slope.F | slope.py | ✅ ported |
| esat_new.F | esat_new.py | ✅ ported (closed-form) |
| esatv.F / esati.F | esatv.py / esati.py | ✅ ported |
| qsatvi.F | qsatvi.py | ✅ ported |
| adjtq.F (1609 lines) | adjtq.py | 🔴 **STUB** (no-op) — the core remaining work |
| tgqr/tgqs/tgqg/tgqi.F (fall speeds) | — | 🔴 not started |
| frzh/conice/adjmlt/qtadj.F | — | 🔴 not started |
| nrmcol/nrmtqw/nrmtqi.F (number bookkeeping) | — | 🔴 not started |
| eqa6/7/9/12/15/18/19/21/22/25/26/27/27r/28.F (warm+ice rates) | — | 🔴 not started |
| eqa5g..eqa22g.F (15 graupel variants; l_graupel=F for mpace_b) | — | 🔴 not started (deprioritize: off for mpace_b) |

`.prol` files are `#include`d parameter/declaration headers — no runtime logic to port.

---

## Known gaps / caveats

- **No oracle.** COAMPS is compiled out by default (`#ifdef COAMPS_MICRO`) and the
  upstream fatal-errors on `l_predict_Nc=.false.`. Validate each ported routine against
  the **F77 source logic** and by conservation/finiteness (Tier A), not bit-vs-Fortran.
- **k-flip.** CLUBB has k=1 at the surface; COAMPS assumes k=1 at the domain top. The
  driver builds `_flip` copies before `adjtq` and un-flips after. adjtq does no advection/
  sedimentation, so the flip is a no-op for the stub — but ported process rates that index
  neighbours (fall speeds) must respect it. Keep the flip in the driver; write adjtq to the
  top-down convention it expects.
- **Number-concentration unit conversions at the COAMPS boundary** (driver, F90:596-610 /
  846-899): `nc3=Ncm/cm3_per_m3`, `nr3=Nrm/cm3_per_m3`, `ni3=Nim*rho`; inverted on the way
  out. These are already in the ported driver — reuse them; don't double-convert inside adjtq.
- **hm_metadata is Morrison's 8-field layout** (a bootstrap shortcut). COAMPS predicts
  rr/Nr/ri/Ni/rs/rg (+ diagnostic Ns, Nccnm, Ncm); Ns/Ng stay 0. Revisit if a dedicated
  `coamps_hm_metadata` is warranted (e.g. Nccnm as a real prognostic).
- **`real*4` everywhere in COAMPS.** The F77 is single precision; the JAX port is float64
  (more accurate, differentiable). Expect small numeric offsets vs a hypothetical
  single-precision COAMPS — judge under Tier-C, per DESIGN's precision rule.

---

## Top-3 next steps

1. **Port `adjtq.F` incrementally, warm-pass first.** Start with the `l_ice_microphys=F`
   liquid-only path (pcond/praut/pracw/prevp via eqa6/eqa7/eqa9/eqa12 + qtadj saturation
   adjustment + slope) so cloud water ↔ rain works, then add the ice pass (conice/pint/
   pdepi/psdep/psaci/…). Port the leaf rate `eqa*.F` files as pure jnp functions and have
   `adjtq` orchestrate them, replacing `adjtq_stub`. Keep graupel (`*g`) last (off for mpace_b).
2. **Add a per-routine conservation/finiteness test** (no oracle): feed a synthetic
   saturated column into each ported rate + into `adjtq`, assert water+energy conservation
   and `rr/ri/rs/rg,Nx >= 0` (Tier A). Use the mpace_b cloudy state (rcm≈5e-4) as the fixture.
3. **Wire the hydrometeor transport + sedimentation** in `coamps_microphys_step.py` (the
   `Vrr/VNr/Vrs/Vri/Vrg` fall speeds adjtq returns), mirroring how morrison_microphys_step.py
   uses `advance_one_hydrometeor` — currently the step does a plain Euler update (fine while
   tendencies are 0). Then compare the JAX MPACE-B climatology to the published COAMPS-LES /
   Klein et al. (2009) LWP as the Tier-D honest gate.
