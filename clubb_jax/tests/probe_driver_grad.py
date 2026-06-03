#!/usr/bin/env python3
"""probe_driver_grad.py — B5 probe (REFACTOR B5): whole-DRIVER reverse-mode grad.

Builds a real arm `state` via init_clubb_case, warms up a few steps, then takes `jax.grad`
of a scalar loss (sum of thlm after ONE more `advance_clubb_to_end` step, stats OFF) w.r.t. an
input thlm perturbation. Reports the first tracer blocker (file:line) if grad is not yet finite,
else confirms grad is finite + finite-difference-correct. This is the B5 analogue of the B4
fixture validator (test_full_timestep_grad.py) but exercises the FULL driver step, not just core.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_JAX_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))
for p in (_JAX_ROOT, _JAX_ROOT + "/clubb_release", _JAX_ROOT + "/clubb_release/clubb_python_api"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end

_CASE = sys.argv[1] if len(sys.argv) > 1 else "arm"
_WARMUP = 3   # run a few concrete steps to pass step-1 specials, then grad through step _WARMUP+1


def _namelist_for(case):
    """First existing aggregate namelist in a WORKING dir for the case. Never use the golden copy:
    init_clubb_case truncates the case's stats .nc beside its namelist, which would corrupt the golden
    oracle (Tier-B baseline). Prefer the compare-run working copy, then the output root."""
    cands = [
        os.path.join(_JAX_ROOT, "clubb_jax", "output", f"{case}_compare_jax", f"{case}.in"),
        os.path.join(_JAX_ROOT, "clubb_jax", "output", f"{case}.in"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"no working-dir namelist for case {case!r}; tried {cands} "
                            "(run compare_cases/run_scm once to generate one)")


def _build_state():
    state = init_clubb_case(_namelist_for(_CASE))
    # Stats / diagnostics OFF — the standard differentiable-forward config (no NetCDF, no sampling).
    state['l_stats'] = False
    state['stats_writer'] = None
    if _WARMUP > 0:
        advance_clubb_to_end(state, l_stdout=False, max_steps=_WARMUP)
    return state


def _loss_from(base_state, in_key, out_key):
    """Returns loss(x): one driver step from a fresh shallow copy, scalar = 0.5*sum(out_key**2).

    The nonlinear (squared) loss gives a per-level-varying gradient (not the trivial ~1 of a
    near-conserved column sum), and choosing in_key='um'/out_key='um' exercises the new differentiable
    arm surface momentum-flux path (um -> ustar/upwp_sfc -> core -> um)."""
    x0 = np.asarray(base_state[in_key], dtype=np.float64)

    def loss(x_in):
        s = dict(base_state)              # shallow copy: top-level writebacks land in s, not base
        s[in_key] = x_in
        s['l_stats'] = False
        s['stats_writer'] = None
        advance_clubb_to_end(s, l_stdout=False, max_steps=1)
        return 0.5 * jnp.sum(_aj(s[out_key]) ** 2)

    return loss, x0


def _aj(x):
    return x if isinstance(x, jax.core.Tracer) else jnp.asarray(x)


def _check_one(base, in_key, out_key):
    """Forward + grad + FD-correctness for d(0.5*sum(out_key^2))/d(in_key). Returns worst FD rel (or None on block)."""
    loss, x0 = _loss_from(base, in_key, out_key)
    x0_j = jnp.asarray(x0)
    tag = f"d(.5*sum {out_key}^2)/d {in_key}"

    f0 = float(loss(x0_j))
    print(f"[B5] {tag}: forward = {f0:.6e}  (finite={np.isfinite(f0)})")
    try:
        g = np.asarray(jax.grad(loss)(x0_j))
    except Exception as e:
        import traceback
        frames = [l for l in traceback.format_exc().splitlines()
                  if "CLUBB-JAX" in l and "probe_driver_grad" not in l]
        print(f"[B5] {tag}: grad BLOCKED: {type(e).__name__}: {str(e)[:140]}")
        if frames:
            print("[B5] blocker frame:", frames[-1].strip())
        return None
    n_finite = int(np.isfinite(g).sum())
    print(f"[B5] {tag}: grad finite {n_finite}/{g.size}  |g|_max={np.nanmax(np.abs(g)):.3e}")
    if n_finite != g.size:
        return None

    eps = 1e-6
    gflat, flat0 = g.ravel(), x0.ravel()
    nz = flat0.size
    worst = 0.0
    for i in [nz // 4, nz // 2, 3 * nz // 4]:
        pp = flat0.copy(); pp[i] += eps
        pm = flat0.copy(); pm[i] -= eps
        fd = (float(loss(jnp.asarray(pp.reshape(x0.shape))))
              - float(loss(jnp.asarray(pm.reshape(x0.shape))))) / (2 * eps)
        an = float(gflat[i])
        rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-30)
        worst = max(worst, rel)
        print(f"[B5] {tag}: level {i:3d}: analytic={an:+.4e} fd={fd:+.4e} rel={rel:.2e}")
    print(f"[B5] {tag}: worst FD rel = {worst:.2e}")
    return worst


def main():
    base = _build_state()
    # thlm->thlm: thermodynamic transport path; um->um: also stresses the new differentiable arm
    # surface momentum-flux path (um -> ustar/upwp_sfc -> core).
    results = {(ik, ok): _check_one(base, ik, ok) for ik, ok in [('thlm', 'thlm'), ('um', 'um')]}
    bad = [k for k, v in results.items() if v is None or v >= 1e-3]
    if bad:
        print(f"[B5] INCOMPLETE — failing probes: {bad}")
        return 1
    print(f"[B5] B5 COMPLETE — whole-driver grad finite + FD-correct for {list(results)} "
          f"(worst rel {max(results.values()):.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
