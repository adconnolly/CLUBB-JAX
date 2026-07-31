#!/usr/bin/env python3
"""probe_coeff_grad.py — de-risk for the tune-to-obs task.

Takes jax.grad of a short (~N-step) trajectory loss w.r.t. the 5 tuning
coefficients (C1, C11, c_K, gamma_coef, mult_coef) by injecting them into a
traced state['clubb_params'] array, then checks the gradient is finite and
finite-difference-correct. Clears both de-risking checks at once:
  (1) whole-driver grad valid through N steps (not just 1);
  (2) the 5 coefficients flow as traceable inputs.

Usage: python probe_coeff_grad.py [case] [N]
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "../.."))
for p in (_REPO, _REPO + "/clubb_release"):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end
from clubb_jax.src.CLUBB_core.parameter_indices import iC1, iC11, ic_K, igamma_coef, imult_coef

CASE = sys.argv[1] if len(sys.argv) > 1 else "arm"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 30
WARMUP = 3
IDX   = [iC1, iC11, ic_K, igamma_coef, imult_coef]
NAMES = ["C1", "C11", "c_K", "gamma_coef", "mult_coef"]

def _namelist(case):
    for c in (f"{_REPO}/clubb_jax/output/{case}_compare_jax/{case}.in",
              f"{_REPO}/clubb_jax/output/{case}.in"):
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"no working namelist for {case!r}; run compare_cases/run_scm once")

def _aj(x):
    return x if isinstance(x, jax.core.Tracer) else jnp.asarray(x)

# Build + warm up a concrete state
state = init_clubb_case(_namelist(CASE))
state['l_stats'] = False; state['stats_writer'] = None
if os.environ.get("FILL_GLOBAL"):
    # Route around the non-differentiable sliding-window hole-fill (dynamic-bound
    # fori_loop) by using the vectorized global fill (differentiable).
    state['flags'] = state['flags']._replace(fill_holes_type=1)  # global_fill
    print("fill_holes_type -> global_fill (1)")
advance_clubb_to_end(state, l_stdout=False, max_steps=WARMUP)
base_params = np.asarray(state['clubb_params'], dtype=np.float64)   # (ngrdcol, 102)
theta0 = jnp.asarray(base_params[0, IDX])                            # per-col-uniform defaults
print(f"case={CASE} N={N} warmup={WARMUP}  theta0={dict(zip(NAMES, np.asarray(theta0)))}")

def loss(theta):
    s = dict(state)                       # shallow copy; writebacks land in s
    p = jnp.asarray(base_params)
    for j, idx in enumerate(IDX):
        p = p.at[:, idx].set(theta[j])    # broadcast scalar over all columns
    s['clubb_params'] = p
    s['l_stats'] = False; s['stats_writer'] = None
    advance_clubb_to_end(s, l_stdout=False, max_steps=N)
    return 0.5 * jnp.sum(_aj(s['thlm']) ** 2)

f0, g = jax.value_and_grad(loss)(theta0)
f0 = float(f0); g = np.asarray(g)
print(f"\nforward loss = {f0:.6e}  finite={np.isfinite(f0)}")
print(f"grad finite {int(np.isfinite(g).sum())}/{g.size}")
for n, gi in zip(NAMES, g):
    print(f"  dL/d{n:11s} = {gi:+.6e}")

# Finite-difference check per coefficient
print("\nFD check (eps=1e-6):")
eps = 1e-6; t0 = np.asarray(theta0); worst = 0.0
for j, n in enumerate(NAMES):
    tp = t0.copy(); tp[j] += eps
    tm = t0.copy(); tm[j] -= eps
    fd = (float(loss(jnp.asarray(tp))) - float(loss(jnp.asarray(tm)))) / (2 * eps)
    an = float(g[j]); rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-30)
    worst = max(worst, rel)
    print(f"  {n:11s}: analytic={an:+.4e} fd={fd:+.4e} rel={rel:.2e}")
print(f"\nworst FD rel = {worst:.2e}  ->  {'PASS' if worst < 1e-3 and np.all(np.isfinite(g)) else 'CHECK'}")
