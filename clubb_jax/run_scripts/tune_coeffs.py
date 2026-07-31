#!/usr/bin/env python3
"""tune_coeffs.py — differentiable gradient-descent tuning of 5 CLUBB closure
coefficients (C1, C11, c_K, gamma_coef, mult_coef) via Adam.

Pipeline validation mode (default): generate a target trajectory from perturbed
coefficients, then recover them from the defaults. Proves the optimizer +
per-field-normalized loss + multi-step reverse-mode gradient work end-to-end
before swapping in real observations.

  python tune_coeffs.py [case] [N_steps] [n_iter]
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end
from clubb_jax.src.CLUBB_core.parameter_indices import iC1, iC11, ic_K, igamma_coef, imult_coef

REPO = "/burg-archive/glab/users/ac5006/CLUBB-JAX"
CASE = sys.argv[1] if len(sys.argv) > 1 else "arm"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 20
NIT  = int(sys.argv[3]) if len(sys.argv) > 3 else 40
IDX  = [iC1, iC11, ic_K, igamma_coef, imult_coef]
NAMES = ["C1", "C11", "c_K", "gamma_coef", "mult_coef"]
# Loss field groups: mean profiles, turbulence moments/fluxes (surface flux/BL
# depth stand in via wpthlp_sfc through wpthlp[0]); equal group weights.
FIELDS = {"mean": ["thlm", "rtm", "um", "vm"],
          "turb": ["wp2", "wpthlp", "wprtp"]}

def _aj(x): return x if isinstance(x, jax.core.Tracer) else jnp.asarray(x)

state = init_clubb_case(f"{REPO}/clubb_jax/output/{CASE}_compare_jax/{CASE}.in")
state['l_stats'] = False; state['stats_writer'] = None
state['flags'] = state['flags']._replace(fill_holes_type=1)  # differentiable global fill
advance_clubb_to_end(state, l_stdout=False, max_steps=3)
base = np.asarray(state['clubb_params'], dtype=np.float64)
theta_def = jnp.asarray(base[0, IDX])                      # defaults

def forward(theta):
    """Return dict of final-time profiles after N steps for coeff vector theta."""
    s = dict(state)
    p = jnp.asarray(base)
    for j, idx in enumerate(IDX):
        p = p.at[:, idx].set(theta[j])
    s['clubb_params'] = p
    s['l_stats'] = False; s['stats_writer'] = None
    advance_clubb_to_end(s, l_stdout=False, max_steps=N)
    return {k: _aj(s[k]).ravel() for grp in FIELDS.values() for k in grp}

# --- build a synthetic target from perturbed coefficients ------------------
theta_true = theta_def * jnp.asarray([1.15, 1.20, 0.85, 1.10, 1.0])  # mult_coef inert
target = jax.tree_util.tree_map(lambda a: jax.lax.stop_gradient(a), forward(theta_true))
# per-field normalizer: std of the target field (so heterogeneous units compare)
norm = {k: float(jnp.std(v)) + 1e-30 for k, v in target.items()}

def loss(theta):
    pred = forward(theta)
    total = 0.0
    for grp, keys in FIELDS.items():
        g = 0.0
        for k in keys:
            g = g + jnp.mean(((pred[k] - target[k]) / norm[k]) ** 2)
        total = total + g / len(keys)          # equal weight within group
    return total / len(FIELDS)                  # equal weight across groups

# Eager value_and_grad (NOT jax.jit): the per-case forcings run outside the core
# jit and write state via np.asarray, which a top-level jit can't trace. The
# per-step advance_clubb_core_jit still accelerates the physics kernels.
val_and_grad = jax.value_and_grad(loss)

# --- Adam (hand-rolled; no optax dependency) -------------------------------
theta = theta_def
m = jnp.zeros_like(theta); v = jnp.zeros_like(theta)
lr, b1, b2, epsA = 0.05, 0.9, 0.999, 1e-8
print(f"case={CASE} N={N} iters={NIT}")
print("theta_true/theta_def:", dict(zip(NAMES, np.asarray(theta_true / theta_def))))
for it in range(1, NIT + 1):
    L, g = val_and_grad(theta)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** it); vh = v / (1 - b2 ** it)
    theta = theta - lr * mh / (jnp.sqrt(vh) + epsA)
    if it == 1 or it % 5 == 0 or it == NIT:
        err = np.asarray(theta / theta_true - 1.0)
        print(f"  it{it:3d}  loss={float(L):.4e}  |theta/true-1|_max={np.max(np.abs(err[:4])):.3e} "
              f"theta={np.asarray(theta)}")
print("\nfinal theta   :", dict(zip(NAMES, np.asarray(theta))))
print("target theta  :", dict(zip(NAMES, np.asarray(theta_true))))
