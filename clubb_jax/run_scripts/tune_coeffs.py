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
# --morrison / MORR=1: tune the Morrison cloud-droplet number Nc_in_cloud
# (single log-scale param) instead of the 5 CLUBB closure coeffs.
MORR = bool(os.environ.get("MORR"))
if MORR:
    NAMES[:] = ["Nc_in_cloud"]; IDX[:] = [0]
    nc0 = np.asarray(state['Nc_in_cloud'], np.float64)
theta_def = jnp.asarray([1.0]) if MORR else jnp.asarray(base[0, IDX])   # defaults

# Tune in LOG space: theta = theta_def * exp(u), u unconstrained (starts at 0).
def theta_of(u):
    return theta_def * jnp.exp(u)

def forward(theta):
    """Return dict of final-time profiles after N steps for the param vector theta."""
    s = dict(state)
    if MORR:
        s['Nc_in_cloud'] = jnp.asarray(nc0) * theta[0]   # theta[0]=exp(u): multiplicative Nc scale
    else:
        p = jnp.asarray(base)
        for j, idx in enumerate(IDX):
            p = p.at[:, idx].set(theta[j])
        s['clubb_params'] = p
    s['l_stats'] = False; s['stats_writer'] = None
    advance_clubb_to_end(s, l_stdout=False, max_steps=N)
    return {k: _aj(s[k]).ravel() for grp in FIELDS.values() for k in grp}

# --- target: real obs netCDF (--obs / 4th arg) or synthetic recovery -------
OBS = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("OBS")
if OBS:
    # Real observations: mean-state fields only (VARANAL has no turbulence).
    from clubb_jax.run_scripts.obs_target import load_obs_target
    FIELDS = {"mean": ["thlm", "rtm", "um", "vm"]}
    tgt = load_obs_target(OBS, state, time_target_s=N * float(state['dt_main']))
    target = {k: jnp.asarray(v).ravel() for k, v in tgt.items()}
    theta_true = None
    print(f"OBS target: {OBS}")
else:
    theta_true = jnp.asarray([1.5]) if MORR else \
        theta_def * jnp.asarray([1.15, 1.20, 0.85, 1.10, 1.0])  # MORR: Nc*1.5; else per-coeff
    target = jax.tree_util.tree_map(lambda a: jax.lax.stop_gradient(a), forward(theta_true))
# per-field normalizer: std of the target field (so heterogeneous units compare)
norm = {k: float(jnp.std(v)) + 1e-30 for k, v in target.items()}

def loss(u):
    pred = forward(theta_of(u))
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

# --- Adam (hand-rolled; no optax dependency) with lr decay -----------------
u = jnp.zeros(len(IDX))
m = jnp.zeros_like(u); v = jnp.zeros_like(u)
lr0, b1, b2, epsA = 0.08, 0.9, 0.999, 1e-8
print(f"case={CASE} N={N} iters={NIT}  (log-space, lr0={lr0} decay)  mode={'OBS' if OBS else 'synthetic-recovery'}")
if theta_true is not None:
    print("target theta/def:", dict(zip(NAMES, np.round(np.asarray(theta_true / theta_def), 3))))
for it in range(1, NIT + 1):
    L, g = val_and_grad(u)
    lr = lr0 * (0.5 ** (it / 25.0))            # halve every 25 iters
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** it); vh = v / (1 - b2 ** it)
    u = u - lr * mh / (jnp.sqrt(vh) + epsA)
    if it == 1 or it % 5 == 0 or it == NIT:
        th = np.asarray(theta_of(u))
        if theta_true is not None:
            err = th / np.asarray(theta_true) - 1.0
            print(f"  it{it:3d} loss={float(L):.3e} "
                  f"err%=[" + " ".join(f"{n}:{100*e:+5.1f}" for n, e in zip(NAMES, err)) + "]")
        else:  # real obs: no ground truth — report loss + coeff/default ratios
            print(f"  it{it:3d} loss={float(L):.3e} "
                  f"theta/def=[" + " ".join(f"{n}:{r:.3f}" for n, r in zip(NAMES, th / np.asarray(theta_def))) + "]")
print("\nfinal  theta:", dict(zip(NAMES, np.round(np.asarray(theta_of(u)), 4))))
if theta_true is not None:
    print("target theta:", dict(zip(NAMES, np.round(np.asarray(theta_true), 4))))
