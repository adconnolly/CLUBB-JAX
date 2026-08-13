#!/usr/bin/env python3
"""Isolate which Nc->thlm route is detached: after 1 mc3e step, grad the
cloud-sed tendency (route 1) and the morrison tendency (route 2) w.r.t. Nc."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end
REPO = "/burg-archive/glab/users/ac5006/CLUBB-JAX"
state = init_clubb_case(f"{REPO}/clubb_jax/output/mc3e_compare_jax/mc3e.in")
state['l_stats'] = False; state['stats_writer'] = None
state['flags'] = state['flags']._replace(fill_holes_type=1)
advance_clubb_to_end(state, l_stdout=False, max_steps=30)   # develop real cloud
print("after warmup: rcm_max=%.2e cloud_frac_max=%.2e" % (
    float(np.max(state['rcm'])), float(np.max(state['cloud_frac']))))
nc0 = np.asarray(state['Nc_in_cloud'], np.float64)
def _s(x): return x if isinstance(x, jax.core.Tracer) else jnp.asarray(x)
def run(scale, key):
    s = dict(state); s['Nc_in_cloud'] = jnp.asarray(nc0) * scale
    s['l_stats'] = False; s['stats_writer'] = None
    advance_clubb_to_end(s, l_stdout=False, max_steps=1)
    v = s.get(key)
    return jnp.sum(_s(v) ** 2) if v is not None else jnp.asarray(0.0)
for key, label in (('rcm_mc', 'route1 cloud_drop_sed'), ('_morr_thlm_mc', 'route2 morrison')):
    g = float(jax.grad(lambda x: run(x, key))(jnp.asarray(1.0)))
    print(f"{label:26s} d sum({key}^2)/dNc = {g:+.6e}  finite={np.isfinite(g)}")
