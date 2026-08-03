#!/usr/bin/env python3
"""Feasibility probe: is the JAX Morrison driver differentiable w.r.t. the cloud
droplet number Ncm? Capture real driver inputs from a warmed mc3e state, then
grad sum of the thlm/rcm/rvm tendencies w.r.t. Ncm."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
if "--nanhunt" in sys.argv:
    jax.config.update("jax_debug_nans", True)
from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end
from clubb_jax.src.Microphys.morrison_microphys_module import morrison_microphys_driver

REPO = "/burg-archive/glab/users/ac5006/CLUBB-JAX"
state = init_clubb_case(f"{REPO}/clubb_jax/output/mc3e_compare_jax/mc3e.in")
state['l_stats'] = False; state['stats_writer'] = None
state['flags'] = state['flags']._replace(fill_holes_type=1)
advance_clubb_to_end(state, l_stdout=False, max_steps=4)

hmm = state['hm_metadata']; g = lambda k: np.asarray(state[k], np.float64)
hydromet = g('hydromet'); pick = lambda i: jnp.asarray(hydromet[..., int(i)])
rcm = jnp.asarray(g('rcm')); thlm = jnp.asarray(g('thlm')); cf = jnp.asarray(g('cloud_frac'))
Nc_in_cloud = jnp.asarray(g('Nc_in_cloud'))
rvm = jnp.asarray(g('rtm')) - rcm
exner = jnp.asarray(g('exner')); rho = jnp.asarray(g('rho')); pres = jnp.asarray(g('p_in_Pa'))
from clubb_jax.src.CLUBB_core.clubb_constants import Lv, Cp
T_in_K = thlm * exner + (Lv / Cp) * rcm
dt = float(state['dt_main']); dzq = jnp.asarray(np.asarray(state['gr'].dzm, np.float64)[:, 1:])
I = lambda n: int(getattr(hmm, n))

def loss(nc_scale):
    Ncm = Nc_in_cloud * nc_scale * cf
    out = morrison_microphys_driver(
        rcm, Ncm, pick(I('iirr')), pick(I('iiNr')), pick(I('iiri')), pick(I('iiNi')),
        pick(I('iirs')), pick(I('iiNs')), pick(I('iirg')), pick(I('iiNg')),
        thlm, rvm, T_in_K, exner, pres, rho, cf, dzq, dt)
    return jnp.sum(out['thlm_mc']) + jnp.sum(out['rcm_mc']) + jnp.sum(out['rvm_mc'])

try:
    f0, g0 = jax.value_and_grad(loss)(jnp.asarray(1.0))
    print(f"forward (sum of *_mc) = {float(f0):.6e}  finite={np.isfinite(float(f0))}")
    print(f"d/d(Nc_scale) = {float(g0):.6e}  finite={np.isfinite(float(g0))}")
    eps = 1e-3
    fd = (float(loss(jnp.asarray(1.0+eps))) - float(loss(jnp.asarray(1.0-eps)))) / (2*eps)
    print(f"FD(eps={eps}) = {fd:.6e}  rel = {abs(fd-float(g0))/(abs(fd)+abs(float(g0))+1e-30):.2e}")
except Exception as e:
    import traceback
    fr = [l.strip() for l in traceback.format_exc().splitlines()
          if "module_mp_graupel.py" in l or "morrison" in l.lower()]
    print("NaN/err. Morrison frames (deepest last):")
    for x in fr[-8:]: print("   ", x)
    print("ERR:", str(e)[:160])
