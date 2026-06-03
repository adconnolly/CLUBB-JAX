import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_JAX_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))
for p in (_JAX_ROOT, _JAX_ROOT + "/clubb_release", _JAX_ROOT + "/clubb_release/clubb_python_api"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from clubb_jax.src.clubb_driver import init_clubb_case
from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end

case = sys.argv[1] if len(sys.argv) > 1 else "bomex"
# Working-dir namelist only — never the golden copy (init_clubb_case truncates the stats .nc beside it).
nl = os.path.join(_JAX_ROOT, "clubb_jax", "output", f"{case}_compare_jax", f"{case}.in")
if not os.path.isfile(nl):
    nl = os.path.join(_JAX_ROOT, "clubb_jax", "output", f"{case}.in")
state = init_clubb_case(nl); state['l_stats'] = False; state['stats_writer'] = None
advance_clubb_to_end(state, l_stdout=False, max_steps=3)
x0 = jnp.asarray(np.asarray(state['thlm'], dtype=np.float64))

def loss(x):
    s = dict(state); s['thlm'] = x; s['l_stats'] = False; s['stats_writer'] = None
    advance_clubb_to_end(s, l_stdout=False, max_steps=1)
    return 0.5 * jnp.sum(jnp.asarray(s['thlm']) ** 2)

jax.config.update("jax_debug_nans", True)   # only for the grad — skip benign masked nans in init
try:
    g = jax.grad(loss)(x0)
    print("no nan; |g|max", float(jnp.nanmax(jnp.abs(g))))
except FloatingPointError as e:
    import traceback
    tb = traceback.format_exc()
    frames = [l for l in tb.splitlines() if "CLUBB-JAX" in l and "_nanhunt" not in l]
    print("FloatingPointError:", str(e)[:120])
    for fr in frames[-6:]:
        print("  ", fr.strip())
