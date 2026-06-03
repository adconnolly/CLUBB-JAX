"""tracer_numpy.py — tracer-transparent numpy shim (REFACTOR B4/B5).

The toolkit that lets the imperative numpy orchestration thread reverse-mode ``jax.grad`` WITHOUT a
rewrite: each helper behaves **exactly like numpy** for concrete arrays (so normal runs are bit-identical
— the bit-faithful suite is provably unaffected) but routes to ``jnp`` under a JAX trace (so the autodiff
graph survives). Shared by ``advance_clubb_core_module`` (B4) and ``advance_clubb_to_end`` (B5) so the
tracer-transparent toolkit is defined once.

Usage: replace `np.asarray(`→`_asarray(`, the inline ufuncs `np.maximum(`→`_xp.maximum(`, the `_like`
constructors `np.zeros_like(`→`_xp.zeros_like(`, and `arr[idx] = v`→`arr = _iset(arr, np.s_[idx], v)`.
Never store a tracer in module-global state (guard with `_is_tracer_arg`). Use `_safe_sqrt` for
`sqrt(maximum(0,·))` (the bare form's reverse grad is inf at the clip).
"""
import numpy as np
import jax
import jax.numpy as jnp

_np_asarray = np.asarray


def _is_tracer_arg(a):
    if isinstance(a, jax.core.Tracer):
        return True
    if isinstance(a, (list, tuple)):
        return any(isinstance(e, jax.core.Tracer) for e in a)
    return False


def _asarray(x, dtype=None):
    """Tracer-transparent ``np.asarray``: jnp under a JAX trace, else exactly numpy (bit-identical)."""
    if isinstance(x, jax.core.Tracer):
        return jnp.asarray(x, dtype)
    return _np_asarray(x, dtype)


def _iset(arr, idx, val):
    """Immutable-safe ``arr[idx] = val``: under a trace promote to jnp + ``.at[idx].set``; else mutate
    the numpy array in place and return it (bit-identical). Use with ``np.s_[...]`` for slice indices."""
    if _is_tracer_arg([arr, val]):
        return jnp.asarray(arr).at[idx].set(val)
    # Concrete path: a numpy view of a jax array (np.asarray of a non-tracer jax.Array) is read-only,
    # so copy before mutating. Same values → bit-identical; only affects already-immutable inputs.
    if getattr(arr, "flags", None) is not None and not arr.flags.writeable:
        arr = np.array(arr)
    arr[idx] = val
    return arr


def _safe_sqrt(x):
    """``sqrt(max(x,0))`` with a finite (0) gradient at x<=0 (double-where) — the bare
    ``jnp.sqrt(jnp.maximum(0,x))`` has an inf reverse-mode gradient at the clip. Forward-identical."""
    xp = jnp.maximum(x, 0.0)
    safe = jnp.where(xp > 0.0, xp, 1.0)
    return jnp.where(xp > 0.0, jnp.sqrt(safe), 0.0)


def _safe_pow(x, p):
    """``max(x,0)**p`` with a finite (0) gradient at x<=0 (double-where) — for a FRACTIONAL p<1 the bare
    ``jnp.maximum(x,0)**p`` has an inf reverse-mode gradient at the clip (d/dx = p*x**(p-1) -> inf as x->0).
    Forward-identical to ``jnp.maximum(x,0.0)**p`` for p>0 (0**p == 0). Use when p can be < 1."""
    xp = jnp.maximum(x, 0.0)
    safe = jnp.where(xp > 0.0, xp, 1.0)
    return jnp.where(xp > 0.0, safe ** p, 0.0)


class _TracerXP:
    """numpy shim: ``_xp.<fn>(...)`` dispatches to jnp when any arg (or list/tuple element) is a tracer,
    else to numpy (bit-identical). Non-callables / dtypes / constants fall through to numpy."""
    def __getattr__(self, name):
        np_fn = getattr(np, name)
        jnp_fn = getattr(jnp, name, None)
        if jnp_fn is None or not callable(np_fn):
            return np_fn

        def dispatch(*args, **kw):
            return (jnp_fn if any(_is_tracer_arg(a) for a in args) else np_fn)(*args, **kw)
        return dispatch


_xp = _TracerXP()
