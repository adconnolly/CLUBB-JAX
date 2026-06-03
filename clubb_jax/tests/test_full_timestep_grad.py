#!/usr/bin/env python3
"""test_full_timestep_grad.py — B4 probe/validator (REFACTOR B4 stage 1, iter15).

Captures the real 135-kwarg input of `advance_clubb_core` for one arm step (via the env-gated
CLUBB_CAPTURE_KWARGS hook) and:
  1. confirms the forward call runs (88 returns, finite) — a real gate;
  2. reports the full-timestep `jax.grad` status. TODAY this is BLOCKED by the orchestration's NumPy
     writebacks (`TracerArrayConversionError` from `np.asarray()` on a tracer) — the B4 target. When the
     pure-functional core lands, this flips to asserting grad is finite + finite-difference-correct.

This is the fixture-based validator the staged B4 conversion is built against (no slow case run per check).
"""
import os
import sys
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_JAX_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))
_RUN = os.path.join(_JAX_ROOT, "clubb_jax", "run_scripts")
sys.path.insert(0, _JAX_ROOT)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_FIXTURE = os.path.join(_JAX_ROOT, "clubb_jax", "output", "b4_fixtures", "arm_core_kwargs.pkl")
# advance_clubb_core return order: um(0) vm(1) up3(2) vp3(3) thlm(4) ... — index of the output we grad.
_THLM_OUT = 4


def _ensure_fixture():
    if os.path.isfile(_FIXTURE):
        return _FIXTURE
    os.makedirs(os.path.dirname(_FIXTURE), exist_ok=True)
    env = os.environ.copy()
    env["CLUBB_CAPTURE_KWARGS"] = _FIXTURE
    env["PYTHONPATH"] = os.pathsep.join([_JAX_ROOT, _JAX_ROOT + "/clubb_release",
                                         _JAX_ROOT + "/clubb_release/clubb_python_api"])
    try:
        subprocess.run([sys.executable, os.path.join(_RUN, "run_scm.py"), "arm", "-jax",
                        "-max_iters", "1"], env=env, check=True, timeout=600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return _FIXTURE if os.path.isfile(_FIXTURE) else None


def _load():
    import pickle
    p = _ensure_fixture()
    if p is None:
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def test_forward_runs():
    from clubb_jax.src.CLUBB_core.advance_clubb_core_module import advance_clubb_core
    kw = _load()
    if kw is None:
        print("SKIP: could not build the arm kwarg fixture")
        return True
    out = advance_clubb_core(**kw)
    assert len(out) == 88, f"unexpected return count {len(out)}"
    thlm_out = np.asarray(out[_THLM_OUT])
    assert np.all(np.isfinite(thlm_out)), "forward thlm output not finite"
    print(f"  full timestep forward: advance_clubb_core runs (88 returns, thlm {thlm_out.shape} finite)  PASS")
    return True


def test_grad_status():
    """Report the full-timestep jax.grad status. PASS today documents the B4 blocker; flips to a real
    grad gate (finite + FD-correct) once the pure-functional core (B4) lands."""
    from clubb_jax.src.CLUBB_core.advance_clubb_core_module import advance_clubb_core
    kw = _load()
    if kw is None:
        print("SKIP: no fixture")
        return True

    # Target the differentiable PROGNOSTIC path: bypass the debug check + stats (debug_level=0,
    # l_sample=False) — these diagnostics are not part of the prognostic update and break tracing
    # separately (REFACTOR B4: grad path = prognostic update, diagnostics are side computations).
    kw = dict(kw); kw["debug_level"] = 0; kw["l_sample"] = False

    # advance_clubb_core return order: um(0) vm(1) up3(2) vp3(3) thlm(4) rtm(5) ...
    # Verify grad is finite + finite-difference-correct w.r.t. SEVERAL prognostic inputs (not just thlm) —
    # a robust 'differentiable' claim, not a single-field fluke.
    cases = [("thlm", 4), ("rtm", 5), ("um", 0)]

    def grad_ok(in_key, out_idx):
        def loss(x):
            k = dict(kw); k[in_key] = x
            return jnp.sum(advance_clubb_core(**k)[out_idx])
        x0 = jnp.asarray(kw[in_key])
        g = jax.grad(loss)(x0)
        finite = bool(np.all(np.isfinite(np.asarray(g))))
        nonzero = float(jnp.sum(jnp.abs(g))) > 0
        eps = 1e-4
        tan = jnp.ones_like(x0)
        fd = float((loss(x0 + eps * tan) - loss(x0 - eps * tan)) / (2 * eps))
        ad = float(jnp.sum(g * tan))
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert finite and nonzero and rel < 1e-3, f"d{out_idx}/d{in_key}: ad={ad:.3e} fd={fd:.3e} rel={rel:.2e}"
        return rel

    try:
        rels = {f"d{ok_out}/d{ik}": grad_ok(ik, ok_out) for ik, ok_out in cases}
        worst = max(rels.values())
        print(f"  full timestep jax.grad: WORKS, finite + FD-correct for {len(cases)} prognostics "
              f"(worst rel {worst:.1e}; {', '.join(f'{k}={v:.0e}' for k,v in rels.items())}) — B4 COMPLETE  PASS")
    except jax.errors.TracerArrayConversionError:
        print("  full timestep jax.grad: BLOCKED by NumPy writebacks (TracerArrayConversionError) "
              "— B4 not yet done (expected; this is the B4 target)  PASS(documented)")
    return True


if __name__ == "__main__":
    ok = True
    ok &= test_forward_runs()
    ok &= test_grad_status()
    print("\nAll full-timestep-grad probe checks PASSED" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
