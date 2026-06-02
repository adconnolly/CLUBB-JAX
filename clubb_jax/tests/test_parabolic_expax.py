"""Validate the JAX `expax_U` port against the oracle's epss=1e-4 parabolic-cylinder values.

`expax_U(a, x)` reproduces `Dv_fnc(order, arg)` for arg>0 (= U(a,x), a=-order-0.5, x=arg) at the
SCM-run tolerance epss=1e-4 — the value the do/ds covar source actually needs (Iter316). The
reference values below were produced by the standalone `parab` harness (tools/parab_harness,
which links the oracle's ACM-850 `parab` at epss=1e-4); regenerate with:
    printf "%s\\n" "-3.0 32.0" | tools/parab_harness/dvtest   # -> order arg Dv(1e-4) Dv(1e-15)
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from clubb_jax.src.Microphys.KK_microphys.parabolic_expax import expax_U

# (order, arg, Dv_oracle@epss=1e-4) with arg>0, so Dv = U(-order-0.5, arg) = expax_U(...).
_REF = [
    (-3.0, 32.0, 2.007378491577508804e-116),
    (-2.5, 25.0, 4.401662473937484121e-072),
    (-3.5, 40.0, 4.708325153738463532e-180),
    (-2.0, 49.0, 8.586244034883043203e-265),
]


def test_expax_U_matches_oracle_epss_1e4():
    worst = 0.0
    for order, arg, dv in _REF:
        u = float(expax_U(-order - 0.5, arg))
        rel = abs((u - dv) / dv)
        worst = max(worst, rel)
        assert rel < 1e-12, f"expax_U(a={-order-0.5}, x={arg}) = {u:.6e} vs oracle {dv:.6e} (rel {rel:.1e})"
    print(f"  expax_U vs oracle epss=1e-4: worst rel {worst:.2e}  PASS")


def test_expax_U_vmaps_and_is_finite():
    a = jnp.array([1.5, 2.5, 3.5])
    x = jnp.array([20.0, 33.0, 47.0])
    out = jax.vmap(expax_U)(a, x)
    assert np.all(np.isfinite(np.asarray(out))), "expax_U not finite under vmap"
    # jit-able
    out_j = jax.jit(jax.vmap(expax_U))(a, x)
    assert np.allclose(np.asarray(out), np.asarray(out_j), rtol=0, atol=0), "jit changes expax_U"
    print("  expax_U vmap+jit finite + identical  PASS")


def test_expax_U_differentiable():
    g = jax.grad(lambda x: jnp.log(expax_U(2.5, x)))(33.0)  # log to keep the tiny value scaled
    assert np.isfinite(float(g)) and float(g) != 0.0, "expax_U gradient not finite/nonzero"
    print(f"  d/dx log U(2.5,x) finite+nonzero ({float(g):.3e})  PASS")


if __name__ == "__main__":
    test_expax_U_matches_oracle_epss_1e4()
    test_expax_U_vmaps_and_is_finite()
    test_expax_U_differentiable()
    print("ALL PASS")
