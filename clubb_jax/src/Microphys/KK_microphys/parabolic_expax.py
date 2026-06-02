"""Faithful JAX port of the oracle's `expax` parabolic-cylinder routine (Parabolic.f90:3259).

WHY THIS EXISTS
---------------
The KK upscaled-covariance integrals call `Dv_fnc` (KK_utilities.F90), which computes the
parabolic cylinder function `D_v` via ACM-850 `parab`. In an SCM run the oracle uses
`epss = 1.0e-4` (the module default `l_high_accuracy_parab_cyl_fnc=.false.`, Parabolic.f90:20).
The existing `parabolic_cylinder.dv_parabolic_cylinder` is a DLMF re-derivation that is
bit-faithful to the *true* D_v (≈ epss=1e-15) — i.e. MORE accurate than the run, which makes
the `dycoms2_rf02_do/ds` covar source differ from the oracle by ~1e-6 (amplified ~16x by the
covar near-cancellation → the 1e-5–1e-4 failures; proven Iter310). Per the project mandate
("be faithful to the original, do not improvise") the faithful choice is to reproduce `parab`
at epss=1e-4 — and for the do/ds regime that is EXACTLY `expax`, called directly with no
recursion (traced Iter316: a∈[1.5,4], x∈[31,37] hits the `x>=30`, `abs(a)<150` → `expax` branch).

`expax` is the Poincaré-type asymptotic expansion for U(a,x), V(a,x). For mode=0 (what Dv_fnc
uses) it is a self-contained convergent loop summing per-term recurrences, truncated when the
relative term `err < epss`. For the do/ds argument range it converges in 2-3 terms.

For `Dv_fnc(order, arg)` with arg > 0 (the chi<0 half — the ONLY branch the epss truncation
shifts; arg<0 is dominated by the growing V term and is already bit-exact), the return value is
`U(a,x)` with `a = -order-0.5`, `x = arg`. `expax_U` below computes exactly that.

VALIDATION (Iter316): bit-exact (rel 0.0) vs the standalone `parab` harness epss=1e-4 column
(tools/parab_harness) across the do/ds (order,arg) range; the masked fixed-iteration loop
reproduces the Fortran's `DO WHILE (err>eps)` truncation. jit/vmap/grad-compatible (no
data-dependent control flow). See tests/test_parabolic_expax.py.
"""
import jax
import jax.numpy as jnp

# Parabolic_constants.f90: dwarf = TINY*1000, giant = HUGE/1000, over = giant*1e-5.
_DWARF = jnp.finfo(jnp.float64).tiny * 1000.0
# expax converges in <=3 terms for the do/ds range (x~31-37, eps=1e-4); 64 is a safe ceiling.
_MAXIT = 64


def expax_U(a, x, eps=1.0e-4):
    """U(a,x) via `expax` (mode=0), faithful to Parabolic.f90:3259 at tolerance `eps`.

    a, x: scalars or broadcastable arrays (x > 0). Returns U(a,x) = facto1 * y1, where the
    asymptotic series y1 is truncated when ALL FOUR component errors (U, V, U', V') fall below
    `eps` — exactly the oracle's coupled stopping rule (the loop advances all four together).
    """
    a = jnp.asarray(a, jnp.float64)
    x = jnp.asarray(x, jnp.float64)
    aph = a + 0.5
    amh = a - 0.5
    x2 = x * x
    sqrtx = jnp.sqrt(x)
    phiax = jnp.exp(0.25 * x * x + a * jnp.log(x))   # mode=0
    facto1 = 1.0 / (sqrtx * phiax)

    def body(_i, st):
        y1, y2, y1p, y2p, a2, b2, x2k, k, done = st
        l = 2.0 * k
        aphl = aph + l
        amhl = amh - l
        a2p = -amhl * (amh + l - 1.0) * a2
        a2n = -(aphl - 2.0) * (aphl - 1.0) * a2
        b2n = (amhl + 2.0) * (amhl + 1.0) * b2
        b2p = jnp.where(jnp.abs(aph - l) < _DWARF, 0.0, aphl / (aph - l) * b2n)
        x2kn = l * x2 * x2k
        acof = a2n / x2kn
        bcof = b2n / x2kn
        acofd = a2p / x2kn
        bcofd = b2p / x2kn
        ny1, ny2 = y1 + acof, y2 + bcof
        ny1p, ny2p = y1p + acofd, y2p + bcofd
        e1 = jnp.abs(acof / ny1)
        e2 = jnp.abs(bcof / ny2)
        e1p = jnp.abs(acofd / ny1p)
        e2p = jnp.abs(bcofd / ny2p)
        ndone = (e1 < eps) & (e2 < eps) & (e1p < eps) & (e2p < eps)
        # Freeze the whole state once converged — matches the Fortran loop terminating, and
        # prevents the (divergent) asymptotic recurrence from overflowing past convergence.
        sel = lambda old, new: jnp.where(done, old, new)
        return (sel(y1, ny1), sel(y2, ny2), sel(y1p, ny1p), sel(y2p, ny2p),
                sel(a2, a2n), sel(b2, b2n), sel(x2k, x2kn), sel(k, k + 1.0), done | ndone)

    one = jnp.ones_like(a + x)
    st0 = (one, one, one, one, one, one, one, one, jnp.zeros_like(a + x, dtype=bool))
    st = jax.lax.fori_loop(0, _MAXIT, body, st0)
    return facto1 * st[0]
