"""JAX port of pos_definite_module.F90 — Smolarkiewicz (1989) positive-definite flux limiter.

Mirrors `clubb_release/src/CLUBB_core/pos_definite_adj`: a flux-conservative positive-definite renormalization
of the advective flux (Smolarkiewicz 1989, MWR 117, 2626), applied to `rtm` (gated by `l_pos_def`, OFF in the
default config — the gated suite uses `mono_flux_limiter`). This is a completeness port, bit-validated against
the f2py `f2py_pos_definite_adj` oracle (`tests/test_pos_definite.py`).

Vectorized for the **ascending grid** (`grid_dir_indx == 1`, every gated case + the f2py setup); a descending
grid raises (the mirrored index branches are untested without a descending oracle). Pure jnp → `jax.grad`-able
(the max/min/where are subgradient-differentiable; the divide is floored at `eps`).
"""
import jax.numpy as jnp

from clubb_jax.src.CLUBB_core.constants_clubb import zero_threshold, eps
from clubb_jax.src.CLUBB_core.grid_class import ddzm_jax


def pos_definite_adj_jax(gr, dt, field_np1, flux_np1, field_n):
    """Positive-definite flux adjustment (Smolarkiewicz 1989), flux on zm / field on zt.

    Args:
      gr:        grid (uses .invrs_dzt, .grid_dir; ascending only).
      dt:        timestep [s] (scalar).
      field_np1: post-solve field, (ngrdcol, nzt).
      flux_np1:  post-solve flux, (ngrdcol, nzm).
      field_n:   field at the start of the step, (ngrdcol, nzt).
    Returns:
      (field_np1_adj, flux_lim, field_pd, flux_pd) — the renormalized field (zt) + flux (zm) and the two
      diagnostic tendencies (zt / zm), matching the Fortran out-args.
    """
    if int(getattr(gr, "grid_dir_indx", 1)) != 1:
        raise NotImplementedError("pos_definite_adj_jax: only the ascending grid (grid_dir_indx==1) is ported.")
    field_np1 = jnp.asarray(field_np1); flux_np1 = jnp.asarray(flux_np1); field_n = jnp.asarray(field_n)
    dt = jnp.asarray(dt)
    ngrdcol, nzm = flux_np1.shape
    nzt = nzm - 1

    # Smolarkiewicz F+ / F- on the flux (zm) levels (eqn 2).
    flux_plus = jnp.maximum(zero_threshold, flux_np1)
    flux_minus = -jnp.minimum(zero_threshold, flux_np1)

    # dz/dt on the field (zt) levels (grid_dir=+1 ascending).
    dz_over_dt = (1.0 / (gr.grid_dir * gr.invrs_dzt)) / dt

    # Total outward flux per zt level k (eqn A4): F+ at the k+1/2 (=k+1) flux level + F- at the k-1/2 (=k)
    # flux level, floored at eps. zt level j -> flux indices (j+1) above, j below.
    fout = jnp.maximum(flux_plus[:, 1:nzm] + flux_minus[:, 0:nzt], eps)   # (ngrdcol, nzt)

    # Limited flux at the interior flux levels m=1..nzm-2 (eqn 10), vectorized:
    #   flux_lim[m] = max( min( flux_np1[m], (F+[m]/fout[m-1])*field_n[m-1]*dz[m-1] ),
    #                      -( (F-[m]/fout[m])  *field_n[m]  *dz[m-1] ) )
    upper = (flux_plus[:, 1:nzm - 1] / fout[:, 0:nzt - 1]) * field_n[:, 0:nzt - 1] * dz_over_dt[:, 0:nzt - 1]
    lower = -(flux_minus[:, 1:nzm - 1] / fout[:, 1:nzt]) * field_n[:, 1:nzt] * dz_over_dt[:, 0:nzt - 1]
    interior = jnp.maximum(jnp.minimum(flux_np1[:, 1:nzm - 1], upper), lower)
    # Boundary conditions: flux_lim[0]=flux_np1[0], flux_lim[nzm-1]=flux_np1[nzm-1].
    flux_lim = jnp.concatenate([flux_np1[:, 0:1], interior, flux_np1[:, nzm - 1:nzm]], axis=1)

    # Diagnostic flux tendency — only where the column had a below-zero field (input field_np1).
    neg_in = jnp.any(field_np1 < 0.0, axis=1, keepdims=True)
    flux_pd = jnp.where(neg_in, (flux_lim - flux_np1) / dt, 0.0)

    # Apply the flux correction to the field: field += -dt * ddzm(flux_lim - flux_np1).
    field_nonlim = field_np1
    field_np1_adj = -dt * ddzm_jax(flux_lim - flux_np1, gr) + field_np1

    # Diagnostic field tendency — only where the UPDATED field went below zero.
    neg_out = jnp.any(field_np1_adj < 0.0, axis=1, keepdims=True)
    field_pd = jnp.where(neg_out, (field_np1_adj - field_nonlim) / dt, 0.0)

    return field_np1_adj, flux_lim, field_pd, flux_pd
