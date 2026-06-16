"""JAX-side mirrors of `src/CLUBB_core` modules.

Porting deviations:
- This package marker has no Fortran counterpart. It exists only so Python can
  import the per-module JAX mirrors under `clubb_jax.src.CLUBB_core`.
"""
