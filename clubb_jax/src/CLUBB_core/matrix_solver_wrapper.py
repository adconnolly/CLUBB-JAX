"""Matrix-solver wrapper — JAX mirror of matrix_solver_wrapper.F90.

The Fortran matrix_solver_wrapper.F90 is a thin dispatch layer that `use`s the concrete
LU solvers and exposes the generic `tridiag_solve` / `band_solve` interfaces to the rest of
CLUBB. Mirroring that, the JAX solver implementations live in their own Fortran-named modules:
  tridiag_lu_solve_jax / tridiag_lu_solve → CLUBB_core/tridiag_lu_solver.py ↔ tridiag_lu_solver.F90
  penta_lu_solve_jax   / penta_lu_solve   → CLUBB_core/penta_lu_solver.py   ↔ penta_lu_solver.F90

They are re-exported here so callers can keep importing the solver entry points from the wrapper,
exactly as Fortran code calls them through `use matrix_solver_wrapper`.
"""

from __future__ import annotations

from clubb_jax.src.CLUBB_core.tridiag_lu_solver import (
    tridiag_lu_solve_jax,
    tridiag_lu_solve,
)
from clubb_jax.src.CLUBB_core.penta_lu_solver import (
    penta_lu_solve_jax,
    penta_lu_solve,
)


__all__ = [
    "tridiag_lu_solve_jax", "tridiag_lu_solve",
    "penta_lu_solve_jax", "penta_lu_solve",
]
