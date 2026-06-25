"""JAX floating-point precision control — analog of ``CLUBB_core/clubb_precision.F90``.

The Fortran ``clubb_precision`` module defines the working real kinds
(``core_rknd`` = double or single per the ``-precision`` build flag). JAX has no
kind parameters; it selects precision at PROCESS start via the global
``jax_enable_x64`` config, which must be set before the first JAX operation. This
module is the JAX counterpart: it centralizes that decision so every CLUBB-JAX
module enables the *same* precision (an inconsistent per-module setting would let
the last import win).

Controlled by the env var ``CLUBB_JAX_PRECISION`` (read once at import):

  * ``"double"`` (default) -> float64  (x64 ON). The bit-faithful,
    conservation-safe, differentiable path that the correctness gates validate.
  * ``"single"`` / ``"float32"`` -> float32 (x64 OFF). Faster on GPU (the V100S
    fp32 peak is ~2x its fp64 peak) and ~half the device memory, but LOWER
    fidelity: single precision is known to break the ~1e-5 conservation
    (cancellation) contracts and to diverge from the double-precision Fortran
    oracle. Intended for performance exploration / inference where reduced
    accuracy is acceptable, not for the faithfulness gate. (Note: the few spots
    that rely on complex128 — e.g. ``cubic_solve`` — drop to complex64 here.)

Usage: call :func:`configure_jax_precision` at module import, immediately after
``import jax`` and before any array is created. Idempotent.
"""
import os

import jax

# Single-precision spellings accepted in CLUBB_JAX_PRECISION.
_SINGLE_ALIASES = ("single", "float32", "f32", "32", "real4", "sp")

CLUBB_JAX_PRECISION = os.environ.get("CLUBB_JAX_PRECISION", "double").strip().lower()
USE_X64 = CLUBB_JAX_PRECISION not in _SINGLE_ALIASES


def configure_jax_precision() -> None:
    """Set ``jax_enable_x64`` per ``CLUBB_JAX_PRECISION``.

    Must run before the first JAX op. Idempotent — every call sets the same
    value, so repeated invocation across module imports is harmless.
    """
    jax.config.update("jax_enable_x64", USE_X64)
