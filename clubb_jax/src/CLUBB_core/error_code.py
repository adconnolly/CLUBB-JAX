"""JAX-side mirror of `src/CLUBB_core/error_code.F90`."""

CLUBB_NO_ERROR = 0
CLUBB_GENERALIZED_GRD_TEST_ERR = 50
CLUBB_FATAL_ERROR = 99

_debug_level = 0


def set_debug_level(level: int):
    """Set the JAX-side CLUBB debug verbosity level."""
    global _debug_level
    _debug_level = max(int(level), 0)


def clubb_at_least_debug_level(level: int) -> bool:
    """Return whether CLUBB debug verbosity is at least `level`."""
    return int(level) <= _debug_level
