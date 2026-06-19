"""Compatibility re-export for the CLUBB_core grid type."""

from clubb_jax.src.CLUBB_core.grid_class import (
    GRID_TYPE_EVEN,
    GRID_TYPE_STRETCHED_ZM,
    GRID_TYPE_STRETCHED_ZT,
    Grid,
    setup_grid,
)

__all__ = [
    "Grid",
    "setup_grid",
    "GRID_TYPE_EVEN",
    "GRID_TYPE_STRETCHED_ZT",
    "GRID_TYPE_STRETCHED_ZM",
]
