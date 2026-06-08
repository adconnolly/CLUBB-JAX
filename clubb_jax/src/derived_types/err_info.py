"""Python representation of err_info_type derived-type data."""

from typing import NamedTuple, Optional

from clubb_jax.src.derived_types.common import Array


class ErrInfo(NamedTuple):
    """JAX runtime err_info — the subset of the Fortran err_info_type fields the driver uses (Gunther
    short-name idiom; the same subset the f2py-oracle tests bridge)."""

    ngrdcol: int
    chunk_idx: int = 1
    mpi_rank: int = 0
    lat: Optional[Array] = None
    lon: Optional[Array] = None
    err_code: Optional[Array] = None
