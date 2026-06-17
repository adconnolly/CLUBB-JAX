"""Compatibility re-export for CLUBB_core err_info reason codes."""

from clubb_jax.src.CLUBB_core.err_info_codes import (
    ERR_NONE,
    ERR_XP2_XPYP_INVALID_C_UU,
    ERR_XP2_XPYP_MULTIPLE_LHS_REQUIRED,
    messages_for,
)

__all__ = [
    "ERR_NONE",
    "ERR_XP2_XPYP_INVALID_C_UU",
    "ERR_XP2_XPYP_MULTIPLE_LHS_REQUIRED",
    "messages_for",
]
