# run_scripts/debug/ — per-term f2py bit-comparison harnesses

These scripts feed **captured JAX inputs** into a single Fortran CLUBB_core routine (via the `clubb_f2py`
`.so`, bypassing the out-of-sync Python wrapper) and diff the outputs **bit-for-bit**. They were the
decisive localisers during the bit-faithful port (e.g. the stretched-grid `weights_zm2zt` column-swap).

They are **debugging tools, not part of the regression gate** (REFACTOR.md §3.3 C4). Under the
numerical-accuracy standard the gate is `compare_cases.py` / `validate_case.py` (Tier A/B/C); reach for
these only when a *real* divergence needs to be pinned to one routine.

- `cmp_terms_f2py.py` — per-LHS-term diff (`term_ma_zm_lhs`, `diffusion_zm_lhs`, `xpyp_term_ta_pdf_lhs`).
- `cmp_mfl_f2py.py` — monotonic-flux-limiter input-matched diff.
- `compare_xm_wpxp_f2py.py` — full `advance_xm_wpxp` input-matched diff (needs `XMWP_CAP=1` capture).

They require the compiled `clubb_release/clubb_python_api/*.so` and use a hardcoded API path; adjust the
`API = Path(...)` line if your checkout differs.
