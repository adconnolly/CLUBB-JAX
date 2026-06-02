"""Guard the "entirely in JAX" property (Iter279): the JAX driver must import + run the faithful
cases with NO `clubb_python` (f90-wrapped Fortran) dependency.

We install a sys.meta_path import-blocker on `clubb_python` BEFORE importing the JAX driver, then
init + step a couple of faithful cases. If any module-level (or faithful-path) `clubb_python` import
sneaks back in, the import/run fails here. This generalizes testing beyond the Fortran-comparison
harness to "the JAX is genuinely standalone."

Run: python clubb_jax/tests/test_standalone_jax.py   (needs clubb_release/ as a sibling of clubb_jax/)
"""
import gc
import importlib.abc
import os
import sys
import jax
import numpy as np


class _ClubbPythonBlocker(importlib.abc.MetaPathFinder):
    """A sys.meta_path finder (modern find_spec protocol, Py3.4+) that makes any `import
    clubb_python[...]` raise ImportError, proving the JAX driver doesn't touch the Fortran package."""
    def find_spec(self, name, path=None, target=None):
        if name == "clubb_python" or name.startswith("clubb_python."):
            raise ImportError(f"clubb_python is BLOCKED (proving entirely-in-JAX): {name}")
        return None


def _repo_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    jax_root = os.path.abspath(os.path.join(here, "..", ".."))          # contains clubb_jax/
    clubb_release = os.path.abspath(os.path.join(jax_root, "clubb_release"))
    return jax_root, clubb_release


def test_faithful_cases_run_without_clubb_python():
    """Multiple faithful cases (different grids) init + step 3× IN ONE PROCESS with clubb_python blocked
    → finite prognostics. Proves both (a) the JAX driver is standalone (no Fortran import), and (b) it is
    REENTRANT — init_clubb_case resets the cross-timestep core state (`_prev_adg1_j25`, Iter281), so case 2
    doesn't inherit case 1's grid shape (which previously caused a broadcast error)."""
    sys.meta_path.insert(0, _ClubbPythonBlocker())
    # also evict any already-imported clubb_python so the block is real
    for m in [k for k in sys.modules if k == "clubb_python" or k.startswith("clubb_python.")]:
        del sys.modules[m]
    jax_root, clubb_release = _repo_paths()
    # jax_root MUST win over clubb_release on sys.path: the clubb_release checkout contains an
    # unrelated `clubb_jax/` scaffold (different naming convention) that would shadow our package and
    # make `clubb_jax.src` unresolvable. Insert clubb_release first, then jax_root, so jax_root is at [0].
    for p in (clubb_release, jax_root):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    from clubb_jax.src.clubb_standalone import init_clubb_case
    from clubb_jax.src.advance_clubb_to_end import advance_clubb_to_end

    # These cases are genuinely entirely-in-JAX (forcings ported) — they must run with clubb_python blocked.
    # After Iter289 (rico tndcy ported) NO case uses the Fortran `prescribe_forcings` fallback — every
    # case's forcings run entirely in JAX. (rico exercises the KK-microphysics path too, also pure JAX.)
    for case in ("arm", "jun25_altocu", "bomex", "neutral", "gabls2", "wangara", "dycoms2_rf01",
                 "dycoms2_rf01_fixed_sst", "dycoms2_rf02_nd", "dycoms2_rf02_so", "atex", "atex_long", "rico",
                 "mpace_a"):
        nml = os.path.join(clubb_release, "input", "case_setups", f"{case}_model.in")
        state = init_clubb_case(nml)
        advance_clubb_to_end(state, l_stdout=False, max_steps=3)
        thlm = np.asarray(state["thlm"])
        assert np.all(np.isfinite(thlm)), f"{case}: non-finite thlm with clubb_python blocked"
        assert thlm.shape[1] == state["nzt"], f"{case}: wrong thlm shape"
        print(f"  {case}: ran 3 steps with clubb_python BLOCKED, thlm finite (nzt={state['nzt']})  PASS")
        # 13 cases (varied grids; rico compiles the memory-heavy KK path) in one process accumulate
        # jit caches → OOM. Drop them between cases so the run stays bounded.
        del state
        jax.clear_caches()
        gc.collect()

    # confirm the blocker is actually active (would-be import raises)
    blocked = False
    try:
        import clubb_python  # noqa: F401
    except ImportError:
        blocked = True
    assert blocked, "clubb_python import was NOT blocked — test is not proving standalone"
    print("  clubb_python import correctly blocked — JAX driver is standalone  PASS")


if __name__ == "__main__":
    print("Standalone (entirely-in-JAX) validation:")
    test_faithful_cases_run_without_clubb_python()
    print("All standalone tests PASSED.")
