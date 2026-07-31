#!/usr/bin/env python3
"""obs_target.py — load an observational target for tune_coeffs.py from an
ARM VARANAL-style netCDF, mapped onto the model grid.

Expected obs netCDF (ARM VARANAL / Xie forcing conventions; names are matched
flexibly with fallbacks): a (time, lev) grid with
  lev / p / pressure / plev  : pressure levels (mb/hPa or Pa)
  time                       : seconds/hours/days since a reference
  T / temp / temperature     : temperature [K]         -> target thlm (via theta)
  q / qv / sphum / vap_mixing : water vapor [g/kg or kg/kg] -> target rtm
  u, v                       : winds [m/s]             -> target um, vm

If the file uses other names, pass a `varmap` dict. The loader picks the time
index nearest `time_target_s`, interpolates each field from the obs pressure
levels onto the model's `p_in_Pa` profile, and converts to CLUBB mean-state
targets. Turbulence/flux targets are only used if present (VARANAL has none).

`python obs_target.py --selftest` round-trips a synthetic VARANAL-like file.
"""
import numpy as np

P0 = 1.0e5          # reference pressure [Pa]
KAPPA = 287.04 / 1004.67   # Rd/cp

_CANDS = {
    "lev":  ["lev", "p", "pressure", "plev", "lev_p", "levels"],
    "time": ["time", "tsec", "bdate_time"],
    "T":    ["T", "temp", "temperature", "t"],
    "q":    ["q", "qv", "sphum", "vap_mixing_ratio", "shum", "wvmr"],
    "u":    ["u", "U", "u_wind", "uwind"],
    "v":    ["v", "V", "v_wind", "vwind"],
}

def _pick(ds, key, varmap):
    if varmap and key in varmap:
        return ds.variables[varmap[key]]
    for n in _CANDS[key]:
        if n in ds.variables:
            return ds.variables[n]
    raise KeyError(f"obs netCDF: no variable for {key!r} (tried {_CANDS[key]}); pass varmap")

def load_obs_target(obs_path, state, time_target_s, varmap=None):
    """Return {'thlm','rtm','um','vm'} arrays on the model thermo grid (nzt)."""
    from netCDF4 import Dataset
    ds = Dataset(obs_path)
    lev = np.asarray(_pick(ds, "lev", varmap)[:], float).squeeze()
    lev_pa = lev * 100.0 if np.nanmax(lev) < 2000.0 else lev          # mb -> Pa
    tvar = _pick(ds, "time", varmap)
    tsec = np.asarray(tvar[:], float).squeeze()
    # normalize time to seconds if it looks like hours/days
    units = getattr(tvar, "units", "").lower()
    if "day" in units:   tsec = tsec * 86400.0
    elif "hour" in units: tsec = tsec * 3600.0
    it = int(np.argmin(np.abs(tsec - (tsec[0] + time_target_s))))
    def prof(key):
        a = np.asarray(_pick(ds, key, varmap)[:], float)
        a = a[it] if a.ndim == 2 else a           # (time,lev)->(lev,)
        return np.asarray(a).squeeze()
    T, q, u, v = prof("T"), prof("q"), prof("u"), prof("v")
    ds.close()

    if np.nanmax(q) > 1.0:      # g/kg -> kg/kg
        q = q / 1000.0
    theta = T * (P0 / lev_pa) ** KAPPA          # potential temp [K]
    rt = q / (1.0 - q)                          # water vapor mixing ratio [kg/kg]

    # obs pressure decreases with height; np.interp needs ascending x, so flip.
    p_model = np.asarray(state["p_in_Pa"]).ravel()   # (nzt,) Pa, top->? ensure order
    order = np.argsort(lev_pa)
    def to_grid(f):
        return np.interp(p_model, lev_pa[order], f[order])
    return {"thlm": to_grid(theta)[None, :], "rtm": to_grid(rt)[None, :],
            "um": to_grid(u)[None, :], "vm": to_grid(v)[None, :]}

# --------------------------------------------------------------------------
def _write_synthetic_obs(path, state):
    """Write a synthetic VARANAL-like obs file FROM the model grid (for the
    round-trip self-test): T/q/u/v on (time, lev[mb]) built from thlm/rtm/um/vm."""
    from netCDF4 import Dataset
    p = np.asarray(state["p_in_Pa"]).ravel()
    exn = np.asarray(state["exner"]).ravel()
    thlm = np.asarray(state["thlm"]).ravel(); rtm = np.asarray(state["rtm"]).ravel()
    T = thlm * exn                                   # theta*exner = T
    q = rtm / (1.0 + rtm) * 1000.0                   # kg/kg -> g/kg specific-ish
    u = np.asarray(state["um"]).ravel(); v = np.asarray(state["vm"]).ravel()
    ds = Dataset(path, "w")
    ds.createDimension("time", 2); ds.createDimension("lev", len(p))
    ds.createVariable("lev", "f8", ("lev",))[:] = p / 100.0        # mb
    tv = ds.createVariable("time", "f8", ("time",)); tv.units = "seconds since start"; tv[:] = [0.0, 1e5]
    for nm, arr in (("T", T), ("q", q), ("u", u), ("v", v)):
        ds.createVariable(nm, "f8", ("time", "lev"))[:] = np.vstack([arr, arr])
    ds.close()

if __name__ == "__main__":
    import sys, os
    if "--selftest" in sys.argv:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        REPO = "/burg-archive/glab/users/ac5006/CLUBB-JAX"
        sys.path.insert(0, REPO)
        from clubb_jax.src.clubb_driver import init_clubb_case
        s = init_clubb_case(f"{REPO}/clubb_jax/output/arm_compare_jax/arm.in")
        tmp = "/tmp/obs_selftest.nc"; _write_synthetic_obs(tmp, s)
        tgt = load_obs_target(tmp, s, time_target_s=0.0)
        for k in ("thlm", "rtm", "um", "vm"):
            ref = np.asarray(s[k]).ravel(); got = tgt[k].ravel()
            rel = np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30)
            print(f"  {k:5s} round-trip max rel err = {rel:.2e}  {'OK' if rel < 1e-6 else 'CHECK'}")
