"""Combine the optical properties of clouds, aerosols, Rayleigh, gas continuum and gas absorption
into the clear-sky and all-sky band optical depth / SSA / asymmetry — port of comscp1.F + comscp2.F.

comscp1 combines the SCATTERING components (Rayleigh + aerosol + ice/water cloud) into the
clear/all-sky optical depth, the scattering-weighted SSA numerators `fwclr/fwcld`, and the
scattering-weighted asymmetry. comscp2 then adds the gas absorption optical depth `tg` and forms
the final single-scattering albedos (capped at 0.999999).
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def comscp1(taer, tcldi, tcldw, tgm, tray, waer, wcldi, wcldw, wray, asyaer, asycldi, asycldw):
    """Combine scattering optical properties (comscp1.F). All inputs (ncol, nlm). Returns
    (tccld1, tcclr1, asycld, asyclr, fwcld, fwclr)."""
    a = lambda x: jnp.asarray(x, dtype=jnp.float64)
    taer, tcldi, tcldw, tgm, tray = a(taer), a(tcldi), a(tcldw), a(tgm), a(tray)
    waer, wcldi, wcldw, wray = a(waer), a(wcldi), a(wcldw), a(wray)
    asyaer, asycldi, asycldw = a(asyaer), a(asycldi), a(asycldw)
    tcclr1 = tgm + tray + taer
    tccld1 = tcclr1 + tcldi + tcldw
    wwray = wray * tray; wwaer = waer * taer
    wwcldi = wcldi * tcldi; wwcldw = wcldw * tcldw
    fwclr = wwray + wwaer
    fwcld = fwclr + wwcldi + wwcldw
    asyclr = jnp.where(fwclr > 1.0e-10,
                       (asyaer * wwaer) / jnp.where(fwclr > 1.0e-10, fwclr, 1.0), 1.0)
    asycld = jnp.where(fwcld > 1.0e-10,
                       (asyaer * wwaer + asycldi * wwcldi + asycldw * wwcldw)
                       / jnp.where(fwcld > 1.0e-10, fwcld, 1.0), 1.0)
    return tccld1, tcclr1, asycld, asyclr, fwcld, fwclr


def comscp2(tg, fwcld, fwclr, tccld1, tcclr1):
    """Add gas absorption `tg` and form the final SSAs (comscp2.F). Returns (tccld, tcclr, wccld,
    wcclr); SSAs capped at 0.999999."""
    a = lambda x: jnp.asarray(x, dtype=jnp.float64)
    tg, fwcld, fwclr, tccld1, tcclr1 = a(tg), a(fwcld), a(fwclr), a(tccld1), a(tcclr1)
    tcclr = tcclr1 + tg
    tccld = tccld1 + tg
    wcclr = jnp.where(tcclr > 0.0, fwclr / jnp.where(tcclr > 0.0, tcclr, 1.0), 0.0)
    wcclr = jnp.minimum(0.999999, wcclr)
    wccld = jnp.where(tccld > 0.0, fwcld / jnp.where(tccld > 0.0, tccld, 1.0), 0.0)
    wccld = jnp.minimum(0.999999, wccld)
    return tccld, tcclr, wccld, wcclr
