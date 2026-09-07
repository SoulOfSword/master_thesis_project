"""Cosmological calculations and utilities."""

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from .config import load_config, get_cosmology_params


def get_cosmology(config):
    """Create an astropy cosmology object from config parameters.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary. If None, loads default config.

    Returns
    -------
    FlatLambdaCDM
        Astropy cosmology object.
    """
    params = get_cosmology_params(config)

    return FlatLambdaCDM(
        H0=params.get("H0", 67.74),
        Om0=params.get("Omega_m", 0.3089),
        Ob0=params.get("Omega_b", 0.0486),
    )


def redshift_to_lookback_time(z, cosmo = None):
    """Convert redshift to lookback time in Gyr."""
    if cosmo is None:
        cosmo = get_cosmology()
    return cosmo.lookback_time(z).value


def critical_density(z = 0, cosmo = None):
    """Critical density at redshift z in Msun/kpc^3."""
    if cosmo is None:
        cosmo = get_cosmology()
    # astropy returns in g/cm^3, convert to Msun/kpc^3
    rho_crit = cosmo.critical_density(z).to("Msun/kpc^3").value
    return rho_crit


def delta_c_bryan_norman(z, cosmo):
    """Virial overdensity Delta_c(z) w.r.t. the CRITICAL density.

    Bryan & Norman (1998) fitting formula for a flat LambdaCDM universe:

        Delta_c = 18*pi^2 + 82*x - 39*x^2 ,   x = Omega_m(z) - 1

    i.e. Delta_c -> 18*pi^2 ~ 178 in the matter-dominated (high-z) limit and
    ~ 100 today. Used to relate virial mass, radius and velocity across
    redshift (M_vir proportional to V_vir^3 * Delta_c^-1/2 * H^-1).

    Args:
        z: redshift (scalar or array).
        cosmo: astropy FlatLambdaCDM (from `get_cosmology(config)`).

    Returns:
        Delta_c(z), same shape as z.
    """
    x = cosmo.Om(z) - 1.0
    return 18.0 * np.pi ** 2 + 82.0 * x - 39.0 * x ** 2
