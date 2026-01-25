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
