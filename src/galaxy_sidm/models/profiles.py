"""Dark matter halo density profiles (JAX implementation)."""

import jax.numpy as jnp
from jax import jit


@jit
def nfw_density(r, rho_s, r_s):
    """NFW density profile.

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_s : float
        Scale density in Msun/kpc^3.
    r_s : float
        Scale radius in kpc.

    Returns
    -------
    array
        Density at each radius in Msun/kpc^3.
    """
    x = r / r_s
    return rho_s / (x * (1 + x) ** 2)


@jit
def nfw_mass(r, rho_s, r_s):
    """Enclosed mass for NFW profile.

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_s : float
        Scale density in Msun/kpc^3.
    r_s : float
        Scale radius in kpc.

    Returns
    -------
    array
        Enclosed mass at each radius in Msun.
    """
    x = r / r_s
    return 4 * jnp.pi * rho_s * r_s**3 * (jnp.log(1 + x) - x / (1 + x))


@jit
def nfw_circular_velocity(r, rho_s, r_s):
    """Circular velocity for NFW profile.

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_s : float
        Scale density in Msun/kpc^3.
    r_s : float
        Scale radius in kpc.

    Returns
    -------
    array
        Circular velocity in km/s.
    """
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    M_enc = nfw_mass(r, rho_s, r_s)
    return jnp.sqrt(G * M_enc / r)


@jit
def cored_nfw_density(r, rho_s, r_s, r_core):
    """Cored NFW profile (approximation for SIDM halos).

    Uses the Read et al. (2016) coreNFW parameterization where
    SIDM scattering creates a constant-density core.

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_s : float
        Scale density in Msun/kpc^3.
    r_s : float
        Scale radius in kpc.
    r_core : float
        Core radius in kpc.

    Returns
    -------
    array
        Density at each radius in Msun/kpc^3.
    """
    x = r / r_s
    f_core = jnp.tanh(r / r_core)
    return rho_s / ((r / r_core + f_core * x) * (1 + x) ** 2)


@jit
def einasto_density(r, rho_s, r_s, alpha=0.18):
    """Einasto density profile.

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_s : float
        Scale density in Msun/kpc^3.
    r_s : float
        Scale radius in kpc.
    alpha : float
        Shape parameter (typically 0.16-0.20 for CDM halos).

    Returns
    -------
    array
        Density at each radius in Msun/kpc^3.
    """
    x = r / r_s
    return rho_s * jnp.exp(-(2 / alpha) * (x**alpha - 1))


@jit
def isothermal_core_density(r, rho_0, r_core):
    """Pseudo-isothermal core profile (often used for SIDM cores).

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_0 : float
        Central density in Msun/kpc^3.
    r_core : float
        Core radius in kpc.

    Returns
    -------
    array
        Density at each radius in Msun/kpc^3.
    """
    return rho_0 / (1 + (r / r_core) ** 2)


@jit
def concentration_duffy08(M200, z=0.0):
    """Concentration-mass relation from Duffy et al. (2008).

    Parameters
    ----------
    M200 : float or array
        Halo mass in Msun.
    z : float
        Redshift.

    Returns
    -------
    float or array
        Concentration c200.
    """
    # Duffy+08 relaxed halos, 200c
    A, B, C = 5.71, -0.084, -0.47
    M_pivot = 2e12  # Msun/h (ignoring h for simplicity)
    return A * (M200 / M_pivot) ** B * (1 + z) ** C


@jit
def nfw_scale_density(M200, c200):
    """Compute NFW scale density from M200 and concentration.

    Parameters
    ----------
    M200 : float
        Virial mass in Msun.
    c200 : float
        Concentration.

    Returns
    -------
    float
        Scale density rho_s in Msun/kpc^3.
    """
    # R200 from M200 (assuming rho_crit ~ 127 Msun/kpc^3 at z=0)
    rho_crit = 127.0  # Msun/kpc^3
    R200 = (M200 / (4/3 * jnp.pi * 200 * rho_crit)) ** (1/3)
    r_s = R200 / c200

    # rho_s from mass normalization
    rho_s = M200 / (4 * jnp.pi * r_s**3 * (jnp.log(1 + c200) - c200 / (1 + c200)))
    return rho_s
