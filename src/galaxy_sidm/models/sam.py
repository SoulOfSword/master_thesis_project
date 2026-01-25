"""Semi-analytical model for galaxy evolution with SIDM (JAX implementation).

This module contains the core semi-analytical model (SAM) that
predicts galaxy properties based on halo properties and SIDM physics.
"""

import jax.numpy as jnp
from jax import jit, vmap
from dataclasses import dataclass

from .profiles import nfw_mass, concentration_duffy08, nfw_scale_density
from .sidm import effective_cross_section, core_formation_timescale


# Physical constants
G_CONST = 4.302e-6  # kpc (km/s)^2 / Msun
RHO_CRIT_Z0 = 127.0  # Critical density at z=0 in Msun/kpc^3


@dataclass
class SAMParameters:
    """Parameters for the semi-analytical model.

    These parameters control the baryonic physics prescriptions
    that connect halo properties to galaxy observables.
    """

    # Star formation efficiency (double power-law SHMR)
    f_star_0: float = 0.02       # Peak efficiency
    M_peak: float = 1e12         # Halo mass at peak efficiency (Msun)
    beta_low: float = 2.0        # Low-mass slope
    beta_high: float = 0.5       # High-mass slope

    # Size-mass relation
    f_R: float = 0.02            # R_half / R_vir ratio at M_peak
    alpha_R: float = 0.25        # Mass dependence exponent

    # Angular momentum retention
    f_j: float = 0.6             # j_star / j_halo ratio

    # SIDM effects
    sidm_core_boost: float = 1.0  # Boost to sizes in SIDM halos


@jit
def stellar_mass_halo_mass(M_halo, f_star_0, M_peak, beta_low, beta_high):
    """Stellar-to-halo mass relation (SHMR).

    Double power-law parameterization following Behroozi et al.

    Parameters
    ----------
    M_halo : array
        Halo mass M200 in Msun.
    f_star_0 : float
        Peak star formation efficiency.
    M_peak : float
        Halo mass at peak efficiency.
    beta_low : float
        Low-mass power-law slope.
    beta_high : float
        High-mass power-law slope.

    Returns
    -------
    array
        Stellar mass in Msun.
    """
    x = M_halo / M_peak
    f_star = f_star_0 * 2 / (x ** (-beta_low) + x ** beta_high)
    return f_star * M_halo


@jit
def virial_radius(M_halo, z=0.0):
    """Compute virial radius R200 from halo mass.

    Parameters
    ----------
    M_halo : array
        Halo mass M200 in Msun.
    z : float
        Redshift.

    Returns
    -------
    array
        Virial radius R200 in kpc.
    """
    # rho_crit(z) ~ rho_crit(0) * E(z)^2, simplified for z=0
    rho_crit = RHO_CRIT_Z0 * (1 + z) ** 3  # Approximate
    return (M_halo / (4/3 * jnp.pi * 200 * rho_crit)) ** (1/3)


@jit
def galaxy_size_cdm(M_halo, R_vir, f_R, alpha_R, M_pivot=1e12):
    """Predict galaxy half-mass radius for CDM halos.

    Parameters
    ----------
    M_halo : array
        Halo mass in Msun.
    R_vir : array
        Virial radius in kpc.
    f_R : float
        R_half / R_vir ratio at pivot mass.
    alpha_R : float
        Mass dependence exponent.
    M_pivot : float
        Pivot mass for scaling.

    Returns
    -------
    array
        Half-mass radius in kpc.
    """
    mass_factor = (M_halo / M_pivot) ** alpha_R
    return f_R * R_vir * mass_factor


@jit
def galaxy_size_sidm(M_halo, R_vir, f_R, alpha_R, r_core, M_pivot=1e12):
    """Predict galaxy half-mass radius for SIDM halos.

    SIDM halos have larger cores which can affect galaxy sizes.

    Parameters
    ----------
    M_halo : array
        Halo mass in Msun.
    R_vir : array
        Virial radius in kpc.
    f_R : float
        R_half / R_vir ratio at pivot mass.
    alpha_R : float
        Mass dependence exponent.
    r_core : array
        SIDM core radius in kpc.
    M_pivot : float
        Pivot mass for scaling.

    Returns
    -------
    array
        Half-mass radius in kpc.
    """
    R_half_cdm = galaxy_size_cdm(M_halo, R_vir, f_R, alpha_R, M_pivot)
    # Simple model: size is at least as large as core
    return jnp.maximum(R_half_cdm, r_core)


@jit
def rotation_velocity_nfw(M_halo, R_half, c200):
    """Rotation velocity at R_half for an NFW halo.

    Parameters
    ----------
    M_halo : array
        Halo mass M200 in Msun.
    R_half : array
        Galaxy half-mass radius in kpc.
    c200 : array
        Halo concentration.

    Returns
    -------
    array
        Rotation velocity in km/s.
    """
    # Get NFW parameters
    rho_s = nfw_scale_density(M_halo, c200)
    R_vir = virial_radius(M_halo)
    r_s = R_vir / c200

    # Enclosed mass at R_half
    M_enc = nfw_mass(R_half, rho_s, r_s)

    # V_circ = sqrt(G * M / r)
    return jnp.sqrt(G_CONST * M_enc / R_half)


@jit
def rotation_velocity_cored(M_halo, R_half, c200, r_core):
    """Rotation velocity at R_half for a cored halo.

    For SIDM halos, the central core reduces the enclosed mass
    and hence the rotation velocity at small radii.

    Parameters
    ----------
    M_halo : array
        Halo mass M200 in Msun.
    R_half : array
        Galaxy half-mass radius in kpc.
    c200 : array
        Halo concentration.
    r_core : array
        Core radius in kpc.

    Returns
    -------
    array
        Rotation velocity in km/s.
    """
    # For now, use a simple core correction factor
    # More sophisticated: integrate cored density profile
    V_nfw = rotation_velocity_nfw(M_halo, R_half, c200)

    # Core reduces V_rot when R_half < r_core
    core_factor = jnp.sqrt(1 - jnp.exp(-(R_half / r_core) ** 2))

    return V_nfw * core_factor


@jit
def specific_angular_momentum_halo(M_halo, lambda_spin=0.035):
    """Specific angular momentum of the halo.

    j_halo = sqrt(2) * lambda * V_vir * R_vir

    Parameters
    ----------
    M_halo : array
        Halo mass in Msun.
    lambda_spin : float
        Halo spin parameter (default 0.035).

    Returns
    -------
    array
        Specific angular momentum in kpc km/s.
    """
    R_vir = virial_radius(M_halo)
    V_vir = jnp.sqrt(G_CONST * M_halo / R_vir)

    return jnp.sqrt(2) * lambda_spin * V_vir * R_vir


@jit
def specific_angular_momentum_galaxy(M_halo, f_j, lambda_spin=0.035):
    """Specific angular momentum of the galaxy.

    j_star = f_j * j_halo

    Parameters
    ----------
    M_halo : array
        Halo mass in Msun.
    f_j : float
        Angular momentum retention fraction.
    lambda_spin : float
        Halo spin parameter.

    Returns
    -------
    array
        Specific angular momentum in kpc km/s.
    """
    j_halo = specific_angular_momentum_halo(M_halo, lambda_spin)
    return f_j * j_halo


def predict_galaxy_properties(M_halo, params, is_sidm=False, r_core=None):
    """Predict all galaxy observables from halo mass.

    This is the main SAM function that computes stellar mass,
    size, rotation velocity, and angular momentum.

    Parameters
    ----------
    M_halo : array
        Halo mass M200 in Msun.
    params : SAMParameters
        Model parameters.
    is_sidm : bool
        Whether to use SIDM corrections.
    r_core : array, optional
        Core radius for SIDM halos (required if is_sidm=True).

    Returns
    -------
    dict
        Dictionary with M_star, R_half, V_rot, j_star.
    """
    M_halo = jnp.atleast_1d(M_halo)

    # Stellar mass from SHMR
    M_star = stellar_mass_halo_mass(
        M_halo, params.f_star_0, params.M_peak,
        params.beta_low, params.beta_high
    )

    # Halo properties
    R_vir = virial_radius(M_halo)
    c200 = concentration_duffy08(M_halo)

    # Galaxy size
    if is_sidm and r_core is not None:
        R_half = galaxy_size_sidm(M_halo, R_vir, params.f_R, params.alpha_R, r_core)
        V_rot = rotation_velocity_cored(M_halo, R_half, c200, r_core)
    else:
        R_half = galaxy_size_cdm(M_halo, R_vir, params.f_R, params.alpha_R)
        V_rot = rotation_velocity_nfw(M_halo, R_half, c200)

    # Angular momentum
    j_star = specific_angular_momentum_galaxy(M_halo, params.f_j)

    return {
        "M_star": M_star,
        "R_half": R_half,
        "V_rot": V_rot,
        "j_star": j_star,
        "M_halo": M_halo,
        "R_vir": R_vir,
        "c200": c200,
    }
