"""Kinematic observables from simulation data (JAX implementation).

Note: Some functions here work with particle data from simulations.
For operations that are inherently non-JIT-able (like variable-size binning),
both JAX and NumPy versions are provided.
"""

import jax.numpy as jnp
from jax import jit, vmap
import numpy as np


# Constants
G_CONST = 4.302e-6  # kpc (km/s)^2 / Msun


@jit
def circular_velocity_from_mass(r, M_enc):
    """Compute circular velocity from enclosed mass.

    Parameters
    ----------
    r : array
        Radius in kpc.
    M_enc : array
        Enclosed mass at each radius in Msun.

    Returns
    -------
    array
        Circular velocity in km/s.
    """
    return jnp.sqrt(G_CONST * M_enc / r)


@jit
def velocity_dispersion_3d(velocities, masses=None):
    """Compute 3D velocity dispersion.

    Parameters
    ----------
    velocities : array, shape (N, 3)
        Particle velocities in km/s.
    masses : array, shape (N,), optional
        Particle masses for mass-weighted dispersion.

    Returns
    -------
    float
        3D velocity dispersion in km/s.
    """
    if masses is None:
        # Unweighted
        v_mean = jnp.mean(velocities, axis=0)
        v_rel = velocities - v_mean
        sigma_3d = jnp.sqrt(jnp.mean(jnp.sum(v_rel**2, axis=1)))
    else:
        # Mass-weighted
        total_mass = jnp.sum(masses)
        v_mean = jnp.sum(velocities * masses[:, None], axis=0) / total_mass
        v_rel = velocities - v_mean
        sigma_3d = jnp.sqrt(jnp.sum(masses * jnp.sum(v_rel**2, axis=1)) / total_mass)

    return sigma_3d


@jit
def velocity_dispersion_1d(velocities, masses=None):
    """Compute 1D velocity dispersion.

    Parameters
    ----------
    velocities : array, shape (N, 3)
        Particle velocities in km/s.
    masses : array, optional
        Particle masses for mass-weighted dispersion.

    Returns
    -------
    float
        1D velocity dispersion in km/s.
    """
    sigma_3d = velocity_dispersion_3d(velocities, masses)
    return sigma_3d / jnp.sqrt(3)


@jit
def specific_angular_momentum_vector(positions, velocities, masses):
    """Compute specific angular momentum vector.

    Parameters
    ----------
    positions : array, shape (N, 3)
        Particle positions in kpc.
    velocities : array, shape (N, 3)
        Particle velocities in km/s.
    masses : array, shape (N,)
        Particle masses in Msun.

    Returns
    -------
    array, shape (3,)
        Specific angular momentum vector in kpc km/s.
    """
    # L = sum(m * r x v)
    L_vec = jnp.sum(masses[:, None] * jnp.cross(positions, velocities), axis=0)
    total_mass = jnp.sum(masses)

    return L_vec / total_mass


@jit
def specific_angular_momentum(positions, velocities, masses):
    """Compute magnitude of specific angular momentum.

    Parameters
    ----------
    positions : array, shape (N, 3)
        Particle positions in kpc.
    velocities : array, shape (N, 3)
        Particle velocities in km/s.
    masses : array, shape (N,)
        Particle masses in Msun.

    Returns
    -------
    float
        Magnitude of specific angular momentum in kpc km/s.
    """
    j_vec = specific_angular_momentum_vector(positions, velocities, masses)
    return jnp.linalg.norm(j_vec)


@jit
def v_over_sigma(V_rot, sigma):
    """Compute V/sigma ratio (rotational support parameter).

    Parameters
    ----------
    V_rot : float
        Rotation velocity in km/s.
    sigma : float
        Velocity dispersion in km/s.

    Returns
    -------
    float
        V/sigma ratio.
    """
    return V_rot / sigma


def rotation_curve_from_particles(r_bins, positions, velocities, masses):
    """Compute rotation curve from particle data.

    This function uses NumPy for binning operations which are
    not easily JIT-able due to variable bin sizes.

    Parameters
    ----------
    r_bins : array
        Radial bin edges in kpc.
    positions : array, shape (N, 3)
        Particle positions in kpc.
    velocities : array, shape (N, 3)
        Particle velocities in km/s.
    masses : array, shape (N,)
        Particle masses in Msun.

    Returns
    -------
    r_mid : array
        Bin centers in kpc.
    V_circ : array
        Circular velocity at each radius in km/s.
    V_rot : array
        Mean tangential velocity at each radius in km/s.
    """
    # Convert to numpy for binning
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    masses = np.asarray(masses)

    # Compute radii
    radii = np.linalg.norm(positions, axis=1)

    n_bins = len(r_bins) - 1
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    V_circ = np.zeros(n_bins)
    V_rot = np.zeros(n_bins)

    # Compute enclosed mass and mean tangential velocity in each bin
    for i in range(n_bins):
        # Enclosed mass for circular velocity
        mask_enc = radii < r_bins[i + 1]
        M_enc = np.sum(masses[mask_enc])
        if r_mid[i] > 0:
            V_circ[i] = np.sqrt(G_CONST * M_enc / r_mid[i])

        # Tangential velocity in this bin
        mask_bin = (radii >= r_bins[i]) & (radii < r_bins[i + 1])
        if np.any(mask_bin):
            pos_bin = positions[mask_bin]
            vel_bin = velocities[mask_bin]
            mass_bin = masses[mask_bin]

            # Tangential velocity: v_tan = |r x v| / |r|
            r_cross_v = np.cross(pos_bin, vel_bin)
            v_tan = np.linalg.norm(r_cross_v, axis=1) / np.linalg.norm(pos_bin, axis=1)

            # Mass-weighted mean
            V_rot[i] = np.sum(mass_bin * v_tan) / np.sum(mass_bin)

    return r_mid, V_circ, V_rot


@jit
def half_mass_radius(radii, masses):
    """Compute half-mass radius.

    Parameters
    ----------
    radii : array
        Particle radii in kpc (sorted).
    masses : array
        Particle masses in Msun (same order as radii).

    Returns
    -------
    float
        Half-mass radius in kpc.
    """
    # Sort by radius
    sort_idx = jnp.argsort(radii)
    radii_sorted = radii[sort_idx]
    masses_sorted = masses[sort_idx]

    # Cumulative mass
    M_cumsum = jnp.cumsum(masses_sorted)
    M_total = M_cumsum[-1]

    # Find where cumulative mass exceeds half
    half_mass_idx = jnp.searchsorted(M_cumsum, 0.5 * M_total)

    return radii_sorted[half_mass_idx]


@jit
def lambda_R(positions, velocities, masses):
    """Compute the lambda_R parameter (ATLAS3D).

    lambda_R = sum(R * |V|) / sum(R * sqrt(V^2 + sigma^2))

    This is a proxy for the angular momentum content.

    Parameters
    ----------
    positions : array, shape (N, 3)
        Particle positions in kpc.
    velocities : array, shape (N, 3)
        Particle velocities in km/s.
    masses : array, shape (N,)
        Particle masses in Msun.

    Returns
    -------
    float
        lambda_R parameter (0 to 1).
    """
    # Project to 2D (assuming z is line of sight)
    R = jnp.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    V_los = velocities[:, 2]  # Line-of-sight velocity

    # Remove mean velocity
    V_mean = jnp.sum(masses * V_los) / jnp.sum(masses)
    V_los = V_los - V_mean

    # Velocity dispersion
    sigma2 = jnp.sum(masses * V_los**2) / jnp.sum(masses)

    # lambda_R
    numerator = jnp.sum(masses * R * jnp.abs(V_los))
    denominator = jnp.sum(masses * R * jnp.sqrt(V_los**2 + sigma2))

    return numerator / denominator
