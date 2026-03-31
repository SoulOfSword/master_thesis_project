"""Kinematic observables from simulation data."""

import numpy as np


# Constants
G_CONST = 4.302e-6  # kpc (km/s)^2 / Msun



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
    return np.sqrt(G_CONST * M_enc / r)



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
        v_mean = np.mean(velocities, axis=0)
        v_rel = velocities - v_mean
        sigma_3d = np.sqrt(np.mean(np.sum(v_rel**2, axis=1)))
    else:
        # Mass-weighted
        total_mass = np.sum(masses)
        v_mean = np.sum(velocities * masses[:, None], axis=0) / total_mass
        v_rel = velocities - v_mean
        sigma_3d = np.sqrt(np.sum(masses * np.sum(v_rel**2, axis=1)) / total_mass)

    return sigma_3d



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
    return sigma_3d / np.sqrt(3)



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
    L_vec = np.sum(masses[:, None] * np.cross(positions, velocities), axis=0)
    total_mass = np.sum(masses)

    return L_vec / total_mass



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
    return np.linalg.norm(j_vec)



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
    sort_idx = np.argsort(radii)
    radii_sorted = radii[sort_idx]
    masses_sorted = masses[sort_idx]

    # Cumulative mass
    M_cumsum = np.cumsum(masses_sorted)
    M_total = M_cumsum[-1]

    # Find where cumulative mass exceeds half
    half_mass_idx = np.searchsorted(M_cumsum, 0.5 * M_total)

    return radii_sorted[half_mass_idx]



def lambda_R(positions, velocities, masses, los=None):
    """Compute the lambda_R parameter (ATLAS3D).

    lambda_R = sum(m * R * |V_los|) / sum(m * R * sqrt(V_los^2 + sigma_los^2))

    Args:
        positions: Particle positions in kpc, shape (N, 3).
        velocities: Particle velocities in km/s, shape (N, 3).
        masses: Particle masses in Msun, shape (N,).
        los: Line-of-sight unit vector, shape (3,). If None, uses [0, 0, 1]
            (z-axis).

    Returns:
        lambda_R parameter (float, 0 to 1).
    """
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    masses = np.asarray(masses)

    if los is None:
        los = np.array([0.0, 0.0, 1.0])
    los = np.asarray(los, dtype=np.float64)
    los = los / np.linalg.norm(los)

    # Project positions perpendicular to LOS -> cylindrical radius
    pos_along_los = np.outer(positions @ los, los)
    R = np.linalg.norm(positions - pos_along_los, axis=1)

    # LOS velocity
    V_los = velocities @ los
    V_mean = np.sum(masses * V_los) / np.sum(masses)
    V_los = V_los - V_mean

    sigma2 = np.sum(masses * V_los**2) / np.sum(masses)

    numerator = np.sum(masses * R * np.abs(V_los))
    denominator = np.sum(masses * R * np.sqrt(V_los**2 + sigma2))

    if denominator == 0:
        return np.nan
    return numerator / denominator


def compute_circularity(positions, velocities, masses, potential, r_half,
                        n_bins=50):
    """Compute orbital circularity for each star particle.

    The circularity parameter epsilon = j_z / j_circ(E) measures how circular
    each particle's orbit is:
        - epsilon ~ +1: co-rotating circular orbit (disc)
        - epsilon ~  0: radial orbit (bulge/halo)
        - epsilon ~ -1: counter-rotating circular orbit

    The rotation axis is defined from the mass-weighted angular momentum of
    stars within r_half. j_circ(E) is estimated as the maximum |j_z| in
    equal-count energy bins.

    Uses NumPy (not JIT-able) due to variable-size energy binning.

    Args:
        positions: Particle positions relative to galaxy centre, shape (N, 3),
            in kpc.
        velocities: Particle velocities in the galaxy rest frame, shape (N, 3),
            in km/s.
        masses: Particle masses, shape (N,), in Msun.
        potential: Gravitational potential per particle, shape (N,), in
            (km/s)^2.
        r_half: Stellar half-mass radius in kpc, used to select inner particles
            for defining the rotation axis.
        n_bins: Number of equal-count energy bins for j_circ estimation.

    Returns:
        Circularity array epsilon, shape (N,).
    """
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    masses = np.asarray(masses)
    potential = np.asarray(potential)

    r = np.linalg.norm(positions, axis=1)

    # Rotation axis from mass-weighted angular momentum of stars within r_half
    inner = r < r_half
    if inner.sum() < 10:
        return np.full(len(masses), np.nan)

    j_inner = np.cross(positions[inner], velocities[inner])
    L = np.sum(j_inner * masses[inner, None], axis=0)
    L_norm = np.linalg.norm(L)
    if L_norm == 0:
        return np.full(len(masses), np.nan)
    z_hat = L / L_norm

    # j_z for all particles
    j_all = np.cross(positions, velocities)
    j_z = j_all @ z_hat

    # Specific binding energy
    E = 0.5 * np.sum(velocities**2, axis=1) + potential

    # j_circ(E): maximum |j_z| in equal-count energy bins
    sort_idx = np.argsort(E)
    jz_sorted = j_z[sort_idx]

    bins = np.array_split(np.arange(len(E)), n_bins)
    j_circ_sorted = np.zeros(len(E))
    for b in bins:
        if len(b) == 0:
            continue
        jc = np.max(np.abs(jz_sorted[b]))
        j_circ_sorted[b] = jc if jc > 0 else 1.0

    j_circ = np.zeros(len(E))
    j_circ[sort_idx] = j_circ_sorted

    return j_z / j_circ


def disc_fraction(circularity, masses, threshold=0.7):
    """Compute the disc-to-total mass ratio from circularity values.

    Args:
        circularity: Circularity array epsilon = j_z / j_circ, shape (N,).
        masses: Particle masses, shape (N,), in Msun.
        threshold: Circularity threshold for disc classification. Particles
            with epsilon > threshold are counted as disc. Default 0.7
            (standard in TNG literature, e.g. Pillepich+2019).

    Returns:
        D/T ratio (float between 0 and 1), or NaN if input contains NaN.
    """
    circularity = np.asarray(circularity)
    masses = np.asarray(masses)

    if np.any(np.isnan(circularity)):
        return np.nan

    return float(np.sum(masses[circularity > threshold]) / np.sum(masses))
