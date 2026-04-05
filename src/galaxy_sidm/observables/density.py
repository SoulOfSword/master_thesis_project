"""Density profile measurement from particle data."""

import numpy as np


def measure_density_profile(positions, masses, r_min, r_max, n_bins=40):
    """Measure a spherically averaged density profile from particles.

    Bins particles into logarithmically spaced spherical shells and computes
    the density as total mass / shell volume in each bin.

    Args:
        positions: Particle positions relative to halo centre, shape (N, 3),
            in kpc.
        masses: Particle masses, shape (N,), in Msun. For equal-mass particles
            (e.g. DM), pass an array of identical values.
        r_min: Inner edge of first radial bin in kpc.
        r_max: Outer edge of last radial bin in kpc.
        n_bins: Number of logarithmically spaced bins.

    Returns:
        Tuple of (r_mid, rho, r_edges) where:
            r_mid: Bin centres (geometric mean), shape (n_bins,), in kpc.
            rho: Density in each shell, shape (n_bins,), in Msun/kpc^3.
            r_edges: Bin edges, shape (n_bins+1,), in kpc.
    """
    positions = np.asarray(positions)
    masses = np.asarray(masses)

    radii = np.linalg.norm(positions, axis=1)

    r_edges = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    r_mid = np.sqrt(r_edges[:-1] * r_edges[1:])  # geometric mean

    rho = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (radii >= r_edges[i]) & (radii < r_edges[i + 1])
        shell_mass = np.sum(masses[mask])
        shell_vol = (4.0 / 3.0) * np.pi * (r_edges[i + 1]**3 - r_edges[i]**3)
        rho[i] = shell_mass / shell_vol

    return r_mid, rho, r_edges


def measure_inner_slope(r_mid, rho, r_inner, r_outer):
    """Measure the logarithmic slope of a density profile in a radial range.

    Fits a power law rho ~ r^gamma in log-log space using linear regression
    over the specified radial range.

    Args:
        r_mid: Bin centres in kpc, shape (n_bins,).
        rho: Density in each bin in Msun/kpc^3, shape (n_bins,).
        r_inner: Inner radius of the fitting range in kpc.
        r_outer: Outer radius of the fitting range in kpc.

    Returns:
        gamma (float): Logarithmic slope d(log rho)/d(log r), or NaN if
        fewer than 2 bins fall in the range or densities are zero.
    """
    mask = (r_mid >= r_inner) & (r_mid <= r_outer) & (rho > 0)
    if mask.sum() < 2:
        return np.nan

    log_r = np.log10(r_mid[mask])
    log_rho = np.log10(rho[mask])

    # Linear regression in log-log space
    coeffs = np.polyfit(log_r, log_rho, 1)
    return coeffs[0]  # slope = gamma
