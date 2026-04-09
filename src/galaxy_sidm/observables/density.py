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
    return coeffs[0]


def compute_gamma_dm(catalogs, model_profiles, models, r_fit_min, min_ndm=1000):
    """Compute inner DM slope for all well-resolved halos across models.

    Follows Despali+26 Fig. 5: gamma_DM is measured via a power-law fit in
    log-log space over the radial range 1 kpc <= r <= max(0.03*R200c, 10 kpc).

    Args:
        catalogs: Dict of group catalogs per model. Each must have keys
            'N_dm', 'M200c', 'R200c'.
        model_profiles: Dict of pre-computed profile dicts per model, as
            returned by ``load_precomputed_profiles``.
        models: List of model names to process.
        r_fit_min: Minimum reliable radius in kpc (not used in current
            implementation — kept for API compatibility).
        min_ndm: Minimum number of DM particles.

    Returns:
        Dict per model with arrays: 'M200c', 'R200c', 'gamma_dm', 'halo_ids'.
    """
    results = {}
    for name in models:
        cat = catalogs[name]
        profs = model_profiles[name]

        sel = (cat["N_dm"] >= min_ndm) & (cat["M200c"] > 0)
        halo_ids = np.where(sel)[0]

        gamma = np.full(len(halo_ids), np.nan)
        for i, hid in enumerate(halo_ids):
            if hid not in profs:
                continue
            prof = profs[hid]
            r = prof["r_mid"]
            rho = prof["prof_dm"]
            if rho is None:
                continue
            r200 = cat["R200c"][hid]

            r_inner = 1.0  # kpc, following Despali+26
            r_outer = max(0.03*r200, 10.0)  # kpc

            if r_outer <= r_inner:
                continue

            gamma[i] = measure_inner_slope(r, rho, r_inner=r_inner, r_outer=r_outer)

        valid = ~np.isnan(gamma)
        results[name] = {
            "M200c": cat["M200c"][halo_ids][valid],
            "R200c": cat["R200c"][halo_ids][valid],
            "gamma_dm": gamma[valid],
            "halo_ids": halo_ids[valid],
        }
        print(f"{name}: {valid.sum()}/{len(halo_ids)} halos with valid gamma_DM")

    return results
