"""Density profile measurement from particle data."""

import numpy as np
import illustris_python as il

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


def collect_profiles(profiles, halo_ids, r_common, components,
                     sf_gas_cache=None):
    """Interpolate and sum density profile components onto a common radial grid.

    Args:
        profiles: Dict mapping halo ID to profile dict.
        halo_ids: Array of halo IDs to process.
        r_common: Common radial grid in physical kpc.
        components: List of profile keys to sum, e.g. ["prof_dm"],
            ["prof_dm", "prof_stars"], or ["prof_dm", "prof_stars", "prof_sfgas"].
            Use "prof_sfgas" to pull from sf_gas_cache instead of the profile dict.
        sf_gas_cache: Dict mapping halo ID to SF gas density array.
            Required when "prof_sfgas" is in components.

    Returns:
        Array of shape (N_halos, len(r_common)) with log10(rho), or None.
    """
    log_r_common = np.log10(r_common)
    all_profiles = []

    for hid in halo_ids:
        if hid not in profiles:
            continue
        p = profiles[hid]
        r_mid = p["r_mid"]

        rho_total = np.zeros(len(r_mid))
        skip = False
        for comp in components:
            if comp == "prof_sfgas":
                if sf_gas_cache is None or hid not in sf_gas_cache:
                    skip = True
                    break
                # SF gas is in Msun/kpc^3 (true density); catalog profiles
                # use mass/(r^3 diff) without the 4pi/3 prefactor, so we
                # multiply to match.
                rho_total += sf_gas_cache[hid] * (4.0 / 3.0 * np.pi)
            else:
                rho_k = p[comp]
                if rho_k is None:
                    skip = True
                    break
                rho_total += rho_k
        if skip:
            continue

        valid = rho_total > 0
        if valid.sum() < 2:
            continue

        log_rho_interp = np.interp(
            log_r_common, np.log10(r_mid[valid]), np.log10(rho_total[valid]),
            left=np.nan, right=np.nan,
        )
        all_profiles.append(log_rho_interp)

    if not all_profiles:
        return None
    return np.array(all_profiles)


def measure_sf_gas_profile(basePath, snap, halo_id, r_edges, h=0.6774):
    """Compute the density profile of star-forming gas for a single halo.

    Loads PartType0 particles, selects those with StarFormationRate > 0,
    and bins them into spherical shells defined by r_edges.

    Args:
        basePath: Path to the simulation output directory.
        snap: Snapshot number.
        halo_id: FoF group index.
        r_edges: Radial bin edges in physical kpc, shape (n_bins+1,).
        h: Hubble parameter for unit conversion.

    Returns:
        Density profile in each shell (Msun/kpc^3), shape (n_bins,).
        Returns zeros if no star-forming gas is found.
    """
    
    halo = il.groupcat.loadSingle(basePath, snap, haloID=halo_id)
    halo_pos = halo["GroupPos"]  # ckpc/h

    gas = il.snapshot.loadHalo(
        basePath, snap, halo_id, "gas",
        fields=["Coordinates", "Masses", "StarFormationRate"],
    )

    n_bins = len(r_edges) - 1
    if gas is None or (isinstance(gas, dict) and gas.get("count", 0) == 0):
        return np.zeros(n_bins)

    if isinstance(gas, dict):
        coords = gas["Coordinates"]
        masses = gas["Masses"]
        sfr = gas["StarFormationRate"]
    else:
        # Single field returns array directly
        return np.zeros(n_bins)

    # Filter star-forming gas
    sf_mask = sfr > 0
    if sf_mask.sum() == 0:
        return np.zeros(n_bins)

    pos = (coords[sf_mask] - halo_pos) / h  # physical kpc, centred
    mass = masses[sf_mask] * 1e10 / h  # Msun

    radii = np.linalg.norm(pos, axis=1)

    rho = np.zeros(n_bins)
    for i in range(n_bins):
        in_shell = (radii >= r_edges[i]) & (radii < r_edges[i + 1])
        shell_mass = np.sum(mass[in_shell])
        shell_vol = (4.0 / 3.0) * np.pi * (r_edges[i + 1]**3 - r_edges[i]**3)
        rho[i] = shell_mass / shell_vol

    return rho
