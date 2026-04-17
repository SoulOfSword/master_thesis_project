"""Density profile measurement from particle data."""

import numpy as np
import illustris_python as il

def measure_density_profile(positions, masses, r_min=None, r_max=None,
                            n_bins=40, r_edges=None):
    """Measure a spherically averaged density profile from particles.

    Bin 0 is a central sphere [0, r_edges[0]]. Bins 1+ are shells
    [r_edges[i-1], r_edges[i]]. Returns n = len(r_edges)-1 bins.

    Args:
        positions: Particle positions relative to halo centre, shape (N, 3),
            in physical kpc.
        masses: Particle masses, shape (N,), in Msun.
        r_min, r_max, n_bins: Used to build log-spaced bins if r_edges not given.
        r_edges: Explicit bin edges in kpc. Overrides r_min/r_max/n_bins.

    Returns:
        (r_label, rho, r_edges) — outer edge of each bin in kpc,
        density in Msun/kpc^3, bin edges in kpc.
    """
    positions = np.asarray(positions)
    masses = np.asarray(masses)
    radii = np.linalg.norm(positions, axis=1)

    if r_edges is None:
        r_edges = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    else:
        r_edges = np.asarray(r_edges)

    n = len(r_edges) - 1
    rho = np.zeros(n)

    # Bin 0: sphere [0, r_edges[0]]
    mask = radii < r_edges[0]
    shell_mass = np.sum(masses[mask])
    shell_vol = (4.0 / 3.0) * np.pi * r_edges[0]**3
    rho[0] = shell_mass / shell_vol if shell_vol > 0 else 0.0

    # Bins 1+: shells [r_edges[i-1], r_edges[i]]
    for i in range(1, n):
        mask = (radii >= r_edges[i-1]) & (radii < r_edges[i])
        shell_mass = np.sum(masses[mask])
        shell_vol = (4.0 / 3.0) * np.pi * (r_edges[i]**3 - r_edges[i-1]**3)
        rho[i] = shell_mass / shell_vol

    r_label = r_edges[:-1]

    return r_label, rho, r_edges
# which for some reason are off by this factor. No idea where it comes from, but it makes the profiles match the catalog values, 
# so we apply it here for consistency.

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


def compute_gamma_dm(catalogs, model_profiles, models, r_fit_min=None,
                     min_ndm=1000, mstar_min=None):
    """Compute inner DM slope for all well-resolved halos across models.

    Args:
        catalogs: Dict of group catalogs per model. Each must have keys
            'N_dm', 'M200c', 'R200c', and optionally 'Mstar'.
        model_profiles: Dict of pre-computed profile dicts per model.
        models: List of model names to process.
        r_fit_min: Unused, kept for compatibility.
        min_ndm: Minimum number of DM particles.
        mstar_min: If set, only include halos with Mstar >= this value.

    Returns:
        Dict per model with arrays: 'M200c', 'R200c', 'gamma_dm', 'halo_ids'.
    """
    results = {}
    for name in models:
        cat = catalogs[name]
        profs = model_profiles[name]

        sel = (cat["N_dm"] >= min_ndm) & (cat["M200c"] > 0)
        if mstar_min is not None and "Mstar" in cat:
            sel &= cat["Mstar"] >= mstar_min
        halo_ids = np.where(sel)[0]

        gamma = np.full(len(halo_ids), np.nan)
        for i, hid in enumerate(halo_ids):
            if hid not in profs:
                continue
            prof = profs[hid]
            r = prof["r_outer"]
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
        r = p["r_outer"]

        rho_total = np.zeros(len(r))
        skip = False
        for comp in components:
            if comp == "prof_sfgas":
                if sf_gas_cache is None or hid not in sf_gas_cache:
                    skip = True
                    break
                rho_total += sf_gas_cache[hid]
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
            log_r_common, np.log10(r[valid]), np.log10(rho_total[valid]),
            left=np.nan, right=np.nan,
        )
        all_profiles.append(log_rho_interp)

    if not all_profiles:
        return None
    return np.array(all_profiles)


def measure_sf_gas_profile(basePath, snap, halo_id, r_edges, h=0.6774):
    """Compute the density profile of star-forming gas for a single halo.

    Loads PartType0 particles, selects those with StarFormationRate > 0,
    and delegates binning to ``measure_density_profile``.

    Args:
        basePath: Path to the simulation output directory.
        snap: Snapshot number.
        halo_id: FoF group index.
        r_edges: Radial bin edges in physical kpc, shape (n_bins+1,).
        h: Hubble parameter for unit conversion.

    Returns:
        Density profile in each shell (scaled to catalog normalization),
        shape (n_bins,). Returns zeros if no star-forming gas is found.
    """
    n_bins = len(r_edges) - 1

    halo = il.groupcat.loadSingle(basePath, snap, haloID=halo_id)
    halo_pos = halo["GroupPos"]

    gas = il.snapshot.loadHalo(
        basePath, snap, halo_id, "gas",
        fields=["Coordinates", "Masses", "StarFormationRate"],
    )

    if not isinstance(gas, dict) or gas.get("count", 0) == 0:
        return np.zeros(n_bins)

    sf_mask = gas["StarFormationRate"] > 0
    if sf_mask.sum() == 0:
        return np.zeros(n_bins)

    pos = (gas["Coordinates"][sf_mask] - halo_pos) / h
    mass = gas["Masses"][sf_mask] * 1e10 / h

    _, rho, _ = measure_density_profile(pos, mass, r_edges=r_edges)
    return rho


def compute_halo_sfe(basePath, snap, halo_id, h=0.6774):
    """Compute SFE = total SFR / total SF gas mass for a single halo.

    Args:
        basePath: Path to the simulation output directory.
        snap: Snapshot number.
        halo_id: FoF group index.
        h: Hubble parameter.

    Returns:
        SFE in 1/yr, or np.nan if no star-forming gas is found.
    """
    gas = il.snapshot.loadHalo(basePath, snap, halo_id, "gas",
        fields=["Masses", "StarFormationRate"])
    if not isinstance(gas, dict) or gas.get("count", 0) == 0:
        return np.nan
    sfr = gas["StarFormationRate"]
    sf = sfr > 0
    if sf.sum() == 0:
        return np.nan
    total_sfr = sfr[sf].sum()
    m_sfgas = gas["Masses"][sf].sum() * 1e10 / h
    return total_sfr / m_sfgas
