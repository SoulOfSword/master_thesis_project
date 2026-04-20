"""Density profile measurement from particle data."""

import numpy as np
import illustris_python as il


def measure_density_profile(radii, masses, r_edges):
    """Spherically binned mass density profile.

    Bin 0 is a central sphere [0, r_edges[0]]; bin i>=1 is a shell
    [r_edges[i-1], r_edges[i]].

    Args:
        radii: Particle radii from halo centre, shape (N,), physical kpc.
        masses: Particle masses, shape (N,), Msun.
        r_edges: Bin edges, physical kpc.

    Returns:
        (r_label, rho, r_edges) — outer edge per bin in kpc,
        density in Msun/kpc^3, bin edges in kpc.
    """
    radii = np.asarray(radii)
    masses = np.asarray(masses)
    r_edges = np.asarray(r_edges)

    n = len(r_edges) - 1
    rho = np.zeros(n)

    mask = radii < r_edges[0]
    vol = (4.0 / 3.0) * np.pi * r_edges[0]**3
    rho[0] = masses[mask].sum() / vol if vol > 0 else 0.0

    for i in range(1, n):
        mask = (radii >= r_edges[i-1]) & (radii < r_edges[i])
        vol = (4.0 / 3.0) * np.pi * (r_edges[i]**3 - r_edges[i-1]**3)
        rho[i] = masses[mask].sum() / vol

    return r_edges[:-1], rho, r_edges


def measure_inner_slope(r_mid, rho, r_inner, r_outer):
    """Logarithmic slope d(log rho)/d(log r) over a radial range.

    Args:
        r_mid: Bin outer edges in kpc, shape (n_bins,).
        rho: Density in each bin, Msun/kpc^3.
        r_inner, r_outer: Fit range in kpc.

    Returns:
        Slope, or NaN if fewer than 2 bins fall in the range.
    """
    mask = (r_mid >= r_inner) & (r_mid <= r_outer) & (rho > 0)
    if mask.sum() < 2:
        return np.nan
    return np.polyfit(np.log10(r_mid[mask]), np.log10(rho[mask]), 1)[0]


def compute_gamma_dm(catalogs, model_profiles, models, r_fit_min=None,
                     min_ndm=1000, mstar_min=None):
    """Inner DM slope for all well-resolved halos across models."""
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
            rho = prof["prof_dm"]
            if rho is None:
                continue
            r = prof["r_outer"]
            r200 = cat["R200c"][hid]

            r_inner = 1.0
            r_outer = max(0.03 * r200, 10.0)
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
    """Interpolate and sum profile components onto a common radial grid."""
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


def measure_gas_density_profile(basePath, snap, halo_id, r_edges,
                                a=1.0, h=0.6774, box=None, mask=None):
    """Gas density profile for one halo, with optional mask.

    Args:
        basePath: Path to simulation output directory.
        snap: Snapshot number.
        halo_id: FoF group index.
        r_edges: Radial bin edges, physical kpc.
        a: Scale factor = 1/(1+z). Default 1 (z=0).
        h: Hubble parameter.
        box: Box size in ckpc/h. If None, read from snapshot header (slow).
        mask: Optional boolean array over gas cells.

    Returns:
        Density per shell, Msun/kpc^3, shape (len(r_edges)-1,).
    """
    n_bins = len(r_edges) - 1
    gas = il.snapshot.loadHalo(basePath, snap, halo_id, "gas",
                               fields=["Coordinates", "Masses"])
    if not isinstance(gas, dict) or gas.get("count", 0) == 0:
        return np.zeros(n_bins)

    halo = il.groupcat.loadSingle(basePath, snap, haloID=halo_id)
    if box is None:
        import h5py
        with h5py.File(f"{basePath}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5", "r") as f:
            box = float(f["Header"].attrs["BoxSize"])

    dx = gas["Coordinates"] - halo["GroupPos"]
    dx -= box * np.round(dx / box)
    rad_kpc = np.linalg.norm(dx, axis=1) * a / h
    mass_msun = gas["Masses"] * 1e10 / h

    if mask is not None:
        rad_kpc = rad_kpc[mask]
        mass_msun = mass_msun[mask]
    if len(rad_kpc) == 0:
        return np.zeros(n_bins)

    _, rho, _ = measure_density_profile(rad_kpc, mass_msun, r_edges)
    return rho


def measure_sf_gas_profile(basePath, snap, halo_id, r_edges,
                           a=1.0, h=0.6774, box=None):
    """SF gas density profile (SFR > 0 mask)."""
    n_bins = len(r_edges) - 1
    gas = il.snapshot.loadHalo(basePath, snap, halo_id, "gas",
                               fields=["Coordinates", "Masses", "StarFormationRate"])
    if not isinstance(gas, dict) or gas.get("count", 0) == 0:
        return np.zeros(n_bins)

    sf = gas["StarFormationRate"] > 0
    if sf.sum() == 0:
        return np.zeros(n_bins)

    halo = il.groupcat.loadSingle(basePath, snap, haloID=halo_id)
    if box is None:
        import h5py
        with h5py.File(f"{basePath}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5", "r") as f:
            box = float(f["Header"].attrs["BoxSize"])

    dx = gas["Coordinates"][sf] - halo["GroupPos"]
    dx -= box * np.round(dx / box)
    rad_kpc = np.linalg.norm(dx, axis=1) * a / h
    mass_msun = gas["Masses"][sf] * 1e10 / h

    _, rho, _ = measure_density_profile(rad_kpc, mass_msun, r_edges)
    return rho


def measure_cold_gas_profile(basePath, snap, halo_id, r_edges,
                             T_thresh=10**4.5, a=1.0, h=0.6774, box=None,
                             sim=None):
    """Cold-gas density profile.

    Non-SF cells with T < T_thresh contribute their full mass. SF cells
    contribute `mass * coldfrac` when `sim` is given (SH03 two-phase via
    `sim.units.densToSH03TwoPhase`); otherwise SF cells contribute their
    full mass (~10% overcount relative to SH03).
    """
    n_bins = len(r_edges) - 1
    fields = ["Coordinates", "Masses", "InternalEnergy",
              "ElectronAbundance", "StarFormationRate"]
    if sim is not None:
        fields += ["Density", "GFM_Metals"]

    gas = il.snapshot.loadHalo(basePath, snap, halo_id, "gas", fields=fields)
    if not isinstance(gas, dict) or gas.get("count", 0) == 0:
        return np.zeros(n_bins)

    m_p = 1.672622e-24
    k_B = 1.380650e-16
    X_H_primordial = 0.76
    gamma_eos = 5.0 / 3.0
    mu = 4.0 / (1.0 + 3.0*X_H_primordial + 4.0*X_H_primordial*gas["ElectronAbundance"])
    T = (gamma_eos - 1.0) * mu * m_p * (gas["InternalEnergy"] * 1e10) / k_B

    sfr = gas["StarFormationRate"]
    is_sf = sfr > 0

    w = np.zeros(len(T))
    w[(~is_sf) & (T < T_thresh)] = 1.0
    if is_sf.any():
        if sim is not None:
            X_H_cell = gas["GFM_Metals"][:, 0]
            nH = sim.units.codeDensToPhys(gas["Density"], cgs=True, numDens=True) * X_H_cell
            coldfrac, _ = sim.units.densToSH03TwoPhase(nH[is_sf], sfr[is_sf])
            w[is_sf] = coldfrac
        else:
            w[is_sf] = 1.0

    m_cold = gas["Masses"] * w * 1e10 / h
    keep = m_cold > 0
    if keep.sum() == 0:
        return np.zeros(n_bins)

    halo = il.groupcat.loadSingle(basePath, snap, haloID=halo_id)
    if box is None:
        import h5py
        with h5py.File(f"{basePath}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5", "r") as f:
            box = float(f["Header"].attrs["BoxSize"])

    dx = gas["Coordinates"][keep] - halo["GroupPos"]
    dx -= box * np.round(dx / box)
    rad_kpc = np.linalg.norm(dx, axis=1) * a / h

    _, rho, _ = measure_density_profile(rad_kpc, m_cold[keep], r_edges)
    return rho


def compute_halo_sfe(basePath, snap, halo_id, h=0.6774):
    """SFE = total SFR / total SF gas mass for a halo."""
    gas = il.snapshot.loadHalo(basePath, snap, halo_id, "gas",
                               fields=["Masses", "StarFormationRate"])
    if not isinstance(gas, dict) or gas.get("count", 0) == 0:
        return np.nan
    sfr = gas["StarFormationRate"]
    sf = sfr > 0
    if sf.sum() == 0:
        return np.nan
    m_sfgas = gas["Masses"][sf].sum() * 1e10 / h
    return sfr[sf].sum() / m_sfgas
