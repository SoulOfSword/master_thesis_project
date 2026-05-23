"""Density profile measurement from particle data."""

import numpy as np
import illustris_python as il


def measure_density_profile(radii, masses, r_edges):
    """Spherically binned mass density profile.

    Bin 0 is a central sphere [0, r_edges[0]]; bin i>=1 is a shell
    [r_edges[i-1], r_edges[i]].

    Args:
        radii: Particle radii from halo centre, shape (N,), comoving ckpc.
        masses: Particle masses, shape (N,), Msun.
        r_edges: Bin edges, comoving ckpc.

    Returns:
        (r_label, rho, r_edges) — outer edge per bin in ckpc,
        density in Msun/ckpc^3, bin edges in ckpc.
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
        r_mid: Bin outer edges in ckpc, shape (n_bins,).
        rho: Density in each bin, Msun/ckpc^3.
        r_inner, r_outer: Fit range in ckpc.

    Returns:
        Slope, or NaN if fewer than 2 bins fall in the range.
    """
    mask = (r_mid >= r_inner) & (r_mid <= r_outer) & (rho > 0)
    if mask.sum() < 2:
        return np.nan
    return np.polyfit(np.log10(r_mid[mask]), np.log10(rho[mask]), 1)[0]


def compute_r_core_dm(catalogs, model_profiles, models,
                      min_ndm=1000, mstar_min=None,
                      fit_min=1.0, fit_max_factor=1.0):
    """Per-halo cored-NFW fit; returns r_core, r_s, rho_s arrays per model.

    For each qualifying halo, runs a 3-parameter fit of the cored NFW
    profile (NFW * tanh(r/r_core)) to its DM density profile over
    [fit_min, fit_max_factor * R200c] in comoving ckpc. r_core close to
    zero indicates a cuspy NFW-like halo; large r_core indicates a core.

    Args:
        catalogs: {model: catalog_dict} with 'N_dm', 'M200c', 'R200c',
            optionally 'Mstar'.
        model_profiles: {model: {halo_id: profile_dict}} with each
            profile having 'r_outer' and 'prof_dm'.
        models: list of model names to process.
        min_ndm: minimum DM particle count per halo.
        mstar_min: if provided and 'Mstar' in catalog, drop halos below.
        fit_min: inner radius for the fit (ckpc, default 3.0).
        fit_max_factor: outer radius = factor * R200c (default 1.0).

    Returns:
        Dict per model with arrays 'M200c', 'R200c', 'r_core', 'r_s',
        'rho_s', 'chi2', 'halo_ids' (only halos with successful fit).
    """
    from ..models.profiles import fit_nfw, fit_cored_nfw

    results = {}
    for name in models:
        cat = catalogs[name]
        profs = model_profiles[name]

        sel = (cat["N_dm"] >= min_ndm) & (cat["M200c"] > 0)
        if mstar_min is not None and "Mstar" in cat:
            sel &= cat["Mstar"] >= mstar_min
        halo_ids = np.where(sel)[0]

        n = len(halo_ids)
        r_core_arr = np.full(n, np.nan)
        r_s_arr    = np.full(n, np.nan)
        rho_s_arr  = np.full(n, np.nan)
        chi2_arr   = np.full(n, np.nan)
        for i, hid in enumerate(halo_ids):
            if hid not in profs:
                continue
            prof = profs[hid]
            rho = prof["prof_dm"]
            if rho is None:
                continue
            r = prof["r_outer"]
            r200 = cat["R200c"][hid]

            nfw0 = fit_nfw(r, rho, r_fit_min=fit_min,
                           r_fit_max=fit_max_factor * r200)
            seed = nfw0 if nfw0.get("success") else None

            cfit = fit_cored_nfw(r, rho, r_fit_min=fit_min,
                                  r_fit_max=fit_max_factor * r200,
                                  p0_from_nfw=seed)
            if not cfit.get("success"):
                continue

            r_core_arr[i] = cfit["r_core"]
            r_s_arr[i]    = cfit["r_s"]
            rho_s_arr[i]  = cfit["rho_s"]
            chi2_arr[i]   = cfit["chi2"]

        valid = np.isfinite(r_core_arr)
        results[name] = {
            "M200c":    cat["M200c"][halo_ids][valid],
            "R200c":    cat["R200c"][halo_ids][valid],
            "r_core":   r_core_arr[valid],
            "r_s":      r_s_arr[valid],
            "rho_s":    rho_s_arr[valid],
            "chi2":     chi2_arr[valid],
            "halo_ids": halo_ids[valid],
        }
        print(f"{name}: {valid.sum()}/{n} halos with successful cored-NFW fit")

    return results


def compute_gamma_dm(catalogs, model_profiles, models,
                     r_inner=1.0,
                     r_outer_kind="nfw_rs",
                     r_outer_factor=None,
                     r_outer_floor=10.0,
                     min_ndm=1000, mstar_min=None,
                     nfw_fit_min=5.0, nfw_fit_max_factor=1.0):
    """Inner DM slope from per-halo log-log slope measurement.

    Two ways to define the outer slope-fit radius, selected via
    r_outer_kind:

    'nfw_rs' (default):
        Per-halo NFW fit to recover r_s, then r_outer = factor * r_s.
        Default factor = 0.3 (NFW log-slope here is ~ -1.46 — close to
        the asymptotic inner cusp without being unstable from too few
        bins).

    'r200c':
        r_outer = max(factor * R200c, r_outer_floor). Default factor =
        0.03 and floor = 10 ckpc — recovers the original outer-radius
        criterion used before the NFW-r_s switch. NFW fit is skipped.

    Inner slope is then measured as a log-log linear fit of rho(r) over
    [r_inner, r_outer]. All radii in comoving ckpc.

    Args:
        catalogs: {model_name: catalog_dict} with 'N_dm', 'M200c',
            'R200c', optionally 'Mstar'.
        model_profiles: {model_name: {halo_id: profile_dict}}.
        models: list of model names.
        r_inner: inner radius for the slope fit (ckpc, default 1.0).
        r_outer_kind: 'nfw_rs' or 'r200c'.
        r_outer_factor: factor for the chosen kind. If None, uses 0.3
            for 'nfw_rs' and 0.03 for 'r200c'.
        r_outer_floor: floor on r_outer (ckpc). Only used when
            r_outer_kind='r200c'.
        min_ndm: minimum N_dm per halo (default 1000).
        mstar_min: if provided and 'Mstar' in catalog, drop halos below.
        nfw_fit_min: inner radius for the NFW fit when kind='nfw_rs'.
        nfw_fit_max_factor: outer NFW-fit radius = factor * R200c.

    Returns:
        Dict per model with 'M200c', 'R200c', 'r_s', 'gamma_dm',
        'halo_ids'. r_s is NaN when r_outer_kind='r200c'.
    """
    if r_outer_kind not in ("nfw_rs", "r200c"):
        raise ValueError(f"r_outer_kind must be 'nfw_rs' or 'r200c', "
                         f"got {r_outer_kind!r}")
    if r_outer_factor is None:
        r_outer_factor = 0.3 if r_outer_kind == "nfw_rs" else 0.03

    from ..models.profiles import fit_nfw

    results = {}
    for name in models:
        cat = catalogs[name]
        profs = model_profiles[name]

        sel = (cat["N_dm"] >= min_ndm) & (cat["M200c"] > 0)
        if mstar_min is not None and "Mstar" in cat:
            sel &= cat["Mstar"] >= mstar_min
        halo_ids = np.where(sel)[0]

        gamma = np.full(len(halo_ids), np.nan)
        r_s_arr = np.full(len(halo_ids), np.nan)
        for i, hid in enumerate(halo_ids):
            if hid not in profs:
                continue
            prof = profs[hid]
            rho = prof["prof_dm"]
            if rho is None:
                continue
            r = prof["r_outer"]
            r200 = cat["R200c"][hid]

            if r_outer_kind == "nfw_rs":
                fit = fit_nfw(r, rho, r_fit_min=nfw_fit_min,
                              r_fit_max=nfw_fit_max_factor * r200)
                if not fit.get("success") or not np.isfinite(fit.get("r_s", np.nan)):
                    continue
                r_s = float(fit["r_s"])
                r_outer = r_outer_factor * r_s
                r_s_arr[i] = r_s
            else:  # r200c
                r_outer = max(r_outer_factor * r200, r_outer_floor)

            if r_outer <= r_inner:
                continue

            gamma[i] = measure_inner_slope(r, rho, r_inner=r_inner, r_outer=r_outer)

        valid = ~np.isnan(gamma)
        results[name] = {
            "M200c":   cat["M200c"][halo_ids][valid],
            "R200c":   cat["R200c"][halo_ids][valid],
            "r_s":     r_s_arr[valid],
            "gamma_dm": gamma[valid],
            "halo_ids": halo_ids[valid],
        }
        print(f"{name}: {valid.sum()}/{len(halo_ids)} halos with valid gamma_DM")

    return results


def collect_profiles(profiles, halo_ids, r_common, components,
                     sf_gas_cache=None, cold_gas_cache=None):
    """Interpolate and sum profile components onto a common radial grid.

    Components `prof_sfgas` and `prof_coldgas` pull from the respective
    external caches; all other component keys are read from each halo's
    profile dict.
    """
    log_r_common = np.log10(r_common)
    all_profiles = []

    ext_caches = {"prof_sfgas": sf_gas_cache, "prof_coldgas": cold_gas_cache}

    for hid in halo_ids:
        if hid not in profiles:
            continue
        p = profiles[hid]
        r = p["r_outer"]

        rho_total = np.zeros(len(r))
        skip = False
        for comp in components:
            if comp in ext_caches:
                cache = ext_caches[comp]
                if cache is None or hid not in cache:
                    skip = True
                    break
                rho_total += cache[hid]
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
        r_edges: Radial bin edges, comoving ckpc.
        a: Retained for backward-compat but unused; output is comoving.
        h: Hubble parameter.
        box: Box size in code length units. If None, read from snapshot
            header (slow).
        mask: Optional boolean array over gas cells.

    Returns:
        Density per shell, Msun/ckpc^3, shape (len(r_edges)-1,).
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
    rad_ckpc = np.linalg.norm(dx, axis=1) / h
    mass_msun = gas["Masses"] * 1e10 / h

    if mask is not None:
        rad_ckpc = rad_ckpc[mask]
        mass_msun = mass_msun[mask]
    if len(rad_ckpc) == 0:
        return np.zeros(n_bins)

    _, rho, _ = measure_density_profile(rad_ckpc, mass_msun, r_edges)
    return rho


def measure_sf_gas_profile(basePath, snap, halo_id, r_edges,
                           a=1.0, h=0.6774, box=None):
    """SF gas density profile (SFR > 0 mask).

    `a` is retained for backward-compat but unused; output is comoving
    (radii in ckpc, density in Msun/ckpc^3).
    """
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
    rad_ckpc = np.linalg.norm(dx, axis=1) / h
    mass_msun = gas["Masses"][sf] * 1e10 / h

    _, rho, _ = measure_density_profile(rad_ckpc, mass_msun, r_edges)
    return rho


def measure_cold_gas_profile(basePath, snap, halo_id, r_edges,
                             T_thresh=1e4, a=1.0, h=0.6774, box=None,
                             sim=None):
    """Cold-gas density profile.

    Non-SF cells with T < T_thresh contribute their full mass. SF cells
    contribute `mass * coldfrac` when `sim` is given (SH03 two-phase via
    `sim.units.densToSH03TwoPhase`); otherwise SF cells contribute their
    full mass (~10% overcount relative to SH03).

    `a` is retained for backward-compat but unused; radii are output in
    comoving ckpc and densities in Msun/ckpc^3.
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
            # nH stays physical: densToSH03TwoPhase is a sub-grid physics
            # formula calibrated against physical hydrogen number density.
            X_H_cell = gas["GFM_Metals"][:, 0]
            nH = sim.units.codeDensToPhys(
                gas["Density"], cgs=True, numDens=True) * X_H_cell
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
    rad_ckpc = np.linalg.norm(dx, axis=1) / h

    _, rho, _ = measure_density_profile(rad_ckpc, m_cold[keep], r_edges)
    return rho


def compute_halo_total_sfr(basePath, snap, halo_id):
    """Total SFR (Msun/yr) summed over all gas in a halo's FoF group."""
    gas = il.snapshot.loadHalo(basePath, snap, halo_id, "gas",
                               fields=["Masses", "StarFormationRate"])
    if not isinstance(gas, dict) or gas.get("count", 0) == 0:
        return np.nan
    return gas["StarFormationRate"].sum()


def measure_subhalo_cold_gas_all(basePath, snap, subhalo_id, h=0.6774,
                                 T_thresh_high=10**4.5, T_thresh_low=1e4,
                                 T_cold_phase=1000.0, sim=None):
    """Total cold-gas mass in a subhalo for four definitions, single read.

    Args:
        basePath: Simulation output path.
        snap: Snapshot number.
        subhalo_id: Subhalo index (use central from GroupFirstSub).
        h: Hubble parameter h.
        T_thresh_high: Upper temperature cut for the 10^4.5 K and SH03
            non-SF branches, in K.
        T_thresh_low: Lower temperature cut for the 10^4 K branch, in K.
        T_cold_phase: Substituted T (K) for SF cells when computing T_eff
            (mimics temet's `temp_sfcold`).
        sim: temet sim instance. Required for the 'sh03' definition; if
            None, 'sh03' falls back to giving SF cells full mass.

    Returns:
        Dict with keys 'SFR>0', 'T<1e4', 'T<10^4.5', 'sh03' — each the
        total cold-gas mass in Msun. Zeros if the subhalo has no gas.
    """
    fields = ["Masses", "InternalEnergy", "ElectronAbundance",
              "StarFormationRate", "Density", "GFM_Metals"]
    gas = il.snapshot.loadSubhalo(basePath, snap, subhalo_id, "gas",
                                  fields=fields)
    out = {"SFR>0": 0.0, "T<1e4": 0.0, "T<10^4.5": 0.0, "sh03": 0.0}
    if not isinstance(gas, dict) or gas.get("count", 0) == 0:
        return out

    masses_msun = gas["Masses"] * 1e10 / h
    sfr = gas["StarFormationRate"]
    is_sf = sfr > 0

    m_p = 1.672622e-24
    k_B = 1.380650e-16
    X_H = 0.76
    gamma_eos = 5.0 / 3.0
    mu = 4.0 / (1.0 + 3.0*X_H + 4.0*X_H*gas["ElectronAbundance"])
    T_normal = (gamma_eos - 1.0) * mu * m_p * (gas["InternalEnergy"] * 1e10) / k_B
    T_eff = np.where(is_sf, T_cold_phase, T_normal)

    out["SFR>0"]    = float(masses_msun[is_sf].sum())
    out["T<1e4"]    = float(masses_msun[T_eff < T_thresh_low].sum())
    out["T<10^4.5"] = float(masses_msun[T_eff < T_thresh_high].sum())

    w = np.zeros(len(masses_msun))
    w[(~is_sf) & (T_normal < T_thresh_high)] = 1.0
    if is_sf.any():
        if sim is not None:
            # nH stays physical: densToSH03TwoPhase is a sub-grid physics
            # formula calibrated against physical hydrogen number density.
            X_H_cell = gas["GFM_Metals"][:, 0]
            nH = sim.units.codeDensToPhys(
                gas["Density"], cgs=True, numDens=True) * X_H_cell
            coldfrac, _ = sim.units.densToSH03TwoPhase(nH[is_sf], sfr[is_sf])
            w[is_sf] = coldfrac
        else:
            w[is_sf] = 1.0
    out["sh03"] = float((masses_msun * w).sum())

    return out


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
