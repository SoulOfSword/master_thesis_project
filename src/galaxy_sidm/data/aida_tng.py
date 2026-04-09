"""AIDA-TNG simulation data loading.
Data is stored in HDF5 format.
"""

from pathlib import Path
import pickle
import numpy as np
import h5py

CACHE_DIR = Path.home() / "master_thesis_project" / "data" / "profile_cache"


def load_precomputed_profiles(run_path, snap, h=0.6774, use_test=False):
    """Load pre-computed density profiles from the postprocessing catalog.

    Reads ``cat_halo_profiles_{snap}.hdf5`` which contains spherically averaged
    profiles for each FoF group, computed by the AIDA-TNG team.

    The raw catalog stores radial bin edges as log10(r / (ckpc/h)) and profile
    values proportional to density (up to a constant factor that differs
    between FP and DMO catalogs).  This function converts radii to physical
    kpc.  The density values are left as stored — the unknown constant does
    not affect logarithmic slope measurements.

    Args:
        run_path: Path to the simulation run directory
            (e.g. ``AIDA_BASE / "L35n1080_CDM"``).
        snap: Snapshot number.
        h: Dimensionless Hubble parameter (default 0.6774, Planck 2016).
        use_test: If True, load the ``_test`` variant of the catalog
            (required for FP/hydro runs per Despali, priv. comm.).

    Returns:
        Dict mapping FoF index (int) to a dict with keys:

        - ``r_edges``: bin edges in physical kpc, shape (40,)
        - ``r_mid``: geometric-mean bin centres in physical kpc, shape (39,)
        - ``prof_dm``: DM density profile (proportional, not calibrated),
          shape (39,), or None
        - ``prof_gas``: gas density profile, shape (39,), or None
        - ``prof_stars``: stellar density profile, shape (39,), or None
    """
    run_path = Path(run_path)
    suffix = "_test" if use_test else ""
    fpath = run_path / "postprocessing" / f"cat_halo_profiles_{snap:02d}{suffix}.hdf5"
    if not fpath.exists():
        fpath = run_path / "postprocessing" / f"cat_halo_profiles_{snap}{suffix}.hdf5"

    # Check for cached version
    run_name = run_path.name
    cache_name = f"{run_name}_profiles_{snap:03d}{suffix}.pkl"
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        with open(cache_path, "rb") as cf:
            return pickle.load(cf)

    profiles = {}
    with h5py.File(fpath, "r") as f:
        for key in f.keys():
            if not key.startswith("fof_"):
                continue
            fof_id = int(key.split("_")[1])
            grp = f[key]

            log_r_code = grp["r"][:]
            r_edges = 10**log_r_code / h
            r_mid = np.sqrt(r_edges[:-1] * r_edges[1:])

            prof = {
                "r_edges": r_edges,
                "r_mid": r_mid,
                "prof_dm": grp["prof_dm"][:] if "prof_dm" in grp else None,
                "prof_gas": grp["prof_gas"][:] if "prof_gas" in grp else None,
                "prof_stars": grp["prof_stars"][:] if "prof_stars" in grp else None,
            }
            profiles[fof_id] = prof

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as cf:
        pickle.dump(profiles, cf, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached {len(profiles)} profiles to {cache_path}")

    return profiles


def load_snapshot(filepath, fields=None):
    """Load particle data from an AIDA-TNG snapshot.

    Parameters
    ----------
    filepath : Path or str
        Path to the HDF5 snapshot file.
    fields : list of str, optional
        Specific fields to load. If None, loads all available fields.

    Returns
    -------
    dict
        Dictionary with particle data arrays.
    """
    data = {}
    with h5py.File(filepath, "r") as f:
        # TODO: Implement based on actual AIDA-TNG file structure
        # Common particle types: PartType0 (gas), PartType1 (DM), PartType4 (stars)
        pass
    return data


def load_halo_catalog(filepath, simulation="sidm"):
    """Load halo/subhalo catalog from AIDA-TNG.

    Parameters
    ----------
    filepath : Path or str
        Path to the catalog file.
    simulation : str
        Which simulation run: "sidm" or "cdm".

    Returns
    -------
    dict
        Dictionary with halo properties.
    """
    # TODO: Implement based on actual AIDA-TNG catalog structure
    # Expected fields: M200, R200, Vmax, M_star, R_half, etc.
    raise NotImplementedError("Implement based on AIDA-TNG data structure")


def get_matched_halos(cdm_catalog, sidm_catalog):
    """Get indices of matched halos between CDM and SIDM runs.

    AIDA-TNG provides matched halos from the same initial conditions,
    enabling direct CDM vs SIDM comparison.

    Returns
    -------
    cdm_indices, sidm_indices : tuple of arrays
        Matching indices into each catalog.
    """
    # TODO: Implement matching logic based on AIDA-TNG matching tables
    raise NotImplementedError
