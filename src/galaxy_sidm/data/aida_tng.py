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

    Per Despali (priv. comm.), bin 0 is a central sphere [0, r_edges[0]] and
    bin i>=1 is a shell [r_edges[i-1], r_edges[i]]. Each bin is labeled by its
    outer edge, so ``r_mid`` here is really ``r_edges[:-1]`` (the outer edges
    of the 39 bins, dropping the unused last edge).

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
        - ``r_outer``: outer edge of each bin in physical kpc, shape (39,)
        - ``prof_dm``: DM density profile, shape (39,), or None
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
            r_outer = r_edges[:-1]

            prof = {
                "r_edges": r_edges,
                "r_outer": r_outer,
                "prof_dm": grp["prof_dm"][:]*h**2 if "prof_dm" in grp else None,
                "prof_gas": grp["prof_gas"][:]*h**2 if "prof_gas" in grp else None,
                "prof_stars": grp["prof_stars"][:]*h**2 if "prof_stars" in grp else None,
            }
            profiles[fof_id] = prof

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as cf:
        pickle.dump(profiles, cf, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached {len(profiles)} profiles to {cache_path}")

    return profiles
