"""AIDA-TNG simulation data loading.
Data is stored in HDF5 format.
"""

from pathlib import Path
import pickle
import numpy as np
import h5py

CACHE_DIR = Path.home() / "master_thesis_project" / "data" / "profile_cache"

def load_precomputed_profiles(run_path, snap, h=0.6774, use_test=False,
                              halo_ids=None):
    """Load pre-computed density profiles from the postprocessing catalog.

    Reads ``cat_halo_profiles_{snap}.hdf5`` which contains spherically averaged
    profiles for each FoF group, computed by the AIDA-TNG team.

    Bin 0 is a central sphere [0, r_edges[0]] and bin i>=1 is a shell
    [r_edges[i-1], r_edges[i]]. Each bin is labeled by its outer edge.

    Args:
        run_path: Path to the simulation run directory.
        snap: Snapshot number.
        h: Dimensionless Hubble parameter (default 0.6774).
        use_test: If True, load the ``_test`` variant of the catalog
            (required for FP/hydro runs per Despali, priv. comm.).
        halo_ids: If provided, only load profiles for these FoF IDs.
            Significantly faster for large catalogs.

    Returns:
        Dict mapping FoF index (int) to a dict with keys:
        ``r_edges``, ``r_outer``, ``prof_dm``, ``prof_gas``, ``prof_stars``.
    """
    run_path = Path(run_path)
    suffix = "_test" if use_test else ""
    fpath = run_path / "postprocessing" / f"cat_halo_profiles_{snap:02d}{suffix}.hdf5"
    if not fpath.exists():
        fpath = run_path / "postprocessing" / f"cat_halo_profiles_{snap}{suffix}.hdf5"

    # Check for cached version
    run_name = run_path.name
    ids_tag = f"_n{len(halo_ids)}" if halo_ids is not None else ""
    cache_name = f"{run_name}_profiles_{snap:03d}{suffix}{ids_tag}.pkl"
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        with open(cache_path, "rb") as cf:
            return pickle.load(cf)

    h2 = h**2

    profiles = {}
    with h5py.File(fpath, "r") as f:
        if halo_ids is not None:
            keys_to_load = [f"fof_{hid}" for hid in halo_ids]
        else:
            keys_to_load = [k for k in f.keys() if k.startswith("fof_")]

        for key in keys_to_load:
            if key not in f:
                continue
            fof_id = int(key.split("_")[1])
            grp = f[key]

            log_r_code = grp["r"][:]
            r_edges = 10**log_r_code / h
            r_outer = r_edges[:-1]

            prof = {
                "r_edges": r_edges,
                "r_outer": r_outer,
                "prof_dm": grp["prof_dm"][:] * h2 if "prof_dm" in grp else None,
                "prof_gas": grp["prof_gas"][:] * h2 if "prof_gas" in grp else None,
                "prof_stars": grp["prof_stars"][:] * h2 if "prof_stars" in grp else None,
            }
            profiles[fof_id] = prof

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as cf:
        pickle.dump(profiles, cf, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached {len(profiles)} profiles to {cache_path}")

    return profiles
