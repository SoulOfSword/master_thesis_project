"""AIDA-TNG simulation data loading."""

from pathlib import Path
import pickle
import h5py

CACHE_DIR = Path.home() / "master_thesis_project" / "data" / "profile_cache"


def _read_snap_redshift(run_path, snap):
    """Read the redshift of a snapshot from its HDF5 header."""
    run_path = Path(run_path)
    snapdir = run_path / "output" / f"snapdir_{snap:03d}"
    candidates = sorted(snapdir.glob(f"snap_{snap:03d}.*.hdf5"))
    if not candidates:
        candidates = sorted((run_path / "output").glob(f"snap_{snap:03d}.hdf5"))
    if not candidates:
        raise FileNotFoundError(
            f"No snapshot file for snap {snap} under {run_path}; "
            f"pass redshift explicitly."
        )
    with h5py.File(candidates[0], "r") as f:
        z = float(f["Header"].attrs["Redshift"])
    return max(z, 0.0)


def load_precomputed_profiles(run_path, snap, h=0.6774, use_test=False,
                              halo_ids=None, redshift=None):
    """Load pre-computed density profiles from the postprocessing catalog.

    Returns profiles in physical units: radii in physical kpc,
    densities in Msun/kpc^3. Bin 0 is a central sphere [0, r_edges[0]];
    bin i>=1 is a shell [r_edges[i-1], r_edges[i]].

    Args:
        run_path: Path to the simulation run directory.
        snap: Snapshot number.
        h: Dimensionless Hubble parameter (default 0.6774).
        use_test: If True, load the ``_test`` variant of the catalog.
        halo_ids: If provided, only load profiles for these FoF IDs.
        redshift: Snapshot redshift. If None, read from the snapshot header.
            Pass ``sim.redshift`` from a temet sim object to skip the I/O.

    Returns:
        Dict mapping FoF index (int) to a dict with keys
        ``r_edges``, ``r_outer``, ``prof_dm``, ``prof_gas``, ``prof_stars``.
    """
    run_path = Path(run_path)

    suffix = "_test" if use_test else ""
    fpath = run_path / "postprocessing" / f"cat_halo_profiles_{snap:02d}{suffix}.hdf5"
    if not fpath.exists():
        fpath = run_path / "postprocessing" / f"cat_halo_profiles_{snap}{suffix}.hdf5"

    run_name = run_path.name
    ids_tag = f"_n{len(halo_ids)}" if halo_ids is not None else ""
    cache_name = f"{run_name}_profiles_{snap:03d}{suffix}{ids_tag}_phys.pkl"
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        with open(cache_path, "rb") as cf:
            return pickle.load(cf)

    if redshift is None:
        redshift = _read_snap_redshift(run_path, snap)
    a = 1.0 / (1.0 + redshift)
    h2 = h**2
    a3 = a**3

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
            r_edges = 10**log_r_code / h * a
            r_outer = r_edges[:-1]

            prof = {
                "r_edges": r_edges,
                "r_outer": r_outer,
                "prof_dm":    grp["prof_dm"][:]    * h2 / a3 if "prof_dm"    in grp else None,
                "prof_gas":   grp["prof_gas"][:]   * h2 / a3 if "prof_gas"   in grp else None,
                "prof_stars": grp["prof_stars"][:] * h2 / a3 if "prof_stars" in grp else None,
            }
            profiles[fof_id] = prof

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as cf:
        pickle.dump(profiles, cf, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached {len(profiles)} profiles to {cache_path}")

    return profiles
