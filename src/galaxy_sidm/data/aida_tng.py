"""AIDA-TNG simulation data loading.
Data is stored in HDF5 format.
"""

from pathlib import Path
import numpy as np
import h5py


def load_precomputed_profiles(run_path, snap, h=0.6774):
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
    fpath = run_path / "postprocessing" / f"cat_halo_profiles_{snap:02d}.hdf5"
    if not fpath.exists():
        fpath = run_path / "postprocessing" / f"cat_halo_profiles_{snap}.hdf5"

    profiles = {}
    with h5py.File(fpath, "r") as f:
        for key in f.keys():
            if not key.startswith("fof_"):
                continue
            fof_id = int(key.split("_")[1])
            grp = f[key]

            # Radii: stored as log10(ckpc/h), convert to physical kpc
            log_r_code = grp["r"][:]
            r_edges = 10**log_r_code / h  # physical kpc
            r_mid = np.sqrt(r_edges[:-1] * r_edges[1:])

            prof = {
                "r_edges": r_edges,
                "r_mid": r_mid,
                "prof_dm": grp["prof_dm"][:] if "prof_dm" in grp else None,
                "prof_gas": grp["prof_gas"][:] if "prof_gas" in grp else None,
                "prof_stars": grp["prof_stars"][:] if "prof_stars" in grp else None,
            }
            profiles[fof_id] = prof

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
