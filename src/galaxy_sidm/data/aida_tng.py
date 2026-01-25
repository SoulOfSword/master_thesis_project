"""AIDA-TNG simulation data loading.
Data is stored in HDF5 format.
"""

from pathlib import Path
import numpy as np
import h5py


def load_snapshot(filepath, fields = None):
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


def load_halo_catalog(filepath, simulation = "sidm"):
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
