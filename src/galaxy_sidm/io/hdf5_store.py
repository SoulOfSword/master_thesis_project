"""HDF5 IO helpers for processed-data files.

Two layouts:
    flat       : top-level datasets of equal length N (catalogs, gamma_dm,
                 rcore fits, anything one-row-per-halo).
    per_halo   : variable-length per-halo subgroups under 'halos/fof_<id>/'
                 (density profiles, anything where the array size varies).

Both layouts store metadata in three subgroups: 'metadata' (file-level),
'cuts' (selection criteria), and 'variants' (algorithm options).
Attributes are written via group.attrs.

The intent is that scripts only need these four functions.
"""

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def save_flat(path, arrays: dict, *, cuts: dict | None = None,
              variants: dict | None = None, metadata: dict | None = None) -> None:
    """Save dict of equal-length 1D arrays plus metadata to HDF5.

    Args:
        path: output HDF5 path (parent created).
        arrays: {name: numpy 1D array}.
        cuts, variants, metadata: dicts written as attrs on
            'cuts', 'variants', 'metadata' subgroups respectively.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in arrays.items():
            f.create_dataset(k, data=np.asarray(v))
        _write_attrs_group(f, "metadata", _with_timestamp(metadata))
        _write_attrs_group(f, "cuts", cuts)
        _write_attrs_group(f, "variants", variants)


def load_flat(path):
    """Load a flat HDF5 file.

    Returns:
        (arrays, attrs) where arrays is {name: ndarray} and attrs is
        {'metadata': dict, 'cuts': dict, 'variants': dict}.
    """
    path = Path(path)
    arrays = {}
    with h5py.File(path, "r") as f:
        for k in f:
            if isinstance(f[k], h5py.Dataset):
                arrays[k] = f[k][...]
        attrs = _read_attrs_groups(f)
    return arrays, attrs


def save_per_halo(path, halo_data: dict, *, cuts: dict | None = None,
                  variants: dict | None = None,
                  metadata: dict | None = None) -> None:
    """Save per-halo variable-length data.

    Args:
        path: output HDF5 path.
        halo_data: {fof_id: {field_name: ndarray}}.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        halos = f.create_group("halos")
        for hid, fields in halo_data.items():
            g = halos.create_group(f"fof_{int(hid)}")
            for k, v in fields.items():
                g.create_dataset(k, data=np.asarray(v))
        _write_attrs_group(f, "metadata", _with_timestamp(metadata))
        _write_attrs_group(f, "cuts", cuts)
        _write_attrs_group(f, "variants", variants)


def load_per_halo(path, halo_ids=None):
    """Load per-halo data, optionally restricted to a subset.

    Args:
        path: HDF5 file.
        halo_ids: optional iterable of int FoF IDs to restrict to.

    Returns:
        (data, attrs). data is {fof_id: {field: ndarray}}.
    """
    path = Path(path)
    out = {}
    with h5py.File(path, "r") as f:
        halos = f["halos"]
        if halo_ids is None:
            keys = list(halos.keys())
        else:
            keys = [f"fof_{int(h)}" for h in halo_ids]
        for k in keys:
            if k not in halos:
                continue
            hid = int(k.split("_", 1)[1])
            g = halos[k]
            out[hid] = {field: g[field][...] for field in g}
        attrs = _read_attrs_groups(f)
    return out, attrs


# --- internals ---------------------------------------------------------

def _with_timestamp(metadata):
    m = dict(metadata or {})
    m.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    return m


def _write_attrs_group(f, name, d):
    if d is None:
        return
    g = f.require_group(name)
    for k, v in d.items():
        g.attrs[k] = _coerce_attr(v)


def _coerce_attr(v):
    if v is None:
        return "None"
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        if any(isinstance(x, str) for x in v):
            return list(map(str, v))
        return list(v)
    return v


def _read_attrs_groups(f):
    return {
        name: (dict(f[name].attrs) if name in f else {})
        for name in ("metadata", "cuts", "variants")
    }
