"""Parse MORDOR's ASCII output into a DataFrame."""

from pathlib import Path
import re
import numpy as np
import pandas as pd


MORDOR_COLS = [
    "filename",
    "Mstar", "Munbound",
    "Mthin", "Mthick", "Mbulge", "Mpbulge", "Mhalo",
    "IsDisc",
    "Ethin", "Ethick", "Ebulge", "Epbulge", "Ehalo",
    "Cthin", "Cthick", "Cbulge", "Cpbulge", "Chalo",
]

_ID_RE = re.compile(r"_(\d+)\.hdf5$")


def _id_from_filename(name):
    m = _ID_RE.search(str(name))
    if not m:
        return -1
    return int(m.group(1))


def parse_mordor_output(path, catalog=None, drop_failed=True):
    """Read an ASCII output file from `run_mordor_batch`.

    Args:
        path: Path to MORDOR's per-galaxy ASCII table.
        catalog: Optional dict whose array values are indexed by
            **subhalo** id. If provided, adds columns 'M200c', 'R200c',
            'N_dm', 'Mstar_cat' from the matching keys when present.
        drop_failed: Remove rows where MORDOR returned Mstar <= 0.

    Returns:
        pandas.DataFrame indexed by subhalo_id.
    """
    path = Path(path)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < len(MORDOR_COLS):
                continue
            rows.append(parts[: len(MORDOR_COLS)])

    if not rows:
        return pd.DataFrame(columns=MORDOR_COLS)

    df = pd.DataFrame(rows, columns=MORDOR_COLS)
    for c in MORDOR_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["subhalo_id"] = df["filename"].map(_id_from_filename)
    df = df[df["subhalo_id"] >= 0].reset_index(drop=True)

    if drop_failed:
        df = df[df["Mstar"] > 0].reset_index(drop=True)

    df["IsDisc"] = df["IsDisc"].astype(int)
    df = df.set_index("subhalo_id")

    if catalog is not None:
        for key, out in [("M200c", "M200c"), ("R200c", "R200c"),
                         ("N_dm", "N_dm"), ("Mstar", "Mstar_cat")]:
            if key in catalog:
                arr = np.asarray(catalog[key])
                df[out] = df.index.to_series().map(
                    lambda s: arr[s] if 0 <= s < len(arr) else np.nan
                )

    return df
