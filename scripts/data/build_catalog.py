"""Build a galaxy / halo catalog HDF5 file for a (model, snap).

For one (model, snap), load the FoF + Subfind catalogs via temet, apply
stellar-mass and DM-particle cuts, and save a flat HDF5 with one row per
qualifying halo. Downstream scripts read these catalogs.

Output (under cfg['paths']['scratch_processed']/catalogs/):
    catalog_<model>_<snap:03d>_mstar<mstar_min>_ndm<ndm_min>.hdf5

Datasets:
    halo_ids, M200c (Msun), R200c (ckpc), Mstar (Msun),
    Rhalf_star (ckpc), N_dm, GroupFirstSub
attrs:
    metadata: model, snap, redshift, h, created_at
    cuts:     mstar_min, mstar_max, n_dm_min
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import temet

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, save_flat


def build(model: str, snap: int, mstar_min: float, mstar_max: float | None,
          n_dm_min: int, h: float):
    """Return (arrays_dict, metadata, cuts) ready for save_flat."""
    sim = temet.sim(run="aida", variant=model, res=1080, snap=snap)
    gc = sim.groupCat(fieldsHalos=[
        "Group_M_Crit200", "Group_R_Crit200",
        "GroupLenType", "GroupFirstSub",
    ])
    mstar_all = sim.units.codeMassToMsun(
        sim.subhalos("SubhaloMassType")[:, 4])
    rhalf_star_all = sim.units.codeLengthToComovingKpc(
        sim.subhalos("SubhaloHalfmassRadType")[:, 4])
    fs = gc["GroupFirstSub"]
    mstar = np.zeros(len(fs))
    rhalf_star = np.zeros(len(fs))
    valid = (fs >= 0) & (fs < len(mstar_all))
    mstar[valid] = mstar_all[fs[valid]]
    rhalf_star[valid] = rhalf_star_all[fs[valid]]

    M200c = sim.units.codeMassToMsun(gc["Group_M_Crit200"])
    R200c = sim.units.codeLengthToComovingKpc(gc["Group_R_Crit200"])
    N_dm = gc["GroupLenType"][:, 1]

    sel = (M200c > 0) & (N_dm >= n_dm_min) & (mstar >= mstar_min)
    if mstar_max is not None:
        sel &= mstar <= mstar_max
    halo_ids = np.where(sel)[0].astype(np.int64)

    arrays = {
        "halo_ids":      halo_ids,
        "M200c":         M200c[halo_ids].astype(np.float64),
        "R200c":         R200c[halo_ids].astype(np.float64),
        "Mstar":         mstar[halo_ids].astype(np.float64),
        "Rhalf_star":    rhalf_star[halo_ids].astype(np.float64),
        "N_dm":          N_dm[halo_ids].astype(np.int64),
        "GroupFirstSub": fs[halo_ids].astype(np.int64),
    }
    metadata = {
        "model":    model,
        "snap":     int(snap),
        "redshift": float(sim.redshift),
        "h":        float(h),
    }
    cuts = {
        "mstar_min":  float(mstar_min),
        "mstar_max":  float(mstar_max) if mstar_max is not None else -1.0,
        "n_dm_min":   int(n_dm_min),
    }
    return arrays, metadata, cuts


def output_name(model: str, snap: int, mstar_min: float, n_dm_min: int) -> str:
    return (f"catalog_{model}_{snap:03d}"
            f"_mstar{mstar_min:.0e}_ndm{n_dm_min}.hdf5")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="Path to config/scripts.yaml (defaults to project root)")
    p.add_argument("--model", required=True)
    p.add_argument("--snap", required=True, type=int)
    p.add_argument("--mstar-min", type=float, default=None)
    p.add_argument("--mstar-max", type=float, default=None)
    p.add_argument("--n-dm-min", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    # Cast defensively: PyYAML 1.1 parses things like "1.0e8" (no exponent sign)
    # as STRINGS, which then breaks numpy comparisons downstream. float() handles
    # both float-like-strings and actual floats.
    mstar_min = float(args.mstar_min if args.mstar_min is not None else cfg["defaults"]["mstar_min"])
    n_dm_min = int(args.n_dm_min if args.n_dm_min is not None else cfg["defaults"]["n_dm_min"])
    out_root = Path(args.out_root or cfg["paths"]["scratch_processed"])

    arrays, metadata, cuts = build(
        model=args.model, snap=args.snap,
        mstar_min=mstar_min, mstar_max=args.mstar_max,
        n_dm_min=n_dm_min, h=cfg["cosmology"]["h"],
    )

    out_path = out_root / "catalogs" / output_name(
        args.model, args.snap, mstar_min, n_dm_min)
    save_flat(out_path, arrays, cuts=cuts, metadata=metadata)
    print(f"[build_catalog] {args.model} snap {args.snap}: "
          f"{len(arrays['halo_ids'])} halos -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
