"""Build a DMO (dark-matter-only) halo catalog matched to an FP catalog.

Takes an existing FP catalog HDF5 (produced by build_catalog.py), matches
its central subhalos to the corresponding DMO run via the
SubhaloMatchingToDark/LHaloTree table, and saves a flat HDF5 with one row
per matched DMO FoF halo. Downstream scripts read these catalogs.

Output (under cfg['paths']['scratch_processed']/catalogs/):
    catalog_<model>-Dark_<snap:03d>_mstar<mstar_min>_ndm<ndm_min>.hdf5

Datasets:
    halo_ids (DMO FoF IDs), M200c (Msun), R200c (ckpc), N_dm
attrs:
    metadata: model (e.g. CDM-Dark), snap, redshift, h, source_catalog, created_at
    cuts:     inherited from the FP catalog (mstar_min, mstar_max, n_dm_min)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import temet

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat, save_flat
from galaxy_sidm.data.aida_tng import match_fp_to_dmo_fof


def build(model: str, snap: int, fp_first_sub: np.ndarray, h: float):
    """Return (arrays_dict, metadata) for the DMO catalog matched to FP.

    Args:
        model: FP model name (e.g. ``CDM``, ``SIDM1``, ``vSIDM``). The DMO
            catalog is tagged as ``<model>-Dark``.
        snap: Snapshot number.
        fp_first_sub: Array of FP central subhalo IDs (``GroupFirstSub``)
            to match into the DMO run.
        h: Dimensionless Hubble parameter, copied into metadata.

    Returns:
        Tuple ``(arrays, metadata)`` ready for ``save_flat``. ``arrays``
        contains ``halo_ids`` (DMO FoF), ``M200c`` (Msun), ``R200c``
        (ckpc), ``N_dm``.
    """
    sim_fp = temet.sim(run="aida", variant=model, res=1080, snap=snap)
    sim_dmo = temet.sim(run="aida_dm", variant=model, res=1080, snap=snap)

    valid_fs = fp_first_sub[fp_first_sub >= 0]
    dmo_fof_ids = match_fp_to_dmo_fof(sim_fp, sim_dmo, valid_fs, snap)

    gc = sim_dmo.groupCat(fieldsHalos=[
        "Group_M_Crit200", "Group_R_Crit200", "GroupLenType",
    ])
    M200c = sim_dmo.units.codeMassToMsun(gc["Group_M_Crit200"])
    R200c = sim_dmo.units.codeLengthToComovingKpc(gc["Group_R_Crit200"])
    N_dm = gc["GroupLenType"][:, 1]

    in_range = (dmo_fof_ids >= 0) & (dmo_fof_ids < len(M200c))
    halo_ids = dmo_fof_ids[in_range].astype(np.int64)

    arrays = {
        "halo_ids": halo_ids,
        "M200c":    M200c[halo_ids].astype(np.float64),
        "R200c":    R200c[halo_ids].astype(np.float64),
        "N_dm":     N_dm[halo_ids].astype(np.int64),
    }
    metadata = {
        "model":    f"{model}-Dark",
        "snap":     int(snap),
        "redshift": float(sim_dmo.redshift),
        "h":        float(h),
    }
    return arrays, metadata


def output_name(model: str, snap: int, mstar_min: float, n_dm_min: int,
                disc_only: bool = False) -> str:
    disc_tag = "_disc" if disc_only else ""
    return (f"catalog_{model}-Dark_{snap:03d}"
            f"_mstar{mstar_min:.0e}_ndm{n_dm_min}{disc_tag}.hdf5")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="Path to config/scripts.yaml (defaults to project root)")
    p.add_argument("--fp-catalog", type=Path, required=True,
                   help="catalog_<model>_<snap>_*.hdf5 produced by build_catalog.py")
    p.add_argument("--model", default=None,
                   help="Sanity-check model name (must match the FP catalog metadata)")
    p.add_argument("--out-root", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.out_root or cfg["paths"]["scratch_processed"])

    fp_arrays, fp_attrs = load_flat(args.fp_catalog)
    fp_meta = fp_attrs["metadata"]
    fp_cuts = fp_attrs["cuts"]
    model = str(fp_meta["model"])
    snap = int(fp_meta["snap"])

    if args.model is not None and args.model != model:
        sys.exit(f"--model {args.model!r} does not match FP catalog model {model!r}")
    if model.endswith("-Dark"):
        sys.exit(f"FP catalog is already DMO ({model}); pass an FP catalog instead.")

    if "GroupFirstSub" not in fp_arrays:
        sys.exit("FP catalog is missing 'GroupFirstSub' — cannot match to DMO.")

    arrays, metadata = build(
        model=model, snap=snap,
        fp_first_sub=fp_arrays["GroupFirstSub"],
        h=cfg["cosmology"]["h"],
    )
    metadata["source_catalog"] = str(args.fp_catalog)

    mstar_min = float(fp_cuts.get("mstar_min", cfg["defaults"]["mstar_min"]))
    n_dm_min = int(fp_cuts.get("n_dm_min", cfg["defaults"]["n_dm_min"]))

    disc_only = bool(fp_cuts.get("disc_only", False))
    out_path = out_root / "catalogs" / output_name(
        model, snap, mstar_min, n_dm_min, disc_only=disc_only)
    save_flat(out_path, arrays, cuts=dict(fp_cuts), metadata=metadata)
    print(f"[build_catalog_dmo] {model}-Dark snap {snap}: "
          f"{len(arrays['halo_ids'])} halos -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
