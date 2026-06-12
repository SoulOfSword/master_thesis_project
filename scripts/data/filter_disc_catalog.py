"""Filter an FP halo catalog down to the MORDOR-disc subset.

Takes an FP catalog HDF5 (from `build_catalog.py`) and the corresponding
MORDOR sample HDF5 (from `build_mordor_sample.py`), keeps only catalog
rows whose halo_id appears in the MORDOR sample with ``IsDisc == 1``,
and writes a new catalog tagged ``_disc`` in the filename plus
``disc_only: True`` in its ``cuts`` attrs.

Downstream scripts (``build_catalog_dmo.py``, ``compute_profiles.py``,
``compute_gamma.py``, ``compute_rcore.py``) inspect ``cuts['disc_only']``
and propagate the ``_disc`` tag into their own output filenames so disc
and all-galaxy outputs coexist without clobbering each other.

Output (under cfg['paths']['scratch_processed']/catalogs/):
    catalog_<model>_<snap:03d>_mstar<mstar_min>_ndm<ndm_min>_disc.hdf5

attrs:
    metadata: model, snap, redshift, h, source_catalog, source_mordor_sample
    cuts:     mstar_min, mstar_max, n_dm_min, disc_only=True

Usage:
    python scripts/data/filter_disc_catalog.py \\
        --catalog $SCRATCH/.../catalogs/catalog_CDM_067_mstar1e+08_ndm0.hdf5 \\
        --mordor-sample $SCRATCH/.../samples/mordor_sample_CDM_067.hdf5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat, save_flat


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--catalog", type=Path, required=True,
                   help="FP catalog_*.hdf5 produced by build_catalog.py")
    p.add_argument("--mordor-sample", type=Path, required=True,
                   help="mordor_sample_*.hdf5 produced by build_mordor_sample.py")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Override processed-data root; default from config.")
    args = p.parse_args()

    cfg = load_config(args.config)

    cat_arrays, cat_attrs = load_flat(args.catalog)
    meta = dict(cat_attrs["metadata"])
    cuts = dict(cat_attrs["cuts"])
    model = str(meta["model"])
    snap = int(meta["snap"])

    if model.endswith("-Dark"):
        sys.exit(f"FP catalog expected; got DMO catalog ({model}). "
                 "Disc filter applies to FP only.")

    md_arrays, _ = load_flat(args.mordor_sample)
    if "IsDisc" not in md_arrays:
        sys.exit(f"MORDOR sample missing IsDisc column: {args.mordor_sample}")
    # MORDOR's `halo_ids` column is misleadingly named: those values are
    # central SUBHALO IDs (from each Gal_<sid>.hdf5 file), NOT FoF group
    # IDs. Match them against the catalog's GroupFirstSub instead.
    md_subhalo_ids = np.asarray(md_arrays["halo_ids"], dtype=np.int64)
    is_disc = np.asarray(md_arrays["IsDisc"]).astype(bool)
    disc_subhalos = set(int(s) for s in md_subhalo_ids[is_disc])

    if "GroupFirstSub" not in cat_arrays:
        sys.exit("FP catalog missing GroupFirstSub — cannot match MORDOR subhalo IDs.")
    fp_first_sub = np.asarray(cat_arrays["GroupFirstSub"], dtype=np.int64)
    keep = np.fromiter((int(s) in disc_subhalos for s in fp_first_sub),
                       dtype=bool, count=len(fp_first_sub))

    if keep.sum() == 0:
        print(f"[filter_disc_catalog] {model} snap {snap}: 0 disc galaxies "
              f"after filter (catalog had {len(cat_halo_ids)}, mordor sample "
              f"had {len(md_halo_ids)} with {int(is_disc.sum())} disc)")
        return 0

    filtered = {k: np.asarray(v)[keep] for k, v in cat_arrays.items()}
    cuts["disc_only"] = True
    meta["source_catalog"] = str(args.catalog)
    meta["source_mordor_sample"] = str(args.mordor_sample)

    out_root = Path(args.out_root or cfg["paths"]["scratch_processed"])
    mstar_min = float(cuts.get("mstar_min", cfg["defaults"]["mstar_min"]))
    n_dm_min = int(cuts.get("n_dm_min", cfg["defaults"]["n_dm_min"]))
    name = (f"catalog_{model}_{snap:03d}"
            f"_mstar{mstar_min:.0e}_ndm{n_dm_min}_disc.hdf5")
    out_path = out_root / "catalogs" / name

    save_flat(out_path, filtered, cuts=cuts, metadata=meta)
    print(f"[filter_disc_catalog] {model} snap {snap}: "
          f"{keep.sum()}/{len(cat_halo_ids)} halos are MORDOR discs "
          f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
