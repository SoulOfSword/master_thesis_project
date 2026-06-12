"""Drop stale rows from MORDOR sample HDF5s.

A per-galaxy HDF5 (`Gal_<id>.hdf5`) was historically written to a flat
per-model directory with no snapshot in the path, and extraction skipped
files already on disk. Any subhalo id reused across snapshots therefore
shares one file, so the snapshot extracted *first* wins and the rest carry
that other snapshot's particles (and MORDOR result). The on-disk file's own
`Header/Redshift` is the ground truth: a sample row is stale iff its Gal
file redshift does not match the sample's snapshot redshift.

This is an interim cleanup so plots are correct before the snapshot-aware
re-extraction lands; rebuilt samples overwrite the masked ones.

Usage:
    python scripts/data/mask_stale_mordor_rows.py            # dry-run
    python scripts/data/mask_stale_mordor_rows.py --apply
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat, save_flat


def gal_header_redshift(gal_path: Path):
    """Header/Redshift of a per-galaxy HDF5, or None if unreadable."""
    if not gal_path.exists():
        return None
    try:
        with h5py.File(gal_path, "r") as f:
            return float(dict(f["Header"].attrs).get("Redshift", np.nan))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--snaps", type=int, nargs="*", default=None,
                   help="Snaps to scan (default: cfg['mosaic_snaps'])")
    p.add_argument("--models", nargs="*", default=None,
                   help="Models to scan (default: cfg['models'])")
    p.add_argument("--z-tol", type=float, default=0.05,
                   help="Max |z_disk - z_snap| to call a row correct")
    p.add_argument("--apply", action="store_true",
                   help="Rewrite samples (default: dry-run, report only)")
    args = p.parse_args()

    cfg = load_config(args.config)
    scratch_mordor = Path(cfg["paths"]["scratch_mordor"])
    sample_dir = scratch_mordor / "samples"
    snaps = args.snaps or list(cfg["mosaic_snaps"])
    models = args.models or list(cfg["models"])

    total_dropped = 0
    for model in models:
        gal_dir = scratch_mordor / model
        for snap in snaps:
            path = sample_dir / f"mordor_sample_{model}_{snap:03d}.hdf5"
            if not path.exists():
                continue
            arrays, attrs = load_flat(path)
            z_snap = float(attrs["metadata"]["redshift"])
            halo_ids = np.asarray(arrays["halo_ids"], dtype=np.int64)

            keep = np.ones(len(halo_ids), dtype=bool)
            stale_ids = []
            for i, hid in enumerate(halo_ids):
                z_disk = gal_header_redshift(gal_dir / f"Gal_{int(hid):06d}.hdf5")
                if z_disk is not None and abs(z_disk - z_snap) > args.z_tol:
                    keep[i] = False
                    stale_ids.append((int(hid), z_disk))

            n_drop = int((~keep).sum())
            if n_drop == 0:
                print(f"  {model} snap {snap:03d} (z={z_snap:.2f}): clean "
                      f"({len(halo_ids)} rows)")
                continue
            total_dropped += n_drop
            ids_str = ", ".join(f"{h}(z{z:.2f})" for h, z in stale_ids[:8])
            more = "" if len(stale_ids) <= 8 else f" +{len(stale_ids)-8} more"
            print(f"  {model} snap {snap:03d} (z={z_snap:.2f}): "
                  f"DROP {n_drop}/{len(halo_ids)} -> ids {ids_str}{more}")

            if args.apply:
                masked = {k: np.asarray(v)[keep] for k, v in arrays.items()}
                meta = dict(attrs["metadata"])
                meta["stale_rows_dropped"] = n_drop
                tmp = path.with_suffix(".tmp.hdf5")
                save_flat(tmp, masked, cuts=attrs["cuts"],
                          variants=attrs["variants"], metadata=meta)
                os.replace(tmp, path)

    mode = "APPLIED" if args.apply else "DRY-RUN (no files changed)"
    print(f"\n[mask_stale_mordor_rows] {mode}: {total_dropped} stale rows total")
    if not args.apply and total_dropped:
        print("Re-run with --apply to rewrite the samples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
