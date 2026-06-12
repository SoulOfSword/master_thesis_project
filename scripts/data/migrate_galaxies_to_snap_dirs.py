"""Migrate flat per-galaxy HDF5s into snapshot-aware subdirectories.

Historically `extract_galaxies.py` wrote `<out_root>/<model>/Gal_<id>.hdf5`
with no snapshot in the path, so a subhalo id reused across snapshots
collided on one file. The pipeline is now snapshot-aware
(`<out_root>/<model>/snap_<NNN>/Gal_<id>.hdf5`); this moves each existing
flat file into the subdir matching its own `Header/Redshift`, so the
already-extracted galaxies are reused instead of re-extracted.

A file lands under snap S iff |z_disk - z(S)| is minimal and < --z-tol.
Files with no matching snapshot are left in place and reported.

Usage:
    python scripts/data/migrate_galaxies_to_snap_dirs.py            # dry-run
    python scripts/data/migrate_galaxies_to_snap_dirs.py --apply
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config


def gal_header_redshift(gal_path: Path):
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
    p.add_argument("--models", nargs="*", default=None,
                   help="Models to migrate (default: cfg['models'])")
    p.add_argument("--z-tol", type=float, default=0.05,
                   help="Max |z_disk - z_snap| to assign a file to a snap")
    p.add_argument("--apply", action="store_true",
                   help="Move files (default: dry-run, report only)")
    args = p.parse_args()

    cfg = load_config(args.config)
    scratch_mordor = Path(cfg["paths"]["scratch_mordor"])
    models = args.models or list(cfg["models"])
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    snaps = np.array(sorted(snap_z))
    zs = np.array([snap_z[s] for s in snaps])

    total_moved = 0
    for model in models:
        gal_dir = scratch_mordor / model
        if not gal_dir.is_dir():
            print(f"  {model}: no dir {gal_dir}")
            continue
        flat = sorted(gal_dir.glob("Gal_*.hdf5"))  # direct children only
        per_snap = Counter()
        unmatched = []
        for gal in flat:
            z_disk = gal_header_redshift(gal)
            if z_disk is None:
                unmatched.append((gal.name, "unreadable"))
                continue
            j = int(np.argmin(np.abs(zs - z_disk)))
            if abs(zs[j] - z_disk) > args.z_tol:
                unmatched.append((gal.name, f"z={z_disk:.3f}"))
                continue
            snap = int(snaps[j])
            per_snap[snap] += 1
            if args.apply:
                dest_dir = gal_dir / f"snap_{snap:03d}"
                dest_dir.mkdir(parents=True, exist_ok=True)
                os.rename(gal, dest_dir / gal.name)

        n = sum(per_snap.values())
        total_moved += n
        layout = ", ".join(f"snap_{s:03d}:{per_snap[s]}"
                           for s in sorted(per_snap))
        print(f"  {model}: {len(flat)} flat files -> {n} placed  [{layout}]")
        if unmatched:
            print(f"    {len(unmatched)} unmatched (left in place): "
                  f"{unmatched[:5]}{' ...' if len(unmatched) > 5 else ''}")

    mode = "APPLIED" if args.apply else "DRY-RUN (no files moved)"
    print(f"\n[migrate_galaxies_to_snap_dirs] {mode}: {total_moved} files")
    if not args.apply and total_moved:
        print("Re-run with --apply to move the files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
