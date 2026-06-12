"""Extract per-galaxy HDF5 files for AIDA-TNG centrals.

For one DM model + snapshot, builds the central-subhalo catalog, filters
by stellar particle count, and writes one Gadget-format HDF5 file per
qualifying central to `<out_root>/<model>/snap_<NNN>/Gal_<subhalo_id>.hdf5`.

The HDF5s are inputs to MORDOR (via `scripts/run_mordor.py`).

Parallel via ProcessPoolExecutor with simple-arg workers (no temet sim
pickling). Idempotent: skips already-extracted files unless --overwrite.

Usage:
    python scripts/extract_galaxies.py --model CDM --snap 67
    python scripts/extract_galaxies.py --model SIDM1 --snap 99 \
        --n-workers 16 --n-star-min 1e4
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

import temet
from galaxy_sidm.data.aida_tng import (
    build_central_subhalo_catalog, qualifying_central_ids,
)
from galaxy_sidm.morphology import extract_galaxy_hdf5


DEFAULT_OUT_ROOT = Path(os.environ.get(
    "SCRATCH", "/leonardo_scratch/large/userexternal/acosta01"
)) / "master_thesis_project" / "data" / "mordor_galaxies"


def _worker(args):
    base_path, snap, sub_id, out_path, h, soft, overwrite = args
    try:
        extract_galaxy_hdf5(
            base_path=base_path, snap=snap, subhalo_id=sub_id,
            out_path=out_path, h=h, soft_phys_kpc=soft, overwrite=overwrite,
        )
        return sub_id, True, ""
    except Exception as e:
        return sub_id, False, repr(e)


def expected_paths(out_root, model, snap, sub_ids):
    out_dir = Path(out_root) / model / f"snap_{int(snap):03d}"
    return [out_dir / f"Gal_{int(s):06d}.hdf5" for s in sub_ids]


def missing_subhalo_ids(out_root, model, snap, sub_ids):
    """Subset of sub_ids whose Gal_<id>.hdf5 is not yet on disk."""
    out_dir = Path(out_root) / model / f"snap_{int(snap):03d}"
    return [int(s) for s in sub_ids
            if not (out_dir / f"Gal_{int(s):06d}.hdf5").exists()]


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-galaxy HDF5 files for AIDA-TNG centrals.",
    )
    parser.add_argument("--model", required=True,
                        choices=["CDM", "SIDM1", "vSIDM", "WDM3", "WDM5"])
    parser.add_argument("--snap", type=int, required=True)
    parser.add_argument("--res", type=int, default=1080,
                        help="AIDA resolution (default 1080)")
    parser.add_argument("--n-star-min", type=float, default=1e4,
                        help="Min subhalo stellar particle count (default 1e4)")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                        help=f"Output root (default: {DEFAULT_OUT_ROOT})")
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--soft-phys-kpc", type=float, default=0.57,
                        help="Plummer-equivalent softening (AIDA 50/A: 0.57)")
    parser.add_argument("-h-cosmo", "--h-cosmo", type=float, default=0.6774,
                        help="Hubble parameter h (default 0.6774)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if output exists")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N qualifying subhalos")
    parser.add_argument("--base-path", type=Path, default=None,
                        help="Override snapshot basePath (e.g. SCRATCH shadow "
                             "tree when snapshot data is not in $WORK)")
    args = parser.parse_args()

    sim = temet.sim(run="aida", variant=args.model,
                    res=args.res, snap=args.snap)
    cat = build_central_subhalo_catalog(sim)
    if args.base_path is not None:
        cat["basePath"] = str(args.base_path).rstrip("/") + "/"
        print(f"[extract_galaxies] basePath overridden -> {cat['basePath']}")
    sub_ids = qualifying_central_ids(cat, n_star_min=args.n_star_min)
    if args.limit is not None:
        sub_ids = sub_ids[:args.limit]
    if len(sub_ids) == 0:
        print(f"No centrals with N_star >= {args.n_star_min:.0f}; nothing to do.")
        return 0

    out_dir = Path(args.out_root) / args.model / f"snap_{args.snap:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        todo = list(sub_ids)
    else:
        todo = missing_subhalo_ids(args.out_root, args.model, args.snap, sub_ids)
    if not todo:
        print(f"All {len(sub_ids)} HDF5s already on disk under {out_dir}/")
        return 0

    print(f"{args.model} snap {args.snap} z={cat['redshift']:.3f}")
    print(f"qualifying centrals: {len(sub_ids)}  to extract: {len(todo)}  "
          f"workers: {args.n_workers}  out: {out_dir}/")

    tasks = [
        (cat["basePath"], int(cat["snap"]), int(s),
         out_dir / f"Gal_{int(s):06d}.hdf5",
         float(args.h_cosmo), float(args.soft_phys_kpc), args.overwrite)
        for s in todo
    ]

    t0 = time.time()
    n_ok = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for sid, ok, err in tqdm(pool.map(_worker, tasks),
                                  total=len(tasks),
                                  desc=f"extract {args.model}",
                                  unit="gal"):
            if ok:
                n_ok += 1
            else:
                failures.append((sid, err))

    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min  "
          f"({n_ok}/{len(tasks)} ok, {len(failures)} failed)")
    if failures:
        print("First 10 failures:")
        for sid, err in failures[:10]:
            print(f"  subhalo {sid}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
