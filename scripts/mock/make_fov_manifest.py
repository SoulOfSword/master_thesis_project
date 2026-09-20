"""Manifest of the galaxies whose emission reaches the cube edge, with a larger FOV.

For every galaxy of a manifest, reads its info.json: if the last final BBarolo fit
has extent_at_edge on either side, the galaxy goes into the output with --factor
times its current cube half-width (cube_half_kpc, or from cube_npix for cubes built
before that key existed), up to --max-half-kpc. Galaxies already at the cap are
listed but not written. Output lines are "model snap subID half_kpc", which
batch.sbatch passes to build_galaxy.py --half-kpc.

Rerun cube and both BBarolo stages for them, then run this again on the output:
the ones still at the edge get the next, larger cube.

Usage:
    python scripts/mock/make_fov_manifest.py                  # <martini>/manifest.txt -> <martini>/manifest_fov.txt
    python scripts/mock/make_fov_manifest.py --manifest <martini>/manifest_fov.txt --out <martini>/manifest_fov2.txt
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import astropy.units as U

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config
from galaxy_sidm.mock import CubeParams


def current_half_kpc(info, px_kpc):
    """FOV half-width of the galaxy's current cube, kpc."""
    if info.get("cube_half_kpc"):
        return float(info["cube_half_kpc"])
    return (int(info["cube_npix"]) - 4) / 2 * px_kpc   # npix = 2 ceil(half / px) + 4


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None,
                   help="input manifest (default <martini>/manifest.txt)")
    p.add_argument("--out", type=Path, default=None,
                   help="output manifest (default <martini>/manifest_fov.txt)")
    p.add_argument("--factor", type=float, default=2.0,
                   help="new half-width = factor x current half-width (default 2)")
    p.add_argument("--max-half-kpc", type=float, default=80.0,
                   help="largest half-width [kpc] a cube may get (default 80)")
    p.add_argument("--chunk", type=int, default=5,
                   help="galaxies per array task, for the printed sbatch command (default 5)")
    args = p.parse_args()

    cfg = load_config(args.config)
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    mart = Path(cfg["paths"]["scratch_processed"]).parent / "martini"
    manifest = args.manifest or mart / "manifest.txt"
    out = args.out or mart / "manifest_fov.txt"
    cp = CubeParams()
    px_kpc = (cp.px_size * cp.distance).to_value(U.kpc, equivalencies=U.dimensionless_angles())

    lines, grown, capped, no_fit = [], Counter(), [], 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        model, snap, sub = line.split()[:3]
        z = snap_z[int(snap)]
        info = json.loads((mart / f"z{z:g}" / model / f"gal_{int(sub):06d}" / "info.json").read_text())
        edge = info.get("extent_at_edge")
        if edge is None:
            no_fit += 1
            continue
        if not any(edge):
            continue
        half = current_half_kpc(info, px_kpc)
        if half >= args.max_half_kpc:
            capped.append(f"{model} {snap} {sub} ({half:.1f} kpc)")
            continue
        lines.append(f"{model} {snap} {sub} {min(args.factor * half, args.max_half_kpc):.2f}")
        grown[f"z{z:g}"] += 1

    out.write_text("".join(l + "\n" for l in lines))
    print(f"read {manifest}: {sum(grown.values())} galaxies reach the cube edge and get a larger cube, "
          f"{len(capped)} already at {args.max_half_kpc:g} kpc, {no_fit} without a final fit")
    print("  by redshift:", dict(sorted(grown.items())))
    if capped:
        print("  at the cap (not written):", "; ".join(capped))
    print(f"wrote {out}")
    if lines:
        # Habrok sets SBATCH_EXPORT=NONE: without --export=ALL the job never sees
        # CHUNK/MANIFEST/STAGES and silently runs batch.sbatch's defaults.
        # A command-line --array replaces the script's %100 limit, so it is repeated here.
        print(f"submit: CHUNK={args.chunk} MANIFEST={out} STAGES=cube,barolo_3rings,barolo "
              f"sbatch --export=ALL --array=0-{math.ceil(len(lines) / args.chunk) - 1}%100 "
              f"scripts/mock/batch.sbatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
