"""Manifest of the galaxies whose emission reaches the cube edge, with a larger FOV.

For every galaxy of a manifest, walks along the major axis of its 3-ring BBarolo
fit's data moment-1 map (barolo.emission_at_border): if the emission comes within
one beam (6 px) of the border on either side, the cube is too small (the beam
margin is there because BBarolo's mask can stop a few pixels short of a border that
cuts through the gas; see emission_at_border), and the galaxy goes into the
output with --factor times its current cube half-width (cube_half_kpc, or from
cube_npix for cubes built before that key existed), up to --max-half-kpc.
Galaxies already at the cap are listed but not written. Galaxies without a 3-ring
fit (no moment-1 map) are listed too: their barolo_3rings stage has to be rerun.
Output lines are "model snap subID half_kpc", which batch.sbatch passes to
build_galaxy.py --half-kpc.

Submit the printed command: it rebuilds the cube and the 3-ring fit with
GROW_FOV=1, so build_galaxy.py --grow-fov keeps doubling each galaxy's cube (up
to the cap) until its emission no longer reaches the border. No further rounds
of this script are needed.

Usage:
    python scripts/mock/make_fov_manifest.py        # <martini>/manifest.txt -> <martini>/manifest_fov_grow.txt
    python scripts/mock/make_fov_manifest.py --manifest <martini>/manifest_fov_grow.txt --out <martini>/check.txt
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
from galaxy_sidm.mock.barolo import emission_at_border


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
                   help="output manifest (default <martini>/manifest_fov_grow.txt)")
    p.add_argument("--factor", type=float, default=2.0,
                   help="new half-width = factor x current half-width (default 2)")
    p.add_argument("--max-half-kpc", type=float, default=80.0,
                   help="largest half-width [kpc] a cube may get (default 80)")
    p.add_argument("--chunk", type=int, default=1,
                   help="galaxies per array task, for the printed sbatch command (default 1)")
    args = p.parse_args()

    cfg = load_config(args.config)
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    mart = Path(cfg["paths"]["scratch_processed"]).parent / "martini"
    manifest = args.manifest or mart / "manifest.txt"
    out = args.out or mart / "manifest_fov_grow.txt"
    cp = CubeParams()
    px_kpc = (cp.px_size * cp.distance).to_value(U.kpc, equivalencies=U.dimensionless_angles())
    margin_px = int(round((cp.beam_fwhm / cp.px_size).decompose().value))   # one beam, in pixels

    lines, grown, capped, no_maps = [], Counter(), [], []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        model, snap, sub = line.split()[:3]
        z = snap_z[int(snap)]
        gal = mart / f"z{z:g}" / model / f"gal_{int(sub):06d}"
        try:
            edge = emission_at_border(gal / "bbarolo_3rings", margin_px)
        except FileNotFoundError:
            no_maps.append(f"{model} {snap} {sub}")
            continue
        if not any(edge):
            continue
        half = current_half_kpc(json.loads((gal / "info.json").read_text()), px_kpc)
        if half >= args.max_half_kpc:
            capped.append(f"{model} {snap} {sub} ({half:.1f} kpc)")
            continue
        lines.append(f"{model} {snap} {sub} {min(args.factor * half, args.max_half_kpc):.2f}")
        grown[f"z{z:g}"] += 1

    out.write_text("".join(l + "\n" for l in lines))
    print(f"read {manifest}: {sum(grown.values())} galaxies have emission within {margin_px} px (one beam) of the "
          f"cube border along the major axis and get a larger cube, {len(capped)} already at {args.max_half_kpc:g} kpc")
    print("  by redshift:", dict(sorted(grown.items())))
    if capped:
        print("  at the cap (not written):", "; ".join(capped))
    if no_maps:
        print(f"  {len(no_maps)} without a 3-ring moment-1 map (rerun their barolo_3rings stage, not written):",
              "; ".join(no_maps))
    print(f"wrote {out}")
    if lines:
        # Habrok sets SBATCH_EXPORT=NONE: without --export=ALL the job never sees
        # CHUNK/MANIFEST/STAGES/GROW_FOV and silently runs batch.sbatch's defaults.
        # A command-line --array replaces the script's %100 limit, so it is repeated here.
        print(f"submit: CHUNK={args.chunk} MANIFEST={out} STAGES=cube,barolo_3rings GROW_FOV=1 "
              f"sbatch --export=ALL --array=0-{math.ceil(len(lines) / args.chunk) - 1}%100 "
              f"--mem=64G --time=08:00:00 scripts/mock/batch.sbatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
