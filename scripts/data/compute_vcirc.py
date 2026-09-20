"""Circular-velocity profiles in the disc plane from the gravitational potential.

For each galaxy in a list, v_circ(R) = sqrt(R dPhi/dR), with Phi the snapshot's
Potential at all the subhalo's particles (gas, DM, stars, black holes), averaged
azimuthally in cylindrical rings inside a thin slab |z| < --slab about the disc
plane. Centre and disc plane are the ones the mock cubes use: SubhaloPos and the
angular momentum of the inner neutral gas (load_galaxy_gas). Rings are --dr wide,
or --dr-frac x R where that is wider, so the sparse outer rings still hold enough
particles for a stable derivative.

Meant for the final sample only (discs in both gas and stars), given as a text
file with one "model snap subID" per line (the mock manifest format).

Output, one file per (model, snap), under cfg['paths']['scratch_processed']/vcirc/
(overwritten on every run):
    vcirc_<model>_<snap:03d>.hdf5
Datasets, one row per galaxy:
    halo_ids
    R_half_star     3-D stellar half-mass radius (SubhaloHalfmassRadType), kpc
    vcirc_5rhalf    v_circ at 5 R_half_star, km/s
    vcirc_10rhalf   v_circ at 10 R_half_star, km/s
    R               (n_gal, n_rings) mean radius of each ring, kpc
    vcirc           (n_gal, n_rings) km/s
    count           (n_gal, n_rings) particles per ring inside the slab
attrs:
    metadata: model, snap, redshift, galaxies_file
    variants: dr_kpc, dr_frac, rmax_kpc, slab_half_kpc, n_azimuth, min_count

Usage:
    python scripts/data/compute_vcirc.py --galaxies discs.txt --ncpu 8
"""

import argparse
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import illustris_python as il

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, save_flat
from galaxy_sidm.data.potential import load_subhalo_potential
from galaxy_sidm.mock import load_galaxy_gas
from galaxy_sidm.observables import vcirc_disc_plane


def base_path(cfg, model):
    # AIDA stores vSIDM under L35n1080_vSIDM_correa, the others as L35n1080_<model>
    run = "L35n1080_vSIDM_correa" if model == "vSIDM" else f"L35n1080_{model}"
    return str(Path(cfg["paths"]["aida_root"]) / run / "output") + "/"


def ring_edges(dr, dr_frac, rmax):
    """Ring edges from 0 to rmax; each ring max(dr, dr_frac x its inner radius) wide."""
    edges = [0.0]
    while edges[-1] < rmax:
        edges.append(edges[-1] + max(dr, dr_frac * edges[-1]))
    return np.array(edges)


def _at(R, v, r):
    """v_circ interpolated at radius r (nan outside the measured range)."""
    good = np.isfinite(R) & np.isfinite(v)
    if good.sum() < 2:
        return float("nan")
    return float(np.interp(r, R[good], v[good], left=np.nan, right=np.nan))


def one_galaxy(task):
    """v_circ profile of one galaxy as a dict (with 'error' set if it failed)."""
    model, snap, sub, base, h, r_edges, slab, n_azimuth, min_count = task
    try:
        gas = load_galaxy_gas(base, snap, sub, h=h)       # disc plane as in the cubes
        xyz, phi = load_subhalo_potential(base, snap, sub, h=h)
        R, v, count = vcirc_disc_plane(xyz, phi, gas.L_hat, r_edges, slab_half=slab,
                                       n_azimuth=n_azimuth, min_count=min_count)
        row = il.groupcat.loadSingle(base, snap, subhaloID=sub)
        r_half = float(row["SubhaloHalfmassRadType"][4]) * gas.a / h
        return dict(sub=sub, R=R, vcirc=v, count=count, R_half_star=r_half,
                    vcirc_5rhalf=_at(R, v, 5 * r_half), vcirc_10rhalf=_at(R, v, 10 * r_half),
                    redshift=gas.redshift, error=None)
    except Exception as e:
        return dict(sub=sub, error=f"{type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--galaxies", type=Path, required=True,
                   help='text file with one "model snap subID" per line')
    p.add_argument("--ncpu", type=int, default=1, help="galaxies processed in parallel")
    p.add_argument("--dr", type=float, default=0.25,
                   help="smallest ring width [kpc] (default 0.25)")
    p.add_argument("--dr-frac", type=float, default=0.1,
                   help="ring width as a fraction of R, where wider than --dr (default 0.1)")
    p.add_argument("--rmax", type=float, default=40.0,
                   help="outer edge of the last ring [kpc] (default 40)")
    p.add_argument("--slab", type=float, default=0.3,
                   help="slab half-thickness, |z| < slab [kpc] (default 0.3)")
    p.add_argument("--n-azimuth", type=int, default=12,
                   help="azimuthal sectors per ring (default 12)")
    p.add_argument("--min-count", type=int, default=10,
                   help="particles a ring needs inside the slab (default 10)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="default: <scratch_processed>/vcirc")
    args = p.parse_args()

    cfg = load_config(args.config)
    h = float(cfg["cosmology"]["h"])
    outdir = args.outdir or Path(cfg["paths"]["scratch_processed"]) / "vcirc"
    r_edges = ring_edges(args.dr, args.dr_frac, args.rmax)

    groups = defaultdict(list)
    for line in args.galaxies.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            model, snap, sub = line.split()[:3]
            groups[(model, int(snap))].append(int(sub))

    for (model, snap), subs in sorted(groups.items()):
        base = base_path(cfg, model)
        tasks = [(model, snap, s, base, h, r_edges, args.slab, args.n_azimuth, args.min_count)
                 for s in subs]
        print(f"[compute_vcirc] {model} snap {snap}: {len(tasks)} galaxies", flush=True)
        with Pool(args.ncpu) as pool:
            results = pool.map(one_galaxy, tasks, chunksize=1)
        for r in results:
            if r["error"]:
                print(f"[compute_vcirc] {model} snap {snap} sub {r['sub']} FAILED: {r['error']}",
                      flush=True)
        ok = [r for r in results if r["error"] is None]
        if not ok:
            continue
        out = outdir / f"vcirc_{model}_{snap:03d}.hdf5"
        save_flat(out, {
            "halo_ids": np.array([r["sub"] for r in ok], dtype=np.int64),
            "R_half_star": np.array([r["R_half_star"] for r in ok]),
            "vcirc_5rhalf": np.array([r["vcirc_5rhalf"] for r in ok]),
            "vcirc_10rhalf": np.array([r["vcirc_10rhalf"] for r in ok]),
            "R": np.vstack([r["R"] for r in ok]),
            "vcirc": np.vstack([r["vcirc"] for r in ok]),
            "count": np.vstack([r["count"] for r in ok]),
        }, metadata={"model": model, "snap": snap, "redshift": ok[0]["redshift"],
                     "galaxies_file": args.galaxies},
           variants={"dr_kpc": args.dr, "dr_frac": args.dr_frac, "rmax_kpc": args.rmax,
                     "slab_half_kpc": args.slab,
                     "n_azimuth": args.n_azimuth, "min_count": args.min_count})
        print(f"[compute_vcirc] wrote {out}  ({len(ok)} galaxies, "
              f"{len(results) - len(ok)} failed)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
