"""Density profiles of the final sample, from each galaxy's own particles.

For every galaxy of the final sample (mock.tables.final_sample, built at run
time), the spherical density of its SUBFIND-bound dark matter, stars (no wind
particles) and gas around SubhaloPos (observables.measure_density_profile: bin 0
is the sphere inside the first edge, then shells). Physical kpc, Msun/kpc^3.

Output, one file per (model, snap), under cfg['paths']['scratch_processed']/profiles/:
    profiles_<model>_<snap:03d>.hdf5
Datasets: halo_ids (subhalo IDs), r_edges, rho_dm, rho_stars, rho_gas

Usage:
    python scripts/data/compute_profiles.py --ncpu 16
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, save_flat
from galaxy_sidm.data.potential import COMPONENTS, load_subhalo_particles, read_softening
from galaxy_sidm.mock.tables import final_sample
from galaxy_sidm.observables import measure_density_profile
from compute_vcirc import base_path

PROFILE_COMPONENTS = ("dm", "stars", "gas")


def one_galaxy(task):
    """Density profiles of one galaxy as a dict (with 'error' set if it failed)."""
    base, snap, sub, soft, edges = task
    try:
        p = load_subhalo_particles(base, snap, sub, soft)
        r = np.linalg.norm(p["xyz"], axis=1)
        out = dict(sub=sub, error=None)
        for comp in PROFILE_COMPONENTS:
            sel = p["comp"] == COMPONENTS.index(comp)
            out[comp] = measure_density_profile(r[sel], p["mass"][sel], edges)[1]
        return out
    except Exception as e:
        return dict(sub=sub, error=f"{type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--ncpu", type=int, default=1, help="galaxies processed in parallel")
    p.add_argument("--r-min", type=float, default=0.1, help="first bin edge [kpc] (default 0.1)")
    p.add_argument("--r-max", type=float, default=50.0, help="last bin edge [kpc] (default 50)")
    p.add_argument("--n-edges", type=int, default=40, help="log-spaced bin edges (default 40)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="default: <scratch_processed>/profiles")
    args = p.parse_args()

    cfg = load_config(args.config)
    h = float(cfg["cosmology"]["h"])
    outdir = args.outdir or Path(cfg["paths"]["scratch_processed"]) / "profiles"
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    edges = np.logspace(np.log10(args.r_min), np.log10(args.r_max), args.n_edges)

    groups, line = final_sample(cfg)
    print(f"[compute_profiles] {sum(map(len, groups.values()))} galaxies: final sample ({line})",
          flush=True)

    for (model, snap), subs in sorted(groups.items()):
        base = base_path(cfg, model)
        soft = read_softening(base, snap)
        if abs(soft["hubble"] - h) > 1e-6:
            raise ValueError(f"{model} snap {snap}: HubbleParam {soft['hubble']} in the snapshot, "
                             f"h = {h} in the config")
        print(f"[compute_profiles] {model} snap {snap}: {len(subs)} galaxies", flush=True)
        with Pool(args.ncpu) as pool:
            results = pool.map(one_galaxy, [(base, snap, s, soft, edges) for s in subs], chunksize=1)
        for r in results:
            if r["error"]:
                print(f"[compute_profiles] {model} snap {snap} sub {r['sub']} FAILED: {r['error']}",
                      flush=True)
        ok = [r for r in results if r["error"] is None]
        if not ok:
            continue
        out = outdir / f"profiles_{model}_{snap:03d}.hdf5"
        arrays = {"halo_ids": np.array([r["sub"] for r in ok], dtype=np.int64), "r_edges": edges}
        arrays.update({f"rho_{c}": np.vstack([r[c] for r in ok]) for c in PROFILE_COMPONENTS})
        save_flat(out, arrays,
                  metadata={"model": model, "snap": snap, "redshift": snap_z[snap], "sample": line},
                  variants={"r_min_kpc": args.r_min, "r_max_kpc": args.r_max, "n_edges": args.n_edges})
        print(f"[compute_profiles] wrote {out}  ({len(ok)} galaxies, "
              f"{len(results) - len(ok)} failed)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
