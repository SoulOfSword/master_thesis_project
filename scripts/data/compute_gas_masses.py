"""Gas masses of the final sample by five definitions, per galaxy and resolved in radius,
height and phase.

Every gas cell of the subhalo (its own SUBFIND-bound cells; no radius or height cut)
counts towards each definition with:
    all       its mass m
    sfr       m if the cell forms stars (SFR > 0)
    sh03      the 'sh03' cold gas of observables.measure_subhalo_cold_gas_all, per cell
              (observables.cold_gas_cells): x m for star-forming cells (x the SH03
              cold-cloud fraction, temet densToSH03TwoPhase), m for the others if
              T < 10^4.5 K, 0 otherwise
    hdec_gas  neutral gas f m, with f the neutral-hydrogen fraction of
              mock.load_galaxy_gas (Hdecompose, Rahmati+13, as in the mock cubes)
    hdec_H    neutral hydrogen f X_H m (load_galaxy_gas's mH_neutral_g; X_H the cell's
              hydrogen mass fraction, GFM_Metals[:, 0]); its total is info.json M_neutral
The cells for sh03 are read as measure_subhalo_cold_gas_all reads them and must be the same
cells as load_galaxy_gas's (count and masses), or the galaxy fails.

Geometry: centre SubhaloPos; the disc plane of mock.cube.disc_normal.

Output, one file per (model, snap): <scratch_processed>/gas_masses/gas_masses_<model>_<snap:03d>.hdf5
    halo_ids, n_cells, zhat (disc normal), R_rings_kpc (outer edge of the outermost BBarolo
    ring of the final fit, at the cube distance)
    M_<def>        total of each definition, Msun
    prof_r_<def>   mass in 3D-radius bins r_edges; r below the first edge counts in the first
                   bin, beyond the last edge only in the total
    map_Rz_<def>   mass in (R, |z|) bins R_edges x z_edges of the disc plane; values below the
                   first edges count in the first bins, beyond the last only in the total
    map_nT_<def>   mass in (log n_H, log T) bins lognH_edges x logT_edges, values outside
                   clipped into the edge bins; n_H and T of load_galaxy_gas
    r_edges, R_edges: 40 log bins 0.1-300 kpc; z_edges: 40 log bins 0.01-100 kpc;
    lognH_edges: -6 to 3 in 0.25 dex; logT_edges: 2 to 8 in 0.1 dex
attrs metadata: model, snap, redshift, definitions, galaxies_file (or the sample's dividing line)

Usage:
    python scripts/data/compute_gas_masses.py --ncpu 16                      # the final sample
    python scripts/data/compute_gas_masses.py --galaxies test.txt --ncpu 4   # a list of galaxies
"""

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import astropy.units as U
import illustris_python as il
import temet

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, save_flat
from galaxy_sidm.mock import CubeParams, disc_coords, disc_normal, load_galaxy_gas
from galaxy_sidm.mock.tables import final_sample
from galaxy_sidm.observables import COLD_GAS_FIELDS, cold_gas_cells
from compute_vcirc import base_path

DEFS = ("all", "sfr", "sh03", "hdec_gas", "hdec_H")
RADIUS_EDGES = np.logspace(-1, np.log10(300.0), 41)
Z_EDGES = np.logspace(-2, 2, 41)
LOGNH_EDGES = np.linspace(-6, 3, 37)
LOGT_EDGES = np.linspace(2, 8, 61)
SIM = None


def init_sim(model, snap):
    global SIM
    SIM = temet.sim(run="aida", variant=model, res=1080, snap=snap)


def one_galaxy(task):
    """Masses and maps of one galaxy as a dict (with 'error' set if it failed)."""
    model, snap, sub, base, h, info_json = task
    try:
        g = il.snapshot.loadSubhalo(base, snap, sub, "gas", fields=COLD_GAS_FIELDS)
        gas = load_galaxy_gas(base, snap, sub, h=h)
        cold = cold_gas_cells(g, h=h, sim=SIM)
        if len(cold["all"]) != len(gas.mgas_g) or not np.array_equal(cold["all"], gas.mgas_g.to_value(U.Msun)):
            raise ValueError("the sh03 read and load_galaxy_gas returned different cells")
        X_H = g["GFM_Metals"][:, 0].astype(np.float64)
        mH = gas.mH_neutral_g.to_value(U.Msun)
        cell = {"all": cold["all"], "sfr": cold["SFR>0"], "sh03": cold["sh03"],
                "hdec_gas": mH / X_H, "hdec_H": mH}
        totals = {d: cell[d].sum() for d in DEFS}

        info = json.loads(Path(info_json).read_text())
        xyz = gas.xyz_g.to_value(U.kpc)
        r = np.linalg.norm(xyz, axis=1)
        zhat = disc_normal(gas)
        z, R = disc_coords(xyz, zhat)
        lognH = np.clip(np.log10(gas.nH_g.to_value(U.cm ** -3)), LOGNH_EDGES[0], LOGNH_EDGES[-1])
        logT = np.clip(np.log10(gas.T_g.to_value(U.K)), LOGT_EDGES[0], LOGT_EDGES[-1])
        r, R = np.maximum(r, RADIUS_EDGES[0]), np.maximum(R, RADIUS_EDGES[0])
        absz = np.maximum(np.abs(z), Z_EDGES[0])
        ring_out = info["rings_arcsec"][-1] + info["ring_width_arcsec"] / 2

        out = dict(sub=sub, redshift=gas.redshift, n_cells=len(mH), zhat=zhat,
                   R_rings_kpc=np.radians(ring_out / 3600) * CubeParams().distance.to_value(U.kpc), error=None)
        for d in DEFS:
            out[f"M_{d}"] = float(totals[d])
            out[f"prof_r_{d}"] = np.histogram(r, bins=RADIUS_EDGES, weights=cell[d])[0].astype(np.float32)
            out[f"map_Rz_{d}"] = np.histogram2d(R, absz, bins=[RADIUS_EDGES, Z_EDGES],
                                                weights=cell[d])[0].astype(np.float32)
            out[f"map_nT_{d}"] = np.histogram2d(lognH, logT, bins=[LOGNH_EDGES, LOGT_EDGES],
                                                weights=cell[d])[0].astype(np.float32)
        return out
    except Exception as e:
        return dict(sub=sub, error=f"{type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--galaxies", type=Path, default=None,
                   help='text file with one "model snap subID" per line '
                        "(default: the final sample, built at run time)")
    p.add_argument("--ncpu", type=int, default=1, help="galaxies processed in parallel")
    p.add_argument("--outdir", type=Path, default=None,
                   help="default: <scratch_processed>/gas_masses")
    args = p.parse_args()

    cfg = load_config(args.config)
    h = float(cfg["cosmology"]["h"])
    outdir = args.outdir or Path(cfg["paths"]["scratch_processed"]) / "gas_masses"
    martini = Path(cfg["paths"]["scratch_processed"]).parent / "martini"
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}

    if args.galaxies:
        groups, source = defaultdict(list), str(args.galaxies)
        for line in args.galaxies.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                model, snap, sub = line.split()[:3]
                groups[(model, int(snap))].append(int(sub))
    else:
        groups, line = final_sample(cfg)
        source = f"final sample built at run time ({line})"
    print(f"[compute_gas_masses] {sum(map(len, groups.values()))} galaxies: {source}", flush=True)

    keys = ["n_cells", "zhat", "R_rings_kpc"] + [f"{p}_{d}" for p in ("M", "prof_r", "map_Rz", "map_nT")
                                                 for d in DEFS]
    for (model, snap), subs in sorted(groups.items()):
        base = base_path(cfg, model)
        tasks = [(model, snap, s, base, h,
                  martini / f"z{snap_z[snap]:g}" / model / f"gal_{s:06d}" / "info.json") for s in subs]
        print(f"[compute_gas_masses] {model} snap {snap}: {len(tasks)} galaxies", flush=True)
        with Pool(args.ncpu, initializer=init_sim, initargs=(model, snap)) as pool:
            results = pool.map(one_galaxy, tasks, chunksize=1)
        for r in results:
            if r["error"]:
                print(f"[compute_gas_masses] {model} snap {snap} sub {r['sub']} FAILED: {r['error']}",
                      flush=True)
        ok = [r for r in results if r["error"] is None]
        if not ok:
            continue

        arrays = {"halo_ids": np.array([r["sub"] for r in ok], dtype=np.int64)}
        arrays.update({k: np.array([r[k] for r in ok]) for k in keys})
        arrays.update(r_edges=RADIUS_EDGES, R_edges=RADIUS_EDGES, z_edges=Z_EDGES,
                      lognH_edges=LOGNH_EDGES, logT_edges=LOGT_EDGES)
        metadata = {"model": model, "snap": snap, "redshift": ok[0]["redshift"],
                    "definitions": list(DEFS), "galaxies_file": source}
        out = outdir / f"gas_masses_{model}_{snap:03d}.hdf5"
        save_flat(out, arrays, metadata=metadata)
        print(f"[compute_gas_masses] wrote {out}  ({len(ok)} galaxies, "
              f"{len(results) - len(ok)} failed)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
