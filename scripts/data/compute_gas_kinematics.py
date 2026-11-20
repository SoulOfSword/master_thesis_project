"""Kinematics of the neutral gas of the MORDOR discs, for the particle-based gas-disc selection.

Per galaxy, from its gas cells (mock.load_galaxy_gas; neutral hydrogen = mH_neutral_g, the gas
MARTINI receives) and its v_circ file (scripts/data/compute_vcirc.py: R_half_star, the 3D
stellar half-mass radius, and the direct-method curve; a galaxy without a curve fails):
  plane   mock.cube.disc_normal: MARTINI's L_align on the cells a cube of the cube-size rule's
          half-width receives (no cube needed); centre SubhaloPos
  cut     mock.cube.disc_cut: |z| < K_Z z_half, with z_half the half-mass height of the neutral
          hydrogen within 2 R_half_star, and R < R_cut, the radius holding the fraction Q_R of
          the neutral hydrogen within that height
  frame   velocities relative to v_bulk, the mass-weighted mean velocity of the neutral
          hydrogen in the cut; v_R, v_phi and v_z about the plane's normal, v_phi > 0 along
          the rotation
Radii, all from the neutral hydrogen within the height cut, before the radial cut
(mock.cube.layer_radii): R_cut (R_99), R_95 and R_half_gas hold 99%, 95% and 50% of it.
kappa_co, the co-rotating share of the kinetic energy (sum of m v_phi^2 over v_phi > 0 / sum of
m v^2) of the neutral hydrogen in the cut within R_cut, R_95 and 2 R_half_gas (capped at R_cut):
kappa_co_r99, kappa_co_r95 and kappa_co_2rhalf; same plane and frame, only the radius differs.
On the neutral hydrogen in the cut:
  rings      from R = 0 to R_cut, RING_DR wide (one beam at the cube's distance); with WIDENING
             a ring is DR_FRAC x its inner radius wide where that is wider (compute_vcirc.ring_edges)
Diagnostics:
  warp_deg       angle between the angular momenta of the inner 30% and the outer 60-90% (by 3D
                 radius, by mass) of the neutral hydrogen in the cut (Downing & Oman 2023's warp)
  gas_star_deg   angle between the plane's normal and the angular momentum of the stars (not the
                 wind particles) within 2 R_half_star
  f_above_zcut   share of the neutral hydrogen within R_cut that lies above the height cut

Galaxies: the mock manifest (<martini>/manifest.txt, all MORDOR discs) or --galaxies.

Output, one file per (model, snap): <scratch_processed>/gas_kinematics/gas_kinematics_<model>_<snap:03d>.hdf5
    halo_ids
    zhat, v_bulk          (n_gal, 3) plane normal; km/s relative to SubhaloVel
    R_half_star, z_half, z_cut, R_cut, R_95, R_half_gas   kpc
    M_neutral, M_cut      neutral hydrogen of the subhalo and in the cut, Msun
    n_cells_cut, kappa_co_r99, kappa_co_r95, kappa_co_2rhalf, f_above_zcut, warp_deg, gas_star_deg
    n_rings               rings out to R_cut
    ring_edges            (n_max + 1) kpc, the same for every galaxy
    ring_mass, ring_n_cells             (n_gal, n_max) Msun, cells
    ring_vphi, ring_vR, ring_vz         mass-weighted means, km/s
    ring_sigma_vphi, ring_sigma_vR, ring_sigma_vz   mass-weighted dispersions, km/s
    ring_K, ring_K_co     kinetic energy, all and co-rotating, Msun (km/s)^2
    ring_vcirc            v_circ at the ring's mid-radius, km/s (nan outside the curve)
    rings past n_rings are padding: 0 mass, cells and energy, nan otherwise
attrs metadata: model, snap, redshift, galaxies_file; variants: widening, ring_dr_kpc, dr_frac,
    k_z, q_R

Usage:
    python scripts/data/compute_gas_kinematics.py --ncpu 16                      # the manifest
    python scripts/data/compute_gas_kinematics.py --galaxies test.txt --ncpu 4   # a list of galaxies
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import astropy.units as U

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.data.aida_tng import base_path
from galaxy_sidm.io import load_config, load_flat, read_galaxy_list, save_flat
from galaxy_sidm.mock import (CubeParams, disc_coords, disc_cut, disc_normal, disc_velocities,
                              layer_radii, load_galaxy_gas, vcirc_at)
from compute_vcirc import ring_edges

WIDENING = False  # True: rings widen to DR_FRAC x R
DR_FRAC = 0.1
RING_DR = (CubeParams().beam_fwhm * CubeParams().distance).to_value(U.kpc, U.dimensionless_angles())
K_Z, Q_R = 5.0, 0.99


def angle_deg(a, b):
    """Angle between the vectors a and b in degrees (nan if either is zero)."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b / norm, -1, 1)))) if norm > 0 else float("nan")


def one_galaxy(task):
    """Kinematics of one galaxy as a dict (with 'error' set if it failed)."""
    model, snap, sub, base, h, r_half, R_curve, v_curve = task
    try:
        gas = load_galaxy_gas(base, snap, sub, h=h)
        zhat = disc_normal(gas)
        keep, z_half, z_cut, R_cut = disc_cut(gas, zhat, r_half, K_Z, Q_R)
        R_half_gas, R_95 = layer_radii(gas, zhat, z_cut, [0.5, 0.95])
        xyz = gas.xyz_g.to_value(U.kpc)
        vel = gas.vxyz_g.to_value(U.km / U.s)
        m_all = gas.mH_neutral_g.to_value(U.Msun)
        z_all, R_all = disc_coords(xyz, zhat)

        R, m = R_all[keep], m_all[keep]
        v_bulk = m @ vel[keep] / m.sum()
        v = vel[keep] - v_bulk
        v_R, v_phi, v_z = disc_velocities(xyz[keep], v, zhat)
        K = 0.5 * m * np.sum(v ** 2, axis=1)
        K_co = np.where(v_phi > 0, 0.5 * m * v_phi ** 2, 0.0)
        kappa = {k: K_co[R < r].sum() / K[R < r].sum()
                 for k, r in (("kappa_co_r99", R_cut), ("kappa_co_r95", R_95), ("kappa_co_2rhalf", 2 * R_half_gas))}

        edges = ring_edges(RING_DR, DR_FRAC if WIDENING else 0.0, R_cut)
        n = len(edges) - 1
        ring = np.digitize(R, edges) - 1

        def ring_sum(w):
            return np.bincount(ring, weights=w, minlength=n)

        mass = ring_sum(m)
        out = dict(sub=sub, redshift=gas.redshift, error=None, zhat=zhat, v_bulk=v_bulk,
                   R_half_star=r_half, z_half=z_half, z_cut=z_cut, R_cut=R_cut, R_95=R_95,
                   R_half_gas=R_half_gas, M_neutral=m_all.sum(), M_cut=m.sum(), n_cells_cut=int(keep.sum()),
                   **kappa, n_rings=n, ring_mass=mass,
                   ring_n_cells=np.bincount(ring, minlength=n), ring_K=ring_sum(K),
                   ring_K_co=ring_sum(K_co),
                   ring_vcirc=vcirc_at(0.5 * (edges[1:] + edges[:-1]), R_curve, v_curve))
        with np.errstate(invalid="ignore", divide="ignore"):
            for name, u in (("vphi", v_phi), ("vR", v_R), ("vz", v_z)):
                mean = ring_sum(m * u) / mass
                out[f"ring_{name}"] = mean
                out[f"ring_sigma_{name}"] = np.sqrt(np.maximum(ring_sum(m * u ** 2) / mass - mean ** 2, 0.0))

        within = R_all < R_cut
        out["f_above_zcut"] = m_all[within & (np.abs(z_all) >= z_cut)].sum() / m_all[within].sum()
        order = np.argsort(np.linalg.norm(xyz[keep], axis=1))
        cum = np.cumsum(m[order]) / m.sum()
        L = m[:, None] * np.cross(xyz[keep], v)
        out["warp_deg"] = angle_deg(L[order[cum <= 0.3]].sum(axis=0), L[order[(cum > 0.6) & (cum <= 0.9)]].sum(axis=0))
        x_s = gas.xyz_s.to_value(U.kpc)
        stars = gas.is_star_s & (np.linalg.norm(x_s, axis=1) < 2 * r_half)
        L_s = gas.m_s.to_value(U.Msun)[stars] @ np.cross(x_s[stars], gas.vxyz_s.to_value(U.km / U.s)[stars])
        out["gas_star_deg"] = angle_deg(zhat, L_s)
        return out
    except Exception as e:
        return dict(sub=sub, error=f"{type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--galaxies", type=Path, default=None,
                   help='text file with one "model snap subID" per line '
                        "(default: the mock manifest, <martini>/manifest.txt)")
    p.add_argument("--ncpu", type=int, default=1, help="galaxies processed in parallel")
    p.add_argument("--outdir", type=Path, default=None,
                   help="default: <scratch_processed>/gas_kinematics")
    args = p.parse_args()

    cfg = load_config(args.config)
    h = float(cfg["cosmology"]["h"])
    processed = Path(cfg["paths"]["scratch_processed"])
    outdir = args.outdir or processed / "gas_kinematics"
    galaxies = args.galaxies or processed.parent / "martini" / "manifest.txt"
    groups = read_galaxy_list(galaxies)
    print(f"[compute_gas_kinematics] {sum(map(len, groups.values()))} galaxies: {galaxies}", flush=True)

    scalars = ["zhat", "v_bulk", "R_half_star", "z_half", "z_cut", "R_cut", "R_95", "R_half_gas",
               "M_neutral", "M_cut", "n_cells_cut", "kappa_co_r99", "kappa_co_r95", "kappa_co_2rhalf",
               "f_above_zcut", "warp_deg", "gas_star_deg", "n_rings"]
    ring_fill = {"ring_mass": 0.0, "ring_n_cells": 0, "ring_K": 0.0, "ring_K_co": 0.0,
                 "ring_vcirc": np.nan,
                 **{f"ring_{s}{c}": np.nan for s in ("", "sigma_") for c in ("vphi", "vR", "vz")}}
    for (model, snap), subs in sorted(groups.items()):
        a, _ = load_flat(processed / "vcirc" / f"vcirc_{model}_{snap:03d}.hdf5")
        curve = {int(s): (r, R, v) for s, r, R, v
                 in zip(a["halo_ids"], a["R_half_star"], a["R_direct"], a["vcirc_total"])}
        for s in subs:
            if s not in curve:
                print(f"[compute_gas_kinematics] {model} snap {snap} sub {s} FAILED: no v_circ curve",
                      flush=True)
        base = base_path(cfg, model)
        tasks = [(model, snap, s, base, h, *curve[s]) for s in subs if s in curve]
        print(f"[compute_gas_kinematics] {model} snap {snap}: {len(tasks)} galaxies", flush=True)
        with Pool(args.ncpu) as pool:
            results = pool.map(one_galaxy, tasks, chunksize=1)
        for r in results:
            if r["error"]:
                print(f"[compute_gas_kinematics] {model} snap {snap} sub {r['sub']} FAILED: {r['error']}",
                      flush=True)
        ok = [r for r in results if r["error"] is None]
        if not ok:
            continue

        n_max = max(r["n_rings"] for r in ok)
        arrays = {"halo_ids": np.array([r["sub"] for r in ok], dtype=np.int64),
                  "ring_edges": ring_edges(RING_DR, DR_FRAC if WIDENING else 0.0,
                                           max(r["R_cut"] for r in ok))}
        arrays.update({k: np.array([r[k] for r in ok]) for k in scalars})
        for k, fill in ring_fill.items():
            arrays[k] = np.full((len(ok), n_max), fill)
            for i, r in enumerate(ok):
                arrays[k][i, :r["n_rings"]] = r[k]
        metadata = {"model": model, "snap": snap, "redshift": ok[0]["redshift"],
                    "galaxies_file": str(galaxies)}
        variants = {"widening": WIDENING, "ring_dr_kpc": RING_DR,
                    "dr_frac": DR_FRAC if WIDENING else 0.0, "k_z": K_Z, "q_R": Q_R}
        out = outdir / f"gas_kinematics_{model}_{snap:03d}.hdf5"
        save_flat(out, arrays, metadata=metadata, variants=variants)
        print(f"[compute_gas_kinematics] wrote {out}  ({len(ok)} galaxies, "
              f"{len(results) - len(ok)} failed)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
