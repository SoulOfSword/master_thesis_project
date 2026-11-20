"""Circular-velocity profiles in the disc plane, for the final sample or a list of galaxies.

Two methods, chosen with --method:
  direct (default)
    v_circ^2(R) = R dPhi/dR on circles of radius R in the disc plane (z = 0),
    averaged exactly over azimuth, from the softened gravity of the subhalo's
    own (SUBFIND-bound) particles summed directly, per component: dark matter,
    stars, gas (all cells), wind particles and black holes; the total is their
    sum (observables.vcirc2_components). The softening is AREPO's, with the
    snapshot's parameters (data.potential.read_softening): cubic spline of
    support h = 2.8 eps for DM, stars and wind; h = GasSoftFactor (2.5) x the
    cell radius on AREPO's ladder for gas; mass-dependent for black holes.
    Gas cells sit at CenterOfMass when the run has CELL_CENTER_GRAVITY (CDM),
    otherwise at Coordinates (SIDM1, vSIDM). The field is that of the
    particles alone (no softening at the evaluation point). Radii: --n-radii
    log-spaced from --r-min to --r-max.
  potential
    v_circ(R) = sqrt(R dPhi/dR), with Phi the snapshot's Potential at all the
    subhalo's particles (gas, DM, stars, black holes), averaged azimuthally in
    cylindrical rings inside a thin slab |z| < --slab about the disc plane.
    Rings are --dr wide, or --dr-frac x R where that is wider, so the sparse
    outer rings still hold enough particles for a stable derivative.
  both
    runs the two.
Centre SubhaloPos. Disc plane of mock.cube.disc_normal: MARTINI's L_align (angular
momentum of the innermost 30% of the neutral gas mass, at least 100 cells) on the gas
cells within mock.cube.SOURCE_RADIUS_FACTOR x the half-width of the cube-size rule
(mock.cube.rule_half_kpc); no cube needed.

Galaxies: by default the final sample (discs in both gas and stars), built at run
time: the gas discs of mock.tables.gas_disc_sample (current residual and asymmetry
tables, the dividing line, config/problematic_discs.yaml) that are MORDOR discs
(IsDisc == 1). --galaxies instead takes a text file with one "model snap subID" per
line (the mock manifest format), e.g. for a few test galaxies.

Output, one file per (model, snap), under cfg['paths']['scratch_processed']/vcirc/
(overwritten on every run):
    vcirc_<model>_<snap:03d>.hdf5
Datasets, one row per galaxy:
    halo_ids
    R_half_star          3-D stellar half-mass radius (SubhaloHalfmassRadType), kpc
  direct method:
    R_direct             (n_gal, n_radii) radii, kpc
    v2_dm, v2_stars, v2_gas, v2_wind, v2_bh
                         (n_gal, n_radii) v_circ^2 of each component, (km/s)^2;
                         negative where the component's net pull points outward
    v2_total             (n_gal, n_radii) sum of the components, (km/s)^2
    vcirc_total          (n_gal, n_radii) sqrt(v2_total), km/s (nan where v2_total <= 0)
    n_part               (n_gal, 5) particles of each component (order: metadata 'components')
    mass                 (n_gal, 5) mass of each component, Msun
    vcirc_direct_5rhalf  vcirc_total at 5 R_half_star, km/s
    vcirc_direct_10rhalf vcirc_total at 10 R_half_star, km/s
  potential method:
    R                    (n_gal, n_rings) mean radius of each ring, kpc
    vcirc                (n_gal, n_rings) km/s
    count                (n_gal, n_rings) particles per ring inside the slab
    vcirc_5rhalf         v_circ at 5 R_half_star, km/s
    vcirc_10rhalf        v_circ at 10 R_half_star, km/s
attrs:
    metadata: model, snap, redshift, galaxies_file (or the sample's dividing line when
              the sample is built here); direct method also components,
              gas_position, G, and the softening used (lengths are the spline support h,
              physical kpc): h_dm_kpc, h_stars_wind_kpc, gas_soft_factor, h_gas_min_kpc,
              gas_soft_spacing, n_gas_soft, bh_mass_dependent and h_bh_types_kpc (types
              1-5) or h_bh_kpc
    variants: method; direct: r_min_kpc, r_max_kpc, n_radii, h_target_kpc, n_gl;
              potential: dr_kpc, dr_frac, rmax_kpc, slab_half_kpc, n_azimuth, min_count

Usage:
    python scripts/data/compute_vcirc.py --ncpu 8                     # the final sample
    python scripts/data/compute_vcirc.py --ncpu 8 --method both
    python scripts/data/compute_vcirc.py --galaxies test.txt --ncpu 8 # a list of galaxies
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
from galaxy_sidm.data.aida_tng import base_path
from galaxy_sidm.data.potential import (COMPONENTS, H_OVER_EPS, h_collisionless,
                                        load_subhalo_particles, load_subhalo_potential,
                                        read_softening, softening_eps)
from galaxy_sidm.mock import disc_normal, load_galaxy_gas
from galaxy_sidm.mock.tables import final_sample
from galaxy_sidm.observables import G_AREPO, N_GL, vcirc2_components, vcirc_disc_plane

# softening of the evaluation point of the direct method: none, the field of
# the particles alone
H_TARGET = 0.0


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


def softening_attrs(soft):
    """The softening of the direct method, for the output metadata (spline
    support h in physical kpc)."""
    p = soft["params"]
    attrs = {
        "components": list(COMPONENTS),
        "gas_position": soft["gas_position"],
        "G": G_AREPO,
        "h_dm_kpc": h_collisionless(soft, 1),
        "h_stars_wind_kpc": h_collisionless(soft, 4),
        "gas_soft_factor": p["GasSoftFactor"],
        "h_gas_min_kpc": H_OVER_EPS * p["MinimumComovingHydroSoftening"] * soft["a"] / soft["hubble"],
        "gas_soft_spacing": p["AdaptiveHydroSofteningSpacing"],
        "n_gas_soft": soft["n_gas_soft"],
        "bh_mass_dependent": soft["bh_mass_dependent"],
    }
    if soft["bh_mass_dependent"]:
        attrs["h_bh_types_kpc"] = [H_OVER_EPS * softening_eps(soft, t) for t in range(1, 6)]
    else:
        attrs["h_bh_kpc"] = h_collisionless(soft, 5)
    return attrs


def one_galaxy(task):
    """v_circ profiles of one galaxy as a dict (with 'error' set if it failed)."""
    model, snap, sub, base, h, opts = task
    try:
        gas = load_galaxy_gas(base, snap, sub, h=h)
        zhat = disc_normal(gas)
        row = il.groupcat.loadSingle(base, snap, subhaloID=sub)
        r_half = float(row["SubhaloHalfmassRadType"][4]) * gas.a / h
        out = dict(sub=sub, R_half_star=r_half, redshift=gas.redshift, error=None)
        del gas # free the cells before loading all the particles

        if opts["method"] in ("direct", "both"):
            radii = opts["radii"]
            p = load_subhalo_particles(base, snap, sub, opts["soft"])
            v2 = vcirc2_components(p["xyz"], p["mass"], p["h"], p["comp"], zhat, radii,
                                   n_comp=len(COMPONENTS), h_target=H_TARGET)
            v2_total = v2.sum(axis=1)
            v_total = np.sqrt(np.where(v2_total > 0, v2_total, np.nan))
            out.update({f"v2_{c}": v2[:, k] for k, c in enumerate(COMPONENTS)})
            out.update(R_direct=radii, v2_total=v2_total, vcirc_total=v_total,
                       n_part=np.bincount(p["comp"], minlength=len(COMPONENTS)),
                       mass=np.bincount(p["comp"], weights=p["mass"], minlength=len(COMPONENTS)),
                       vcirc_direct_5rhalf=_at(radii, v_total, 5 * r_half),
                       vcirc_direct_10rhalf=_at(radii, v_total, 10 * r_half))
            del p

        if opts["method"] in ("potential", "both"):
            xyz, phi = load_subhalo_potential(base, snap, sub, h=h)
            R, v, count = vcirc_disc_plane(xyz, phi, zhat, opts["r_edges"], slab_half=opts["slab"],
                                           n_azimuth=opts["n_azimuth"], min_count=opts["min_count"])
            out.update(R=R, vcirc=v, count=count,
                       vcirc_5rhalf=_at(R, v, 5 * r_half), vcirc_10rhalf=_at(R, v, 10 * r_half))
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
    p.add_argument("--method", choices=["direct", "potential", "both"], default="direct",
                   help="direct: softened gravity of the particles, per component; "
                        "potential: the snapshot's Potential; both (default direct)")
    p.add_argument("--r-min", type=float, default=0.1,
                   help="smallest radius [kpc], direct method (default 0.1)")
    p.add_argument("--r-max", type=float, default=50.0,
                   help="largest radius [kpc], direct method (default 50)")
    p.add_argument("--n-radii", type=int, default=100,
                   help="log-spaced radii, direct method (default 100)")
    p.add_argument("--dr", type=float, default=0.25,
                   help="smallest ring width [kpc], potential method (default 0.25)")
    p.add_argument("--dr-frac", type=float, default=0.1,
                   help="ring width as a fraction of R, where wider than --dr, "
                        "potential method (default 0.1)")
    p.add_argument("--rmax", type=float, default=40.0,
                   help="outer edge of the last ring [kpc], potential method (default 40)")
    p.add_argument("--slab", type=float, default=0.3,
                   help="slab half-thickness, |z| < slab [kpc], potential method (default 0.3)")
    p.add_argument("--n-azimuth", type=int, default=12,
                   help="azimuthal sectors per ring, potential method (default 12)")
    p.add_argument("--min-count", type=int, default=10,
                   help="particles a ring needs inside the slab, potential method (default 10)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="default: <scratch_processed>/vcirc")
    args = p.parse_args()

    cfg = load_config(args.config)
    h = float(cfg["cosmology"]["h"])
    outdir = args.outdir or Path(cfg["paths"]["scratch_processed"]) / "vcirc"
    direct = args.method in ("direct", "both")
    potential = args.method in ("potential", "both")
    radii = np.logspace(np.log10(args.r_min), np.log10(args.r_max), args.n_radii)
    r_edges = ring_edges(args.dr, args.dr_frac, args.rmax)

    if args.galaxies:
        groups, source = defaultdict(list), str(args.galaxies)
        for line in args.galaxies.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                model, snap, sub = line.split()[:3]
                groups[(model, int(snap))].append(int(sub))
    else:
        groups, line = final_sample(cfg)
        source = f"final sample built at run time ({line})"
    print(f"[compute_vcirc] {sum(map(len, groups.values()))} galaxies: {source}", flush=True)

    for (model, snap), subs in sorted(groups.items()):
        base = base_path(cfg, model)
        soft = read_softening(base, snap) if direct else None
        if direct and abs(soft["hubble"] - h) > 1e-6:
            raise ValueError(f"{model} snap {snap}: HubbleParam {soft['hubble']} in the snapshot, "
                             f"h = {h} in the config")
        opts = dict(method=args.method, radii=radii, soft=soft, r_edges=r_edges,
                    slab=args.slab, n_azimuth=args.n_azimuth, min_count=args.min_count)
        tasks = [(model, snap, s, base, h, opts) for s in subs]
        print(f"[compute_vcirc] {model} snap {snap}: {len(tasks)} galaxies, method {args.method}",
              flush=True)
        with Pool(args.ncpu) as pool:
            results = pool.map(one_galaxy, tasks, chunksize=1)
        for r in results:
            if r["error"]:
                print(f"[compute_vcirc] {model} snap {snap} sub {r['sub']} FAILED: {r['error']}",
                      flush=True)
        ok = [r for r in results if r["error"] is None]
        if not ok:
            continue

        names = ["R_half_star"]
        metadata = {"model": model, "snap": snap, "redshift": ok[0]["redshift"],
                    "galaxies_file": source}
        variants = {"method": args.method}
        if direct:
            names += ["R_direct", *[f"v2_{c}" for c in COMPONENTS], "v2_total", "vcirc_total",
                      "n_part", "mass", "vcirc_direct_5rhalf", "vcirc_direct_10rhalf"]
            metadata.update(softening_attrs(soft))
            variants.update(r_min_kpc=args.r_min, r_max_kpc=args.r_max, n_radii=args.n_radii,
                            h_target_kpc=H_TARGET, n_gl=N_GL)
        if potential:
            names += ["R", "vcirc", "count", "vcirc_5rhalf", "vcirc_10rhalf"]
            variants.update(dr_kpc=args.dr, dr_frac=args.dr_frac, rmax_kpc=args.rmax,
                            slab_half_kpc=args.slab, n_azimuth=args.n_azimuth,
                            min_count=args.min_count)
        arrays = {"halo_ids": np.array([r["sub"] for r in ok], dtype=np.int64)}
        arrays.update({k: np.array([r[k] for r in ok]) for k in names})
        out = outdir / f"vcirc_{model}_{snap:03d}.hdf5"
        save_flat(out, arrays, metadata=metadata, variants=variants)
        print(f"[compute_vcirc] wrote {out}  ({len(ok)} galaxies, "
              f"{len(results) - len(ok)} failed)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
