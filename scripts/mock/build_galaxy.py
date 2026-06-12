"""Run the full mock-observation pipeline for one galaxy.

Builds the per-galaxy library entry

    <scratch>/data/martini/z<z>/<model>/gal_<subID>/
        sphviewer.png      face/edge x neutral-gas/stars
        cube.fits          MARTINI neutral-gas datacube
        bbarolo/           3D Barolo tilted-ring fit (rings, maps, PVs)
        kinematics.png     mom0 / mom1 / PV major+minor with model overlay
        info.json          metadata + V, sigma, V/sigma, MORDOR IsDisc

Stages run in order and are individually guarded, so a failure in (say)
BBarolo still leaves the sphviewer + cube outputs in place.

Usage:
    python scripts/mock/build_galaxy.py --model CDM --snap 21 --sub-id 0 \
        --ncpu 16
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat
from galaxy_sidm.mock import (
    load_galaxy_gas, render_face_edge, CubeParams, build_cube,
    run_bbarolo, plot_kinematics,
)
from galaxy_sidm.mock.kinematics import ring_v_over_sigma, ring_kinematics

ALL_STAGES = ["sphview", "cube", "barolo", "kinematics"]


def resolve_base_path(model, snap, res, cfg, override=None):
    """Path to the simulation output/ dir (shadow tree for CDM snap 21)."""
    import temet
    from galaxy_sidm.data.aida_tng import build_central_subhalo_catalog
    sim = temet.sim(run="aida", variant=model, res=res, snap=snap)
    cat = build_central_subhalo_catalog(sim)
    base_path = cat["basePath"]
    if override is not None:
        base_path = str(override).rstrip("/") + "/"
    elif model == "CDM" and snap == 21 and cfg["paths"].get("shadow_cdm"):
        base_path = str(Path(cfg["paths"]["shadow_cdm"]) / "output") + "/"
    return base_path


def mordor_lookup(cfg, model, snap, sub_id):
    """Return {IsDisc, Mstar} for this subhalo from the MORDOR sample."""
    p = (Path(cfg["paths"]["scratch_mordor"]) / "samples"
         / f"mordor_sample_{model}_{snap:03d}.hdf5")
    if not p.exists():
        return {}
    arrs, _ = load_flat(p)
    ids = np.asarray(arrs["halo_ids"], dtype=np.int64)
    hit = np.where(ids == sub_id)[0]
    if not len(hit):
        return {}
    i = int(hit[0])
    return {"IsDisc": int(arrs["IsDisc"][i]),
            "Mstar": float(arrs["Mstar"][i])}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", required=True,
                   choices=["CDM", "SIDM1", "vSIDM"])
    p.add_argument("--snap", required=True, type=int)
    p.add_argument("--sub-id", required=True, type=int,
                   help="Central subhalo id (GroupFirstSub)")
    p.add_argument("--res", type=int, default=1080)
    p.add_argument("--ncpu", type=int, default=1,
                   help="Threads for MARTINI insert / sphviewer / BBarolo")
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help=f"Comma list from {ALL_STAGES} (default all)")
    p.add_argument("--base-path", type=Path, default=None,
                   help="Override simulation output/ dir")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip if kinematics.png + info.json already exist")
    args = p.parse_args()

    cfg = load_config(args.config)
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    z = snap_z.get(args.snap, float("nan"))
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    gal_dir = (Path(cfg["paths"]["scratch_processed"]).parent / "martini"
               / f"z{z:g}" / args.model / f"gal_{args.sub_id:06d}")
    gal_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and (gal_dir / "kinematics.png").exists() \
            and (gal_dir / "info.json").exists():
        print(f"[build_galaxy] skip (exists): {gal_dir}")
        return 0

    info = {"model": args.model, "snap": args.snap, "redshift": z,
            "sub_id": args.sub_id, "dir": str(gal_dir)}
    info.update(mordor_lookup(cfg, args.model, args.snap, args.sub_id))

    base_path = resolve_base_path(args.model, args.snap, args.res, cfg,
                                  override=args.base_path)
    print(f"[build_galaxy] {args.model} snap {args.snap} sub {args.sub_id} "
          f"z={z:g}  base={base_path}")

    gas = load_galaxy_gas(base_path, args.snap, args.sub_id,
                          h=float(cfg["cosmology"]["h"]))
    info["n_gas"] = int(len(gas.mHI_g))
    info["n_star"] = int(len(gas.m_s))
    info["M_neutral"] = float(np.sum(gas.mHI_g.to_value("Msun")))
    print(f"[build_galaxy] gas cells={info['n_gas']} stars={info['n_star']} "
          f"M_neutral={info['M_neutral']:.3e} Msun")

    cube_fits = gal_dir / "cube.fits"

    if "sphview" in stages:
        try:
            lbl = f"subID {args.sub_id}\nz={z:.2f}"
            bb_dir = gal_dir / "bbarolo"
            if bb_dir.exists():
                vsig = ring_v_over_sigma(bb_dir)
                if np.isfinite(vsig):
                    lbl += "\n" + rf"$V/\sigma={vsig:.2f}$"
            render_face_edge(gas, gal_dir / "galaxy_sphviewer.png",
                             num_threads=args.ncpu, label=lbl)
            print(f"[build_galaxy] sphviewer -> {gal_dir/'galaxy_sphviewer.png'}")
        except Exception:
            print("[build_galaxy] sphview FAILED:\n" + traceback.format_exc())

    if "cube" in stages:
        try:
            _, npix = build_cube(gas, cube_fits, CubeParams(), ncpu=args.ncpu)
            info["cube_npix"] = int(npix)
            print(f"[build_galaxy] cube ({npix}px) -> {cube_fits}")
        except Exception:
            print("[build_galaxy] cube FAILED:\n" + traceback.format_exc())

    if "barolo" in stages and cube_fits.exists():
        try:
            res = run_bbarolo(cube_fits, gal_dir / "bbarolo",
                              inc_deg=60.0, pa_deg=90.0,
                              beam_arcsec=30.0, threads=args.ncpu)
            V, sigma, vsig = ring_kinematics(gal_dir / "bbarolo")
            info.update({"V": V, "sigma": sigma, "V_over_sigma": vsig,
                         "bbarolo_rc": res.returncode})
            print(f"[build_galaxy] BBarolo rc={res.returncode} "
                  f"V={V:.1f} sigma={sigma:.1f} V/sigma={vsig:.2f}")
        except Exception:
            print("[build_galaxy] barolo FAILED:\n" + traceback.format_exc())

    if "kinematics" in stages and (gal_dir / "bbarolo").exists():
        try:
            title = (f"{args.model}" + r" $\vert$ " + f"z={z:g}" + r" $\vert$ " + f"subID {args.sub_id}" + r" $\vert$ " + f"IsDisc={info.get('IsDisc','?')}")
            plot_kinematics(gal_dir / "bbarolo", gal_dir / "kinematics.png",
                            suptitle=title)
            print(f"[build_galaxy] kinematics -> {gal_dir/'kinematics.png'}")
        except Exception:
            print("[build_galaxy] kinematics FAILED:\n" + traceback.format_exc())

    (gal_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(f"[build_galaxy] wrote {gal_dir/'info.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
