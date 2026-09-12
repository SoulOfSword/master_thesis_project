"""Run the full mock-observation pipeline for one galaxy.

Builds the per-galaxy library entry

    <scratch>/data/martini/z<z>/<model>/gal_<subID>/
        sphviewer.png      face/edge x neutral-gas/stars
        cube.fits          MARTINI neutral-gas datacube
        bbarolo_3rings/    first 3D Barolo fit: 3 rings for every galaxy (maps, mask, PVs, plots)
        bbarolo/           final fit: rings out to the edge of the data velocity
                           field along the major axis (rings, maps, PVs, plots)
        kinematics.png     mom0 / mom1 / PV major+minor with model overlay
        info.json          metadata + V, sigma, V/sigma, MORDOR IsDisc

Stages (in this order): sphview, cube, barolo_3rings, barolo, kinematics.
Each is individually guarded, so a failure in (say) BBarolo still leaves the
sphviewer + cube outputs in place. Stages can be run separately: info.json is
updated, not rewritten, and when a stage re-runs, the products and info.json
entries of the stages after it are deleted (they belong to the old version).

BBarolo fits: barolo_3rings fits the same 3 rings for every galaxy and gives the
data moment-1 map and the centre; barolo (the final fit) reads that fit and uses
as many one-beam rings as fit between the centre and the last non-NaN pixel of
the moment-1 map along the major axis (farther side, rounded down), with the
centre fixed to the first fit's.

Usage:
    python scripts/mock/build_galaxy.py --model CDM --snap 21 --sub-id 0 \
        --ncpu 16
    python scripts/mock/build_galaxy.py --model CDM --snap 21 --sub-id 0 \
        --ncpu 16 --stages barolo,kinematics      # final fit only, later
"""

import argparse
import json
import shutil
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
from galaxy_sidm.mock.barolo import major_axis_extent
from galaxy_sidm.mock.kinematics import ring_v_over_sigma, ring_kinematics

ALL_STAGES = ["sphview", "cube", "barolo_3rings", "barolo", "kinematics"]

# what each stage writes: (files/dirs in the galaxy dir, info.json keys)
STAGE_PRODUCTS = {
    "cube": (["cube.fits"], ["cube_npix", "cube_signal", "cube_noise_rms", "reference_snr"]),
    "barolo_3rings": (["bbarolo_3rings"], ["bbarolo_3rings_rc"]),
    "barolo": (["bbarolo"], ["extent_arcsec", "fit_centre_px", "nradii",
                             "V", "sigma", "V_over_sigma", "bbarolo_rc"]),
    "kinematics": (["kinematics.png"], []),
}

FIRST_NRADII = 3                                      # rings of the first fit
RING_ARCSEC = CubeParams().beam_fwhm.to_value("arcsec")  # rings are one beam wide

# written by scripts/mock/snr_reference.py; sets the S/N every cube gets
SNR_REFERENCE = ROOT / "config" / "snr_reference.yaml"


def load_reference_snr(params):
    """reference_snr from SNR_REFERENCE, checked against the cube settings."""
    import yaml
    if not SNR_REFERENCE.exists():
        raise FileNotFoundError(
            f"{SNR_REFERENCE} not found: run scripts/mock/snr_reference.py first")
    ref = yaml.safe_load(SNR_REFERENCE.read_text())
    if not (np.isclose(ref["signal_peak_fraction"], params.signal_peak_fraction)
            and np.isclose(ref["noise_rms"], params.noise_rms.to_value("Jy / arcsec2"))):
        raise ValueError(
            f"{SNR_REFERENCE} was made with signal_peak_fraction="
            f"{ref['signal_peak_fraction']}, noise_rms={ref['noise_rms']}, but "
            f"CubeParams has {params.signal_peak_fraction}, {params.noise_rms}: "
            "rerun scripts/mock/snr_reference.py")
    return float(ref["reference_snr"])


def clear_after(stage, gal_dir, info):
    """Delete the products and info.json entries of every stage after `stage`.

    A stage that re-ran makes everything downstream of it stale (e.g. a new cube
    -> the old BBarolo fits describe a cube that no longer exists).
    """
    for later in ALL_STAGES[ALL_STAGES.index(stage) + 1:]:
        names, keys = STAGE_PRODUCTS.get(later, ([], []))
        for name in names:
            path = gal_dir / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        for key in keys:
            info.pop(key, None)


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
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        p.error(f"unknown stage(s) {sorted(unknown)}; choose from {ALL_STAGES}")

    gal_dir = (Path(cfg["paths"]["scratch_processed"]).parent / "martini"
               / f"z{z:g}" / args.model / f"gal_{args.sub_id:06d}")
    gal_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and (gal_dir / "kinematics.png").exists() \
            and (gal_dir / "info.json").exists():
        print(f"[build_galaxy] skip (exists): {gal_dir}")
        return 0

    # update the existing info.json (stages may run separately)
    info_path = gal_dir / "info.json"
    try:
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
    except ValueError:
        info = {}
    info.update({"model": args.model, "snap": args.snap, "redshift": z,
                 "sub_id": args.sub_id, "dir": str(gal_dir)})
    info.update(mordor_lookup(cfg, args.model, args.snap, args.sub_id))

    gas = None
    if "sphview" in stages or "cube" in stages:   # only these need the particles
        base_path = resolve_base_path(args.model, args.snap, args.res, cfg,
                                      override=args.base_path)
        print(f"[build_galaxy] {args.model} snap {args.snap} sub {args.sub_id} "
              f"z={z:g}  base={base_path}")
        gas = load_galaxy_gas(base_path, args.snap, args.sub_id,
                              h=float(cfg["cosmology"]["h"]))
        info["n_gas"] = int(len(gas.mH_neutral_g))
        info["n_star"] = int(len(gas.m_s))
        info["M_neutral"] = float(np.sum(gas.mH_neutral_g.to_value("Msun")))
        print(f"[build_galaxy] gas cells={info['n_gas']} stars={info['n_star']} "
              f"M_neutral={info['M_neutral']:.3e} Msun")

    cube_fits = gal_dir / "cube.fits"
    first_dir = gal_dir / "bbarolo_3rings"
    final_dir = gal_dir / "bbarolo"

    if "sphview" in stages:
        try:
            lbl = f"subID {args.sub_id}\nz={z:.2f}"
            if final_dir.exists():
                vsig = ring_v_over_sigma(final_dir)
                if np.isfinite(vsig):
                    lbl += "\n" + rf"$V/\sigma={vsig:.2f}$"
            render_face_edge(gas, gal_dir / "galaxy_sphviewer.png",
                             num_threads=args.ncpu, label=lbl)
            print(f"[build_galaxy] sphviewer -> {gal_dir/'galaxy_sphviewer.png'}")
        except Exception:
            print("[build_galaxy] sphview FAILED:\n" + traceback.format_exc())

    cube_ok = False
    if "cube" in stages:
        try:
            # same signal/noise for every galaxy; own noise realisation each
            ref_snr = load_reference_snr(CubeParams())
            res = build_cube(gas, cube_fits,
                             CubeParams(reference_snr=ref_snr, noise_seed=args.sub_id),
                             ncpu=args.ncpu)
            info.update({"cube_npix": int(res.npix), "cube_signal": res.signal,
                         "cube_noise_rms": res.noise_rms, "reference_snr": ref_snr})
            clear_after("cube", gal_dir, info)  # old fits belong to the old cube
            cube_ok = True
            print(f"[build_galaxy] cube ({res.npix}px, signal={res.signal:.3e} Jy/beam, "
                  f"noise_rms={res.noise_rms:.3e}) -> {cube_fits}")
        except Exception:
            print("[build_galaxy] cube FAILED:\n" + traceback.format_exc())
    # a failed cube stage must not let BBarolo refit the previous cube.fits
    cube_current = cube_ok or "cube" not in stages

    # tie BBarolo's DISTANCE to the same value MARTINI built the cube with, so
    # RAD(Kpc) is consistent across galaxies (not Vsys-guessed)
    fit = dict(inc_deg=60.0, pa_deg=90.0, beam_arcsec=RING_ARCSEC,
               threads=args.ncpu, distance_mpc=CubeParams().distance.to_value("Mpc"))

    if "barolo_3rings" in stages and cube_fits.exists() and cube_current:
        try:
            # the same few rings for every galaxy -> data maps, mask and centre
            shutil.rmtree(first_dir, ignore_errors=True)  # no stale products
            clear_after("barolo_3rings", gal_dir, info)   # final fit used the old one
            first = run_bbarolo(cube_fits, first_dir, nradii=FIRST_NRADII, **fit)
            info["bbarolo_3rings_rc"] = first.returncode
            print(f"[build_galaxy] BBarolo {FIRST_NRADII} rings rc={first.returncode} -> {first_dir}")
        except Exception:
            print("[build_galaxy] barolo_3rings FAILED:\n" + traceback.format_exc())

    if "barolo" in stages and cube_fits.exists() and cube_current:
        info["bbarolo_rc"] = None  # stays None if this stage breaks
        try:
            if info.get("bbarolo_3rings_rc") != 0 or not first_dir.is_dir():
                raise RuntimeError(f"needs a successful {FIRST_NRADII}-ring fit in "
                                   f"{first_dir} (stage barolo_3rings)")
            # rings out to the last non-NaN pixel of the velocity field along
            # the major axis (farther side, rounded down), same centre
            ext = major_axis_extent(first_dir)
            nradii = max(1, int(max(ext.left_arcsec, ext.right_arcsec) // RING_ARCSEC))
            info.update({"extent_arcsec": [ext.left_arcsec, ext.right_arcsec],
                         "fit_centre_px": [ext.xpos, ext.ypos], "nradii": nradii})
            print(f"[build_galaxy] extent left/right = {ext.left_arcsec:.0f}\"/"
                  f"{ext.right_arcsec:.0f}\" -> NRADII={nradii}")

            shutil.rmtree(final_dir, ignore_errors=True)
            clear_after("barolo", gal_dir, info)  # kinematics.png showed the old fit
            res = run_bbarolo(cube_fits, final_dir, nradii=nradii,
                              extra=[f"XPOS        {ext.xpos}", f"YPOS        {ext.ypos}"],
                              **fit)
            if res.returncode == 0:
                V, sigma, vsig = ring_kinematics(final_dir)
            else:
                V = sigma = vsig = float("nan")  # failed fit -> no stale rings
            info.update({"V": V, "sigma": sigma, "V_over_sigma": vsig,
                         "bbarolo_rc": res.returncode})
            print(f"[build_galaxy] BBarolo rc={res.returncode} "
                  f"V={V:.1f} sigma={sigma:.1f} V/sigma={vsig:.2f}")
        except Exception:
            print("[build_galaxy] barolo FAILED:\n" + traceback.format_exc())

    if ("kinematics" in stages and final_dir.exists()
            and cube_current and info.get("bbarolo_rc") == 0):
        try:
            title = (f"{args.model}" + r" $\vert$ " + f"z={z:g}" + r" $\vert$ " + f"subID {args.sub_id}" + r" $\vert$ " + f"IsDisc={info.get('IsDisc','?')}")
            plot_kinematics(final_dir, gal_dir / "kinematics.png",
                            suptitle=title)
            print(f"[build_galaxy] kinematics -> {gal_dir/'kinematics.png'}")
        except Exception:
            print("[build_galaxy] kinematics FAILED:\n" + traceback.format_exc())

    info_path.write_text(json.dumps(info, indent=2))
    print(f"[build_galaxy] wrote {info_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
