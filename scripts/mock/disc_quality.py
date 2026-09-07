"""Rank MORDOR discs by BBarolo fit residual — an automatic disc-quality flag.

For every MORDOR disc, compares the data cube to BBarolo's rotating-disc model
inside the region the model actually covers, and reports

    resid = RMS(data - model) / noise

over the pixels where the model has flux (noise = std of the cube OUTSIDE the
detection mask). A clean rotating disc -> resid ~ 1 (the disc model reproduces
the data; what's left is just noise). Weird / non-disc gas that a rotating disc
can't reproduce -> resid >> 1. Galaxies whose fit failed (bbarolo_rc != 0, no
usable model) get resid = inf -- genuine non-detections to auto-exclude.

Restricting the residual to the MODELLED region (not the whole mask) matters:
NRADII is a small fixed 5/7, so gas beyond the fitted rings is unmodelled for
*every* galaxy; measuring only where the model has flux isolates "is the fitted
inner disc actually a disc" from "the cube is bigger than the fit".

Outputs a CSV sorted worst-first (review the top instead of all ~4000), and
prints how `resid` separates the discs you already hand-flagged in
config/problematic_discs.yaml from the ones you kept -- i.e. whether a simple
residual cut reproduces your eye, and where the threshold should sit.

Reads only existing BBarolo outputs (cube.fits, MOCKmod_azim.fits, mask.fits),
so no re-running. I/O-heavy (3 cube reads/galaxy) -> run on a node; use
--model to split the work across the three models in parallel.

Usage:
  python scripts/mock/disc_quality.py                    # all discs -> CSV
  python scripts/mock/disc_quality.py --model CDM        # one model (parallelise)
  python scripts/mock/disc_quality.py --out figures/disc_quality.csv
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config


def _resid(gal_dir):
    """RMS(data-model)/noise over the modelled region; inf if no usable fit."""
    bb = os.path.join(gal_dir, "bbarolo")
    try:
        data = np.asarray(fits.getdata(os.path.join(gal_dir, "cube.fits")), float)
        model = np.asarray(fits.getdata(os.path.join(bb, "MOCKmod_azim.fits")), float)
        mask = np.asarray(fits.getdata(os.path.join(bb, "mask.fits")), float)
    except Exception:
        return np.inf
    if data.shape != model.shape or data.shape != mask.shape:
        return np.inf
    data = np.nan_to_num(data)
    model = np.nan_to_num(model)
    mmax = model.max()
    if mmax <= 0:
        return np.inf
    reg = model > 0.05 * mmax            # pixels the disc model actually covers
    if reg.sum() < 20:
        return np.inf
    off = mask < 0.5                      # outside the detection mask = noise
    noise = np.std(data[off]) if off.sum() > 100 else np.std(data)
    if not np.isfinite(noise) or noise <= 0:
        return np.inf
    return float(np.sqrt(np.mean((data[reg] - model[reg]) ** 2)) / noise)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, choices=["CDM", "SIDM1", "vSIDM"],
                   help="Restrict to one model (default: all).")
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "disc_quality.csv")
    p.add_argument("--exclude", type=Path,
                   default=ROOT / "config" / "problematic_discs.yaml",
                   help="Hand-flagged YAML; used only to report the resid split.")
    args = p.parse_args()

    cfg = load_config(None)
    mart = Path(cfg["paths"]["scratch_processed"]).parent / "martini"

    rows = []
    for f in glob.glob(str(mart / "z*/*/gal_*/info.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if int(d.get("IsDisc", 0)) != 1:            # MORDOR discs only
            continue
        if args.model and d.get("model") != args.model:
            continue
        rc = d.get("bbarolo_rc")
        mstar = d.get("Mstar")
        logM = float(np.log10(mstar)) if mstar else float("nan")
        resid = np.inf if rc not in (0, None) else _resid(os.path.dirname(f))
        rows.append((d.get("model"), d.get("redshift"), d.get("sub_id"),
                     logM, resid, rc))

    # worst first: inf (failures / non-detections) at the top, then descending
    rows.sort(key=lambda r: -(r[4] if np.isfinite(r[4]) else 1e30))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("model,redshift,sub_id,logMstar,resid,bbarolo_rc\n")
        for m, z, s, lm, r, rc in rows:
            fh.write(f"{m},{z},{s},{lm:.3f},{r:.4f},{rc}\n")
    print(f"wrote {args.out}  ({len(rows)} discs)")

    # --- calibration against the hand-flagged set ---
    import yaml
    excl = yaml.safe_load(args.exclude.read_text()) or {} if args.exclude.exists() else {}

    def is_flagged(model, z, sub):
        lst = ((excl.get(model) or {}).get(f"z{z:g}")) or []
        return int(sub) in {int(x) for x in lst}

    flagged = [r[4] for r in rows if is_flagged(*r[:3]) and np.isfinite(r[4])]
    kept = [r[4] for r in rows if not is_flagged(*r[:3]) and np.isfinite(r[4])]
    n_inf = sum(1 for r in rows if not np.isfinite(r[4]))
    print(f"\nresid split -- hand-flagged (n={len(flagged)}) vs kept (n={len(kept)}):")
    for lbl, arr in (("flagged", flagged), ("kept", kept)):
        if arr:
            a = np.array(arr)
            print(f"  {lbl:8s} median={np.median(a):5.2f}  "
                  f"75th={np.percentile(a, 75):5.2f}  90th={np.percentile(a, 90):5.2f}")
    print(f"  resid=inf (failed fit -> auto-exclude): {n_inf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
