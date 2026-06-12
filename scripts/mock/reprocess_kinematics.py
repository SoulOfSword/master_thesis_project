"""Replot the kinematics figure and recompute V/sigma for every galaxy that
already has a BBarolo fit. Rings are trimmed to the HI emission, so this
corrects the V/sigma that the data-free outer rings used to inflate. No
re-fit: reads existing bbarolo products only, so it's fast.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config
from galaxy_sidm.mock.kinematics import plot_kinematics, ring_kinematics


def _one(gal_dir):
    gal_dir = Path(gal_dir)
    bb = gal_dir / "bbarolo"
    if not list(bb.rglob("*_0mom.fits")):
        return "nofit", None
    try:
        ij = gal_dir / "info.json"
        info = json.loads(ij.read_text()) if ij.exists() else {}
        V, sigma, vsig = ring_kinematics(bb)
        info["V"], info["sigma"], info["V_over_sigma"] = V, sigma, vsig
        ij.write_text(json.dumps(info, indent=2))
        title = (f"{info.get('model','?')}" + r" $\vert$ " + f"z={info.get('redshift', float('nan')):g}" + r" $\vert$ " + f"subID {info.get('sub_id','?')}" + r" $\vert$ " + f"IsDisc={info.get('IsDisc','?')}")
        plot_kinematics(bb, gal_dir / "kinematics.png", suptitle=title)
        return "ok", vsig
    except Exception:
        return "fail", f"{gal_dir.name}: {traceback.format_exc().splitlines()[-1]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--ncpu", type=int, default=8)
    args = p.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["scratch_processed"]).parent / "martini"
    gals = sorted(str(d) for d in root.glob("z*/*/gal_*") if (d / "bbarolo").is_dir())
    print(f"reprocessing {len(gals)} galaxies on {args.ncpu} cpus", flush=True)

    n = {"ok": 0, "nofit": 0, "fail": 0}
    with Pool(args.ncpu) as pool:
        for i, (status, payload) in enumerate(
                pool.imap_unordered(_one, gals, chunksize=4), 1):
            n[status] += 1
            if status == "fail":
                print("FAIL", payload, flush=True)
            if i % 500 == 0:
                print(f"  {i}/{len(gals)} ...", flush=True)
    print(f"done: ok={n['ok']} nofit={n['nofit']} fail={n['fail']}")


if __name__ == "__main__":
    main()
