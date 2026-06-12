"""Render N interacting galaxies for visual inspection.

Reads a mordor_sample HDF5, selects galaxies with
`sat_mass_ratio >= --m-ratio-min`, sorts by Mstar (ascending) and picks
N evenly-spaced examples in log-mass order. For each picked galaxy
renders a 3-panel (gas coldens, stellar coldens, stellar v_rad) figure
via `temet.vis.halo.renderSingleHalo`.

The notebook drops subhalo_id == 0 (the main central is not informative
here); this script does the same by default — override with
`--include-zero`.

Output (one PDF per galaxy):
    <fig_root>/morphology/interacting_examples/
        <model>_snap<snap>_sub<sid>.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", default="CDM",
                   choices=["CDM", "SIDM1", "vSIDM", "WDM3", "WDM5"])
    p.add_argument("--snap", type=int, default=67)
    p.add_argument("--m-ratio-min", type=float, default=0.2)
    p.add_argument("--n-pick", type=int, default=6)
    p.add_argument("--include-zero", action="store_true",
                   help="Include subhalo_id 0 (excluded by default)")
    p.add_argument("--res", type=int, default=1080)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    scratch_mordor = Path(cfg["paths"]["scratch_mordor"])
    fig_root = Path(cfg["paths"]["fig_root"])

    path = (scratch_mordor / "samples"
            / f"mordor_sample_{args.model}_{args.snap:03d}.hdf5")
    if not path.exists():
        sys.exit(f"missing sample: {path}")
    arrays, _ = load_flat(path)

    halo_ids = arrays["halo_ids"]
    mstar = arrays["Mstar"]
    sat_ratio = arrays["sat_mass_ratio"]

    sel = sat_ratio >= args.m_ratio_min
    if not args.include_zero:
        sel &= halo_ids != 0
    n_int = int(sel.sum())
    n_total = len(halo_ids)
    print(f"{args.model} interacting (sat M*/M* >= {args.m_ratio_min:g}): "
          f"{n_int}/{n_total} ({100 * n_int / max(n_total, 1):.1f}%)")
    if n_int == 0:
        sys.exit("no interacting galaxies in sample")

    ids_int = halo_ids[sel]
    mstar_int = mstar[sel]
    ratio_int = sat_ratio[sel]
    order = np.argsort(mstar_int)
    ids_int = ids_int[order]
    mstar_int = mstar_int[order]
    ratio_int = ratio_int[order]

    n_pick = min(args.n_pick, n_int)
    pick_idx = np.linspace(0, n_int - 1, n_pick).astype(int)
    picks = [(int(ids_int[k]), float(mstar_int[k]), float(ratio_int[k]))
             for k in pick_idx]

    print(f"\nSelected {n_pick} galaxies (log M_star -> subhalo_id, sat_ratio):")
    for sid, ms, r in picks:
        print(f"  log M*={np.log10(ms):.2f}  subhalo {sid}  "
              f"sat_ratio={r:.2f}")

    out_dir = args.out_dir or (fig_root / "morphology"
                                / "interacting_examples")
    out_dir.mkdir(parents=True, exist_ok=True)

    import temet
    sim = temet.sim(run="aida", variant=args.model,
                    res=args.res, snap=args.snap)
    for sid, ms, r in picks:
        save_path = (out_dir / f"{args.model}_snap{args.snap:03d}"
                                f"_sub{sid:06d}.pdf")
        config = {"plotStyle": "edged", "saveFilename": str(save_path)}
        panels = [
            {"partType": "gas",   "partField": "coldens_msunkpc2"},
            {"partType": "stars", "partField": "coldens_msunkpc2"},
            {"partType": "stars", "partField": "vrad",
             "size": 6.0, "sizeType": "rHalfMassStars"},
        ]
        common = dict(
            sP=sim,
            subhaloInd=int(sid),
            size=1.0,
            rVirFracs=[1.5],
            fracsType="rHalfMassStars",
            sizeType="rVirial",
            rotation="edge-on",
            labelScale=True,
        )
        print(f"\n=== subhalo {sid}: log M*={np.log10(ms):.2f}, "
              f"sat_ratio={r:.2f} -> {save_path} ===")
        temet.vis.halo.renderSingleHalo(panels, config, common)
    return 0


if __name__ == "__main__":
    sys.exit(main())
