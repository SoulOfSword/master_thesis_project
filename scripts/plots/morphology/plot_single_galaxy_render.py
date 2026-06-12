"""Single-galaxy 3-panel render: gas coldens, stars coldens, stars v_rad.

Reproduces the `notebooks/AIDA-TNG/04_morphology.ipynb` render:
gas column density, stellar column density, and stellar radial velocity
(v_rad), all rendered via `temet.vis.halo.renderSingleHalo`. Useful for
ad-hoc visual inspection of any (model, snap, subhalo_id).

For the MORDOR (eta, E) phase-space diagnostic see
`plot_eta_E_diagnostic.py` in the same directory.

Usage:
    python plot_single_galaxy_render.py --model CDM --snap 67 \\
        --subhalo-id 0
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", required=True,
                   choices=["CDM", "SIDM1", "vSIDM", "WDM3", "WDM5"])
    p.add_argument("--snap", required=True, type=int)
    p.add_argument("--subhalo-id", required=True, type=int)
    p.add_argument("--res", type=int, default=1080)
    p.add_argument("--size", type=float, default=1.0,
                   help="`size` in rHalfMassStars units (default 1.0)")
    p.add_argument("--rotation", default="edge-on",
                   help="Rotation (default edge-on)")
    p.add_argument("--vrad-size", type=float, default=6.0,
                   help="`size` for the v_rad panel in rHalfMassStars units")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory; default <fig_root>/morphology/")
    args = p.parse_args()

    cfg = load_config(args.config)
    fig_root = Path(cfg["paths"]["fig_root"])
    out_dir = args.out_dir or (fig_root / "morphology")
    out_dir.mkdir(parents=True, exist_ok=True)

    import temet
    sim = temet.sim(run="aida", variant=args.model,
                    res=args.res, snap=args.snap)

    save_path = out_dir / (
        f"render_{args.model}_snap{args.snap:03d}"
        f"_sub{args.subhalo_id:06d}.pdf")
    config = {"plotStyle": "edged", "saveFilename": str(save_path)}
    panels = [
        {"partType": "gas",   "partField": "coldens_msunkpc2"},
        {"partType": "stars", "partField": "coldens_msunkpc2"},
        {"partType": "stars", "partField": "vrad",
         "size": args.vrad_size, "sizeType": "rHalfMassStars"},
    ]
    common = dict(
        sP=sim,
        subhaloInd=int(args.subhalo_id),
        size=args.size,
        rVirFracs=[1.5],
        fracsType="rHalfMassStars",
        sizeType="rVirial",
        rotation=args.rotation,
        labelScale=True,
    )
    print(f"[plot_single_galaxy_render] {args.model} snap {args.snap} "
          f"subhalo {args.subhalo_id} -> {save_path}")
    temet.vis.halo.renderSingleHalo(panels, config, common)
    return 0


if __name__ == "__main__":
    sys.exit(main())
