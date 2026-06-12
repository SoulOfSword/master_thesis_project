"""Mosaic of (M_thin + M_thick) / M_star histograms.

Per-galaxy weighted histograms (weights = 1/N) of
`(Mthin + Mthick) / Mstar` for the three models at each snap in
`cfg['mosaic_snaps']`. Also prints 16th/50th/84th percentiles per model
per snap.

Reads the mordor_sample HDF5s produced by
`scripts/data/build_mordor_sample.py`.

Output: <fig_root>/morphology/mthin_over_mstar_mosaic_redshift.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    scratch_mordor = Path(cfg["paths"]["scratch_mordor"])
    fig_root = Path(cfg["paths"]["fig_root"])
    snaps = list(cfg["mosaic_snaps"])
    models = list(cfg["models"])
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    model_colors = {m: cfg["model_colors"][m] for m in models}

    panel_data = {snap: {} for snap in snaps}
    for snap in snaps:
        for model in models:
            path = (scratch_mordor / "samples"
                    / f"mordor_sample_{model}_{snap:03d}.hdf5")
            if not path.exists():
                print(f"MISSING: {path.name}")
                continue
            arrays, _ = load_flat(path)
            panel_data[snap][model] = arrays

    n_cols = len(snaps)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5),
                              sharey=True, constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    for j, snap in enumerate(snaps):
        ax = axes[j]
        z = snap_z.get(int(snap), float("nan"))
        print(f"\nz = {z:g}  (snap {snap})  "
              f"(Mthin+Mthick)/Mstar percentiles (16, 50, 84):")
        for model in models:
            if model not in panel_data[snap]:
                continue
            arrs = panel_data[snap][model]
            mstar = arrs["Mstar"]
            good = mstar > 0
            if not good.any():
                continue
            x = ((arrs["Mthin"][good] + arrs["Mthick"][good])
                 / mstar[good])
            weights = np.ones_like(x) / len(x)
            ax.hist(x, bins=args.n_bins, weights=weights,
                    histtype="step", lw=2, color=model_colors[model],
                    label=model)
            p16, p50, p84 = np.percentile(x, [16, 50, 84])
            print(f"  {model}: median={p50:.3f}  "
                  f"[16, 84]=[{p16:.3f}, {p84:.3f}]  (N={len(x)})")
        ax.set_xlabel(r"$M_{\rm thin} / M_\star$", fontsize=16)
        ax.set_title(f"$z = {z:g}$", fontsize=16)
        ax.set_xlim(0)
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=12)
    axes[0].set_ylabel("fraction of galaxies", fontsize=16)

    out_path = args.out or (fig_root / "morphology"
                            / "mthin_over_mstar_mosaic_redshift.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[plot_mthin_histogram_mosaic] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
