"""Mosaic of log M_star histograms per (model, snap).

For each snap in `cfg['mosaic_snaps']`, overlays stepped histograms of
log10(M_star) for the three models. Reads the mordor_sample HDF5s
produced by `scripts/data/build_mordor_sample.py`.

Output: <fig_root>/morphology/sample_histogram_mosaic_redshift.pdf
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
    p.add_argument("--n-bins", type=int, default=30)
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
        for model in models:
            if model not in panel_data[snap]:
                continue
            mstar = panel_data[snap][model]["Mstar"]
            mstar = mstar[mstar > 0]
            if len(mstar) == 0:
                continue
            ax.hist(np.log10(mstar), bins=args.n_bins, histtype="step",
                    lw=2, color=model_colors[model],
                    label=f"{model} (N={len(mstar)})")
        ax.set_xlabel(r"$\log_{10}(M_\star\;[M_\odot])$", fontsize=16)
        ax.set_title(f"$z = {z:g}$", fontsize=16)
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=12)
    axes[0].set_ylabel("N galaxies", fontsize=16)

    out_path = args.out or (fig_root / "morphology"
                            / "sample_histogram_mosaic_redshift.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot_sample_histogram_mosaic] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
