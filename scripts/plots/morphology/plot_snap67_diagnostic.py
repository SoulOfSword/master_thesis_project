"""Single-snap diagnostic mosaic.

Three panels at one snap (default 67, z~0.5):
  1. Sample histogram: log10(M_star) per model.
  2. Disc fraction vs log M_star with Wilson 95% CI errorbars; shared
     bins pooled across models within the panel; yerr clipped to >=0.
  3. M_thin / M_star histogram per galaxy, weighted by 1/N.

Also prints (16, 50, 84) percentiles of M_thin / M_star per model.

Reads mordor_sample HDF5s produced by `scripts/data/build_mordor_sample.py`.

Output: <fig_root>/morphology/mordor_overview_snap<snap>.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat
from galaxy_sidm.morphology import disc_fraction_binned


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--snap", type=int, default=67)
    p.add_argument("--bin-step", type=float, default=0.2)
    p.add_argument("--n-min", type=int, default=5)
    p.add_argument("--n-bins-hist", type=int, default=30,
                   help="Sample-histogram bins (panel 1)")
    p.add_argument("--n-bins-mthin", type=int, default=20,
                   help="M_thin/M_star histogram bins (panel 3)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    scratch_mordor = Path(cfg["paths"]["scratch_mordor"])
    fig_root = Path(cfg["paths"]["fig_root"])
    models = list(cfg["models"])
    model_colors = {m: cfg["model_colors"][m] for m in models}

    data = {}
    for model in models:
        path = (scratch_mordor / "samples"
                / f"mordor_sample_{model}_{args.snap:03d}.hdf5")
        if not path.exists():
            print(f"MISSING: {path.name}")
            continue
        arrays, _ = load_flat(path)
        data[model] = arrays

    if not data:
        sys.exit("no mordor_sample HDF5s found for any model")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- panel 1: sample histogram -----------------------------------
    ax = axes[0]
    log_m_min = None
    for model in models:
        if model not in data:
            continue
        mstar = data[model]["Mstar"]
        mstar = mstar[mstar > 0]
        if len(mstar) == 0:
            continue
        log_m = np.log10(mstar)
        if log_m_min is None or log_m.min() < log_m_min:
            log_m_min = float(log_m.min())
        ax.hist(log_m, bins=args.n_bins_hist, histtype="step", lw=2,
                color=model_colors[model],
                label=f"{model} (N={len(mstar)})")
    ax.set_xlabel(r"$\log_{10}(M_\star\;[M_\odot])$", fontsize=16)
    ax.set_ylabel("N galaxies", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12)
    ax.set_title("Sample", fontsize=16)
    if log_m_min is not None:
        ax.set_xlim(log_m_min)

    # --- panel 2: D/T vs M_star with shared per-panel bins -----------
    ax = axes[1]
    all_log_m = []
    for model in models:
        if model not in data:
            continue
        mstar = data[model]["Mstar"]
        mstar = mstar[mstar > 0]
        if len(mstar):
            all_log_m.append(np.log10(mstar))
    if all_log_m:
        pooled = np.concatenate(all_log_m)
        bins = np.arange(pooled.min(),
                          pooled.max() + args.bin_step, args.bin_step)
        for model in models:
            if model not in data:
                continue
            mstar = data[model]["Mstar"]
            good = mstar > 0
            log_m = np.log10(mstar[good])
            is_disc = data[model]["IsDisc"][good]
            centres, fracs, lo, hi = disc_fraction_binned(
                log_m, is_disc, bins, n_min=args.n_min)
            if len(centres) == 0:
                continue
            color = model_colors[model]
            ax.plot(centres, fracs, "-o", color=color, label=model,
                    ms=10, markeredgecolor="grey", markeredgewidth=2)
            ax.errorbar(centres, fracs, yerr=np.array([lo, hi]),
                         fmt="none", color=color, alpha=0.4,
                         capsize=3, capthick=1.5, zorder=0)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel(r"$\log_{10}(M_\star\;[M_\odot])$", fontsize=16)
    ax.set_ylabel("disc fraction", fontsize=16)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12)
    ax.set_title(r"Disc fraction vs $M_\star$", fontsize=16)

    # --- panel 3: M_thin / M_star histogram --------------------------
    ax = axes[2]
    print("Mthin/Mstar percentiles (16, 50, 84):")
    for model in models:
        if model not in data:
            continue
        arrs = data[model]
        mstar = arrs["Mstar"]
        good = mstar > 0
        if not good.any():
            continue
        x = arrs["Mthin"][good] / mstar[good]
        weights = np.ones_like(x) / len(x)
        ax.hist(x, bins=args.n_bins_mthin, weights=weights,
                histtype="step", lw=2, color=model_colors[model],
                label=model)
        p16, p50, p84 = np.percentile(x, [16, 50, 84])
        print(f"  {model}: median={p50:.3f}  "
              f"[16, 84]=[{p16:.3f}, {p84:.3f}]  (N={len(x)})")
    ax.set_xlabel(r"$M_{\rm thin} / M_\star$", fontsize=16)
    ax.set_ylabel("fraction of galaxies", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12)
    ax.set_title("Thin-disc fraction per galaxy", fontsize=16)
    ax.set_xlim(0)

    plt.tight_layout()

    out_path = args.out or (fig_root / "morphology"
                            / f"mordor_overview_snap{args.snap:03d}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[plot_snap67_diagnostic] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
