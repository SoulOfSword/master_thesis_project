"""Mosaic of D/T vs log M_star at multiple redshifts.

For each (snap in `cfg['mosaic_snaps']`, model in `cfg['models']`), reads
the mordor_sample HDF5 produced by `scripts/data/build_mordor_sample.py`
and plots the binned fraction of disc galaxies (IsDisc == 1) with
Wilson 95% CI errorbars.

Bins are **shared per panel**: within each redshift panel, log M_star
from all models is pooled to define a common grid. Where galaxies are
plentiful the grid is the fixed `--bin-step` width (so rich low/mid-z
panels are unchanged); only when a panel is too sparse to fill that grid
(fewer than ~`--n-target` galaxies per model per bin) does it widen to
fewer uniform bins, so high-z panels show a few well-populated points
instead of many empty ones. The disc fraction is computed per model in
those shared bins, with Wilson 95% CI errorbars clipped to non-negative
by `disc_fraction_binned`.
Missing (model, snap) combos are skipped with a printed MISSING line.

Output: <fig_root>/morphology/dt_vs_mstar_mosaic_redshift.pdf
Also prints a per-bin tally to stdout (galaxies and discs in each shared
mass bin, per model and redshift, flagging which clear the n_min plotting
cut) and saves it next to the PDF as
dt_vs_mstar_mosaic_redshift_counts.csv.
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


def load_sample(scratch_mordor: Path, model: str, snap: int):
    p = scratch_mordor / "samples" / f"mordor_sample_{model}_{snap:03d}.hdf5"
    if not p.exists():
        return None, None
    arrays, _ = load_flat(p)
    return arrays, p


def _write_counts_table(rows, n_min, out_csv):
    """Print a per-bin galaxy/disc-count table and save it as CSV.

    Each entry of `rows` is one shared mass bin of one model in one
    redshift panel: (z, snap, model, logM_lo, logM_hi, n_total, n_disc).
    These are exactly the bins the mosaic plots, so the counts here are
    the populations behind each plotted point. The function:
      - prints one human-readable block per redshift, listing for every
        bin the galaxy count N, the disc count, the disc fraction, and a
        yes/no flag for whether N >= n_min (i.e. whether it is actually
        drawn as a point);
      - writes the same numbers to `out_csv` (one row per bin) so the
        exact point populations are recorded next to the figure.
    """
    print("\n" + "=" * 60)
    print("disc-fraction mosaic: galaxies and discs per shared mass bin")
    print("=" * 60)
    csv = ["z,snap,model,logM_lo,logM_hi,N,N_disc,f_disc,plotted"]
    last = None
    for z, snap, model, lo, hi, n, nd in rows:
        f = nd / n if n else float("nan")
        plotted = n >= n_min
        if (z, snap) != last:
            print(f"\nz = {z:g}  (snap {snap})")
            print(f"  {'model':6s} {'bin [logM*)':15s} {'N':>4s} "
                  f"{'discs':>6s} {'f':>6s}  plotted")
            last = (z, snap)
        print(f"  {model:6s} [{lo:5.2f},{hi:5.2f}) {n:>4d} {nd:>6d} "
              f"{f:>6.2f}  {'yes' if plotted else 'no'}")
        csv.append(f"{z:g},{snap},{model},{lo:.4f},{hi:.4f},{n},{nd},"
                   f"{f:.4f},{int(plotted)}")
    out_csv.write_text("\n".join(csv) + "\n")
    print(f"\n[plot_dt_mosaic] wrote counts table -> {out_csv}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--bin-step", type=float, default=0.2,
                   help="Minimum bin width in dex (the low-z floor)")
    p.add_argument("--n-target", type=int, default=10,
                   help="Target galaxies per model per bin; sets the number "
                        "of bins per panel from the sample size (default 10)")
    p.add_argument("--n-min", type=int, default=5,
                   help="Min galaxies per bin to plot (default 5)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PDF (default <fig_root>/morphology/...)")
    args = p.parse_args()

    cfg = load_config(args.config)
    scratch_mordor = Path(cfg["paths"]["scratch_mordor"])
    fig_root = Path(cfg["paths"]["fig_root"])
    snaps = list(cfg["mosaic_snaps"])
    models = list(cfg["models"])
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    model_colors = {m: cfg["model_colors"][m] for m in models}

    # Preload all available samples
    panel_data = {snap: {} for snap in snaps}
    for snap in snaps:
        for model in models:
            arrs, path = load_sample(scratch_mordor, model, snap)
            if arrs is None:
                print(f"MISSING: mordor_sample_{model}_{snap:03d}.hdf5")
                continue
            panel_data[snap][model] = arrs

    n_cols = len(snaps)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5),
                              sharey=True, constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    tally_rows = []   # (z, snap, model, logM_lo, logM_hi, N, N_disc) per bin
    for j, snap in enumerate(snaps):
        ax = axes[j]
        ax.tick_params(labelsize=13)
        z = snap_z.get(int(snap), float("nan"))

        # Shared bin grid pooled across all models in this panel
        all_log_m = []
        for model in models:
            if model not in panel_data[snap]:
                continue
            mstar = panel_data[snap][model]["Mstar"] # (n_galaxies,) array of M_star for this model in this panel
            mstar = mstar[mstar > 0]
            if len(mstar):
                all_log_m.append(np.log10(mstar)) # list of (n_galaxies,) arrays of log M_star for all models in this panel; used to define the shared bin grid
        if not all_log_m:
            ax.set_xlabel(r"$\log_{10}(M_\star\;[M_\odot])$")
            ax.set_title(f"$z = {z:g}$")
            continue
        pooled = np.concatenate(all_log_m) 
        # when a panel is too sparse to fill that grid (~n_target per model per bin) use fewer, wider uniform bins.
        fixed = np.arange(pooled.min(),
                          pooled.max() + args.bin_step, args.bin_step)
        n_count = len(pooled) // (len(all_log_m) * args.n_target) # len(all_log_m) is number of models with data in this panel, so n_count is the number of bins we would get if we used the fixed bin step and aimed for n_target per model per bin
        if n_count >= len(fixed) - 1:
            bins = fixed
        else:
            bins = np.linspace(pooled.min(), pooled.max(), max(2, n_count) + 1)

        for model in models:
            if model not in panel_data[snap]:
                continue
            arrs = panel_data[snap][model]
            mstar = arrs["Mstar"]
            good = mstar > 0
            log_m = np.log10(mstar[good])
            is_disc = np.asarray(arrs["IsDisc"][good]).astype(int)

            # tally galaxies + discs per shared bin (same lo<=x<hi edges
            # disc_fraction_binned uses) for the printed/saved counts table
            for i in range(len(bins) - 1):
                sel = (log_m >= bins[i]) & (log_m < bins[i + 1])
                tally_rows.append((z, int(snap), model, float(bins[i]),
                                   float(bins[i + 1]), int(sel.sum()),
                                   int(is_disc[sel].sum())))

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
        ax.set_title(f"$z = {z:g}$", fontsize=16)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("disc fraction", fontsize=16)
    axes[-1].legend(loc="upper right", fontsize=12)

    out_path = args.out or (fig_root / "morphology"
                            / "dt_vs_mstar_mosaic_redshift.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot_dt_mosaic] saved {out_path}")

    counts_csv = out_path.with_name(out_path.stem + "_counts.csv")
    _write_counts_table(tally_rows, args.n_min, counts_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
