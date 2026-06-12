"""Star-forming main sequence (SFMS) per model across redshift.

For centrals (GroupFirstSub-tagged subhalos) with M* > mstar_min, plot
log10(SFR) vs log10(M*) as median + 16-84 percentile envelope, with one
panel per model and one line per redshift.

Loads data directly from temet.sim (no upstream HDF5).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config


# Snaps used by the notebook for the SFMS panel.
SFMS_SNAPS_DEFAULT = [99, 67, 50, 33, 25, 21, 17]


def _per_snap_data(model, snap, mstar_min):
    """Return (Mstar, SFR) arrays for centrals of one (model, snap)."""
    import temet
    sim = temet.sim(run="aida", variant=model, res=1080, snap=snap)
    first_sub = sim.halos("GroupFirstSub")
    mstar_all = sim.subhalos("mstar_tot")
    sfr_all = sim.subhalos("SubhaloSFR")
    valid = (first_sub >= 0) & (first_sub < len(mstar_all))
    idx = first_sub[valid]
    ms = mstar_all[idx]
    sf = sfr_all[idx]
    keep = ms > mstar_min
    return ms[keep], sf[keep]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--snaps", type=int, nargs="*", default=None,
                   help=f"Default: {SFMS_SNAPS_DEFAULT}")
    p.add_argument("--mstar-min", type=float, default=1e8)
    p.add_argument("--out-fig", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    models = list(cfg["models"])
    snaps = [int(s) for s in (args.snaps or SFMS_SNAPS_DEFAULT)]

    bins = np.arange(8.0, 12.1, 0.25)
    bin_mid = 0.5 * (bins[:-1] + bins[1:])

    sfms_data = {}
    for snap in snaps:
        for model in models:
            # Skip the SIDM1 snap-99 combination — not present in
            # AIDA-TNG, as the notebook flags.
            if model == "SIDM1" and snap == 99:
                continue
            try:
                ms, sf = _per_snap_data(model, snap, args.mstar_min)
            except Exception as exc:
                print(f"MISSING: ({model}, snap {snap}): {exc}")
                continue
            sfms_data[(model, snap)] = (ms, sf)

    if not sfms_data:
        print("[plot_sfms] no data; aborting")
        return 1

    fig, axes = plt.subplots(1, len(models), figsize=(16, 5),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    cmap = plt.cm.viridis
    z_sorted = sorted(set(snap_z.get(s, np.nan) for s in snaps
                          if snap_z.get(s) is not None))
    if not z_sorted:
        z_sorted = sorted(snaps)
    z_colors = {z: cmap(i / max(len(z_sorted) - 1, 1))
                for i, z in enumerate(z_sorted)}

    for col, model in enumerate(models):
        ax = axes[col]
        ax.tick_params(labelsize=13)
        # Order panels by redshift for a clean colour ramp.
        sorted_snaps = sorted(
            [s for s in snaps if (model, s) in sfms_data],
            key=lambda s: snap_z.get(s, np.inf),
        )
        for snap in sorted_snaps:
            z = snap_z.get(snap, np.nan)
            ms, sf = sfms_data[(model, snap)]
            active = sf > 0
            log_m = np.log10(ms[active])
            log_s = np.log10(sf[active])
            medians, p16s, p84s, centres = [], [], [], []
            for j in range(len(bins) - 1):
                in_bin = (log_m >= bins[j]) & (log_m < bins[j + 1])
                if in_bin.sum() >= 5:
                    medians.append(np.median(log_s[in_bin]))
                    p16s.append(np.percentile(log_s[in_bin], 16))
                    p84s.append(np.percentile(log_s[in_bin], 84))
                    centres.append(bin_mid[j])
            if not medians:
                continue
            c = z_colors.get(z, "tab:gray")
            label = f"z={z:.1f}" if np.isfinite(z) else f"snap {snap}"
            ax.plot(centres, medians, "-", color=c, lw=2, label=label)
            ax.fill_between(centres, p16s, p84s, color=c, alpha=0.15)
        ax.set_xlabel(r"$\log_{10}(M_\star\;/\;\mathrm{M_\odot})$",
                      fontsize=16)
        ax.set_title(model, fontsize=16)
        if col == 0:
            ax.set_ylabel(r"$\log_{10}(\mathrm{SFR}\;/\;\mathrm{M_\odot\,yr^{-1}})$",
                          fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)

    axes[0].set_xlim(8, 12)
    axes[0].set_ylim(-3, 2)
    fig.suptitle("SFMS: SubhaloSFR vs mstar_tot (centrals)", y=1.02,
                 fontsize=16)
    fig.tight_layout()

    out_fig = args.out_fig
    if out_fig is None:
        out_fig = (Path(cfg["paths"]["fig_root"]) / "density" / "sfms.pdf")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_sfms] wrote {out_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
