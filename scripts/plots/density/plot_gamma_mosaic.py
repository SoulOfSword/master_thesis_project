"""Mosaic of gamma_DM vs M200c across redshift, optionally with DMO row.

Given a list of gamma_*.hdf5 files (from compute_gamma.py), build a
mosaic of gamma_DM vs log10(M200c). Each panel corresponds to one
(sample, redshift) pair. The sample (DMO / FP) is inferred from the
model name in each file: variants ending in "-Dark" go to the DMO row.

Layout:
    rows = {DMO, FP} if both present, else single row.
    cols = sorted redshifts (low z -> high z), pulled from the config's
           snap_z map using each file's metadata.snap.

The mosaic snaps default to cfg['mosaic_snaps'] ([67, 50, 33, 25, 21,
17]); panels with no available data print a MISSING line and stay
blank.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat
from galaxy_sidm.viz import lighten, running_median


def _strip_dark(model):
    return model[:-5] if model.endswith("-Dark") else model


def _classify_sample(model):
    return "DMO" if model.endswith("-Dark") else "FP"


def _load_gamma_files(paths):
    """Load gamma files, group by (sample, snap, base_model).

    Returns:
        data: {(sample, snap): {base_model: arrays}}, where arrays has
              keys 'log_M200c' and 'gamma_dm' (only finite entries).
        snaps_present: set of snaps seen.
    """
    data = {}
    snaps_present = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        arrays, attrs = load_flat(path)
        meta = attrs.get("metadata", {})
        model = str(meta.get("model"))
        snap = int(meta.get("snap"))
        sample = _classify_sample(model)
        base = _strip_dark(model)
        m200c = np.asarray(arrays["M200c"])
        gamma = np.asarray(arrays["gamma_dm"])
        finite = np.isfinite(gamma) & (m200c > 0)
        data.setdefault((sample, snap), {})[base] = {
            "log_M200c": np.log10(m200c[finite]),
            "gamma_dm":  gamma[finite],
        }
        snaps_present.add(snap)
    return data, snaps_present


def _draw_panel(ax, data_for_panel, base_models, model_colors):
    """Scatter + running-median per base model on one axes."""
    bins = np.arange(9.5, 14.5, 0.3)
    for m in base_models:
        if m not in data_for_panel:
            continue
        d = data_for_panel[m]
        log_m = d["log_M200c"]
        gamma = d["gamma_dm"]
        color = model_colors.get(m, "tab:gray")
        ax.scatter(log_m, gamma, s=15, alpha=0.25, color=color,
                   edgecolors="grey", linewidths=0.15)
        x, y = running_median(log_m, gamma, bins, min_count=5)
        if len(x) > 0:
            ax.plot(x, y, "-",
                    color=lighten(color, 0.3), lw=3, ms=5, label=m,
                    path_effects=[pe.Stroke(linewidth=4, foreground="black"),
                                  pe.Normal()])
    ax.axhline(-1, color="gray", ls="--", lw=1.1)
    ax.axhline(0,  color="gray", ls="--", lw=1.1)
    ax.set_ylim(-3.5, 0.1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--disc-only", action="store_true",
                   help="Use only *_disc.hdf5 inputs (MORDOR-disc subset) "
                        "and tag the output PDF with _disc. Default: "
                        "exclude *_disc.hdf5 files.")
    p.add_argument("--gamma-files", type=Path, nargs="+", required=True,
                   help="One or more gamma_*.hdf5 files (FP and/or DMO)")
    p.add_argument("--snaps", type=int, nargs="*", default=None,
                   help="Snap list (low z -> high z). Defaults to cfg['mosaic_snaps'].")
    p.add_argument("--out-fig", type=Path, default=None,
                   help="Output figure path (default under fig_root/density/)")
    args = p.parse_args()

    cfg = load_config(args.config)
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    snaps = args.snaps or cfg["mosaic_snaps"]
    snaps = [int(s) for s in snaps]
    model_colors = cfg["model_colors"]

    gamma_files = [p for p in args.gamma_files
                   if ("_disc" in str(p)) == args.disc_only]
    if not gamma_files:
        print(f"[plot_gamma_mosaic] no inputs matched "
              f"--disc-only={args.disc_only}; aborting")
        return 1
    data, _present = _load_gamma_files(gamma_files)
    if not data:
        print("[plot_gamma_mosaic] no usable input files; aborting")
        return 1

    samples_present = sorted({s for (s, _) in data.keys()},
                             key=lambda x: 0 if x == "DMO" else 1)
    n_rows = len(samples_present)
    n_cols = len(snaps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 4 * n_rows),
                              sharex=True, sharey=True,
                              constrained_layout=True, dpi=200,
                              squeeze=False)

    # Stable model-order so legends line up across panels.
    base_models = list(cfg["models"])  # e.g. ["CDM", "SIDM1", "vSIDM"]

    for row, sample in enumerate(samples_present):
        for col, snap in enumerate(snaps):
            ax = axes[row, col]
            ax.tick_params(labelsize=13)
            panel_data = data.get((sample, snap))
            if not panel_data:
                ax.text(0.5, 0.5, f"MISSING\n{sample}, snap {snap}",
                        transform=ax.transAxes, ha="center", va="center",
                        color="gray", fontsize=10)
                ax.set_xlim(9.5, 14.5)
                continue
            _draw_panel(ax, panel_data, base_models, model_colors)

            if row == 0:
                z = snap_z.get(snap, np.nan)
                ax.set_title(f"$z = {z:g}$" if np.isfinite(z) else f"snap {snap}",
                             fontsize=17)
            if col == 0:
                ax.set_ylabel(f"{sample}\n" + r"$\gamma_{\rm DM}$",
                              fontsize=17)
            if row == n_rows - 1:
                ax.set_xlabel(r"$\log_{10}(M_{200c}\,[M_\odot])$",
                              fontsize=17)

    axes[0, -1].legend(loc="upper right", fontsize=12)

    out_fig = args.out_fig
    if out_fig is None:
        disc_tag = "_disc" if args.disc_only else ""
        out_fig = (Path(cfg["paths"]["fig_root"]) / "density"
                   / f"gamma_dm_mosaic_redshift{disc_tag}.pdf")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_gamma_mosaic] wrote {out_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
