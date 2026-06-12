"""Mosaic of cored-NFW r_core vs M200c across redshift.

Given a list of rcore_*.hdf5 files (from compute_rcore.py), build a
mosaic of r_core (log y) vs log10(M200c). Same layout/conventions as
plot_gamma_mosaic.py: rows = sample ({DMO, FP}, both if available),
cols = redshifts from cfg['mosaic_snaps'].
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


def _load_rcore_files(paths):
    data = {}
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
        r_core = np.asarray(arrays["r_core"])
        keep = np.isfinite(r_core) & (r_core > 0) & (m200c > 0)
        data.setdefault((sample, snap), {})[base] = {
            "log_M200c": np.log10(m200c[keep]),
            "r_core":    r_core[keep],
        }
    return data


def _draw_panel(ax, data_for_panel, base_models, model_colors):
    bins = np.arange(9.5, 14.5, 0.3)
    for m in base_models:
        if m not in data_for_panel:
            continue
        d = data_for_panel[m]
        log_m = d["log_M200c"]
        rc = d["r_core"]
        color = model_colors.get(m, "tab:gray")
        ax.scatter(log_m, rc, s=15, alpha=0.25, color=color,
                   edgecolors="grey", linewidths=0.15)
        x, y = running_median(log_m, rc, bins, min_count=5)
        if len(x) > 0:
            ax.plot(x, y, "-",
                    color=lighten(color, 0.3), lw=3, ms=5, label=m,
                    path_effects=[pe.Stroke(linewidth=4, foreground="black"),
                                  pe.Normal()])
    ax.set_yscale("log")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--rcore-files", type=Path, nargs="+", required=True,
                   help="One or more rcore_*.hdf5 files (FP and/or DMO)")
    p.add_argument("--disc-only", action="store_true",
                   help="Use only *_disc.hdf5 inputs and tag the output PDF "
                        "with _disc. Default: exclude *_disc.hdf5 files.")
    p.add_argument("--snaps", type=int, nargs="*", default=None,
                   help="Snap list (low z -> high z). Defaults to cfg['mosaic_snaps'].")
    p.add_argument("--out-fig", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    snaps = [int(s) for s in (args.snaps or cfg["mosaic_snaps"])]
    model_colors = cfg["model_colors"]

    rcore_files = [p for p in args.rcore_files
                   if ("_disc" in str(p)) == args.disc_only]
    if not rcore_files:
        print(f"[plot_rcore_mosaic] no inputs matched "
              f"--disc-only={args.disc_only}; aborting")
        return 1
    data = _load_rcore_files(rcore_files)
    if not data:
        print("[plot_rcore_mosaic] no usable input files; aborting")
        return 1

    samples_present = sorted({s for (s, _) in data.keys()},
                             key=lambda x: 0 if x == "DMO" else 1)
    n_rows = len(samples_present)
    n_cols = len(snaps)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.3 * n_cols, 4 * n_rows),
                              sharex=True, sharey=True,
                              constrained_layout=True, dpi=200,
                              squeeze=False)

    base_models = list(cfg["models"])

    for row, sample in enumerate(samples_present):
        for col, snap in enumerate(snaps):
            ax = axes[row, col]
            ax.tick_params(labelsize=13)
            panel_data = data.get((sample, snap))
            if not panel_data:
                ax.text(0.5, 0.5, f"MISSING\n{sample}, snap {snap}",
                        transform=ax.transAxes, ha="center", va="center",
                        color="gray", fontsize=10)
                continue
            _draw_panel(ax, panel_data, base_models, model_colors)

            if row == 0:
                z = snap_z.get(snap, np.nan)
                ax.set_title(f"$z = {z:g}$" if np.isfinite(z) else f"snap {snap}",
                             fontsize=17)
            if col == 0:
                ax.set_ylabel(f"{sample}\n" + r"$r_{\rm core}$ [ckpc]",
                              fontsize=17)
            if row == n_rows - 1:
                ax.set_xlabel(r"$\log_{10}(M_{200c}\,[M_\odot])$",
                              fontsize=17)

    axes[0, -1].legend(loc="upper right", fontsize=12)

    out_fig = args.out_fig
    if out_fig is None:
        disc_tag = "_disc" if args.disc_only else ""
        out_fig = (Path(cfg["paths"]["fig_root"]) / "density"
                   / f"r_core_mosaic_redshift{disc_tag}.pdf")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_rcore_mosaic] wrote {out_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
