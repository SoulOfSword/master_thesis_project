"""Tully-Fisher relation figure from the mock BBarolo rotation curves.

2x3 mosaic (one panel per pipeline redshift, z = 5 -> 0.5): log10 v_flat vs
log10 Mstar for the usable MORDOR discs. The sample comes from
galaxy_sidm.mock.tables.assemble -- IsDisc==1, bbarolo_rc==0, not hand-flagged
in config/problematic_discs.yaml -- and v_flat is the mean of the outermost N
ring VROT (default 3). Colours follow config model_colors (CDM black, SIDM1
blue, vSIDM red). An OLS power-law fit log v = a + b log M is drawn per model;
slope / intercept / vertical scatter / N are printed.

Paper style: matplotlib's bundled cmr10 (Computer Modern; text.usetex is
broken on Leonardo), large labels/ticks/markers, inward ticks on all sides.

Usage:
  python scripts/plots/scaling/plot_tfr.py               # stellar TFR
  python scripts/plots/scaling/plot_tfr.py --baryonic    # Mstar + M_neutral
  python scripts/plots/scaling/plot_tfr.py --n-outer 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config
from galaxy_sidm.mock.tables import assemble, MODELS

plt.rcParams.update({
    # paper font: bundled Computer Modern (cmr10) -- do NOT use usetex here
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    # legibility
    "axes.labelsize": 19,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 5.5, "ytick.major.size": 5.5,
})


def _fit(logM, logV):
    """OLS fit logV = a + b*logM. Returns (b, a, scatter_dex, N) or None."""
    m = np.isfinite(logM) & np.isfinite(logV)
    if m.sum() < 5:
        return None
    b, a = np.polyfit(logM[m], logV[m], 1)
    scatter = float(np.std(logV[m] - (a + b * logM[m])))
    return float(b), float(a), scatter, int(m.sum())


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baryonic", action="store_true",
                   help="Use Mbar = Mstar + M_neutral instead of Mstar.")
    p.add_argument("--n-outer", type=int, default=3,
                   help="Outermost rings averaged for v_flat (default 3).")
    p.add_argument("--exclude", type=Path,
                   default=ROOT / "config" / "problematic_discs.yaml")
    p.add_argument("--outdir", type=Path, default=ROOT / "figures" / "tfr")
    args = p.parse_args()

    cfg = load_config(None)
    colors = cfg["model_colors"]
    rows = assemble(cfg, exclude_yaml=args.exclude, n_outer=args.n_outer)
    rows = [r for r in rows
            if np.isfinite(r["v_flat"]) and r["v_flat"] > 0 and r["Mstar"] > 0]
    if not rows:
        print("no usable discs -- has the pipeline + BBarolo fits run?")
        return 1

    tag = "baryonic" if args.baryonic else "stellar"
    xlabel = (r"$\log_{10}\,(M_\star + M_{\rm neutral})\ \,[M_\odot]$"
              if args.baryonic else r"$\log_{10}\,M_\star\ \,[M_\odot]$")

    def mass(r):
        if args.baryonic and np.isfinite(r["M_neutral"]):
            return r["Mstar"] + r["M_neutral"]
        return r["Mstar"]

    zs = sorted({r["z"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True,
                             constrained_layout=True)
    print(f"\nTFR ({tag}), v_flat = mean of last {args.n_outer} rings")
    print(f"{'z':>5} {'model':6s} {'slope':>6} {'icpt':>7} {'scatter':>8} {'N':>5}")
    for ax, z in zip(axes.flat, zs):
        for model in MODELS:
            sel = [r for r in rows if r["z"] == z and r["model"] == model]
            if not sel:
                continue
            lM = np.log10([mass(r) for r in sel])
            lV = np.log10([r["v_flat"] for r in sel])
            ax.scatter(lM, lV, s=38, c=colors[model], alpha=0.55, lw=0,
                       rasterized=True, label=model)
            fit = _fit(lM, lV)
            if fit:
                b, a, sc, n = fit
                xx = np.linspace(np.nanmin(lM), np.nanmax(lM), 20)
                ax.plot(xx, a + b * xx, c=colors[model], lw=2.4)
                print(f"{z:>5g} {model:6s} {b:6.2f} {a:7.2f} {sc:8.3f} {n:5d}")
        ax.set_title(rf"$z = {z:g}$")
        ax.grid(alpha=0.25, lw=0.6)
        ax.plot(np.linspace(np.nanmin(lM), np.nanmax(lM)), 1/4.06 * (np.linspace(np.nanmin(lM), np.nanmax(lM)) - 1.26), c='k', lw=3, linestyle='--', label="DiTeodoro+23 z=0") #DiTeodoro+23 z=0 reference stellar TFR: log v_flat = (1/4.06)*(log Mstar - 1.26)
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}\,v_{\rm flat}\ \,[{\rm km\,s^{-1}}]$")
    leg = axes.flat[0].legend(frameon=False, markerscale=1.6, handletextpad=0.2,
                              loc="lower right")
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    # clamp the shared y-range to the bulk of the sample: a handful of junk-fit
    # outliers (v_flat of a few km/s) would otherwise squash every panel
    allV = np.array([np.log10(r["v_flat"]) for r in rows])
    lo, hi = np.percentile(allV, [0.5, 99.8])
    axes.flat[0].set_ylim(lo - 0.1, hi + 0.12)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"tfr_{tag}.pdf"
    fig.savefig(out, dpi=250)
    print(f"\nwrote {out}  ({len(rows)} discs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
