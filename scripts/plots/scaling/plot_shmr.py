"""Stellar-halo mass relation figures from the mock disc sample.

STANDARD  (--relation standard): 2x3 mosaic, one panel per pipeline redshift,
log10 Mstar vs log10 M200c for the usable MORDOR discs (the sample from
galaxy_sidm.mock.tables.assemble: IsDisc==1, bbarolo_rc==0, not hand-flagged).
Per DM model: scatter + a running-median line so the model comparison stays
readable where the clouds overlap. Colours from config model_colors.

MODIFIED  (--relation modified): the redshift evolution of the stellar-to-halo
mass ratio at fixed stellar mass,

    y  =  f_Mstar(Mstar, z) / f_Mstar(Mstar, z0) ,      f_Mstar = Mstar / Mvir ,

plotted AGAINST Mstar (per your spec; f_V(z)/f_V(z0) == 1 by assumption, so no
velocities enter anywhere). Galaxies are binned in log10 Mstar; per
(model, z, bin) f_Mstar = median(Mstar/M200c) over the discs in the bin; each
value is divided by the SAME bin's value at the reference z0 (default 0.5 --
the lowest MORDOR sample; there is no z=0 run yet). One panel per DM model,
one line per redshift, unity line = z0. Bins with fewer than --min-count discs
at either z or z0 are left as gaps. The full numeric table is printed.

Default --relation both writes both figures. Paper style: bundled cmr10
(Computer Modern; usetex broken on Leonardo), large labels/ticks/markers.

Usage:
  python scripts/plots/scaling/plot_shmr.py                       # both figures
  python scripts/plots/scaling/plot_shmr.py --relation standard
  python scripts/plots/scaling/plot_shmr.py --relation modified --ref-z 1
  python scripts/plots/scaling/plot_shmr.py --mass-edges 9.4 9.8 10.2 10.6 11.0 11.4 11.8
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


def _running_median(x, y, nbins=12, min_pts=8):
    """(bin centres, median y) over equal-width x bins with >= min_pts points."""
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    cen, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= min_pts:
            cen.append(0.5 * (lo + hi))
            med.append(np.median(y[m]))
    return np.array(cen), np.array(med)


# --------------------------------------------------------------- standard ---
def plot_standard(rows, colors, outdir):
    """Mstar vs M200c mosaic: scatter + running median per model, per z."""
    zs = sorted({r["z"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, z in zip(axes.flat, zs):
        for model in MODELS:
            sel = [r for r in rows if r["z"] == z and r["model"] == model]
            if not sel:
                continue
            lH = np.log10([r["M200c"] for r in sel])
            lS = np.log10([r["Mstar"] for r in sel])
            ax.scatter(lH, lS, s=38, c=colors[model], alpha=0.45, lw=0,
                       rasterized=True, label=model)
            cen, med = _running_median(lH, lS)
            if cen.size:
                ax.plot(cen, med, c=colors[model], lw=3.0, zorder=5)
        ax.set_title(rf"$z = {z:g}$")
        ax.grid(alpha=0.25, lw=0.6)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10}\,M_{200c}\ \,[M_\odot]$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}\,M_\star\ \,[M_\odot]$")
    leg = axes.flat[0].legend(frameon=False, markerscale=1.6, handletextpad=0.2,
                              loc="upper left")
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "shmr_standard.pdf"
    fig.savefig(out, dpi=250)
    print(f"wrote {out}")


# --------------------------------------------------------------- modified ---
def plot_modified(rows, colors, ref_z, edges, min_count, outdir):
    """f_Mstar(Mstar,z)/f_Mstar(Mstar,z0) vs Mstar; lines per z, panel per model."""
    zs = sorted({r["z"] for r in rows}, reverse=True)
    if ref_z not in zs:
        print(f"--ref-z {ref_z} not among available z {zs}; using {min(zs)}")
        ref_z = min(zs)
    zlines = [z for z in zs if z != ref_z]           # high z first
    edges = np.asarray(edges, float)
    centres = 0.5 * (edges[:-1] + edges[1:])

    def fstar_curve(model, z):
        """median(Mstar/M200c) per log10-Mstar bin (nan where < min_count)."""
        sel = [r for r in rows if r["model"] == model and r["z"] == z]
        lm = np.log10([r["Mstar"] for r in sel])
        f = np.array([r["Mstar"] / r["M200c"] for r in sel])
        out = np.full(centres.size, np.nan)
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (lm >= lo) & (lm < hi)
            if m.sum() >= min_count:
                out[j] = np.median(f[m])
        return out

    cmap = plt.get_cmap("plasma")
    zcol = {z: cmap(0.08 + 0.72 * i / max(1, len(zlines) - 1))
            for i, z in enumerate(zlines)}

    fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 5.6), sharex=True,
                             sharey=True, constrained_layout=True)
    print(f"\nMODIFIED SHMR: f_Mstar(Mstar,z) / f_Mstar(Mstar,z0={ref_z:g}) "
          f"per log10 Mstar bin")
    hdr = "  ".join(f"[{lo:.1f},{hi:.1f})" for lo, hi in zip(edges[:-1], edges[1:]))
    for ax, model in zip(np.atleast_1d(axes).ravel(), MODELS):
        ref = fstar_curve(model, ref_z)
        print(f"{model}:  logM* bins: {hdr}")
        for z in zlines:
            ratio = fstar_curve(model, z) / ref
            print(f"  z={z:>3g}: " + "  ".join(
                f"{v:8.3f}" if np.isfinite(v) else f"{'--':>8}" for v in ratio))
            if np.isfinite(ratio).any():        # no line, no legend entry
                ax.plot(centres, ratio, "o-", color=zcol[z], lw=2.4, ms=8,
                        label=rf"$z = {z:g}$")
        ax.axhline(1.0, color="grey", lw=1.2, ls=":")
        ax.set_title(model, color=colors[model])
        ax.set_xlabel(r"$\log_{10}\,M_\star\ \,[M_\odot]$")
        ax.grid(alpha=0.25, lw=0.6)
    ax0 = np.atleast_1d(axes).ravel()[0]
    ax0.set_ylabel(
        rf"$f_{{M_\star}}(M_\star,z)\,/\,f_{{M_\star}}(M_\star,z_0\!=\!{ref_z:g})$")
    ax0.legend(frameon=False, handletextpad=0.4, loc="lower right")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "shmr_modified.pdf"
    fig.savefig(out, dpi=250)
    print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--relation", choices=["standard", "modified", "both"],
                   default="both")
    p.add_argument("--ref-z", type=float, default=0.5,
                   help="Reference redshift z0 for the modified relation.")
    p.add_argument("--mass-edges", type=float, nargs="+",
                   default=list(np.round(np.arange(9.4, 11.81, 0.3), 2)),
                   help="log10(Mstar) bin edges for the modified relation.")
    p.add_argument("--min-count", type=int, default=5,
                   help="Min discs per (model, z, Mstar bin) to form a median.")
    p.add_argument("--exclude", type=Path,
                   default=ROOT / "config" / "problematic_discs.yaml")
    p.add_argument("--outdir", type=Path, default=ROOT / "figures" / "shmr")
    args = p.parse_args()

    cfg = load_config(None)
    colors = cfg["model_colors"]
    rows = assemble(cfg, exclude_yaml=args.exclude)
    rows = [r for r in rows if r["Mstar"] > 0 and r["M200c"] > 0]
    if not rows:
        print("no usable discs -- has the pipeline + BBarolo fits run?")
        return 1

    if args.relation in ("standard", "both"):
        plot_standard(rows, colors, args.outdir)
    if args.relation in ("modified", "both"):
        plot_modified(rows, colors, args.ref_z, args.mass_edges,
                      args.min_count, args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
