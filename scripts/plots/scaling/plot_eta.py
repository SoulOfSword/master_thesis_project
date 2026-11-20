"""Galaxies in the eta_bar - eta_rot plane (Santos-Santos et al. 2020, MNRAS 495, 58).

From the direct-method circular-velocity curves of scripts/data/compute_vcirc.py
(processed/vcirc: v_circ^2 of every component on 100 log radii, 0.1-50 kpc):
    v_max   the largest v_circ of the curve
    r_fid   2 (v_max / 70 km/s) kpc
    eta_rot v_circ(r_fid) / v_max
    eta_bar v_bar^2(r_fid) / v_circ^2(r_fid), the baryons' share of the gravitational
            acceleration at r_fid; v_bar^2 is the sum of the stars, gas, wind and
            black-hole terms (everything but dark matter)
Values at r_fid are interpolated linearly in radius (mock.tables.vcirc_at). The galaxies
are those of the final sample (mock.tables.final_sample, built at run time: MORDOR discs whose
neutral gas passes mock.tables.DISC_RULE) with a successful final BBarolo fit (info.json
bbarolo_rc == 0; the others are counted in the printout) whose
rotation curve has a flat part, by the rule of plot_tfr.py (mock.tables.flat_rings: SPARC_mod at
--flat-tol, else SPARC_mod at --retry-tol, else SPARC at --flat-tol). Galaxies whose v_max
is at the last radius of the curve, so that the true maximum may lie further out, are
counted in the printout.

One panel per redshift (in the order of REDSHIFTS, left to right and top to bottom, two
rows) with all three models: for each model an X at the median eta_bar and the median
eta_rot, and the lines that enclose the fractions LEVELS of its galaxies (68% solid, 95%
dashed). These are contours of a Gaussian KDE (scipy.stats.gaussian_kde, Scott's bandwidth)
in (log10 eta_bar, eta_rot), mirrored at eta_rot = 1 (eta_rot <= 1 by definition, so no
density is left above it), at the KDE density of the galaxies themselves at the 1 - p
quantile. A model with fewer than N_MIN galaxies in a panel is drawn as points, without
contours, and printed. Galaxies with non-finite or non-positive eta_bar or non-finite
eta_rot are left out and counted in the printout.

Paper style as plot_tfr.py.

Usage:
  python scripts/plots/scaling/plot_eta.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from contourpy import contour_generator
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat
from galaxy_sidm.mock.barolo import rings_file
from galaxy_sidm.mock.tables import MODELS, final_sample, flat_rings, vcirc_at
from galaxy_sidm.viz import legend_entries

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

REDSHIFTS = (0, 0.5, 1, 2, 3, 4)
BARYONS = ("stars", "gas", "wind", "bh")
LEVELS = (0.68, 0.95)
N_MIN = 10

pavel_dataset = pd.read_csv(ROOT / "data" / "pavel_etas_2026.csv", header=None, skiprows=2)
mp_filled = pavel_dataset.iloc[:, [0,1]].dropna()
mp_filled.columns = ["eta_bar", "eta_rot"]
mp_unfilled = pavel_dataset.iloc[:, [2,3]].dropna()
mp_unfilled.columns = ["eta_bar", "eta_rot"]

def etas(a, i):
    """(eta_rot, eta_bar, v_max at the last radius) of row i of a vcirc file."""
    R, v = a["R_direct"][i], a["vcirc_total"][i]
    k = int(np.nanargmax(v))
    r_fid = 2 * v[k] / 70
    v2_fid = vcirc_at(r_fid, R, a["v2_total"][i])
    v2_bar = vcirc_at(r_fid, R, sum(a[f"v2_{c}"][i] for c in BARYONS))
    return np.sqrt(v2_fid) / v[k], v2_bar / v2_fid, k == len(R) - 1


def kde_contours(x, y):
    """Lines of the Gaussian KDE of the points (x, y), mirrored at y = 1, that enclose the
    fractions LEVELS of them."""
    kde = gaussian_kde([x, y])

    def density(px, py):
        return np.where(py <= 1, kde([px, py]) + kde([px, 2 - py]), 0)

    pad = 3 * np.sqrt(np.diag(kde.covariance))
    lo, hi = np.min([x, y], axis=1) - pad, np.max([x, y], axis=1) + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
    gen = contour_generator(gx, gy, density(gx.ravel(), gy.ravel()).reshape(gx.shape))
    return [[np.column_stack([line[:, 0], np.minimum(line[:, 1], 1)]) for line in gen.lines(lev)]
            for lev in np.quantile(density(x, y), 1 - np.array(LEVELS))]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flat-tol", type=float, default=0.05,
                   help="Flatness tolerance of SPARC_mod and SPARC, as in plot_tfr.py (default 0.05).")
    p.add_argument("--retry-tol", type=float, default=0.10,
                   help="SPARC_mod tolerance for discs with no flat part at --flat-tol, as in plot_tfr.py (default 0.10).")
    p.add_argument("--outdir", type=Path, default=ROOT / "figures" / "eta")
    args = p.parse_args()

    cfg = load_config(None)
    colors = cfg["model_colors"]
    snap_of = {float(z): int(s) for s, z in cfg["snap_z"].items()}
    vdir = Path(cfg["paths"]["scratch_processed"]) / "vcirc"
    mart = Path(cfg["paths"]["scratch_processed"]).parent / "martini"
    groups, _ = final_sample(cfg)

    ncol = int(np.ceil(len(REDSHIFTS) / 2))
    fig, axes = plt.subplots(2, ncol, figsize=(4 * ncol, 8),
                             sharex=True, sharey=True, constrained_layout=True)
    for ax, z in zip(axes.flat, REDSHIFTS):
        snap = snap_of[z]
        for model in MODELS:
            subs = groups.get((model, snap), [])
            gal = {s: mart / f"z{z:g}" / model / f"gal_{s:06d}" for s in subs}
            fitted = [s for s in subs if json.loads((gal[s] / "info.json").read_text()).get("bbarolo_rc") == 0]
            vrot = {s: np.loadtxt(rings_file(gal[s] / "bbarolo"), usecols=2, ndmin=1) for s in fitted}
            flat = [s for s in fitted if flat_rings(vrot[s], args.flat_tol, args.retry_tol)[1] is not None]
            if not flat:
                continue
            a, _ = load_flat(vdir / f"vcirc_{model}_{snap:03d}.hdf5")
            row_of = {int(s): i for i, s in enumerate(a["halo_ids"])}
            missing = [s for s in flat if s not in row_of]
            if missing:
                raise KeyError(f"{model} z {z:g}: no v_circ curve for {missing}")
            eta = np.array([etas(a, row_of[s]) for s in flat])
            print(f"{model} z = {z:g}: {len(flat)} of {len(subs)} galaxies with a flat rotation curve "
                  f"({len(subs) - len(fitted)} without a successful final fit), "
                  f"{int(eta[:, 2].sum())} of them with v_max at the last radius")
            ok = np.isfinite(eta[:, 0]) & np.isfinite(eta[:, 1]) & (eta[:, 1] > 0)
            if not ok.all():
                print(f"{model} z = {z:g}: {int((~ok).sum())} galaxies with non-finite or non-positive eta_bar "
                      f"or non-finite eta_rot left out")
            rot, bar = eta[ok, 0], eta[ok, 1]
            if len(bar) < N_MIN:
                print(f"{model} z = {z:g}: N = {len(bar)} < N_MIN = {N_MIN}, shown as points")
                ax.scatter(bar, rot, s=15, c=colors[model], lw=0, zorder=3, rasterized=True)
            else:
                for lines, ls in zip(kde_contours(np.log10(bar), rot), ("-", "--"), strict=True):
                    for line in lines:
                        ax.plot(10 ** line[:, 0], line[:, 1], c=colors[model], ls=ls, zorder=3)
            ax.scatter(np.median(bar), np.median(rot), s=250, c=colors[model], marker="X",
                       edgecolors="k", lw=1, zorder=4, label=model)
        ax.scatter(mp_filled["eta_bar"], mp_filled["eta_rot"], s=40, c="darkturquoise", lw=1, edgecolors='k', rasterized=True, marker="D", label="MP+26 (z=0)")
        ax.scatter(mp_unfilled["eta_bar"], mp_unfilled["eta_rot"], s=40, facecolor="none", edgecolors="cyan", lw=2, rasterized=True, marker="D")
        ax.set_title(rf"$z = {z:g}$")
        ax.set_xscale("log")
        ax.set_xticks([0.1, 1])
    for ax in axes.flat[len(REDSHIFTS):]:
        ax.axis("off")
    for top, bottom in axes.T:
        ax = bottom if bottom.axison else top
        ax.set_xlabel(r"$\eta_{\rm bar}=(V_{\rm bar,fid}/V_{\rm fid})^2$")
        ax.tick_params(axis="x", which="both", labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\eta_{\rm rot}=V_{\rm fid}/V_{\rm max}$")

    axes[1, 0].legend(*legend_entries(axes.flat, MODELS), loc="upper left", frameon=False)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "eta_bar_eta_rot.pdf"
    fig.savefig(out, dpi=250)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
