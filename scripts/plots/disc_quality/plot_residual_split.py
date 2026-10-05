"""Residual split of hand-classified calibration galaxies: discs, unsure discs, perturbed.

    res1 = sum_V (D - M)^2 / sigma^2 / N_V  -  1
    res2 = sum_V |D - M|   / sigma   / N_V  -  sqrt(2/pi)

per voxel and with the noise taken out, over the N_V voxels inside the final fit's mask
or where the final model is above 1 sigma (galaxy_sidm.mock.residuals); sigma is the cube
noise (its first and last 3 channels), for the PV too. A perfect fit gives 0 for both.

Each figure has two side-by-side panels (res1 left, res2 right; the axis label is the
formula), one jittered column per class. The calibration galaxies are listed in --labels
(config/residual_calibration.yaml) as discs, unsure_discs and perturbed, per model and
redshift; by default the first model and redshift in that file are used. Near-perfect
fits can come out slightly negative: if a panel has any, its axis is symlog (linear
through 0) instead of log, so no galaxy is dropped.

With --all [pv|cube|both]: the residuals of every galaxy in the mock manifest instead
(all models and redshifts), one column per redshift with the models side by side as
small dots, the calibration galaxies on top, and in each panel the midpoint between
the highest calibration disc and the lowest perturbed galaxy (when they don't
overlap). Written as residual_split_pv_all.pdf / residual_split_cube_all.pdf. The cube
version reads every full cube, so run it on a compute node
(scripts/slurm/residual_split_all.sbatch). Galaxies without a final fit have no
residuals; they are listed and not plotted.

Usage:
  python scripts/plots/disc_quality/plot_residual_split.py                # first model/z in the labels file
  python scripts/plots/disc_quality/plot_residual_split.py --model CDM --z 0.5
  python scripts/plots/disc_quality/plot_residual_split.py --labels my_labels.yaml
  python scripts/plots/disc_quality/plot_residual_split.py --all          # every galaxy, PV
  python scripts/plots/disc_quality/plot_residual_split.py --all cube     # every galaxy, cubes
  python scripts/plots/disc_quality/plot_residual_split.py --all both
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config
from galaxy_sidm.mock.residuals import cube_residuals, pv_residuals

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

CLASSES = ("discs", "unsure_discs", "perturbed")          # keys in the labels file
CLASS_LABEL = {"discs": "discs", "unsure_discs": "unsure discs", "perturbed": "perturbed"}
CLASS_STYLE = {"discs": dict(color="tab:blue"), "unsure_discs": dict(color="tab:orange"),
               "perturbed": dict(color="tab:red")}
RESIDUALS = {"pv": pv_residuals, "cube": cube_residuals}
KIND_TITLE = {"pv": "PV", "cube": "Cube"}
YLABELS = (r"$\sum_{V}(D-M)^{2}/\sigma^{2}\,/\,N_V\;-\;1$",
           r"$\sum_{V}|D-M|/\sigma\,/\,N_V\;-\;\sqrt{2/\pi}$")


def _yscale(ax, values):
    """Log axis, unless values reach 0 or below (near-perfect fits): then symlog, linear
    through 0 with a perfect fit marked at 0, so no galaxy is dropped."""
    v = values[np.isfinite(values)]
    if not len(v) or v.min() > 0:
        ax.set_yscale("log")          # the residuals span decades
        return
    # linear within a tenth of the typical value, log beyond
    med = np.median(np.abs(v))
    e0 = int(np.floor(np.log10(med))) - 1 if med > 0 else 0
    ax.set_yscale("symlog", linthresh=10.0 ** e0)
    # ticks at 0 and at the decades outside the linear range (matplotlib also puts one
    # inside it, which crowds the 0 label)
    top = int(np.ceil(np.log10(max(v.max(), 10.0 ** e0))))
    bot = int(np.ceil(np.log10(max(-v.min(), 10.0 ** e0))))
    ax.yaxis.set_major_locator(FixedLocator(
        [-10.0 ** e for e in range(bot, e0 - 1, -1)] + [0.0]
        + [10.0 ** e for e in range(e0, top + 1)]))
    ax.axhline(0.0, color="grey", lw=1.2, ls=":")


def _figure(values, title, out):
    """One figure: two panels (res1, res2), one jittered column per class.

    values: {class: [(res1, res2), ...]}.
    """
    classes = [c for c in CLASSES if values.get(c)]
    rng = np.random.default_rng(0)                    # reproducible jitter
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)
    for k, ax in enumerate(axes):                     # k = 0: res1, 1: res2
        for xc, cls in enumerate(classes):
            v = np.array([r[k] for r in values[cls]])
            good = np.isfinite(v)
            x = xc + rng.uniform(-0.13, 0.13, size=good.sum())
            ax.scatter(x, v[good], s=80, alpha=0.85, lw=0.8,
                       edgecolors="white", zorder=3, **CLASS_STYLE[cls])
        _yscale(ax, np.array([r[k] for c in classes for r in values[c]]))
        ax.set_ylabel(YLABELS[k])
        ax.set_xlim(-0.5, len(classes) - 0.5)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([CLASS_LABEL[c] for c in classes], fontsize=18)
        ax.grid(alpha=0.25, lw=0.6, axis="y")
    fig.suptitle(title, fontsize=19)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


MODEL_OFFSET = {"CDM": -0.25, "SIDM1": 0.0, "vSIDM": 0.25}   # x offset inside a redshift column


def _all_residuals(cfg, mart, kind, workers):
    """Residuals ('pv' or 'cube') of every manifest galaxy: (models, redshifts, subIDs, vals),
    vals[i, k] = res(k+1) of galaxy i (NaN without a final fit)."""
    residuals = RESIDUALS[kind]
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    gals = []
    for line in (mart / "manifest.txt").read_text().splitlines():
        if line.strip():
            model, snap, sub = line.split()
            gals.append((model, snap_z[int(snap)], int(sub)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        vals = np.array(list(ex.map(
            lambda g: residuals(mart / f"z{g[1]:g}" / g[0] / f"gal_{g[2]:06d}"), gals)))
    return (np.array([g[0] for g in gals]), np.array([g[1] for g in gals]),
            np.array([g[2] for g in gals]), vals)


def _figure_all(cfg, models, zg, subs, vals, calib, calib_model, calib_z, kind, out):
    """Residuals of every manifest galaxy (one kind); calibration galaxies drawn on top.

    vals: (n_galaxies, 2) res1, res2; calib: {class: [(res1, res2), ...]}
    for the calibration galaxies of (calib_model, calib_z), same kind.
    """
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    zs = [snap_z[s] for s in cfg["mosaic_snaps"]]                 # low z -> high z
    name = KIND_TITLE[kind]

    print(f"\n{name}: res1 of all {len(vals)} galaxies: median [16th, 84th percentile]")
    for z in zs:
        cells = []
        for m in MODEL_OFFSET:
            v = vals[(models == m) & (zg == z), 0]
            v = v[np.isfinite(v)]
            if len(v):
                cells.append(f"{m} {np.median(v):7.3g} [{np.percentile(v, 16):.3g}, "
                             f"{np.percentile(v, 84):.3g}] N={len(v)}")
        print(f"  z={z:<4g} " + "   ".join(cells))
    for k in range(2):
        bad = ~np.isfinite(vals[:, k])
        ids = [f"{m} z{zb:g} {s}" for m, zb, s in zip(models[bad], zg[bad], subs[bad])]
        print(f"  res{k + 1}: {len(ids)} galaxies without a value (not plotted)"
              + (": " + ", ".join(ids[:20]) + (" ..." if len(ids) > 20 else "") if ids else ""))

    rng = np.random.default_rng(0)                    # reproducible jitter
    x_calib = zs.index(calib_z) + MODEL_OFFSET[calib_model]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), constrained_layout=True)
    for k, ax in enumerate(axes):                     # k = 0: res1, 1: res2
        for iz, z in enumerate(zs):
            for m, dx in MODEL_OFFSET.items():
                sel = (models == m) & (zg == z) & np.isfinite(vals[:, k])
                ax.scatter(iz + dx + rng.uniform(-0.09, 0.09, sel.sum()), vals[sel, k],
                           s=3, lw=0, alpha=0.4, color=cfg["model_colors"][m], rasterized=True,
                           label=m if k == 0 and iz == 0 else None)
        for cls in CLASSES:
            if not calib.get(cls):
                continue
            v = np.array([r[k] for r in calib[cls]])
            v = v[np.isfinite(v)]
            ax.scatter(x_calib + rng.uniform(-0.09, 0.09, len(v)), v, s=26, lw=0.7,
                       edgecolors="k", zorder=3,
                       label=f"calibration {CLASS_LABEL[cls]}" if k == 0 else None,
                       **CLASS_STYLE[cls])
        if calib.get("discs") and calib.get("perturbed"):
            d_max = np.nanmax([r[k] for r in calib["discs"]])
            p_min = np.nanmin([r[k] for r in calib["perturbed"]])
            if d_max < p_min:
                ax.axhline(0.5 * (d_max + p_min), color="grey", ls="--", lw=1.2,
                           label="calibration midpoint")
        _yscale(ax, vals[:, k])
        ax.set_ylabel(YLABELS[k])
        ax.set_xticks(range(len(zs)))
        ax.set_xticklabels([f"$z={z:g}$" for z in zs])
        ax.set_xlim(-0.5, len(zs) - 0.5)
        ax.grid(alpha=0.25, lw=0.6, axis="y")
    # one legend under the panels, so it never covers points
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="outside lower center",
               ncol=len(handles), frameon=False, fontsize=13, markerscale=2.0)
    fig.suptitle(f"{name}, all galaxies", fontsize=19)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, choices=["CDM", "SIDM1", "vSIDM"],
                   help="calibration model (default: the first one in --labels)")
    p.add_argument("--z", type=float, default=None,
                   help="calibration redshift (default: the first one for that model in --labels)")
    p.add_argument("--labels", type=Path,
                   default=ROOT / "config" / "residual_calibration.yaml")
    p.add_argument("--outdir", type=Path, default=ROOT / "figures" / "disc_quality")
    p.add_argument("--all", nargs="?", const="pv", default=None, choices=["pv", "cube", "both"],
                   help="residuals of every galaxy in the mock manifest (pv, cube or both), with "
                        "the calibration galaxies on top -> residual_split_<pv|cube>_all.pdf")
    p.add_argument("--workers", type=int, default=8,
                   help="galaxies read in parallel with --all (default 8)")
    args = p.parse_args()

    d = yaml.safe_load(args.labels.read_text()) or {}
    model = args.model or next(iter(d), None)
    by_z = d.get(model) or {}
    zkey = f"z{args.z:g}" if args.z is not None else next(iter(by_z), None)
    labels = by_z.get(zkey) or {}
    subs = {cls: [int(s) for s in (labels.get(cls) or [])] for cls in CLASSES}
    if not subs["discs"] or not subs["perturbed"]:
        print(f"no calibration subIDs for {model} {zkey} in {args.labels} -- "
              f"fill the 'discs' and 'perturbed' lists first.")
        return 1
    z = float(zkey[1:])

    cfg = load_config(None)
    mart = Path(cfg["paths"]["scratch_processed"]).parent / "martini"

    if args.all:   # the calibration residuals are only needed for the overlay
        for kind in (("pv", "cube") if args.all == "both" else (args.all,)):
            calib = {cls: [RESIDUALS[kind](mart / zkey / model / f"gal_{s:06d}") for s in ids]
                     for cls, ids in subs.items()}
            models, zg, subs_all, vals = _all_residuals(cfg, mart, kind, args.workers)
            _figure_all(cfg, models, zg, subs_all, vals, calib, model, z, kind,
                        args.outdir / f"residual_split_{kind}_all.pdf")
        return 0

    # vals[kind][class] = [(res1, res2), ...] in the order of subs[class]
    vals = {kind: {cls: [] for cls in CLASSES} for kind in RESIDUALS}
    for cls in CLASSES:
        for sub in subs[cls]:
            g = mart / zkey / model / f"gal_{sub:06d}"
            for kind, residuals in RESIDUALS.items():
                vals[kind][cls].append(residuals(g))

    print(f"\n{model} {zkey}   res1 = sum (D-M)^2/sig^2 / N - 1,  res2 = sum |D-M|/sig / N - 0.798"
          f"   (N: voxels in the final fit's mask or with model > 1 sigma)")
    print(f"{'class':13s} {'subID':>7} {'cube res1':>10} {'cube res2':>10} {'pv res1':>10} {'pv res2':>10}")
    for cls in CLASSES:
        for sub, (c1, c2), (p1, p2) in zip(subs[cls], vals["cube"][cls], vals["pv"][cls]):
            print(f"{cls:13s} {sub:>7d} {c1:10.4g} {c2:10.4g} {p1:10.4g} {p2:10.4g}")

    ztxt = rf"$z={z:g}$"
    for kind in ("cube", "pv"):
        _figure(vals[kind], f"{KIND_TITLE[kind]}  ({model}, {ztxt})",
                args.outdir / f"residual_split_{kind}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
