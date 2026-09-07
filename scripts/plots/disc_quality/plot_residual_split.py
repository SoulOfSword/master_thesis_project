"""Residual split of hand-classified disc vs perturbed galaxies.

    res1 = sum_ij (D_ij - M_ij)^2 / sigma^2      (chi^2-like)
    res2 = sum_ij |D_ij - M_ij|   / sigma        (chi-like)
    res3 = sum_ij |D_ij - M_ij| / sum_ij |D_ij|  (fraction of signal unexplained)

Each figure has three side-by-side panels (res1 left, res2 middle, res3 right; the axis label
is the actual formula). 

Usage:
  python scripts/plots/disc_quality/plot_residual_split.py                # CDM, z=2
  python scripts/plots/disc_quality/plot_residual_split.py --model SIDM1 --z 1
  python scripts/plots/disc_quality/plot_residual_split.py --labels my_labels.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

CLASS_STYLE = {"discs": dict(color="tab:blue"), "perturbed": dict(color="tab:red")}
YLABELS = (r"$\sum_{i,j}\,(D_{ij}-M_{ij})^{2}\,/\,\sigma^{2}$",
           r"$\sum_{i,j}\,|D_{ij}-M_{ij}|\,/\,\sigma$",
           r"$\sum_{i,j}\,|D_{ij}-M_{ij}|\,/\,\sum_{i,j}\,|D_{ij}|$")


def _figure(values, title, out):
    """One figure: two panels (res1, res2), two jittered classes on x."""
    rng = np.random.default_rng(0)                    # reproducible jitter
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
    for k, ax in enumerate(axes):                     # k = 0: res1, 1: res2, 2: res3
        for xc, cls in ((0.0, "discs"), (1.0, "perturbed")):
            v = np.array([r[k] for r in values[cls]])
            good = np.isfinite(v)
            x = xc + rng.uniform(-0.13, 0.13, size=good.sum())
            ax.scatter(x, v[good], s=80, alpha=0.85, lw=0.8,
                       edgecolors="white", zorder=3, **CLASS_STYLE[cls])
        if k < 2:
            ax.set_yscale("log")          # res1/res2 span decades
        else:
            ax.axhline(1.0, color="grey", lw=1.2, ls=":")   # null model (M=0)
        ax.set_ylabel(YLABELS[k])
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["discs", "perturbed"], fontsize=18)
        ax.grid(alpha=0.25, lw=0.6, axis="y")
    fig.suptitle(title, fontsize=19)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="CDM",
                   choices=["CDM", "SIDM1", "vSIDM"])
    p.add_argument("--z", type=float, default=2.0)
    p.add_argument("--labels", type=Path,
                   default=ROOT / "config" / "residual_calibration.yaml")
    p.add_argument("--outdir", type=Path, default=ROOT / "figures" / "disc_quality")
    args = p.parse_args()

    zkey = f"z{args.z:g}"
    d = yaml.safe_load(args.labels.read_text()) or {}
    labels = (d.get(args.model) or {}).get(zkey) or {}
    discs = [int(s) for s in (labels.get("discs") or [])]
    pert = [int(s) for s in (labels.get("perturbed") or [])]
    if not discs or not pert:
        print(f"no calibration subIDs for {args.model} {zkey} in {args.labels} -- "
              f"fill the 'discs' and 'perturbed' lists first.")
        return 1

    cfg = load_config(None)
    mart = Path(cfg["paths"]["scratch_processed"]).parent / "martini"

    cube_vals = {"discs": [], "perturbed": []}
    pv_vals = {"discs": [], "perturbed": []}
    print(f"\n{args.model} {zkey}   res1 = sum (D-M)^2/sig^2,  res2 = sum |D-M|/sig,"
          f"  res3 = sum |D-M| / sum |D|")
    print(f"{'class':10s} {'subID':>7} {'cube res1':>11} {'cube res2':>11} "
          f"{'cube res3':>10} {'pv res1':>11} {'pv res2':>11} {'pv res3':>10}")
    for cls, subs in (("discs", discs), ("perturbed", pert)):
        for sub in subs:
            g = mart / zkey / args.model / f"gal_{sub:06d}"
            c1, c2, c3 = cube_residuals(g)
            p1, p2, p3 = pv_residuals(g)
            cube_vals[cls].append((c1, c2, c3))
            pv_vals[cls].append((p1, p2, p3))
            print(f"{cls:10s} {sub:>7d} {c1:11.4g} {c2:11.4g} {c3:10.4f} "
                  f"{p1:11.4g} {p2:11.4g} {p3:10.4f}")

    ztxt = rf"$z={args.z:g}$"
    _figure(cube_vals, f"Cube  ({args.model}, {ztxt})",
            args.outdir / "residual_split_cube.pdf")
    _figure(pv_vals, f"PV  ({args.model}, {ztxt})",
            args.outdir / "residual_split_pv.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
