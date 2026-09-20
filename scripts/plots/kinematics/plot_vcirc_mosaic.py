"""Mosaic of disc-plane circular-velocity profiles, one figure per model and redshift.

Reads the compute_vcirc.py output (vcirc_<model>_<snap>.hdf5) and the MORDOR
sample for the stellar masses. One panel per log10(M*) bin; every galaxy's
v_circ(R) goes in its bin's panel (thin lines), with the median profile on top.

Usage:
    python scripts/plots/kinematics/plot_vcirc_mosaic.py                  # every vcirc file found
    python scripts/plots/kinematics/plot_vcirc_mosaic.py --model CDM --snap 33
    python scripts/plots/kinematics/plot_vcirc_mosaic.py --mass-edges 9.5 10 10.5 11
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat

plt.rcParams.update({
    # paper font: bundled Computer Modern (cmr10) -- do NOT use usetex here
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "axes.labelsize": 19,
    "axes.titlesize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

DEFAULT_MASS_EDGES = [9.5, 10.0, 10.5, 11.0, 11.5]   # log10(Mstar / Msun)


def plot_file(path, mordor_dir, mass_edges, rmax, outdir):
    arr, attrs = load_flat(path)
    meta, variants = attrs["metadata"], attrs["variants"]
    model, snap, z = str(meta["model"]), int(meta["snap"]), float(meta["redshift"])

    sample, _ = load_flat(mordor_dir / f"mordor_sample_{model}_{snap:03d}.hdf5")
    mstar = dict(zip(np.asarray(sample["halo_ids"], np.int64).tolist(),
                     np.asarray(sample["Mstar"], float)))
    logm = np.array([np.log10(mstar[i]) if mstar.get(i, 0) > 0 else np.nan
                     for i in np.asarray(arr["halo_ids"], np.int64).tolist()])

    bins = list(zip(mass_edges[:-1], mass_edges[1:])) + [(mass_edges[-1], np.inf)]
    ncol = min(3, len(bins))
    nrow = int(np.ceil(len(bins) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.4 * nrow), sharex=True,
                             sharey=True, constrained_layout=True, squeeze=False)
    for ax, (lo, hi) in zip(axes.flat, bins):
        sel = np.flatnonzero((logm >= lo) & (logm < hi))
        for i in sel:
            ax.plot(arr["R"][i], arr["vcirc"][i], c="tab:blue", lw=0.8, alpha=0.35)
        if len(sel) >= 3:
            with warnings.catch_warnings():            # rings empty in every galaxy
                warnings.simplefilter("ignore", RuntimeWarning)
                ax.plot(np.nanmedian(arr["R"][sel], axis=0), np.nanmedian(arr["vcirc"][sel], axis=0),
                        c="k", lw=2.2, label="median")
            ax.legend(frameon=False, loc="lower right")
        hi_txt = f"{hi:.1f}" if np.isfinite(hi) else r"\infty"
        ax.set_title(rf"$\log_{{10}} M_\star \in [{lo:.1f}, {hi_txt})$   $N={len(sel)}$")
        ax.grid(alpha=0.25, lw=0.6)
    for ax in axes.flat[len(bins):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$R$ [kpc]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$v_{\rm circ}$ [km s$^{-1}$]")
    axes[0, 0].set_xlim(0, rmax)
    axes[0, 0].set_ylim(bottom=0)
    fig.suptitle(rf"{model}   $z = {z:.2g}$   disc-plane $v_{{\rm circ}}$, "
                 rf"$|z| < {float(variants['slab_half_kpc']):g}$ kpc", fontsize=19)

    out = outdir / f"vcirc_{model}_{snap:03d}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  ({np.isfinite(logm).sum()} galaxies)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", default=None, choices=["CDM", "SIDM1", "vSIDM"])
    p.add_argument("--snap", type=int, default=None)
    p.add_argument("--mass-edges", type=float, nargs="+", default=DEFAULT_MASS_EDGES,
                   help="log10(Mstar) bin edges; the last bin is open-ended")
    p.add_argument("--rmax", type=float, default=30.0, help="x-axis limit [kpc] (default 30)")
    p.add_argument("--indir", type=Path, default=None, help="default: <scratch_processed>/vcirc")
    p.add_argument("--outdir", type=Path, default=None, help="default: <fig_root>/vcirc")
    args = p.parse_args()

    cfg = load_config(args.config)
    indir = args.indir or Path(cfg["paths"]["scratch_processed"]) / "vcirc"
    outdir = args.outdir or Path(cfg["paths"]["fig_root"]) / "vcirc"
    mordor_dir = Path(cfg["paths"]["scratch_mordor"]) / "samples"

    files = sorted(indir.glob(f"vcirc_{args.model or '*'}_"
                              f"{f'{args.snap:03d}' if args.snap is not None else '*'}.hdf5"))
    if not files:
        print(f"no vcirc files in {indir} -- run scripts/data/compute_vcirc.py first")
        return 1
    for f in files:
        plot_file(f, mordor_dir, args.mass_edges, args.rmax, outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
