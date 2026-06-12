"""Size-mass relation: stellar half-mass radius vs stellar mass.

Reads catalog_*.hdf5 files produced by build_catalog.py for any number of
(model, snap) combinations, groups them by snap (one column per snap in the
order set by cfg['mosaic_snaps']), and plots the size-mass relation with
running medians per stellar-mass bin.

Two styles:
    overlaid : 1xN_snaps, all models overlaid in each panel (default).
    grid     : N_snaps x N_models, one model per column.

Stellar half-mass radius is read directly from the catalog's 'Rhalf_star'
field (comoving ckpc, h-removed; see build_catalog.py).

Output (under cfg['paths']['fig_root']/size_mass/):
    size_mass[.<style>].pdf
"""

import argparse
import colorsys
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat


def lighten(color, amount: float = 0.6):
    """Increase HLS lightness; amount=0 keeps color, amount=1 yields white."""
    r, g, b = mcolors.to_rgb(color)
    h_, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h_, l + (1 - l) * amount, s)


def _load_catalogs(paths):
    """Index input catalogs by (model, snap). Returns dict and a snap->z map."""
    by_key = {}
    snap_z = {}
    for path in paths:
        arrs, attrs = load_flat(path)
        meta = attrs["metadata"]
        model = str(meta["model"])
        snap = int(meta["snap"])
        by_key[(model, snap)] = arrs
        snap_z[snap] = float(meta["redshift"])
    return by_key, snap_z


def _running_median(log_m, log_r, mbins, n_min):
    """Median log_r in each log_m bin with at least n_min entries."""
    centres = 0.5 * (mbins[:-1] + mbins[1:])
    out_c, out_m = [], []
    for j in range(len(mbins) - 1):
        in_bin = (log_m >= mbins[j]) & (log_m < mbins[j + 1])
        if in_bin.sum() >= n_min:
            out_c.append(centres[j])
            out_m.append(np.median(log_r[in_bin]))
    return np.asarray(out_c), np.asarray(out_m)


def _xy(arrs, mstar_floor):
    """Pull (log10 Mstar, log10 Rhalf_star) for valid centrals."""
    m = arrs["Mstar"]
    r = arrs["Rhalf_star"]
    sel = (m > mstar_floor) & (r > 0)
    return np.log10(m[sel]), np.log10(r[sel])


def plot_overlaid(by_key, snaps, snap_z, models, colors, mbins, n_min,
                  mstar_floor, out_path):
    """1xN_snaps, all models overlaid in each panel."""
    fig, axes = plt.subplots(1, len(snaps),
                             figsize=(4.2 * len(snaps), 4.2),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    for col, snap in enumerate(snaps):
        ax = axes[col]
        ax.tick_params(labelsize=13)
        present = [m for m in models if (m, snap) in by_key]
        if not present:
            print(f"[plot_size_mass] MISSING snap {snap}: no catalogs supplied")
            ax.text(0.5, 0.5, "MISSING", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=14)
            ax.set_title(f"snap {snap}", fontsize=16)
            continue

        for model in present:
            log_m, log_r = _xy(by_key[(model, snap)], mstar_floor)
            color = colors.get(model, "grey")
            ax.scatter(log_m, log_r, s=10, alpha=0.3, color=color,
                       edgecolors="grey", linewidths=0.15)
            c, med = _running_median(log_m, log_r, mbins, n_min)
            if c.size:
                ax.plot(c, med, "-", color=color, lw=3.5)
                ax.plot(c, med, "-", color=lighten(color, 0.6), lw=2,
                        label=model)

        z = snap_z.get(snap)
        ax.set_title(rf"$z = {z:.1f}$" if z is not None else f"snap {snap}",
                     fontsize=16)
        ax.set_xlabel(r"$\log_{10}(M_\star\;/\;\mathrm{M_\odot})$",
                      fontsize=16)
        if col == 0:
            ax.set_ylabel(r"$\log_{10}(R_{1/2,\star}\;/\;\mathrm{ckpc})$",
                          fontsize=16)
            ax.legend(fontsize=12)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid(by_key, snaps, snap_z, models, colors, mbins, n_min,
              mstar_floor, out_path):
    """N_snaps x N_models grid, one model per column."""
    fig, axes = plt.subplots(len(snaps), len(models),
                             figsize=(4.2 * len(models), 3.6 * len(snaps)),
                             sharex=True, sharey=True, squeeze=False)

    for row, snap in enumerate(snaps):
        z = snap_z.get(snap)
        any_present = False
        for col, model in enumerate(models):
            ax = axes[row, col]
            ax.tick_params(labelsize=13)
            arrs = by_key.get((model, snap))
            if arrs is None:
                print(f"[plot_size_mass] MISSING ({model}, snap {snap})")
                ax.text(0.5, 0.5, "MISSING", transform=ax.transAxes,
                        ha="center", va="center", color="grey", fontsize=12)
            else:
                any_present = True
                log_m, log_r = _xy(arrs, mstar_floor)
                color = colors.get(model, "grey")
                ax.scatter(log_m, log_r, s=25, alpha=0.5, color=color,
                           edgecolors="grey", linewidths=0.15)
                c, med = _running_median(log_m, log_r, mbins, n_min)
                if c.size:
                    ax.plot(c, med, "-", color=lighten(color, 0.5), lw=4,
                            path_effects=[pe.Stroke(linewidth=7,
                                                    foreground="black"),
                                          pe.Normal()])

            if row == 0:
                ax.set_title(model, fontsize=16)
            if col == 0:
                zlabel = rf"$z = {z:.1f}$" + "\n" if z is not None else ""
                ax.set_ylabel(zlabel + r"$\log_{10}(R_{1/2,\star}\;/\;"
                              r"\mathrm{ckpc})$", fontsize=16)
            if row == len(snaps) - 1:
                ax.set_xlabel(r"$\log_{10}(M_\star\;/\;\mathrm{M_\odot})$",
                              fontsize=16)

        if not any_present:
            print(f"[plot_size_mass] row snap {snap}: nothing to plot")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--catalogs", type=Path, nargs="+", required=True,
                   help="catalog_*.hdf5 files spanning (model, snap) combos")
    p.add_argument("--disc-only", action="store_true",
                   help="Use only *_disc.hdf5 inputs and tag the output PDF "
                        "with _disc. Default: exclude *_disc.hdf5 files.")
    p.add_argument("--style", choices=("overlaid", "grid"), default="overlaid")
    p.add_argument("--out-fig", type=Path, default=None,
                   help="Output figure path; defaults to "
                        "fig_root/size_mass/size_mass[_<style>].pdf")
    p.add_argument("--mstar-floor", type=float, default=1.0e8,
                   help="Lower Mstar cutoff for the scatter (Msun)")
    p.add_argument("--mbin-step", type=float, default=0.3,
                   help="log10(Mstar) bin width for running medians")
    p.add_argument("--mbin-lo", type=float, default=8.0)
    p.add_argument("--mbin-hi", type=float, default=12.1)
    p.add_argument("--n-min-overlaid", type=int, default=3,
                   help="min galaxies per Mstar bin to plot a median (overlaid)")
    p.add_argument("--n-min-grid", type=int, default=5,
                   help="min galaxies per Mstar bin to plot a median (grid)")
    args = p.parse_args()

    cfg = load_config(args.config)
    models = list(cfg["models"])
    colors = dict(cfg["model_colors"])
    snap_order = list(cfg["mosaic_snaps"])

    catalog_paths = [p for p in args.catalogs
                     if ("_disc" in str(p)) == args.disc_only]
    if not catalog_paths:
        print(f"[plot_size_mass] no catalogs matched --disc-only={args.disc_only}; "
              f"aborting")
        return 1
    by_key, snap_z_in = _load_catalogs(catalog_paths)
    # cfg redshift map takes priority; fall back to per-catalog redshift.
    snap_z = {s: cfg["snap_z"].get(s, snap_z_in.get(s)) for s in snap_order}

    fig_root = Path(cfg["paths"]["fig_root"])
    if args.out_fig is None:
        suffix = "" if args.style == "overlaid" else f"_{args.style}"
        disc_tag = "_disc" if args.disc_only else ""
        out_path = fig_root / "size_mass" / f"size_mass{suffix}{disc_tag}.pdf"
    else:
        out_path = args.out_fig

    mbins = np.arange(args.mbin_lo, args.mbin_hi, args.mbin_step)
    if args.style == "overlaid":
        plot_overlaid(by_key, snap_order, snap_z, models, colors, mbins,
                      args.n_min_overlaid, args.mstar_floor, out_path)
    else:
        plot_grid(by_key, snap_order, snap_z, models, colors, mbins,
                  args.n_min_grid, args.mstar_floor, out_path)

    print(f"[plot_size_mass] style={args.style}, "
          f"{len(by_key)} catalogs -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
