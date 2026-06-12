"""Mosaic of log10(r_core / r_half,*) vs log10(M200c) per redshift.

For each (model, snap) pair drawn from --rcore-files, we look up each
halo's central GroupFirstSub from the matching catalog file and grab
SubhaloHalfmassRadType[:, 4] (stellar half-mass radius, ckpc) via
temet. The ratio is dimensionless and directly comparable across
redshifts.

The mapping from rcore-file to its catalog file is done by matching
(model, snap) in metadata; pass either:
    --catalogs path1.hdf5 path2.hdf5 ...           (explicit, paired)
or
    --catalogs-glob "<dir>/catalog_*.hdf5"           (we'll match on meta)
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat
from galaxy_sidm.viz import running_median


def _classify_sample(model):
    return "DMO" if model.endswith("-Dark") else "FP"


def _index_catalogs(catalog_paths):
    """Build {(model, snap): (path, arrays)} from a list of catalog HDF5s.

    Skips any path that lacks model/snap metadata.
    """
    index = {}
    for path in catalog_paths:
        path = Path(path)
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        arrays, attrs = load_flat(path)
        meta = attrs.get("metadata", {})
        model = str(meta.get("model", ""))
        snap_meta = meta.get("snap", None)
        if not model or snap_meta is None:
            print(f"[plot_rcore_over_rhalf] skipping {path}: no model/snap meta")
            continue
        index[(model, int(snap_meta))] = (path, arrays)
    return index


def _load_rhalf_star(model, snap):
    """SubhaloHalfmassRadType[:, 4] in ckpc via temet."""
    import temet
    run = "aida_dm" if model.endswith("-Dark") else "aida"
    variant = model[:-5] if model.endswith("-Dark") else model
    sim = temet.sim(run=run, variant=variant, res=1080, snap=snap)
    return sim.units.codeLengthToComovingKpc(
        sim.subhalos("SubhaloHalfmassRadType")[:, 4])


def _draw_panel(ax, base_to_data, model_colors):
    bins = np.arange(10.0, 14.5, 0.5)
    for m, d in base_to_data.items():
        log_m = d["log_M200c"]
        log_ratio = d["log_ratio"]
        color = model_colors.get(m, "tab:gray")
        ax.scatter(log_m, log_ratio, s=4, alpha=0.3, color=color)
        x, y = running_median(log_m, log_ratio, bins, min_count=5)
        if len(x) > 0:
            ax.plot(x, y, "-", color=color, lw=2, label=m)
    ax.axhline(0, ls="--", color="gray", alpha=0.5)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--rcore-files", type=Path, nargs="+", required=True)
    p.add_argument("--catalogs", type=Path, nargs="*", default=None,
                   help="Catalog files to consult for GroupFirstSub")
    p.add_argument("--catalogs-glob", type=str, default=None,
                   help="Glob expression for catalog files (alternative to --catalogs)")
    p.add_argument("--snaps", type=int, nargs="*", default=None,
                   help="Snap list (low z -> high z). Defaults to cfg['mosaic_snaps'].")
    p.add_argument("--disc-only", action="store_true",
                   help="Use only *_disc.hdf5 inputs and tag the output PDF "
                        "with _disc. Default: exclude *_disc.hdf5 files.")
    p.add_argument("--sample", choices=["FP", "DMO", "AUTO"], default="FP",
                   help=("Which sample to plot. AUTO=use whatever is in "
                         "the rcore files; default FP."))
    p.add_argument("--out-fig", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    snaps = [int(s) for s in (args.snaps or cfg["mosaic_snaps"])]
    model_colors = cfg["model_colors"]

    catalog_paths = list(args.catalogs or [])
    if args.catalogs_glob:
        catalog_paths.extend(Path(p) for p in glob.glob(args.catalogs_glob))
    catalog_paths = [p for p in catalog_paths
                     if ("_disc" in str(p)) == args.disc_only]
    if not catalog_paths:
        print(f"[plot_rcore_over_rhalf] no catalog inputs matched "
              f"--disc-only={args.disc_only}; aborting")
        return 1
    cat_index = _index_catalogs(catalog_paths)

    rcore_paths = [p for p in args.rcore_files
                   if ("_disc" in str(p)) == args.disc_only]
    rcore_index = {}  # {(model, snap): arrays}
    for path in rcore_paths:
        path = Path(path)
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        arrays, attrs = load_flat(path)
        meta = attrs.get("metadata", {})
        model = str(meta.get("model"))
        snap = int(meta.get("snap"))
        if args.sample != "AUTO" and _classify_sample(model) != args.sample:
            continue
        rcore_index[(model, snap)] = arrays

    if not rcore_index:
        print("[plot_rcore_over_rhalf] no matching rcore files; aborting")
        return 1

    # Build per-snap, per-base-model data.
    panel = {}   # {snap: {base_model: {log_M200c, log_ratio}}}
    for (model, snap), rc_arr in rcore_index.items():
        if (model, snap) not in cat_index:
            print(f"MISSING catalog for ({model}, {snap})")
            continue
        cat_path, cat_arr = cat_index[(model, snap)]
        cat_hids = np.asarray(cat_arr["halo_ids"], dtype=np.int64)
        first_sub_flat = np.asarray(cat_arr["GroupFirstSub"], dtype=np.int64)
        cat_lookup = dict(zip(cat_hids.tolist(), first_sub_flat.tolist()))

        rhalf_all = _load_rhalf_star(model, snap)

        rc_hids = np.asarray(rc_arr["halo_ids"], dtype=np.int64)
        rc_m200 = np.asarray(rc_arr["M200c"])
        rc_rcore = np.asarray(rc_arr["r_core"])

        first_subs = np.array([cat_lookup.get(int(h), -1) for h in rc_hids])
        valid = (first_subs >= 0) & (first_subs < len(rhalf_all)) & (rc_rcore > 0)
        if not valid.any():
            continue
        rhalf_vals = rhalf_all[np.where(valid, first_subs, 0)]
        keep = valid & (rhalf_vals > 0)
        if not keep.any():
            continue
        log_m = np.log10(rc_m200[keep])
        log_ratio = np.log10(rc_rcore[keep] / rhalf_vals[keep])
        base = model[:-5] if model.endswith("-Dark") else model
        panel.setdefault(snap, {})[base] = {
            "log_M200c": log_m,
            "log_ratio": log_ratio,
        }

    n_cols = len(snaps)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols + 2, 4.5),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    for col, snap in enumerate(snaps):
        ax = axes[col]
        ax.tick_params(labelsize=13)
        z = snap_z.get(snap, np.nan)
        if snap not in panel or not panel[snap]:
            ax.text(0.5, 0.5, f"MISSING\nsnap {snap}",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=10)
            ax.set_title(f"z={z:g}" if np.isfinite(z) else f"snap {snap}",
                         fontsize=16)
            continue

        _draw_panel(ax, panel[snap], model_colors)
        ax.set_title(f"z={z:g}" if np.isfinite(z) else f"snap {snap}",
                     fontsize=16)
        ax.set_xlabel(r"$\log_{10}(M_{200c}\,/\,\mathrm{M}_\odot)$",
                      fontsize=16)
        if col == 0:
            ax.set_ylabel(r"$\log_{10}(r_{\rm core}\,/\,r_{1/2,\star})$",
                          fontsize=16)
        ax.legend(fontsize=12, loc="lower left")

    fig.tight_layout()
    out_fig = args.out_fig
    if out_fig is None:
        disc_tag = "_disc" if args.disc_only else ""
        out_fig = (Path(cfg["paths"]["fig_root"]) / "density"
                   / f"rcore_over_rhalf_star_mosaic_redshift{disc_tag}.pdf")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_rcore_over_rhalf] wrote {out_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
