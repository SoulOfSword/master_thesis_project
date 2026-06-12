"""Density-profile mosaic — three layouts, one script.

Each panel is (model curves) at a given (snap, component-set, mass-bin)
triple. The CLI picks a *layout* that decides how those triples are
arranged into a grid:

    --layout standard   (default)
        rows = snaps, cols = mass-bins.
        One component-set per panel (the FIRST set in --component-sets,
        or --components if given).

    --layout big
        rows = snaps, cols = component-sets × mass-bins.
        Each component-set occupies a contiguous block of mass-bin
        columns with a header label spanning the block.

    --layout components
        rows = snaps, cols = component-sets, at ONE mass-bin chosen by
        --mass-bin (index into --mstar-bins).

For each panel the script computes, per model, the median and 16-84
envelope of the chosen profile composition across halos in the bin.
Cold-gas inclusion (component 'coldgas') triggers a live particle-data
computation via temet/illustris_python and is cached per (model, snap).

Examples
--------
Standard, one component-set:
    python scripts/plots/density/plot_density_mosaic.py \\
        --catalogs $SCRATCH/.../catalogs/*.hdf5 \\
        --profiles $SCRATCH/.../profiles/*.hdf5 \\
        --components dm,stars

Big notebook-style mosaic with all three component-sets:
    python scripts/plots/density/plot_density_mosaic.py \\
        --catalogs ... --profiles ... \\
        --layout big \\
        --component-sets "dm;dm,stars;dm,stars,coldgas"

Component-set comparison at the highest-mass bin:
    python scripts/plots/density/plot_density_mosaic.py \\
        --catalogs ... --profiles ... \\
        --layout components --mass-bin 2 \\
        --component-sets "dm;dm,stars;dm,stars,coldgas"
"""

import argparse
import glob
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat, load_per_halo
from galaxy_sidm.observables import collect_profiles, measure_cold_gas_profile


# Notebook-side defaults — kept so behaviour matches without explicit args.
MSTAR_BINS_DEFAULT = [(1e8, 1e9), (1e9, 1e10), (1e10, 1e12)]
MSTAR_LABELS_DEFAULT = [r"$10^8 - 10^9$",
                        r"$10^9 - 10^{10}$",
                        r"$10^{10} - 10^{12}$"]
R_COMMON_DEFAULT = np.logspace(np.log10(1.0), np.log10(500.0), 50)
DEFAULT_COMPONENT_SETS_BIG = ["dm", "dm,stars", "dm,stars,coldgas"]

COMPONENT_LOOKUP = {
    "dm":      "prof_dm",
    "stars":   "prof_stars",
    "gas":     "prof_gas",
    "coldgas": "prof_coldgas",
}
COMPONENT_LABEL = {
    "prof_dm":      "DM",
    "prof_stars":   "stars",
    "prof_gas":     "gas",
    "prof_coldgas": "cold gas",
}


# ---- CLI parsing helpers --------------------------------------------

def _parse_component_set(spec):
    """Parse 'dm,stars,coldgas' into ['prof_dm', 'prof_stars', 'prof_coldgas']."""
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    if not parts:
        raise ValueError("Empty component set")
    out = []
    for s in parts:
        if s not in COMPONENT_LOOKUP:
            raise ValueError(f"Unknown component {s!r}; known: {list(COMPONENT_LOOKUP)}")
        out.append(COMPONENT_LOOKUP[s])
    return out


def _parse_component_sets_arg(spec):
    """Parse 'dm;dm,stars;dm,stars,coldgas' into a list of component lists."""
    return [_parse_component_set(s) for s in spec.split(";") if s.strip()]


def _parse_mstar_bins(spec):
    """Parse '1e8,1e9;1e9,1e10;1e10,1e12' into list of (lo, hi) tuples + labels."""
    bins, labels = [], []
    for piece in spec.split(";"):
        if not piece.strip():
            continue
        lo, hi = [float(x) for x in piece.split(",")]
        bins.append((lo, hi))
        labels.append(rf"${lo:g} - {hi:g}$")
    return bins, labels


def _set_label(component_keys):
    """Human-readable label for a list of prof_* keys."""
    return " + ".join(COMPONENT_LABEL.get(c, c) for c in component_keys)


# ---- Data loading helpers -------------------------------------------

def _index_by_model_snap(paths, kind):
    """Map (model, snap) -> path for catalogs / profiles.

    kind: 'catalog' (load with load_flat, return (path, arrays)) or
          'profiles' (just record path, load lazily later).
    """
    index = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"MISSING ({kind}): {path}")
            continue
        if kind == "catalog":
            arrays, attrs = load_flat(path)
            meta = attrs.get("metadata", {})
            index[(str(meta.get("model")), int(meta.get("snap")))] = (path, arrays)
        else:
            with h5py.File(path, "r") as f:
                meta = dict(f["metadata"].attrs) if "metadata" in f else {}
            index[(str(meta.get("model")), int(meta.get("snap")))] = path
    return index


def _profs_as_dict(per_halo):
    """load_per_halo output -> the shape collect_profiles expects."""
    out = {}
    for hid, d in per_halo.items():
        out[int(hid)] = {
            "r_edges":    np.asarray(d["r_edges"]),
            "r_outer":    np.asarray(d["r_outer"]),
            "prof_dm":    np.asarray(d["prof_dm"]) if d["prof_dm"].size > 0 else None,
            "prof_gas":   np.asarray(d["prof_gas"]) if d["prof_gas"].size > 0 else None,
            "prof_stars": np.asarray(d["prof_stars"]) if d["prof_stars"].size > 0 else None,
        }
    return out


def _load_mstar(cat_arrays):
    """Flat array m[hid] -> Mstar (zeros for absent ids)."""
    halo_ids = np.asarray(cat_arrays["halo_ids"], dtype=np.int64)
    if len(halo_ids) == 0:
        return np.zeros(0)
    n = int(halo_ids.max()) + 1
    out = np.zeros(n, dtype=np.float64)
    out[halo_ids] = np.asarray(cat_arrays["Mstar"])
    return out


# ---- Cold-gas cache --------------------------------------------------

def _resolve_basepath(sim, model, snap, cfg):
    """Return a basePath that actually has snapdir_<snap>/ on disk.

    `temet.sim().simPath` points at the canonical $WORK path. For combos
    where the snapshot was copied into a SCRATCH shadow tree (e.g. CDM
    snap 21), $WORK is missing the snapdir and we fall back to the shadow
    path declared in cfg['paths']['shadow_<model_lowercase>']/output.
    Falls through to the default if no shadow override resolves either.
    """
    default = Path(sim.simPath)
    if (default / f"snapdir_{snap:03d}").exists():
        return str(default)
    shadow_key = f"shadow_{model.lower()}"
    shadow_root = cfg.get("paths", {}).get(shadow_key)
    if shadow_root is not None:
        shadow_base = Path(shadow_root) / "output"
        if (shadow_base / f"snapdir_{snap:03d}").exists():
            print(f"[plot_density_mosaic] {model} snap {snap}: "
                  f"using SCRATCH shadow basePath ({shadow_base})")
            return str(shadow_base) + "/"
    return str(default)


def _one_cold(args):
    basePath, snap, hid, r_edges, h_val, box = args
    return hid, measure_cold_gas_profile(basePath, snap, hid, r_edges,
                                         a=1.0, h=h_val, box=box)


def _cold_gas_cache_for(model, snap, basePath, box, halo_ids,
                        profs_per_halo, n_workers, h_val):
    """Cold-gas profiles per halo, parallelised."""
    tasks = []
    for hid in halo_ids:
        if hid not in profs_per_halo:
            continue
        r_edges = np.asarray(profs_per_halo[hid]["r_edges"])
        tasks.append((basePath, snap, int(hid), r_edges, h_val, box))
    cache = {}
    if not tasks:
        return cache
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_one_cold, t): t[2] for t in tasks}
        for fut in as_completed(futs):
            hid, rho = fut.result()
            cache[hid] = rho
    return cache


# ---- Panel grid construction ----------------------------------------

def _build_panel_grid(layout, snaps, mstar_bins, mstar_labels,
                       component_sets, mass_bin_idx):
    """Return (n_rows, n_cols, panels, group_headers).

    panels      : list of (row, col, snap, component_set, mass_bin_tuple,
                  col_title_or_None, row_label_or_None)
    group_headers : for 'big' layout, list of (col_first, col_last,
                  group_label, group_color_optional). For other layouts,
                  empty list.
    """
    n_rows = len(snaps)
    panels = []
    group_headers = []

    if layout == "standard":
        n_cols = len(mstar_bins)
        cs = component_sets[0]
        for r, snap in enumerate(snaps):
            for c, (mb, mlbl) in enumerate(zip(mstar_bins, mstar_labels)):
                col_title = (mlbl + r" $M_\odot$") if r == 0 else None
                row_label = f"snap {snap}" if c == 0 else None
                panels.append((r, c, snap, cs, mb, col_title, row_label))

    elif layout == "big":
        n_cols = len(component_sets) * len(mstar_bins)
        for r, snap in enumerate(snaps):
            for csi, cs in enumerate(component_sets):
                for mbi, (mb, mlbl) in enumerate(zip(mstar_bins, mstar_labels)):
                    c = csi * len(mstar_bins) + mbi
                    col_title = (mlbl + r" $M_\odot$") if r == 0 else None
                    row_label = f"snap {snap}" if c == 0 else None
                    panels.append((r, c, snap, cs, mb, col_title, row_label))
        for csi, cs in enumerate(component_sets):
            c_lo = csi * len(mstar_bins)
            c_hi = c_lo + len(mstar_bins) - 1
            group_headers.append((c_lo, c_hi, _set_label(cs)))

    elif layout == "components":
        if not (0 <= mass_bin_idx < len(mstar_bins)):
            raise ValueError(f"--mass-bin {mass_bin_idx} out of range "
                             f"[0, {len(mstar_bins) - 1}]")
        n_cols = len(component_sets)
        mb = mstar_bins[mass_bin_idx]
        mlbl = mstar_labels[mass_bin_idx]
        for r, snap in enumerate(snaps):
            for c, cs in enumerate(component_sets):
                col_title = _set_label(cs) if r == 0 else None
                row_label = f"snap {snap}" if c == 0 else None
                panels.append((r, c, snap, cs, mb, col_title, row_label))
        # use the figure suptitle to indicate the mass bin
        group_headers.append((0, n_cols - 1, mlbl + r" $M_\odot$"))

    else:
        raise ValueError(f"Unknown layout {layout!r}")

    return n_rows, n_cols, panels, group_headers


# ---- Plotting --------------------------------------------------------

def _plot_panel(ax, data, components, mlo, mhi, R_COMMON, log_R,
                model_colors, models, cold_cache_for_panel):
    """Draw curves for all models in one panel; return True if anything drawn."""
    any_drawn = False
    for model in models:
        if model not in data:
            continue
        d = data[model]
        sel = (d["mstar"] >= mlo) & (d["mstar"] < mhi)
        halo_ids = np.where(sel)[0]
        if len(halo_ids) == 0:
            continue

        need_coldgas = "prof_coldgas" in components
        cache = None
        if need_coldgas:
            cc = cold_cache_for_panel.get(model, {})
            cache = {hid: cc[hid] for hid in halo_ids if hid in cc}

        prof_arr = collect_profiles(d["profiles"], halo_ids, R_COMMON,
                                    components, cold_gas_cache=cache)
        if prof_arr is None or len(prof_arr) < 3:
            continue
        median = np.nanmedian(prof_arr, axis=0)
        p16 = np.nanpercentile(prof_arr, 16, axis=0)
        p84 = np.nanpercentile(prof_arr, 84, axis=0)
        valid = np.isfinite(median)
        color = model_colors.get(model, "tab:gray")
        ax.plot(log_R[valid], median[valid], color=color, lw=1.5,
                label=model)
        ax.fill_between(log_R[valid], p16[valid], p84[valid],
                        color=color, alpha=0.15)

        # When showing a composite, also overlay individual components
        # to mirror the notebook's behaviour (stars dashed, cold gas dotted).
        if need_coldgas:
            for comp_key, ls in (("prof_stars", "--"),
                                  ("prof_coldgas", ":")):
                comp_cache = cache if comp_key == "prof_coldgas" else None
                comp_arr = collect_profiles(d["profiles"], halo_ids, R_COMMON,
                                            [comp_key], cold_gas_cache=comp_cache)
                if comp_arr is None or len(comp_arr) < 3:
                    continue
                cmed = np.nanmedian(comp_arr, axis=0)
                cvalid = np.isfinite(cmed)
                ax.plot(log_R[cvalid], cmed[cvalid], color=color, ls=ls, lw=0.8)
        any_drawn = True
    return any_drawn


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--catalogs", type=Path, nargs="+", required=True,
                   help="catalog_*.hdf5 files (any model/snap)")
    p.add_argument("--catalogs-glob", type=str, default=None)
    p.add_argument("--profiles", type=Path, nargs="+", required=True,
                   help="profiles_*.hdf5 files (any model/snap)")
    p.add_argument("--profiles-glob", type=str, default=None)
    p.add_argument("--snaps", type=int, nargs="*", default=None,
                   help="Snap list (low z -> high z). Defaults to cfg['mosaic_snaps'].")
    p.add_argument("--components", type=str, default=None,
                   help="Single component-set (e.g. 'dm,stars'). Equivalent to "
                        "--component-sets with one set.")
    p.add_argument("--component-sets", type=str, default=None,
                   help="Semicolon-separated list of component-sets, e.g. "
                        "'dm;dm,stars;dm,stars,coldgas'. Required for layout "
                        "'big' or 'components' (default = all three sets).")
    p.add_argument("--mstar-bins", type=str, default=None,
                   help="Semicolon-separated '<lo>,<hi>' pairs, e.g. "
                        "'1e8,1e9;1e9,1e10;1e10,1e12'.")
    p.add_argument("--layout", type=str,
                   choices=["standard", "big", "components"],
                   default="standard")
    p.add_argument("--mass-bin", type=int, default=0,
                   help="Index of the mass bin to use for layout 'components'")
    p.add_argument("--disc-only", action="store_true",
                   help="Use only *_disc.hdf5 inputs (MORDOR-disc subset) "
                        "and tag the output PDF with _disc. Default: "
                        "exclude *_disc.hdf5 files from inputs.")
    p.add_argument("--n-workers", type=int, default=16,
                   help="Workers for cold-gas computation (if needed)")
    p.add_argument("--out-fig", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    snaps = [int(s) for s in (args.snaps or cfg["mosaic_snaps"])]
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    model_colors = cfg["model_colors"]
    models = list(cfg["models"])
    h_val = cfg["cosmology"]["h"]

    # --- resolve component sets ---
    if args.components and args.component_sets:
        sys.exit("Pass --components OR --component-sets, not both.")
    if args.component_sets:
        component_sets = _parse_component_sets_arg(args.component_sets)
    elif args.components:
        component_sets = [_parse_component_set(args.components)]
    else:
        if args.layout == "standard":
            component_sets = [_parse_component_set("dm")]
        else:
            component_sets = [_parse_component_set(s)
                              for s in DEFAULT_COMPONENT_SETS_BIG]

    if args.layout == "standard" and len(component_sets) > 1:
        print(f"NOTE: --layout standard uses only the first component-set "
              f"({_set_label(component_sets[0])}). For all sets together, "
              f"use --layout big or components.")
        component_sets = component_sets[:1]

    # --- mass bins ---
    if args.mstar_bins:
        mstar_bins, mstar_labels = _parse_mstar_bins(args.mstar_bins)
    else:
        mstar_bins = MSTAR_BINS_DEFAULT
        mstar_labels = MSTAR_LABELS_DEFAULT

    # --- gather input paths ---
    cat_paths = list(args.catalogs)
    if args.catalogs_glob:
        cat_paths.extend(Path(p) for p in glob.glob(args.catalogs_glob))
    prof_paths = list(args.profiles)
    if args.profiles_glob:
        prof_paths.extend(Path(p) for p in glob.glob(args.profiles_glob))
    cat_paths = [p for p in cat_paths
                 if ("_disc" in str(p)) == args.disc_only]
    prof_paths = [p for p in prof_paths
                  if ("_disc" in str(p)) == args.disc_only]
    if not cat_paths or not prof_paths:
        print(f"[plot_density_mosaic] no inputs matched "
              f"--disc-only={args.disc_only}; aborting")
        return 1
    cat_index = _index_by_model_snap(cat_paths, "catalog")
    prof_index = _index_by_model_snap(prof_paths, "profiles")

    R_COMMON = R_COMMON_DEFAULT
    log_R = np.log10(R_COMMON)

    # --- decide cold-gas needs (any component set with coldgas) ---
    any_coldgas = any("prof_coldgas" in cs for cs in component_sets)

    # --- pre-cache per (model, snap) ---
    mosaic_data = {}     # (model, snap) -> {'mstar', 'profiles'}
    cold_caches = {}     # (model, snap) -> {hid: rho_cold}
    for snap in snaps:
        for model in models:
            key = (model, snap)
            if key not in cat_index or key not in prof_index:
                print(f"MISSING for ({model}, snap {snap}); skipping")
                continue
            _, cat_arrays = cat_index[key]
            mstar = _load_mstar(cat_arrays)
            per_halo, _ = load_per_halo(prof_index[key])
            profs = _profs_as_dict(per_halo)
            mosaic_data[key] = {"mstar": mstar, "profiles": profs}

            if any_coldgas:
                import temet
                sim = temet.sim(run="aida", variant=model, res=1080, snap=snap)
                basePath = _resolve_basepath(sim, model, snap, cfg)
                box = sim.boxSize
                halo_ids = [hid for hid in np.where(mstar >= 1e8)[0]
                            if hid in profs]
                c = _cold_gas_cache_for(model, snap, basePath, box, halo_ids,
                                        profs, args.n_workers, h_val)
                cold_caches[key] = c
                print(f"  cold-gas cache: {model} snap {snap}: {len(c)} halos"
                      f"  (basePath={basePath})")

    if not mosaic_data:
        print("[plot_density_mosaic] no data; aborting")
        return 1

    # --- build the panel grid ---
    n_rows, n_cols, panels, group_headers = _build_panel_grid(
        args.layout, snaps, mstar_bins, mstar_labels,
        component_sets, args.mass_bin)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    # --- draw each panel ---
    for row, col, snap, components, (mlo, mhi), col_title, row_label in panels:
        ax = axes[row, col]
        ax.tick_params(labelsize=13)
        # data for this row by model
        data_for_row = {m: mosaic_data[(m, snap)] for m in models
                        if (m, snap) in mosaic_data}
        cold_for_row = {m: cold_caches.get((m, snap), {}) for m in models}
        any_drawn = _plot_panel(
            ax, data_for_row, components, mlo, mhi, R_COMMON, log_R,
            model_colors, models, cold_for_row)
        if not any_drawn:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=9)
        ax.set_xlim(0)
        if col_title is not None:
            ax.set_title(col_title, fontsize=16)
        if row_label is not None:
            z = snap_z.get(snap, np.nan)
            z_lbl = f"$z = {z:g}$" if np.isfinite(z) else row_label
            ax.set_ylabel(z_lbl + "\n" + r"$\log_{10}(\rho)$", fontsize=16)
        if row == n_rows - 1:
            ax.set_xlabel(r"$\log_{10}(r\;/\;\mathrm{ckpc})$", fontsize=16)
        if row == 0 and col == n_cols - 1:
            ax.legend(fontsize=12)

    # --- group headers (big layout) / suptitle (components layout) ---
    if args.layout == "big" and group_headers:
        for c_lo, c_hi, label in group_headers:
            c_mid = (c_lo + c_hi) // 2
            ax_top = axes[0, c_mid]
            ax_top.text(0.5, 1.25, label, transform=ax_top.transAxes,
                        ha="center", fontsize=14, fontweight="bold")
    if args.layout == "components" and group_headers:
        label = group_headers[0][2]
        fig.suptitle(label, fontsize=16, fontweight="bold", y=1.005)

    # --- shared stars/cold-gas style legend on the last panel ---
    if any_coldgas:
        extra = [
            Line2D([0], [0], color="gray", ls="--", lw=1, label="stars"),
            Line2D([0], [0], color="gray", ls=":",  lw=1, label="cold gas"),
        ]
        handles, _ = axes.flat[-1].get_legend_handles_labels()
        axes.flat[-1].legend(handles=handles + extra,
                             fontsize=12, loc="lower left")

    fig.tight_layout()

    # --- save ---
    if args.out_fig is None:
        tag_parts = [args.layout]
        if args.layout == "components":
            tag_parts.append(f"mb{args.mass_bin}")
        sets_tag = "-".join("_".join(c.replace("prof_", "") for c in cs)
                             for cs in component_sets)
        tag_parts.append(sets_tag)
        if args.disc_only:
            tag_parts.append("disc")
        out_fig = (Path(cfg["paths"]["fig_root"]) / "density"
                   / f"density_mosaic_{'_'.join(tag_parts)}.pdf")
    else:
        out_fig = args.out_fig
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_density_mosaic] wrote {out_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
