"""3D asymmetry (He+2026) of the mock cubes: calibration classes and all galaxies.

    A = sum |I(i,j,k) - I(-i,-j,-k)| / sum |I(i,j,k) + I(-i,-j,-k)|

over BBarolo's symmetrised mask, with the pairs taken about the cube centre and the
central channel of BBarolo's VSYS (galaxy_sidm.mock.asymmetry). Two versions:

    A        as He+2026 define it
    A_corr   A - A_noise, where A_noise is what a perfectly symmetric galaxy gets
             from the noise alone

He+2026 call a galaxy asymmetric above A = 0.35 (dashed line). The dotted line is the
midpoint between the highest calibration disc and the lowest perturbed galaxy (when
they don't overlap). The calibration galaxies are listed in --labels
(config/residual_calibration.yaml); by default the first model and redshift there.

Modes:
  (default)            every galaxy in the mock manifest. Reads every full cube, so run
                       it on a compute node (scripts/slurm/asymmetry_all.sbatch). Writes
                       <processed>/gas_discs/asymmetry_3d.csv, asymmetry_split.pdf and
                       asymmetry_all.pdf.
  --calibration-only   only the galaxies of --labels -> asymmetry_split.pdf
  --from-table         both figures again from the saved table, without reading cubes

Galaxies whose final fit found no gas on the major axis (extent_arcsec [0, 0]), or
that have no usable fit, get no A; they are printed with the reason, which also goes
in the table's 'skipped' column.

Usage:
  python scripts/plots/disc_quality/plot_asymmetry.py --calibration-only
  python scripts/plots/disc_quality/plot_asymmetry.py --from-table
  sbatch scripts/slurm/asymmetry_all.sbatch
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config
from galaxy_sidm.mock.asymmetry import HE26_THRESHOLD, asymmetry_3d

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
MODEL_OFFSET = {"CDM": -0.25, "SIDM1": 0.0, "vSIDM": 0.25}   # x offset inside a redshift column
VALUES = ("A", "A_corr")
YLABELS = {"A": r"$A$", "A_corr": r"$A-A_{\rm noise}$"}
# the definitions, above the panels (I' = I_{-i,-j,-k}, N = voxels in the mask)
PANEL_TITLES = {"A": r"$A=\sum|I_{i,j,k}-I'|\,/\,\sum|I_{i,j,k}+I'|,\quad I'=I_{-i,-j,-k}$",
                "A_corr": r"$A_{\rm noise}=(2/\sqrt{\pi})\,\sigma N\,/\,\sum|I_{i,j,k}+I'|$"}
FIELDS = ("model", "z", "subID", "A", "A_noise", "A_corr", "n_voxels",
          "central_channel", "vsys", "skipped")


def _measure(mart, model, z, sub):
    """Table row of one galaxy; 'skipped' says why A is missing."""
    g = mart / f"z{z:g}" / model / f"gal_{sub:06d}"
    row = dict(model=model, z=z, subID=sub, skipped="")
    try:
        ext = json.loads((g / "info.json").read_text()).get("extent_arcsec")
        if ext is not None and max(ext) == 0:
            row["skipped"] = "no gas on the major axis (extent_arcsec 0)"
            return row
        row.update(asdict(asymmetry_3d(g)))
    except (OSError, ValueError, KeyError) as e:
        row["skipped"] = f"{type(e).__name__}: {e}"
    return row


def _measure_all(mart, gals, workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda g: _measure(mart, *g), gals))


def _write_table(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def _read_table(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["z"], r["subID"] = float(r["z"]), int(r["subID"])
            for k in ("A", "A_noise", "A_corr", "central_channel", "vsys"):
                r[k] = float(r[k]) if r[k] else float("nan")
            r["n_voxels"] = int(r["n_voxels"]) if r["n_voxels"] else 0
            rows.append(r)
    return rows


def _value(r, v):
    """r[v] as a float, nan for a skipped galaxy."""
    return float(r.get(v, float("nan"))) if not r["skipped"] else float("nan")


def _lines(ax, calib, v, label):
    """He+2026's threshold and, if the classes don't overlap, the calibration midpoint."""
    ax.axhline(HE26_THRESHOLD, color="k", ls="--", lw=1.2,
               label=f"He+2026 threshold ({HE26_THRESHOLD:g})" if label else None)
    d = [_value(r, v) for r in calib.get("discs", [])]
    p = [_value(r, v) for r in calib.get("perturbed", [])]
    if np.isfinite(d).any() and np.isfinite(p).any() and np.nanmax(d) < np.nanmin(p):
        ax.axhline(0.5 * (np.nanmax(d) + np.nanmin(p)), color="grey", ls=":", lw=1.4,
                   label="calibration midpoint" if label else None)


def _figure_split(calib, title, out):
    """Two panels (A, A_corr), one jittered column per calibration class."""
    classes = [c for c in CLASSES if calib.get(c)]
    rng = np.random.default_rng(0)                    # reproducible jitter
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    for i, (v, ax) in enumerate(zip(VALUES, axes)):
        for xc, cls in enumerate(classes):
            y = np.array([_value(r, v) for r in calib[cls]])
            y = y[np.isfinite(y)]
            ax.scatter(xc + rng.uniform(-0.13, 0.13, len(y)), y, s=80, alpha=0.85, lw=0.8,
                       edgecolors="white", zorder=3, **CLASS_STYLE[cls])
        _lines(ax, calib, v, label=i == 0)
        ax.set_ylabel(YLABELS[v])
        ax.set_title(PANEL_TITLES[v], fontsize=15)
        ax.set_xlim(-0.5, len(classes) - 0.5)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([CLASS_LABEL[c] for c in classes], fontsize=18)
        ax.grid(alpha=0.25, lw=0.6, axis="y")
    axes[0].legend(frameon=False, fontsize=12, loc="upper left")
    fig.suptitle(title, fontsize=19)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def _figure_all(cfg, rows, calib, calib_model, calib_z, out):
    """Two panels (A, A_corr): every galaxy by redshift, models side by side, calibration on top."""
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    zs = [snap_z[s] for s in cfg["mosaic_snaps"]]                 # low z -> high z
    models = np.array([r["model"] for r in rows])
    zg = np.array([r["z"] for r in rows])
    rng = np.random.default_rng(0)
    x_calib = zs.index(calib_z) + MODEL_OFFSET[calib_model]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), constrained_layout=True)
    for i, (v, ax) in enumerate(zip(VALUES, axes)):
        vals = np.array([_value(r, v) for r in rows])
        for iz, z in enumerate(zs):
            for m, dx in MODEL_OFFSET.items():
                sel = (models == m) & (zg == z) & np.isfinite(vals)
                ax.scatter(iz + dx + rng.uniform(-0.09, 0.09, sel.sum()), vals[sel],
                           s=3, lw=0, alpha=0.4, color=cfg["model_colors"][m], rasterized=True,
                           label=m if i == 0 and iz == 0 else None)
        for cls in CLASSES:
            y = np.array([_value(r, v) for r in calib.get(cls, [])])
            y = y[np.isfinite(y)]
            ax.scatter(x_calib + rng.uniform(-0.09, 0.09, len(y)), y, s=26, lw=0.7,
                       edgecolors="k", zorder=3,
                       label=f"calibration {CLASS_LABEL[cls]}" if i == 0 else None,
                       **CLASS_STYLE[cls])
        _lines(ax, calib, v, label=True)
        ax.set_ylabel(YLABELS[v])
        ax.set_title(PANEL_TITLES[v], fontsize=15)
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
               ncol=4, frameon=False, fontsize=13, markerscale=2.0)
    fig.suptitle(f"3D asymmetry, all galaxies", fontsize=19)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")

    print(f"\nfraction above He+2026's {HE26_THRESHOLD:g}, per model and redshift:")
    for z in zs:
        for m in MODEL_OFFSET:
            sel = [r for r in rows if r["model"] == m and r["z"] == z and not r["skipped"]]
            if sel:
                a = np.array([_value(r, "A") for r in sel])
                c = np.array([_value(r, "A_corr") for r in sel])
                print(f"  {m:6s} z={z:<4g} N={len(sel):4d}   A: {np.mean(a > HE26_THRESHOLD):4.0%}"
                      f"   A_corr: {np.mean(c > HE26_THRESHOLD):4.0%}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--calibration-only", action="store_true",
                      help="only the galaxies of --labels -> asymmetry_split.pdf")
    mode.add_argument("--from-table", action="store_true",
                      help="replot from the saved table instead of reading the cubes")
    p.add_argument("--model", default=None, choices=["CDM", "SIDM1", "vSIDM"],
                   help="calibration model (default: the first one in --labels)")
    p.add_argument("--z", type=float, default=None,
                   help="calibration redshift (default: the first one for that model in --labels)")
    p.add_argument("--labels", type=Path,
                   default=ROOT / "config" / "residual_calibration.yaml")
    p.add_argument("--manifest", type=Path, default=None,
                   help="galaxies to measure (default <martini>/manifest.txt)")
    p.add_argument("--table", type=Path, default=None,
                   help="table to write / read (default <processed>/gas_discs/asymmetry_3d.csv)")
    p.add_argument("--outdir", type=Path, default=ROOT / "figures" / "disc_quality")
    p.add_argument("--workers", type=int, default=8, help="galaxies read in parallel (default 8)")
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
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    mart = Path(cfg["paths"]["scratch_processed"]).parent / "martini"
    table = args.table or Path(cfg["paths"]["scratch_processed"]) / "gas_discs" / "asymmetry_3d.csv"

    if args.calibration_only:
        rows = _measure_all(mart, [(model, z, s) for cls in CLASSES for s in subs[cls]], args.workers)
    elif args.from_table:
        rows = _read_table(table)
        print(f"read {table}: {len(rows)} galaxies")
    else:
        gals = []
        for line in (args.manifest or mart / "manifest.txt").read_text().splitlines():
            if line.strip():
                m, snap, sub = line.split()[:3]
                gals.append((m, snap_z[int(snap)], int(sub)))
        rows = _measure_all(mart, gals, args.workers)
        _write_table(rows, table)

    skipped = [r for r in rows if r["skipped"]]
    print(f"\n{len(skipped)} of {len(rows)} galaxies without A:")
    for r in skipped:
        print(f"  {r['model']} z{r['z']:g} {r['subID']}: {r['skipped']}")

    by_id = {(r["model"], r["z"], r["subID"]): r for r in rows}
    missing = [s for cls in CLASSES for s in subs[cls] if (model, z, s) not in by_id]
    if missing:
        print(f"\ncalibration galaxies not measured: {missing}")
    calib = {cls: [by_id[(model, z, s)] for s in subs[cls] if (model, z, s) in by_id]
             for cls in CLASSES}

    print(f"\ncalibration {model} {zkey}")
    print(f"{'class':13s} {'subID':>7} {'A':>7} {'A_noise':>8} {'A_corr':>7} {'voxels':>8} {'central ch':>10}")
    for cls in CLASSES:
        for r in calib[cls]:
            if r["skipped"]:
                print(f"{cls:13s} {r['subID']:>7d}   no A: {r['skipped']}")
            else:
                print(f"{cls:13s} {r['subID']:>7d} {r['A']:7.3f} {r['A_noise']:8.3f} {r['A_corr']:7.3f} "
                      f"{r['n_voxels']:8d} {r['central_channel']:10.1f}")

    _figure_split(calib, rf"3D asymmetry  ({model}, $z={z:g}$)", args.outdir / "asymmetry_split.pdf")
    if not args.calibration_only:
        _figure_all(cfg, rows, calib, model, z, args.outdir / "asymmetry_all.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
