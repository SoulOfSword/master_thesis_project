"""Tully-Fisher relation figure from the circular velocity at the mock BBarolo rings.

One panel per redshift (z = 0 -> 5, two rows, galaxy_sidm.viz.panel_grid): log10 Mstar (or
Mbar = Mstar + 1.36 M_cut, M_cut the neutral hydrogen inside the gas cut, processed/gas_kinematics)
against log10 v_flat for the usable discs, with the Di Teodoro+23 z = 0 relation and its
perpendicular scatter (0.1 dex stellar, 0.07 dex baryonic), drawn as a band of vertical
half-width scatter x sqrt(1 + slope^2). The sample comes from
galaxy_sidm.mock.tables.assemble -- the final sample (tables.final_sample: MORDOR discs whose
neutral gas passes tables.DISC_RULE) with bbarolo_rc==0. v_flat is the
mean of v_circ (processed/vcirc, scripts/data/compute_vcirc.py) on the BBarolo rings where
the rotation curve is flat (tables.flat_rings): the rings of SPARC_mod (tables.sparc_mod_vflat) at --flat-tol,
else SPARC_mod at --retry-tol, else SPARC (tables.sparc_rings) at --flat-tol, drawn as
circles, diamonds and crosses. Discs with none of these, with rings outside the v_circ
curve or without a v_circ curve are left out and counted. --vrot uses the mean v_rot on the
same rings instead of v_circ, for the same discs, and writes tfr_<tag>_vrot.pdf.
--retry-bias draws no points but the fits of the SPARC_mod discs at --flat-tol and at
--retry-tol separately (SPARC discs left out), each with the 16th-84th percentile band of
its residuals, and adds the suffix _retry.
Colours follow config model_colors (CDM black, SIDM1
blue, vSIDM red). An OLS power-law fit log M = a + b log v is drawn per model;
slope / intercept / scatter of log M about it / N are printed.

Paper style: matplotlib's bundled cmr10 (Computer Modern; text.usetex is
broken on Leonardo), large labels/ticks/markers, inward ticks on all sides.

Usage:
  python scripts/plots/scaling/plot_tfr.py               # stellar TFR
  python scripts/plots/scaling/plot_tfr.py --baryonic    # Mstar + M_neutral
  python scripts/plots/scaling/plot_tfr.py --flat-tol 0.1
  python scripts/plots/scaling/plot_tfr.py --retry-tol 0.15
  python scripts/plots/scaling/plot_tfr.py --vrot        # v_flat from v_rot
  python scripts/plots/scaling/plot_tfr.py --retry-bias  # 5% vs 10% SPARC_mod fits
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config
from galaxy_sidm.mock.barolo import rings_file
from galaxy_sidm.mock.tables import assemble, flat_rings, MODELS, vcirc_at, vcirc_curves
from galaxy_sidm.viz import legend_entries, panel_grid

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


def _fit(logV, logM):
    """OLS fit logM = a + b*logV. Returns (b, a, scatter_dex, N) or None."""
    m = np.isfinite(logV) & np.isfinite(logM)
    if m.sum() < 5:
        return None
    b, a = np.polyfit(logV[m], logM[m], 1)
    scatter = float(np.std(logM[m] - (a + b * logV[m])))
    return float(b), float(a), scatter, int(m.sum())


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baryonic", action="store_true",
                   help="Use Mbar = Mstar + 1.36 M_cut (neutral hydrogen inside the gas cut, with helium) "
                        "instead of Mstar.")
    p.add_argument("--flat-tol", type=float, default=0.05,
                   help="Flatness tolerance of SPARC_mod and SPARC (default 0.05).")
    p.add_argument("--retry-tol", type=float, default=0.10,
                   help="SPARC_mod tolerance for discs with no flat part at --flat-tol (default 0.10).")
    p.add_argument("--vrot", action="store_true",
                   help="v_flat from v_rot on the same rings instead of v_circ (output suffix _vrot).")
    p.add_argument("--retry-bias", action="store_true",
                   help="No points: the fits of the --flat-tol and the --retry-tol SPARC_mod discs "
                        "separately (SPARC discs left out), each with the 16th-84th percentile band "
                        "of its residuals (output suffix _retry).")
    p.add_argument("--outdir", type=Path, default=ROOT / "figures" / "tfr")
    args = p.parse_args()
    if args.retry_tol <= args.flat_tol:
        p.error("--retry-tol must be larger than --flat-tol")

    cfg = load_config(None)
    colors = cfg["model_colors"]
    mart = Path(cfg["paths"]["scratch_processed"]).parent / "martini"
    methods = {rf"$v_{{\rm rot}}$ tol {100 * args.flat_tol:g}%": "o", rf"$v_{{\rm rot}}$ tol {100 * args.retry_tol:g}%": "D",
               "SPARC": "x"}
    rows = assemble(cfg, flat_tol=args.flat_tol)
    curves, missing, neither, outside = {}, [], 0, 0
    for r in rows:
        r["v_flat_circ"], r["v_flat_rot"], r["method"] = float("nan"), float("nan"), None
        key = (r["model"], r["snap"])
        if key not in curves:
            curves[key] = vcirc_curves(cfg, *key)
        if r["sub_id"] not in curves[key]:
            missing.append(f"{r['model']} z{r['z']:g} {r['sub_id']}")
            continue
        bdir = mart / f"z{r['z']:g}" / r["model"] / f"gal_{r['sub_id']:06d}" / "bbarolo"
        R, vrot = np.loadtxt(rings_file(bdir), usecols=(0, 2), ndmin=2).T
        used, k = flat_rings(vrot, args.flat_tol, args.retry_tol)
        if k is None:
            neither += 1
            continue
        vc = vcirc_at(R[used], *curves[key][r["sub_id"]])
        if np.isnan(vc).any():
            outside += 1
            continue
        r["v_flat_circ"], r["v_flat_rot"], r["method"] = float(vc.mean()), float(vrot[used].mean()), list(methods)[k]
    rows = [r for r in rows
            if np.isfinite(r["v_flat_circ"]) and r["v_flat_circ"] > 0 and r["Mstar"] > 0]
    vkey = "v_flat_rot" if args.vrot else "v_flat_circ"
    if not rows:
        print("no usable discs -- has the pipeline + BBarolo fits run?")
        return 1

    tag = "baryonic" if args.baryonic else "stellar"
    mlabel = (r"$\log_{10}\,(M_\star + 1.36\,M_{\rm neutral})\ \,[M_\odot]$"
              if args.baryonic else r"$\log_{10}\,M_\star\ \,[M_\odot]$")
    ref_slope, ref_icpt, ref_perp = (3.58, 2.52, 0.07) if args.baryonic else (4.06, 1.26, 0.1)
    ref_band = ref_perp * np.sqrt(1 + ref_slope ** 2) # y band is perpendicular scatter * sqrt(1 + slope^2)

    def mass(r):
        if args.baryonic:
            return r["Mstar"] + 1.36 * r["M_cut"] # 1.36 factor for He for standard ISM, use 1.33 for primordial He fraction
        return r["Mstar"]

    zs = sorted({r["z"] for r in rows})
    fig, axes, bottom, left = panel_grid(len(zs), (16 / 3, 5), constrained_layout=True)
    print(f"\nTFR ({tag}), v_flat = mean {'v_rot' if args.vrot else 'v_circ'} on the rings where v_rot is flat: "
          + ", ".join(f"{sum(r['method'] == m for r in rows)} by {m}" for m in methods)
          + f"; left out: {neither} with no flat part of v_rot, {outside} with rings outside the "
          f"v_circ curve, {len(missing)} without a v_circ curve")
    if missing:
        print("without a v_circ curve:", ", ".join(missing))
    print(f"{'z':>5} {'model':6s} {'slope':>6} {'icpt':>7} {'scatter':>8} {'N':>5}")
    samples = list(zip(list(methods)[:2], (args.flat_tol, args.retry_tol), ("-", "--")))
    for ax, z in zip(axes, zs, strict=True):
        lv_panel = []
        for model in MODELS:
            sel = [r for r in rows if r["z"] == z and r["model"] == model]
            if not sel:
                continue
            lM = np.log10([mass(r) for r in sel])
            lV = np.log10([r[vkey] for r in sel])
            lv_panel.extend(lV)
            label = model
            if args.retry_bias:
                for method, tol, ls in samples:
                    m = np.array([r["method"] == method for r in sel])
                    fit = _fit(lV[m], lM[m])
                    if fit:
                        b, a, sc, n = fit
                        p16, p84 = np.percentile(lM[m] - (a + b * lV[m]), [16, 84])
                        xx = np.linspace(lV[m].min(), lV[m].max(), 20)
                        ax.plot(xx, a + b * xx, c=colors[model], lw=2.4, ls=ls, label=label)
                        ax.fill_between(xx, a + b * xx + p16, a + b * xx + p84, color=colors[model], alpha=0.15, lw=0)
                        label = None
                        print(f"{z:>5g} {model:6s} {b:6.2f} {a:7.2f} {sc:8.3f} {n:5d}  ({100 * tol:g}%)")
                continue
            for method, marker in methods.items():
                m = np.array([r["method"] == method for r in sel])
                if m.any():
                    ax.scatter(lV[m], lM[m], s=38, c=colors[model], marker=marker, alpha=0.55,
                               lw=1.5 if marker == "x" else 0, rasterized=True, label=label)
                    label = None
            fit = _fit(lV, lM)
            if fit:
                b, a, sc, n = fit
                xx = np.linspace(lV.min(), lV.max(), 20)
                ax.plot(xx, a + b * xx, c=colors[model], lw=2.4)
                print(f"{z:>5g} {model:6s} {b:6.2f} {a:7.2f} {sc:8.3f} {n:5d}")
        ax.set_title(rf"$z = {z:g}$")
        # ax.grid(alpha=0.25, lw=0.6)
        xx = np.linspace(np.nanmin(lv_panel), np.nanmax(lv_panel), 50)
        ax.plot(xx, ref_slope * xx + ref_icpt, c='k', lw=3, linestyle='--', label="DiTeodoro+23 z=0") #DiTeodoro+23 z=0 reference TFR, stellar or baryonic
        ax.fill_between(xx, ref_slope * xx + ref_icpt - ref_band, ref_slope * xx + ref_icpt + ref_band, color='k', alpha=0.1)
    for ax in bottom:
        ax.set_xlabel(r"$\log_{10}\,v_{\rm flat}\ \,[{\rm km\,s^{-1}}]$")
    for ax in left:
        ax.set_ylabel(mlabel)
    leg = axes[0].legend(*legend_entries(axes, MODELS), frameon=False, markerscale=1.6,
                              handletextpad=0.2, loc="lower right")
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    if args.retry_bias:
        handles = [Line2D([], [], ls=ls, color="0.35", lw=2.4) for _, _, ls in samples]
        names = [method for method, _, _ in samples]
    else:
        handles = [Line2D([], [], ls="none", marker=mk, color="0.35", markersize=8, mew=1.5)
                   for mk in methods.values()]
        names = list(methods)
    axes[1].legend(handles, names, frameon=False, handletextpad=0.2, loc="lower right")
    # clamp the shared v_flat range to the bulk of the sample: a handful of junk-fit
    # outliers (v_flat of a few km/s) would otherwise squash every panel
    # allV = np.array([np.log10(r[vkey]) for r in rows])
    # lo, hi = np.percentile(allV, [0.5, 99.8])
    # axes.flat[0].set_xlim(lo - 0.1, hi + 0.12)
    # axes.flat[0].set_ylim(10**9, 10**12)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"tfr_{tag}{'_vrot' if args.vrot else ''}{'_retry' if args.retry_bias else ''}.pdf"
    fig.savefig(out, dpi=250)
    print(f"\nwrote {out}  ({len(rows)} discs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
