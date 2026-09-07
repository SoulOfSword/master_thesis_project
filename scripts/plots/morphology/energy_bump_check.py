"""Quantify extra "bumps" in a galaxy's stellar binding-energy histogram
and check whether MORDOR's Ecut lands on the dominant valley.

Why
---
MORDOR splits the bound stars at an energy Ecut into a more-bound
(bulge/pseudo-bulge) and a less-bound (thick disc / halo) part. It finds
Ecut as a *minimum* of the energy histogram (`decomposition.FindMin`), and
when several minima exist it keeps the lowest-count one. A clean disc has a
bimodal energy histogram (one obvious valley). Some galaxies show a THIRD
mode; FindMin can then snap Ecut to the small valley next to that bump
instead of the dominant valley, so Ecut is misplaced and most stars get
labelled "more bound". This script measures both things, per galaxy.

What it does (for one galaxy)
-----------------------------
1. Runs the real MORDOR decomposition (`run_mordor_single`) to get the
   exact per-star binding energy `te` and the Ecut MORDOR actually chose
   (parsed from the `Ecut = ...` line `morph` prints).
2. Builds the COUNT histogram of the bound stars' `te`, normalised to
   [-1, 0] exactly as `morph` feeds it to FindMin.
3. Counts modes by topographic **prominence** (`scipy.signal.find_peaks`):
   `n_modes >= 3` flags a third bump. Prominence is the non-arbitrary
   "is this a real bump" measure (height above the saddle to a taller
   peak), so noise wiggles score ~0.
4. Finds the **dominant valley** = the most prominent minimum (the cut a
   human would draw) and compares it to MORDOR's Ecut.
5. Saves a diagnostic plot and prints the metrics.

Run on a COMPUTE NODE (it loads a galaxy and decomposes it):
    srun ... python scripts/plots/morphology/energy_bump_check.py \
        --hdf5 <...>/SIDM1/snap_021/Gal_001913.hdf5
"""

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

import matplotlib
# --all is a headless batch job (no plots); single-galaxy uses WebAgg to view live
matplotlib.use("Agg" if "--all" in sys.argv else "WebAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from galaxy_sidm.morphology import run_mordor_single

MORDOR_DIR = Path.home() / "software" / "mordor"
if str(MORDOR_DIR) not in sys.path:
    sys.path.insert(0, str(MORDOR_DIR))

DEFAULT_GAL = ("/leonardo_scratch/large/userexternal/acosta01/master_thesis_project"
               "/data/mordor_galaxies/SIDM1/snap_021/Gal_001913.hdf5")


def mordor_te_and_ecut(hdf5_path, mode="cosmo_sim", soft_phys_kpc=0.57):
    """Decompose one galaxy and return (te_norm, mass, ecut_mordor).

    `te_norm` is the bound stars' binding energy, offset by the least-bound
    energy and scaled by the most-bound one -> [-1, 0], i.e. the exact array
    `morph` hands to FindMin. `ecut_mordor` is parsed from MORDOR's printout.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):                 # capture 'Ecut = ...'
        gal = run_mordor_single(hdf5_path, mode=mode,
                                soft_phys_kpc=soft_phys_kpc)
    m = re.search(r"Ecut\s*=\s*\[?\s*([-\d.eE+]+)", buf.getvalue())
    ecut_mordor = float(m.group(1)) if m else float("nan")

    te_all = np.asarray(gal.s["te"], dtype=float)
    bound = np.asarray(gal.s["morph"]) != 0               # classified = bound
    mass = np.asarray(gal.s["mass"], dtype=float)[bound]
    te = te_all[bound] - te_all.max()                     # least-bound -> 0
    te = te / np.abs(te).max()                            # -> [-1, 0]
    return te, mass, ecut_mordor


def mode_metrics(te, nbins=60, prom_frac=0.1):
    """Count energy modes by prominence and locate the dominant valley.

    Returns a dict with the histogram, the peak/valley positions, `n_modes`,
    the dominant Ecut (most prominent minimum), and MORDOR's own FindMin
    minima over the full range (for the overlay).
    """
    hist, edges = np.histogram(te, bins=nbins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    prom = prom_frac * hist.max() # prominence floor

    peaks, _ = find_peaks(hist, prominence=prom)
    valleys, vprops = find_peaks(-hist, prominence=prom)

    if len(valleys):  # the "visual" cut
        ecut_dom = float(centres[valleys[np.argmax(vprops["prominences"])]])
    else:
        ecut_dom = float("nan")

    import decomposition
    mins, _ = decomposition.FindMin(te, float(te.min()), float(te.max()), nbins) 
    # can use np.quantile(te, 0.90) instead of te.max() to ignore the long tail of very weakly bound stars

    return {"centres": centres, "hist": hist,
            "peaks": centres[peaks], "valleys": centres[valleys],
            "n_modes": int(len(peaks)), "ecut_dominant": ecut_dom,
            "findmin_minima": np.asarray(mins, dtype=float)}


def plot_diagnostic(mm, ecut_mordor, out_path, title=""):
    """Energy histogram with modes, MORDOR's Ecut (red), dominant valley (green)."""
    c, h = mm["centres"], mm["hist"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(c, h, step="mid", alpha=0.18, color="grey")
    ax.step(c, h, where="mid", color="k", lw=1.2)
    if len(mm["peaks"]):
        ax.plot(mm["peaks"], np.interp(mm["peaks"], c, h), "^",
                color="steelblue", ms=10, label=f"modes (n={mm['n_modes']})")
    if np.isfinite(mm["ecut_dominant"]):
        ax.axvline(mm["ecut_dominant"], color="green", lw=2,
                   label=f"dominant valley = {mm['ecut_dominant']:.3f}")
    if np.isfinite(ecut_mordor):
        ax.axvline(ecut_mordor, color="red", lw=2, ls="--",
                   label=f"MORDOR Ecut = {ecut_mordor:.3f}")
    ax.set_xlabel(r"e")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    #fig.savefig(out_path, dpi=150)
    #plt.close(fig)
    plt.show()


GAL_ROOT = Path("/leonardo_scratch/large/userexternal/acosta01/"
                "master_thesis_project/data/mordor_galaxies")


def _galaxy_row(task):
    """Worker: compute the bump metrics for ONE galaxy -> a flat dict row.

    Catches per-galaxy failures (decomposition can blow up on a bad
    galaxy) so one failure can't kill the whole batch; the failure is
    recorded in the row's `error` field with n_modes=-1.
    """
    hdf5, sub_id, nbins, prom_frac, mode, soft = task
    try:
        te, _mass, ecut_mordor = mordor_te_and_ecut(
            hdf5, mode=mode, soft_phys_kpc=soft)
        mm = mode_metrics(te, nbins=nbins, prom_frac=prom_frac)
        bw = (te.max() - te.min()) / nbins
        dom = mm["ecut_dominant"]
        misplaced = int(bool(np.isfinite(ecut_mordor) and np.isfinite(dom)
                             and abs(ecut_mordor - dom) > 2 * bw))
        return dict(sub_id=sub_id, n_bound=len(te), n_modes=mm["n_modes"],
                    ecut_mordor=ecut_mordor, ecut_dominant=dom,
                    ecut_offset=ecut_mordor - dom, misplaced=misplaced, error="")
    except Exception as exc:
        return dict(sub_id=sub_id, n_bound=0, n_modes=-1, ecut_mordor=np.nan,
                    ecut_dominant=np.nan, ecut_offset=np.nan, misplaced=-1,
                    error=repr(exc)[:140])


def run_all(args):
    """Decompose every qualifying central of (model, snap) in parallel and
    write one CSV row of bump metrics per galaxy, plus a summary line.

    The galaxy list is the same selection MORDOR uses (central subhaloes
    with N_star >= n_star_min); each `Gal_<id>.hdf5` is processed by
    `_galaxy_row` in a worker pool. No plots — this is the population table
    you then compare across models (e.g. n_modes>=3 fraction, SIDM1 vs CDM).
    """
    import csv
    from multiprocessing import Pool
    import temet
    from galaxy_sidm.data.aida_tng import (build_central_subhalo_catalog,
                                           qualifying_central_ids)

    sim = temet.sim(run="aida", variant=args.model, res=1080, snap=args.snap)
    sub_ids = qualifying_central_ids(build_central_subhalo_catalog(sim),
                                     n_star_min=args.n_star_min)
    gdir = args.gal_root / args.model / f"snap_{args.snap:03d}"
    tasks = [(gdir / f"Gal_{int(s):06d}.hdf5", int(s), args.nbins,
              args.prom_frac, args.mode, args.soft_phys_kpc)
             for s in sub_ids if (gdir / f"Gal_{int(s):06d}.hdf5").exists()]
    print(f"[energy_bump] {args.model} snap {args.snap}: "
          f"{len(tasks)}/{len(sub_ids)} galaxies on {args.ncpu} cpus", flush=True)

    rows = []
    with Pool(args.ncpu) as pool:
        for i, row in enumerate(pool.imap_unordered(_galaxy_row, tasks), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} ...", flush=True)
    rows.sort(key=lambda r: r["sub_id"])

    out_csv = args.out_csv or (gdir / f"energy_modes_{args.model}_{args.snap:03d}.csv")
    fields = ["sub_id", "n_bound", "n_modes", "ecut_mordor", "ecut_dominant",
              "ecut_offset", "misplaced", "error"]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader(); wr.writerows(rows)

    ok = [r for r in rows if r["n_modes"] >= 0]
    d = max(len(ok), 1)
    n3 = sum(r["n_modes"] >= 3 for r in ok)
    nmis = sum(r["misplaced"] == 1 for r in ok)
    print(f"\n[energy_bump] wrote {out_csv}")
    print(f"  ok={len(ok)}/{len(rows)} | n_modes>=3: {n3} ({100*n3/d:.1f}%) | "
          f"Ecut misplaced: {nmis} ({100*nmis/d:.1f}%)")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdf5", type=Path, default=Path(DEFAULT_GAL))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--mode", default="cosmo_sim")
    p.add_argument("--soft-phys-kpc", type=float, default=0.57)
    p.add_argument("--nbins", type=int, default=60)
    p.add_argument("--prom-frac", type=float, default=0.05,
                   help="Min peak prominence as a fraction of the histogram peak")
    # --- batch mode (--all): every qualifying central of one (model, snap) ---
    p.add_argument("--all", action="store_true",
                   help="Batch: process every qualifying central of --model/--snap")
    p.add_argument("--model", default=None)
    p.add_argument("--snap", type=int, default=None)
    p.add_argument("--ncpu", type=int, default=8)
    p.add_argument("--n-star-min", type=float, default=1e4)
    p.add_argument("--gal-root", type=Path, default=GAL_ROOT)
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    if args.all:
        if not (args.model and args.snap is not None):
            sys.exit("--all requires --model and --snap")
        return run_all(args)

    if not args.hdf5.exists():
        sys.exit(f"galaxy file not found: {args.hdf5}")
    out = args.out or args.hdf5.with_name(args.hdf5.stem + "_energy_modes.png")

    te, mass, ecut_mordor = mordor_te_and_ecut(
        args.hdf5, mode=args.mode, soft_phys_kpc=args.soft_phys_kpc)
    mm = mode_metrics(te, nbins=args.nbins, prom_frac=args.prom_frac)

    bw = (te.max() - te.min()) / args.nbins
    misplaced = bool(np.isfinite(ecut_mordor) and np.isfinite(mm["ecut_dominant"])
                     and abs(ecut_mordor - mm["ecut_dominant"]) > 2 * bw)

    print(f"galaxy            : {args.hdf5.name}")
    print(f"bound stars       : {len(te)}")
    print(f"n_modes (prom)    : {mm['n_modes']}   (>=3 -> third bump)")
    print(f"peak positions    : {np.round(mm['peaks'], 3)}")
    print(f"FindMin minima    : {np.round(mm['findmin_minima'], 3)}")
    print(f"dominant valley   : {mm['ecut_dominant']:.3f}")
    print(f"MORDOR Ecut       : {ecut_mordor:.3f}")
    print(f"Ecut misplaced?   : {misplaced}  (|MORDOR - dominant| = "
          f"{abs(ecut_mordor - mm['ecut_dominant']):.3f}, bin={bw:.3f})")

    title = (f"{args.hdf5.stem}   n_modes={mm['n_modes']}   "
             f"misplaced={misplaced}")
    plot_diagnostic(mm, ecut_mordor, out, title=title)
    #print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())