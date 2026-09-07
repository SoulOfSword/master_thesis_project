"""Prove that `stellar_te` returns MORDOR's `te`, bit-for-bit.

Earlier this script compared a hand-rolled cheap `te` (KE from a global mean
bulk velocity + Phi/a) against MORDOR and found a few-percent disagreement on
the massive centrals -- because the global mean is the wrong bulk velocity.
`stellar_te` no longer approximates anything: it runs MORDOR's own
`load_and_align` (same hybrid centering) and MORDOR's own `te = ke + phi`
lines. This script checks that claim end to end, over a SAMPLE of galaxies
spanning redshift (scale factor a) and model.

For each galaxy it computes:
  - te_fast = `stellar_te(hdf5)`              (the function the plots use)
  - te_full = `run_mordor_single(hdf5).s['te']`  (a full decomposition)
both normalised to [-1, 0] the way MORDOR does, then reports the per-star
correlation and the mean/max |difference|. If `stellar_te` really is MORDOR's
te, these differences should sit at floating-point noise (~1e-10), NOT the
0.16 the old approximation hit. The summary line is the worst case over the
whole sample.

`decomposition.debug` is forced False so the full run stays headless.

Usage (compute node):
    python scripts/plots/morphology/validate_te.py --n-per-cell 2 --ncpu 6
"""

import argparse
import contextlib
import io
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path.home() / "software" / "mordor"))
import decomposition
decomposition.debug = False                 # no debug popups
matplotlib.use("Agg", force=True)           # decomposition imports flip the
#   backend to WebAgg; force it back or a stray figure spawns a blocking server
from galaxy_sidm.morphology import run_mordor_single, stellar_te

GAL_ROOT = Path("/leonardo_scratch/large/userexternal/acosta01/"
                "master_thesis_project/data/mordor_galaxies")


def _norm(te):
    te = te - te.max()
    return te / np.abs(te).max()


def _one(hdf5):
    """Compare stellar_te vs a full run_mordor_single te for one galaxy."""
    import decomposition
    import matplotlib
    decomposition.debug = False
    matplotlib.use("Agg", force=True)       # workers inherit WebAgg via fork
    hdf5 = Path(hdf5)
    try:
        te_fast, _ = stellar_te(str(hdf5))
        with contextlib.redirect_stdout(io.StringIO()):
            gal = run_mordor_single(str(hdf5))
        te_full = np.asarray(gal.s["te"], dtype=np.float64)
        n = min(len(te_fast), len(te_full))
        a, b = _norm(te_fast[:n]), _norm(te_full[:n])
        return dict(gal=str(hdf5.relative_to(GAL_ROOT)), n=n,
                    corr=float(np.corrcoef(a, b)[0, 1]),
                    meandiff=float(np.abs(a - b).mean()),
                    maxdiff=float(np.abs(a - b).max()), err="")
    except Exception as exc:
        return dict(gal=str(hdf5), n=0, corr=np.nan, meandiff=np.nan,
                    maxdiff=np.nan, err=repr(exc)[:90])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["CDM", "SIDM1", "vSIDM"])
    p.add_argument("--snaps", type=int, nargs="+", default=[17, 33, 67],
                   help="z = 5, 2, 0.5 (span of scale factor a)")
    p.add_argument("--n-per-cell", type=int, default=2,
                   help="galaxies per (model, snap)")
    p.add_argument("--ncpu", type=int, default=6)
    args = p.parse_args()

    gals = []
    for model in args.models:
        for snap in args.snaps:
            d = GAL_ROOT / model / f"snap_{snap:03d}"
            gals += sorted(d.glob("Gal_*.hdf5"))[: args.n_per_cell]
    print(f"checking stellar_te == full-MORDOR te on {len(gals)} galaxies "
          f"({args.models} x snaps {args.snaps}) on {args.ncpu} cpus\n",
          flush=True)

    rows = []
    with Pool(args.ncpu) as pool:
        for r in pool.imap_unordered(_one, [str(g) for g in gals]):
            rows.append(r)
            tag = r["err"] or (f"corr={r['corr']:.6f} mean|d|={r['meandiff']:.2e} "
                               f"max|d|={r['maxdiff']:.2e}")
            print(f"  {r['gal']:28s} N={r['n']:7d}  {tag}", flush=True)

    ok = [r for r in rows if not r["err"]]
    if ok:
        print(f"\nSUMMARY over {len(ok)} galaxies:")
        print(f"  worst correlation : {min(r['corr'] for r in ok):.6f}")
        print(f"  worst mean|diff|  : {max(r['meandiff'] for r in ok):.2e}")
        print(f"  worst max|diff|   : {max(r['maxdiff'] for r in ok):.2e}")
    if len(ok) < len(rows):
        print(f"  ({len(rows) - len(ok)} failed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
