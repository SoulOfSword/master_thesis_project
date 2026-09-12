"""Save one PNG of the stellar binding-energy distribution for every MORDOR
galaxy, over the FULL energy range (no bump-finding, no Ecut — just the
histogram you'd stare at).

The binding energy is MORDOR's own `te`, not a re-derivation: each galaxy is
loaded, potential-fixed, and hybrid-centred by `galaxy_sidm.morphology`'s
`stellar_te`, which runs the *same* `load_and_align` preamble as
`run_mordor_single` and then forms `te = ke + phi` with MORDOR's verbatim
lines. So the histogram here is exactly the one MORDOR's decomposition sees
(bit-identical to `gal.s['te']`), including for massive centrals where a naive
global-mean bulk velocity would be dragged off by satellites/ICL.

It is therefore real MORDOR work (full pynbody load + centering per galaxy),
NOT a cheap HDF5 read — run it on a compute node and keep `--ncpu` modest,
since a big central can use several GB on its own.

Output: <out-dir>/<model>/snap_<NN>/Gal_<id>_edist.png, one per galaxy.

Usage (compute node):
    python scripts/plots/morphology/energy_dist_all.py --ncpu 8
    python scripts/plots/morphology/energy_dist_all.py --model SIDM1 --snap 21
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

# single-threaded numerics per worker (we parallelise over galaxies, not BLAS)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")                       # headless: save, never show
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from galaxy_sidm.io import load_config
from galaxy_sidm.morphology import stellar_te

GAL_ROOT = Path(load_config()["paths"]["scratch_mordor"])


def _norm(te):
    """MORDOR's energy normalisation: offset by the least-bound star, scale by
    the most-bound -> [-1, 0]. Affine, so the histogram shape is unchanged."""
    te = te - te.max()
    return te / np.abs(te).max()


def _one(task):
    """Worker: plot one galaxy's MORDOR energy histogram to PNG."""
    hdf5, out_png, nbins = task
    try:
        te, _mass = stellar_te(str(hdf5))
        if te is None or len(te) < 2:
            return "skip"
        te = _norm(te)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(te, bins=nbins, range=(te.min(), te.max()),
                color="steelblue", alpha=0.85)
        ax.set_xlabel(r"normalized binding energy $\tilde{e}$ "
                      r"($-1$ = most bound, $0$ = least)")
        ax.set_ylabel("star count")
        ax.set_title(f"{hdf5.parent.parent.name}  {hdf5.stem}   "
                     f"N$_\\star$={len(te)}")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        return "ok"
    except Exception as exc:
        return f"fail {hdf5.name}: {repr(exc)[:100]}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gal-root", type=Path, default=GAL_ROOT)
    p.add_argument("--out-dir", type=Path, default=GAL_ROOT / "energy_dists")
    p.add_argument("--model", default=None, help="Restrict to one model")
    p.add_argument("--snap", type=int, default=None, help="Restrict to one snap")
    p.add_argument("--nbins", type=int, default=80)
    p.add_argument("--ncpu", type=int, default=8,
                   help="galaxies in flight; memory-bound (big centrals)")
    args = p.parse_args()

    pat = (f"{args.model or '*'}/snap_"
           f"{(f'{args.snap:03d}' if args.snap is not None else '*')}/Gal_*.hdf5")
    gals = sorted(args.gal_root.glob(pat))
    if not gals:
        sys.exit(f"no galaxies matched {args.gal_root / pat}")

    tasks = [(g, args.out_dir / g.parent.parent.name / g.parent.name
              / f"{g.stem}_edist.png", args.nbins) for g in gals]
    print(f"plotting {len(tasks)} MORDOR energy distributions on {args.ncpu} "
          f"cpus -> {args.out_dir}", flush=True)

    n = {"ok": 0, "skip": 0, "fail": 0}
    with Pool(args.ncpu) as pool:
        for i, st in enumerate(pool.imap_unordered(_one, tasks), 1):
            key = st if st in ("ok", "skip") else "fail"
            n[key] += 1
            if key == "fail":
                print(st, flush=True)
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)} ...", flush=True)
    print(f"done: ok={n['ok']} skip={n['skip']} fail={n['fail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
