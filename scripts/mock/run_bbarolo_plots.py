"""Run BBarolo's own plotscripts for every fitted galaxy, headless.

Each galaxy's `bbarolo/plotscripts/` holds BBarolo-written scripts
(plot_kinmaps, plot_pvs, plot_pvslices, plot_chanmaps, plot_parameters)
with ABSOLUTE paths baked in -- but our batch run never executed them, so
the MOCK_*.pdf plots are missing everywhere except where you hand-ran one.

This runs each script directly, under a forced headless backend, in parallel
over galaxies. Fits made on Leonardo carry that cluster's martini root in the
baked-in paths; it is swapped for the current root in memory (the files on
disk are not modified), so the transferred library still plots. Per galaxy the scripts
run SEQUENTIALLY -- we do NOT use BBarolo's `plot_all.py`, because it
launches them with `&` (background) and those jobs would be orphaned/killed
when the launcher returns inside a batch. We also SKIP:
  - `plot_all.py`, `plot_utils.py`  (a launcher and a helper, not plots)
  - `plot_pvs_old.py`               (deprecated and broken)
Failures are caught per-script, so one bad plot can't stop a galaxy or the
batch; the PDFs land in each galaxy's `bbarolo/` dir (where the scripts
write them).

Usage (compute node):
    python scripts/mock/run_bbarolo_plots.py --ncpu 16
    python scripts/mock/run_bbarolo_plots.py --model SIDM1 --zdir z4
"""

import argparse
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config

MARTINI = Path(load_config()["paths"]["scratch_processed"]).parent / "martini"
# martini root baked into the plotscripts of fits made on Leonardo
LEGACY_MARTINI = ("/leonardo_scratch/large/userexternal/acosta01/"
                  "master_thesis_project/data/martini")
SKIP = {"plot_all.py", "plot_utils.py", "plot_pvs_old.py"}


def _one(task):
    """Run every non-skipped plot_*.py in one galaxy's plotscripts dir.

    Returns (galaxy_label, [failures]). Each script gets a forced Agg
    backend via the env (so a WebAgg matplotlibrc can't pop a server /
    block), runs to completion, and a non-zero exit or exception is
    recorded rather than raised. The script source is run with the legacy
    Leonardo root replaced by `martini_root`, from inside `psdir` so its
    `from plot_utils import *` still resolves.
    """
    psdir, martini_root = task
    psdir = Path(psdir)
    scripts = sorted(p for p in psdir.glob("plot_*.py") if p.name not in SKIP)
    env = dict(os.environ, MPLBACKEND="Agg")
    fails = []
    for s in scripts:
        try:
            code = s.read_text().replace(LEGACY_MARTINI, str(martini_root))
            r = subprocess.run([sys.executable, "-c", code], cwd=str(psdir), env=env,
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                last = (r.stderr.strip().splitlines() or ["rc!=0"])[-1]
                fails.append(f"{s.name}:{last[:80]}")
        except Exception as exc:
            fails.append(f"{s.name}:{repr(exc)[:80]}")
    label = str(psdir.parent.parent.relative_to(martini_root))
    return label, fails


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--martini-root", type=Path, default=MARTINI)
    p.add_argument("--model", default=None, help="Restrict to one model")
    p.add_argument("--zdir", default=None, help="Restrict to one z dir, e.g. z4")
    p.add_argument("--ncpu", type=int, default=8)
    args = p.parse_args()

    z = args.zdir or "z*"
    m = args.model or "*"
    psdirs = sorted(args.martini_root.glob(f"{z}/{m}/gal_*/bbarolo/plotscripts"))
    if not psdirs:
        sys.exit(f"no plotscripts dirs under {args.martini_root}/{z}/{m}")
    print(f"running BBarolo plots for {len(psdirs)} galaxies on {args.ncpu} cpus",
          flush=True)

    nok = nfail = 0
    with Pool(args.ncpu) as pool:
        for i, (label, fails) in enumerate(
                pool.imap_unordered(_one, [(str(d), args.martini_root)
                                           for d in psdirs]), 1):
            if fails:
                nfail += 1
                print(f"  {label}: {len(fails)} script fail(s): {fails}", flush=True)
            else:
                nok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(psdirs)} ...", flush=True)
    print(f"done: clean={nok}  with-failures={nfail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
