"""Run MORDOR for all (model, snap) combinations sequentially.

For each combination, sets snap-tagged output and work directories so
runs at different snapshots do not overwrite each other. Skips combos
whose output ASCII already exists and is non-empty (use --force to
override). After each combo, reports DONE / FAILED and continues.

Defaults:
  snaps  = 67, 50, 33, 25, 17  (z = 0.5, 1, 2, 3, 5)
  models = CDM, SIDM1, vSIDM

Output layout (per combo):
  <out-root>/outputs/snap_<NN>/morphology_<MODEL>.txt
  <out-root>/chunks/snap_<NN>/<MODEL>/...

Usage:
  python scripts/run_mordor_all.py
  python scripts/run_mordor_all.py --max-workers 32 --mem-per-worker 8G
  python scripts/run_mordor_all.py --snaps 50 33 --models CDM
  python scripts/run_mordor_all.py --force
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SCRATCH = Path(os.environ.get(
    "SCRATCH", "/leonardo_scratch/large/userexternal/acosta01"
))
DEFAULT_OUT_ROOT = SCRATCH / "master_thesis_project" / "data" / "mordor_galaxies"
RUN_MORDOR = Path(__file__).resolve().parent / "run_mordor.py"

DEFAULT_SNAPS = [67, 50, 33, 25, 17]
DEFAULT_MODELS = ["CDM", "SIDM1", "vSIDM"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snaps", type=int, nargs="+", default=DEFAULT_SNAPS)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   choices=["CDM", "SIDM1", "vSIDM", "WDM3", "WDM5"])
    p.add_argument("--max-workers", type=int, default=64)
    p.add_argument("--mem-per-worker", default="4G")
    p.add_argument("--mode", default="cosmo_sim")
    p.add_argument("--soft-phys-kpc", type=float, default=0.57)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--force", action="store_true",
                   help="Rerun even if output ASCII already exists.")
    p.add_argument("--base-path", type=Path, default=None,
                   help="Override snapshot basePath (passed through to "
                        "run_mordor.py; for SCRATCH shadow trees, applies "
                        "to all (model, snap) combos in this invocation)")
    args = p.parse_args()

    todo = []
    for snap in args.snaps:
        for model in args.models:
            out_dir = args.out_root / "outputs" / f"snap_{snap:03d}"
            work_dir = args.out_root / "chunks" / f"snap_{snap:03d}"
            out_file = out_dir / f"morphology_{model}.txt"
            if (out_file.exists() and out_file.stat().st_size > 0
                    and not args.force):
                print(f"SKIP  {model} snap {snap}: exists at {out_file}")
                continue
            todo.append((model, snap, out_dir, work_dir, out_file))

    if not todo:
        print("\nNothing to do. Use --force to rerun all combinations.")
        return 0

    print(f"\nWill run {len(todo)} combination(s):")
    for m, s, _, _, _ in todo:
        print(f"  {m} snap {s}")

    t0 = time.time()
    failures = []
    for i, (model, snap, out_dir, work_dir, out_file) in enumerate(todo, 1):
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(todo)}] {model} snap {snap}  ->  {out_file}")
        print(f"{'=' * 70}\n", flush=True)

        cmd = [
            sys.executable, str(RUN_MORDOR),
            "--model", model,
            "--snap", str(snap),
            "--max-workers", str(args.max_workers),
            "--mem-per-worker", args.mem_per_worker,
            "--mode", args.mode,
            "--soft-phys-kpc", f"{args.soft_phys_kpc}",
            "--out-root", str(args.out_root),
            "--output-dir", str(out_dir),
            "--work-dir", str(work_dir),
            "--resume",
        ]
        if args.base_path is not None:
            cmd += ["--base-path", str(args.base_path)]

        t_start = time.time()
        rc = subprocess.run(cmd).returncode
        dt = time.time() - t_start

        if rc == 0:
            print(f"\n[{i}/{len(todo)}] OK in {dt/60:.1f} min")
        else:
            print(f"\n[{i}/{len(todo)}] FAILED rc={rc} after {dt/60:.1f} min")
            failures.append((model, snap, rc))

    total = (time.time() - t0) / 60
    print(f"\n{'=' * 70}")
    print(f"DONE. Total elapsed: {total:.1f} min  "
          f"({len(todo) - len(failures)} ok, {len(failures)} failed)")
    if failures:
        print("Failures:")
        for m, s, rc in failures:
            print(f"  {m} snap {s}: rc={rc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
