"""Run MORDOR over an AIDA-TNG sample with memory-aware parallel workers.

Master spawns up to N worker subprocesses, each processing a chunk of
per-galaxy HDF5 files in-process via `run_mordor_single`. Each worker
appends one ASCII row per galaxy (success or `# FAILED ...`) to its own
chunk output file. Master tracks progress with tqdm by tailing those
files. After all workers exit, master verifies each chunk's output is
complete and concatenates everything into the final ASCII table.

If the per-galaxy HDF5s for the requested model are not all on disk,
the master invokes `scripts/extract_galaxies.py` first to make them.

The output ASCII has the same column layout as MORDOR's CLI in list
mode (see `format_mordor_row`).

Usage:
    python scripts/run_mordor.py --model CDM --snap 67
    python scripts/run_mordor.py --model SIDM1 --snap 99 \
        --mem-per-worker 4G --max-workers 32 --resume
"""

import argparse
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm


SCRATCH = Path(os.environ.get(
    "SCRATCH", "/leonardo_scratch/large/userexternal/acosta01"
))
DEFAULT_OUT_ROOT = SCRATCH / "master_thesis_project" / "data" / "mordor_galaxies"
DEFAULT_OUTPUT_DIR = DEFAULT_OUT_ROOT / "outputs"

EXTRACT_SCRIPT = Path(__file__).resolve().parent / "extract_galaxies.py"


# ---------- size helpers ---------------------------------------------------

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([KMGT]?)B?\s*$", re.IGNORECASE)


def parse_size(s: str) -> int:
    """Parse strings like '8G', '512M', '2.5GB' to bytes."""
    m = _SIZE_RE.match(s)
    if not m:
        raise argparse.ArgumentTypeError(f"can't parse size {s!r}")
    val, unit = float(m.group(1)), m.group(2).upper()
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[unit]
    return int(val * mult)


def human_bytes(n: int) -> str:
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}P"


# ---------- sample selection ------------------------------------------------

def list_qualifying_hdf5s(model: str, snap: int, res: int, n_star_min: float,
                          out_root: Path) -> Tuple[List[Path], List[int]]:
    """Catalog → subhalo IDs → expected HDF5 paths.

    Returns (paths, sub_ids) in the same order. Paths may not exist yet.
    """
    import temet
    from galaxy_sidm.data.aida_tng import (
        build_central_subhalo_catalog, qualifying_central_ids,
    )
    sim = temet.sim(run="aida", variant=model, res=res, snap=snap)
    cat = build_central_subhalo_catalog(sim)
    sub_ids = qualifying_central_ids(cat, n_star_min=n_star_min)
    out_dir = Path(out_root) / model
    paths = [out_dir / f"Gal_{int(s):06d}.hdf5" for s in sub_ids]
    return paths, [int(s) for s in sub_ids]


# ---------- bin packing -----------------------------------------------------

def estimate_resident_bytes(hdf5_path: Path, factor: float) -> int:
    """Pessimistic resident-memory estimate from on-disk size × factor."""
    return int(hdf5_path.stat().st_size * factor)


def plan_chunks(paths: List[Path], sizes: List[int],
                mem_per_worker: int, max_workers: int,
                fat_factor: float = 1.5):
    """Distribute galaxies into worker chunks.

    Galaxies whose estimated resident bytes exceed `mem_per_worker` go
    each into their own 'fat' chunk (one galaxy per worker). The rest
    are distributed greedily over `max_workers - n_fat` normal workers,
    balanced by total bytes per chunk.

    Returns: list of dicts {id, paths, budget_bytes, kind}.
    """
    fat = []
    normal = []
    for p, sz in zip(paths, sizes):
        (fat if sz > mem_per_worker else normal).append((p, sz))

    n_fat = len(fat)
    n_normal = max(1, max_workers - n_fat) if normal else 0

    # Sort normal descending by size, greedy bin-pack
    normal.sort(key=lambda x: x[1], reverse=True)
    chunks = [[] for _ in range(n_normal)]
    loads = [0] * n_normal
    for p, sz in normal:
        i = int(np.argmin(loads))
        chunks[i].append((p, sz))
        loads[i] += sz

    plan = []
    cid = 0
    for ch in chunks:
        if not ch:
            continue
        plan.append({
            "id": cid,
            "paths": [p for p, _ in ch],
            "sizes": [sz for _, sz in ch],
            "budget_bytes": mem_per_worker,
            "kind": "normal",
        })
        cid += 1
    for p, sz in fat:
        plan.append({
            "id": cid,
            "paths": [p],
            "sizes": [sz],
            "budget_bytes": int(sz * fat_factor),
            "kind": "fat",
        })
        cid += 1
    return plan


# ---------- worker entry point ----------------------------------------------

def worker_main(args):
    """Process the chunk filelist; one row per galaxy in --output."""
    from galaxy_sidm.morphology import run_mordor_single, format_mordor_row

    paths = [Path(p.strip()) for p in args.chunk.read_text().splitlines()
             if p.strip()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: if file already has rows for some galaxies, skip them.
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            tok = line.split(maxsplit=2)
            if line.startswith("# FAILED") and len(tok) >= 3:
                done.add(tok[2].split()[0])
            elif not line.startswith("#") and tok:
                done.add(tok[0])

    with open(out_path, "a", buffering=1) as fout:
        for p in paths:
            key = str(p.resolve())
            if key in done or p.name in done:
                continue
            try:
                gal = run_mordor_single(
                    p, mode=args.mode, soft_phys_kpc=args.soft_phys_kpc,
                )
                fout.write(format_mordor_row(gal, key) + "\n")
            except Exception as e:
                fout.write(f"# FAILED {key} :: {repr(e)}\n")
            fout.flush()
    return 0


# ---------- master ----------------------------------------------------------

def maybe_extract(model: str, snap: int, res: int, n_star_min: float,
                  out_root: Path, paths: List[Path], n_workers: int,
                  soft_phys_kpc: float, base_path: Path = None):
    """Invoke extract_galaxies.py if any expected HDF5 is missing."""
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return
    print(f"\n[run_mordor] {len(missing)}/{len(paths)} HDF5s missing — "
          f"calling extract_galaxies.py...\n", flush=True)
    cmd = [
        sys.executable, str(EXTRACT_SCRIPT),
        "--model", model, "--snap", str(snap), "--res", str(res),
        "--n-star-min", f"{n_star_min:g}",
        "--out-root", str(out_root),
        "--n-workers", str(n_workers),
        "--soft-phys-kpc", f"{soft_phys_kpc}",
    ]
    if base_path is not None:
        cmd += ["--base-path", str(base_path)]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(f"extract_galaxies.py failed with code {rc}")
    still_missing = [p for p in paths if not p.exists()]
    if still_missing:
        print(f"WARNING: {len(still_missing)} HDF5s still missing after "
              f"extraction:\n  " + "\n  ".join(str(p) for p in still_missing[:10]))
        if len(still_missing) > 10:
            print(f"  ... and {len(still_missing) - 10} more")
        sys.exit(1)


def count_processed_lines(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.strip())


def planned_galaxy_count(plan) -> int:
    return sum(len(c["paths"]) for c in plan)


def master_main(args):
    paths, _ = list_qualifying_hdf5s(args.model, args.snap, args.res,
                                      args.n_star_min, args.out_root)
    if not paths:
        sys.exit("no qualifying centrals; nothing to do")
    print(f"[run_mordor] model={args.model} snap={args.snap}  "
          f"qualifying centrals: {len(paths)}")

    maybe_extract(args.model, args.snap, args.res, args.n_star_min,
                  args.out_root, paths, args.extract_workers,
                  args.soft_phys_kpc, base_path=args.base_path)

    sizes = [estimate_resident_bytes(p, args.mem_factor) for p in paths]
    print(f"[run_mordor] HDF5 sizes: median={human_bytes(int(np.median([p.stat().st_size for p in paths])))}, "
          f"max={human_bytes(max(p.stat().st_size for p in paths))}; "
          f"resident estimate factor={args.mem_factor}")

    plan = plan_chunks(paths, sizes, args.mem_per_worker, args.max_workers)
    n_normal = sum(1 for c in plan if c["kind"] == "normal")
    n_fat = sum(1 for c in plan if c["kind"] == "fat")
    total_budget = sum(c["budget_bytes"] for c in plan)
    print(f"[run_mordor] chunks: {n_normal} normal + {n_fat} fat = "
          f"{len(plan)} workers; budget total {human_bytes(total_budget)}")

    chunk_dir = args.work_dir / args.model
    chunk_dir.mkdir(parents=True, exist_ok=True)
    final_output = args.output_dir / f"morphology_{args.model}.txt"
    final_output.parent.mkdir(parents=True, exist_ok=True)

    # Write per-chunk filelists; remove output files unless --resume
    for ch in plan:
        flist = chunk_dir / f"chunk_{ch['id']:03d}.txt"
        out = chunk_dir / f"chunk_{ch['id']:03d}.out"
        flist.write_text("\n".join(str(p.resolve()) for p in ch["paths"]) + "\n")
        if not args.resume and out.exists():
            out.unlink()
        ch["filelist"] = flist
        ch["out"] = out

    # Spawn workers
    procs = []
    for ch in plan:
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker",
            "--chunk", str(ch["filelist"]),
            "--output", str(ch["out"]),
            "--mode", args.mode,
            "--soft-phys-kpc", f"{args.soft_phys_kpc}",
        ]
        log_path = chunk_dir / f"chunk_{ch['id']:03d}.log"
        log_fp = open(log_path, "w")
        worker_env = os.environ.copy()
        worker_env.update({'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
        proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=worker_env)
        procs.append({"proc": proc, "log_fp": log_fp, "chunk": ch})

    total = planned_galaxy_count(plan)
    t0 = time.time()
    pbar = tqdm(total=total, unit="gal", desc=f"MORDOR {args.model}")
    last = 0
    try:
        while any(pp["proc"].poll() is None for pp in procs):
            done = sum(count_processed_lines(pp["chunk"]["out"]) for pp in procs)
            pbar.update(done - last)
            last = done
            time.sleep(2.0)
        # final tick
        done = sum(count_processed_lines(pp["chunk"]["out"]) for pp in procs)
        pbar.update(done - last)
    finally:
        pbar.close()
        for pp in procs:
            pp["log_fp"].close()

    # Verify completion
    incomplete = []
    failed_codes = []
    for pp in procs:
        ch = pp["chunk"]
        rc = pp["proc"].returncode
        n_done = count_processed_lines(ch["out"])
        n_expected = len(ch["paths"])
        if rc != 0:
            failed_codes.append((ch["id"], rc))
        if n_done < n_expected:
            incomplete.append((ch["id"], n_done, n_expected))

    elapsed = time.time() - t0
    print(f"\n[run_mordor] elapsed: {elapsed/60:.1f} min")

    if failed_codes:
        print(f"WARNING: {len(failed_codes)} workers exited non-zero "
              f"(see chunk logs): " + ", ".join(f"{i}={rc}" for i, rc in failed_codes[:8]))
    if incomplete:
        print(f"INCOMPLETE: {len(incomplete)} chunks short of expected count.")
        for cid, nd, ne in incomplete[:10]:
            print(f"  chunk {cid:03d}: {nd}/{ne} galaxies")
        print("Re-run with --resume to retry the missing galaxies.")
        sys.exit(2)

    # Concatenate
    with open(final_output, "w") as fout:
        for ch in plan:
            if ch["out"].exists():
                fout.write(ch["out"].read_text())
    print(f"[run_mordor] wrote {final_output}  "
          f"({count_processed_lines(final_output)} galaxies)")
    return 0


# ---------- arg parsing -----------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Run MORDOR over an AIDA-TNG sample in parallel.",
    )
    p.add_argument("--worker", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--chunk", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--output", type=Path, help=argparse.SUPPRESS)

    p.add_argument("--model", choices=["CDM", "SIDM1", "vSIDM", "WDM3", "WDM5"])
    p.add_argument("--snap", type=int)
    p.add_argument("--res", type=int, default=1080)
    p.add_argument("--n-star-min", type=float, default=1e4,
                   help="Min stellar particle count for the central (default 1e4)")

    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                   help="Where per-galaxy HDF5s live")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Where the final ASCII table is written")
    p.add_argument("--work-dir", type=Path,
                   default=DEFAULT_OUT_ROOT / "chunks",
                   help="Per-chunk filelists, outputs, and worker logs")

    p.add_argument("--mem-per-worker", type=parse_size, default=parse_size("4G"),
                   help="Per-worker resident memory cap; HDF5s estimated to "
                   "exceed this go to fat workers (default 4G)")
    p.add_argument("--mem-factor", type=float, default=5.0,
                   help="Resident bytes / on-disk size pessimistic factor (default 5)")
    p.add_argument("--max-workers", type=int, default=64,
                   help="Maximum number of normal worker subprocesses (default 64)")
    p.add_argument("--extract-workers", type=int, default=16,
                   help="Workers for extract_galaxies.py if it gets called (default 16)")

    p.add_argument("--mode", default="cosmo_sim",
                   help="MORDOR potential mode (default cosmo_sim)")
    p.add_argument("--soft-phys-kpc", type=float, default=0.57)
    p.add_argument("--base-path", type=Path, default=None,
                   help="Override snapshot basePath (passed through to "
                        "extract_galaxies.py; for SCRATCH shadow trees)")
    p.add_argument("--resume", action="store_true",
                   help="Skip galaxies already in chunk outputs")
    return p


def main():
    args = build_parser().parse_args()
    if args.worker:
        return worker_main(args)
    if args.model is None or args.snap is None:
        sys.exit("--model and --snap are required for the master")
    return master_main(args)


if __name__ == "__main__":
    sys.exit(main())
