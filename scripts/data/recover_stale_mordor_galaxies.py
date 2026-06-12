"""Recover the stale (cross-snapshot) galaxies in the MORDOR tables.

After `migrate_galaxies_to_snap_dirs.py`, each `<model>/snap_<NNN>/` holds
the correctly-extracted galaxies for that snapshot. Any qualifying central
that is *missing* from its per-snap dir is one whose `Gal_<id>.hdf5` was a
different snapshot's file (the historic flat-dir collision). This:

    1. re-extracts each missing central at the correct snapshot,
    2. runs MORDOR on it (in-process; needs a compute node),
    3. splices the fresh row into the per-snap morphology table,
       replacing any stale row for that subhalo id.

The morphology table is backed up to `<name>.bak` before rewriting.
Rebuild the samples afterwards with `build_mordor_sample.py`.

CDM snap 21 is read from the SCRATCH shadow tree (`paths.shadow_cdm`)
since its snapshot is not in $WORK.

Usage:
    python scripts/data/recover_stale_mordor_galaxies.py             # dry-run
    python scripts/data/recover_stale_mordor_galaxies.py --apply
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config
from galaxy_sidm.morphology import run_mordor_single, format_mordor_row
from galaxy_sidm.morphology.extract import extract_galaxy_hdf5
from galaxy_sidm.data.aida_tng import (
    build_central_subhalo_catalog, qualifying_central_ids,
)

_ID_RE = re.compile(r"_(\d+)\.hdf5$")


def _id_from(text):
    m = _ID_RE.search(str(text))
    return int(m.group(1)) if m else None


def ids_present(snap_dir: Path):
    return {_id_from(p.name) for p in snap_dir.glob("Gal_*.hdf5")}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--snaps", type=int, nargs="*", default=None,
                   help="Snaps to recover (default: cfg['mosaic_snaps'])")
    p.add_argument("--res", type=int, default=1080)
    p.add_argument("--apply", action="store_true",
                   help="Extract + MORDOR + patch tables (default: dry-run)")
    args = p.parse_args()

    cfg = load_config(args.config)
    scratch_mordor = Path(cfg["paths"]["scratch_mordor"])
    shadow_cdm = cfg["paths"].get("shadow_cdm")
    n_star_min = float(cfg["defaults"].get("n_star_min_mordor", 1e4))
    soft = float(cfg["softening"]["phys_kpc"])
    h = float(cfg["cosmology"]["h"])
    models = args.models or list(cfg["models"])
    snaps = args.snaps or list(cfg["mosaic_snaps"])

    import temet
    total_recovered = 0
    for model in models:
        for snap in snaps:
            sim = temet.sim(run="aida", variant=model, res=args.res, snap=snap)
            cat = build_central_subhalo_catalog(sim)
            base_path = cat["basePath"]
            if model == "CDM" and snap == 21 and shadow_cdm:
                base_path = str(Path(shadow_cdm) / "output").rstrip("/") + "/"
                cat["basePath"] = base_path

            qual = {int(s) for s in qualifying_central_ids(
                cat, n_star_min=n_star_min)}
            snap_dir = scratch_mordor / model / f"snap_{snap:03d}"
            missing = sorted(qual - ids_present(snap_dir))
            if not missing:
                print(f"  {model} snap {snap:03d}: complete")
                continue
            print(f"  {model} snap {snap:03d}: {len(missing)} to recover "
                  f"-> {missing}")
            total_recovered += len(missing)
            if not args.apply:
                continue

            snap_dir.mkdir(parents=True, exist_ok=True)
            new_rows = []
            for sid in missing:
                gal_path = snap_dir / f"Gal_{sid:06d}.hdf5"
                extract_galaxy_hdf5(
                    base_path=base_path, snap=snap, subhalo_id=sid,
                    out_path=gal_path, h=h, soft_phys_kpc=soft, overwrite=True)
                try:
                    gal = run_mordor_single(
                        gal_path, mode="cosmo_sim", soft_phys_kpc=soft)
                    new_rows.append(format_mordor_row(gal, str(gal_path.resolve())))
                except Exception as e:
                    new_rows.append(f"# FAILED {gal_path} :: {repr(e)}")

            txt = (scratch_mordor / "outputs" / f"snap_{snap:03d}"
                   / f"morphology_{model}.txt")
            miss = set(missing)
            old = txt.read_text().splitlines() if txt.exists() else []
            kept = [ln for ln in old
                    if ln.startswith("#") or _id_from(ln.split()[0] if ln.split()
                                                      else "") not in miss]
            shutil.copy2(txt, txt.with_suffix(".txt.bak"))
            txt.write_text("\n".join(kept + new_rows) + "\n")
            n_fail = sum(1 for r in new_rows if r.startswith("#"))
            print(f"    patched {txt.name}: kept {len(kept)}, "
                  f"added {len(new_rows)} ({n_fail} MORDOR failures)")

    mode = "APPLIED" if args.apply else "DRY-RUN (nothing changed)"
    print(f"\n[recover_stale_mordor_galaxies] {mode}: "
          f"{total_recovered} galaxies across {len(models)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
