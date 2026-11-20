"""Build the work list for the mock pipeline batch.

One line per galaxy: <model> <snap> <sub_id>. Takes every MORDOR disc
(IsDisc == 1) at all redshifts. Prints a per-bin tally.

Until 2026-09-19 the manifest also held the MORDOR non-discs at z >= 4 (155
galaxies: 93 at z=4, 62 at z=5; 4237 lines in total, backed up as
manifest_4237_with_highz_nondiscs.txt next to manifest.txt). Their cubes and
BBarolo fits are still on disk. To bring them back, see the note at the
selection below.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat

MODELS = ["CDM", "SIDM1", "vSIDM"]


def main():
    cfg = load_config(None)
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    mdir = Path(cfg["paths"]["scratch_mordor"]) / "samples"
    out = (Path(cfg["paths"]["scratch_processed"]).parent / "martini"
           / "manifest.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    print(f"{'model':6s} {'snap':>4s} {'z':>4s} {'discs':>5s}")
    for model in MODELS:
        for snap, z in sorted(snap_z.items()):
            p = mdir / f"mordor_sample_{model}_{snap:03d}.hdf5"
            if not p.exists():
                print(f"{model:6s} {snap:>4d} {z:>4.1f}   (missing)")
                continue
            arrs, _ = load_flat(p)
            ids = np.asarray(arrs["halo_ids"], dtype=np.int64)
            isd = np.asarray(arrs["IsDisc"]).astype(int)
            # discs only. To add the non-discs at z >= 4 again, select
            #   ids[(isd == 1) | ((isd == 0) & (z >= 4.0))]
            for sid in ids[isd == 1]:
                lines.append(f"{model} {snap} {int(sid)}")
            print(f"{model:6s} {snap:>4d} {z:>4.1f} {int((isd == 1).sum()):>5d}")
    out.write_text("\n".join(lines) + "\n")
    print(f"\n{len(lines)} galaxies -> {out}")


if __name__ == "__main__":
    main()
