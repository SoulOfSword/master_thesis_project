"""Build the work list for the mock pipeline batch.

One line per galaxy: <model> <snap> <sub_id>. Takes every MORDOR disc at
all redshifts plus the non-discs at z>=4. Prints a per-bin tally.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat

MODELS = ["CDM", "SIDM1", "vSIDM"]
HIGHZ = 4.0          # non-discs are kept only at z >= this


def main():
    cfg = load_config(None)
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    mdir = Path(cfg["paths"]["scratch_mordor"]) / "samples"
    out = (Path(cfg["paths"]["scratch_processed"]).parent / "martini"
           / "manifest.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    print(f"{'model':6s} {'snap':>4s} {'z':>4s} {'disc':>5s} {'nonZ':>5s}")
    for model in MODELS:
        for snap, z in sorted(snap_z.items()):
            p = mdir / f"mordor_sample_{model}_{snap:03d}.hdf5"
            if not p.exists():
                print(f"{model:6s} {snap:>4d} {z:>4.1f}   (missing)")
                continue
            arrs, _ = load_flat(p)
            ids = np.asarray(arrs["halo_ids"], dtype=np.int64)
            isd = np.asarray(arrs["IsDisc"]).astype(int)
            sel_disc = isd == 1
            sel_nonz = (isd == 0) & (z >= HIGHZ)
            for sid in ids[sel_disc | sel_nonz]:
                lines.append(f"{model} {snap} {int(sid)}")
            print(f"{model:6s} {snap:>4d} {z:>4.1f} "
                  f"{int(sel_disc.sum()):>5d} {int(sel_nonz.sum()):>5d}")
    out.write_text("\n".join(lines) + "\n")
    print(f"\n{len(lines)} galaxies -> {out}")


if __name__ == "__main__":
    main()
