"""Assemble the per-galaxy scaling-relation table for the mock disc sample.

Merges, per galaxy, the MORDOR sample (Mstar, M200c, IsDisc, halo_ids) with
the mock pipeline's info.json (bbarolo_rc, M_neutral) and the BBarolo
rotation curve (v_flat = mean of the last N ring VROT), then keeps only the
usable DISCS:
  * MORDOR IsDisc == 1,
  * a successful BBarolo fit (bbarolo_rc == 0),
  * NOT hand-flagged in config/problematic_discs.yaml.

Used by scripts/plots/scaling/plot_tfr.py and plot_shmr.py. Masses in Msun,
v_flat in km/s. Returns a list of plain dict rows so the plotting scripts
stay simple.
"""

import json
from pathlib import Path

import numpy as np

from ..io import load_flat

MODELS = ("CDM", "SIDM1", "vSIDM")
SNAPS = (17, 21, 25, 33, 50, 67)   # z = 5, 4, 3, 2, 1, 0.5


def v_flat(bbarolo_dir, n_outer=3):
    """Mean of the last `n_outer` ring VROT (km/s) from rings_final1.txt.

    Returns nan if the file/ring data is missing. Column 3 of rings_final1.txt
    is VROT(km/s).
    """
    rf = Path(bbarolo_dir) / "rings_final1.txt"
    if not rf.exists():
        return float("nan")
    vrot = []
    for ln in rf.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) > 3:
            try:
                vrot.append(float(parts[2]))
            except ValueError:
                pass
    if not vrot:
        return float("nan")
    return float(np.mean(vrot[-n_outer:]))


def load_excluded(exclude_yaml):
    """{model: {zkey: set(subIDs)}} from the problematic-discs YAML (or {})."""
    if not exclude_yaml or not Path(exclude_yaml).exists():
        return {}
    import yaml
    d = yaml.safe_load(Path(exclude_yaml).read_text()) or {}
    return {m: {zk: {int(x) for x in (v or [])}
                for zk, v in (zz or {}).items()}
            for m, zz in d.items()}


def assemble(cfg, models=MODELS, snaps=SNAPS, exclude_yaml=None,
             n_outer=3, discs_only=True, drop_failed=True):
    """Build the usable-disc table.

    Args:
        cfg: loaded project config (for snap_z + scratch paths).
        models, snaps: which (model, snap) to include.
        exclude_yaml: path to config/problematic_discs.yaml (hand flags).
        n_outer: rings averaged for v_flat.
        discs_only: keep only MORDOR IsDisc==1.
        drop_failed: drop galaxies with bbarolo_rc != 0 (failed/non-detection).

    Returns:
        list of dicts with keys: model, snap, z, sub_id, Mstar, M200c,
        M_neutral, v_flat, IsDisc, bbarolo_rc.
    """
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    mart = Path(cfg["paths"]["scratch_processed"]).parent / "martini"
    mdir = Path(cfg["paths"]["scratch_mordor"]) / "samples"
    excl = load_excluded(exclude_yaml)

    rows = []
    for model in models:
        for snap in snaps:
            z = snap_z.get(snap)
            sp = mdir / f"mordor_sample_{model}_{snap:03d}.hdf5"
            if z is None or not sp.exists():
                continue
            a, _ = load_flat(sp)
            ids = np.asarray(a["halo_ids"], np.int64)
            isd = np.asarray(a["IsDisc"]).astype(int)
            mstar = np.asarray(a["Mstar"], float)
            m200 = np.asarray(a["M200c"], float)
            zkey = f"z{z:g}"
            ex = excl.get(model, {}).get(zkey, set())
            for i in range(len(ids)):
                sub = int(ids[i])
                if discs_only and isd[i] != 1:
                    continue
                if sub in ex:
                    continue
                gdir = mart / zkey / model / f"gal_{sub:06d}"
                rc, m_neu = None, float("nan")
                info = gdir / "info.json"
                if info.exists():
                    try:
                        d = json.load(open(info))
                        rc = d.get("bbarolo_rc")
                        m_neu = float(d.get("M_neutral", float("nan")))
                    except Exception:
                        pass
                if drop_failed and rc != 0:
                    continue
                rows.append(dict(
                    model=model, snap=snap, z=z, sub_id=sub,
                    Mstar=float(mstar[i]), M200c=float(m200[i]),
                    M_neutral=m_neu, v_flat=v_flat(gdir / "bbarolo", n_outer),
                    IsDisc=int(isd[i]), bbarolo_rc=rc))
    return rows
