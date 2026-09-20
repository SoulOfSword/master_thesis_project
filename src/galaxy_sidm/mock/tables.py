"""Assemble the per-galaxy scaling-relation table for the mock disc sample.

Merges, per galaxy, the MORDOR sample (Mstar, M200c, IsDisc, halo_ids) with
the mock pipeline's info.json (bbarolo_rc, M_neutral) and the BBarolo
rotation curve (v_flat by the SPARC rule, see v_flat), then keeps only the
usable DISCS:
  * MORDOR IsDisc == 1,
  * a successful BBarolo fit (bbarolo_rc == 0),
  * NOT hand-flagged in config/problematic_discs.yaml.

Used by scripts/plots/scaling/plot_tfr.py and plot_shmr.py. Masses in Msun,
v_flat in km/s. Returns a list of plain dict rows so the plotting scripts
stay simple.
"""

import json
import warnings
from pathlib import Path

import numpy as np

from ..io import load_flat
from .barolo import rings_file

MODELS = ("CDM", "SIDM1", "vSIDM")
SNAPS = (17, 21, 25, 33, 50, 67)   # z = 5, 4, 3, 2, 1, 0.5


def sparc_vflat(vrot, tol=0.05):
    """V_flat of a rotation curve (ordered by radius) as in SPARC.

    Lelli, McGaugh & Schombert (2016, ApJ 816, L14): start from the mean of the
    two outermost points, add the next point inwards while it differs from the
    running mean by at most `tol` x the mean, and return the mean at the first
    point that does not. Curves not flat within `tol` over at least three points
    (the third-outermost point already fails) get nan.
    """
    v = [float(x) for x in vrot]
    if len(v) < 3:
        return float("nan")
    flat = v[-2:]
    for vi in reversed(v[:-2]):
        mean = np.mean(flat)
        if mean <= 0 or abs(vi - mean) > tol * mean:
            break
        flat.append(vi)
    return float(np.mean(flat)) if len(flat) >= 3 else float("nan")


def v_flat(bbarolo_dir, tol=0.05):
    """SPARC V_flat (km/s, see sparc_vflat) of a BBarolo fit.

    Rings from barolo.rings_file (rings_final2.txt: one VSYS for all rings, see
    kinematics._rings); column 3 is VROT(km/s). Returns nan if the file/ring
    data is missing or the curve is not flat.
    """
    try:
        rf = rings_file(bbarolo_dir)
    except FileNotFoundError as e:
        warnings.warn(str(e))
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
    return sparc_vflat(vrot, tol)


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
             flat_tol=0.05, discs_only=True, drop_failed=True):
    """Build the usable-disc table.

    Args:
        cfg: loaded project config (for snap_z + scratch paths).
        models, snaps: which (model, snap) to include.
        exclude_yaml: path to config/problematic_discs.yaml (hand flags).
        flat_tol: flatness tolerance of the SPARC v_flat rule (fraction).
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
                    M_neutral=m_neu, v_flat=v_flat(gdir / "bbarolo", flat_tol),
                    IsDisc=int(isd[i]), bbarolo_rc=rc))
    return rows
