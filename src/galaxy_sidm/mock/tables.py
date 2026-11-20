"""Assemble the per-galaxy scaling-relation table for the mock disc sample.

Merges, per galaxy, the MORDOR sample (Mstar, M200c, IsDisc, halo_ids) with
the mock pipeline's info.json (bbarolo_rc, M_neutral), the neutral hydrogen
inside the gas cut (M_cut, processed/gas_kinematics) and the BBarolo rotation
curve (v_flat by the SPARC rule, see v_flat), then keeps only the usable DISCS:
  * the final sample (final_sample): MORDOR discs (IsDisc == 1) whose neutral
    gas passes DISC_RULE (kinematic_sample),
  * a successful BBarolo fit (bbarolo_rc == 0).

The earlier selection, the disc side of the dividing line in the (A_sq, mix)
plane of the cubes (gas_disc_sample) without the galaxies hand-flagged in
config/problematic_discs.yaml, is residual_asymmetry_sample.

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

CONFIG = Path(__file__).resolve().parents[3] / "config"
# calibration galaxy left out of the disc/perturbed line: perturbed, but it sits among the
# unsure discs (notebooks/residuals_asymmetry.ipynb)
LINE_LEAVE_OUT = (87165,)

MODELS = ("CDM", "SIDM1", "vSIDM")
SNAPS = (17, 21, 25, 33, 50, 67, 99)   # z = 5, 4, 3, 2, 1, 0.5, 0
# particle-based gas-disc rule (kinematic_sample)
DISC_RULE = dict(kappa_min=0.8, kappa_key="kappa_co_r95", z_cut_max=10.0, angle_max=30.0)


def sparc_rings(vrot, tol=0.05):
    """Number of outermost points in the SPARC flat part of a rotation curve
    (ordered by radius), 0 if it is not flat; the rule is in sparc_vflat."""
    v = [float(x) for x in vrot]
    if len(v) < 3:
        return 0
    flat = v[-2:]
    for vi in reversed(v[:-2]):
        mean = np.mean(flat)
        if mean <= 0 or abs(vi - mean) > tol * mean:
            break
        flat.append(vi)
    return len(flat) if len(flat) >= 3 else 0


def sparc_vflat(vrot, tol=0.05):
    """V_flat of a rotation curve (ordered by radius) as in SPARC.

    Lelli, McGaugh & Schombert (2016, ApJ 816, L14): start from the mean of the
    two outermost points, add the next point inwards while it differs from the
    running mean by at most `tol` x the mean, and return the mean at the first
    point that does not. Curves not flat within `tol` over at least three points
    (the third-outermost point already fails) get nan.
    """
    n = sparc_rings(vrot, tol)
    return float(np.mean([float(x) for x in vrot][-n:])) if n else float("nan")


def sparc_mod_vflat(vrot, tol=0.05, min_rings=3):
    """V_flat of a rotation curve (ordered by radius) by our rule, SPARC_mod.

    Every stretch of at least min_rings neighbouring points whose speeds are all within tol
    of each other (fastest <= (1 + tol) x slowest) is a candidate; the longest wins, and
    between equally long ones the faster. Returns (v_flat, indices of the stretch's
    points), or (nan, no indices) if there is no such stretch.
    """
    v = np.asarray(vrot, dtype=float)
    best, best_key = np.array([], dtype=int), None
    for i in range(len(v)):
        j = i
        while j + 1 < len(v) and v[i:j + 2].max() <= (1 + tol) * v[i:j + 2].min():
            j += 1
        stretch = np.arange(i, j + 1)
        if len(stretch) < min_rings or v[stretch].min() <= 0:
            continue
        key = (len(stretch), v[stretch].mean())
        if best_key is None or key > best_key:
            best, best_key = stretch, key
    return (float(v[best].mean()) if len(best) else float("nan")), best


def flat_rings(vrot, tol, retry_tol):
    """Rings where the rotation curve is flat, and which rule found them: 0 SPARC_mod at tol,
    1 SPARC_mod at retry_tol, 2 SPARC at tol; (no rings, None) if none finds a flat part."""
    for k, t in enumerate((tol, retry_tol)):
        _, used = sparc_mod_vflat(vrot, t)
        if len(used):
            return used, k
    n = sparc_rings(vrot, tol)
    if n:
        return np.arange(len(vrot) - n, len(vrot)), 2
    return np.array([], dtype=int), None


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


def vcirc_curves(cfg, model, snap):
    """{subID: (R, v_circ)} of the direct-method curves in processed/vcirc
    (scripts/data/compute_vcirc.py: R_direct, vcirc_total)."""
    a, _ = load_flat(Path(cfg["paths"]["scratch_processed"]) / "vcirc" / f"vcirc_{model}_{snap:03d}.hdf5")
    return {int(s): (R, v) for s, R, v in zip(a["halo_ids"], a["R_direct"], a["vcirc_total"])}


def vcirc_at(r, R, v):
    """v_circ at radii r, linear between the points of the curve (R, v); nan outside it."""
    return np.interp(r, R, v, left=np.nan, right=np.nan)


def load_excluded(exclude_yaml):
    """{model: {zkey: set(subIDs)}} from the problematic-discs YAML (or {})."""
    if not exclude_yaml or not Path(exclude_yaml).exists():
        return {}
    import yaml
    d = yaml.safe_load(Path(exclude_yaml).read_text()) or {}
    return {m: {zk: {int(x) for x in (v or [])}
                for zk, v in (zz or {}).items()}
            for m, zz in d.items()}


def calibration_classes(calibration_yaml=CONFIG / "residual_calibration.yaml"):
    """{(model, z, subID): class} of the hand-classified calibration galaxies (discs,
    unsure_discs or perturbed)."""
    import yaml
    labels = yaml.safe_load(Path(calibration_yaml).read_text())
    return {(m, float(zk[1:]), int(s)): c for m, by_z in labels.items()
            for zk, classes in (by_z or {}).items()
            for c, subs in (classes or {}).items() for s in (subs or [])}


def gas_disc_sample(cfg, exclude_yaml=CONFIG / "problematic_discs.yaml",
                    calibration_yaml=CONFIG / "residual_calibration.yaml"):
    """Galaxies with a gas disc, rebuilt from the current residual and asymmetry tables.

    Fitted on the calibration galaxies as in notebooks/residuals_asymmetry.ipynb: the
    LDA mix of the PV and cube res2 residuals (three classes weighted equally), and
    the line in the (A_sq, mix) plane between the perturbed ones and the rest (linear
    SVM, C = 1e6, both axes scaled, LINE_LEAVE_OUT left out). A galaxy is in the
    sample if it has both residuals and A_sq, lies on the disc side of the line and
    is not in exclude_yaml (where the grey region, below the line with mix > 4, is
    checked by eye).

    Args:
        cfg: loaded project config (scratch_processed path).
        exclude_yaml: hand-flagged galaxies (config/problematic_discs.yaml); None: none.
        calibration_yaml: classes of the calibration galaxies.

    Returns:
        DataFrame, one row per galaxy of the residual tables: model, z, subID, the
        residuals, A_sq, mix, perturbed, flagged, in_sample. attrs: a_pv (weight of
        the PV res2 in the mix), line_intercept and line_slope (mix = intercept +
        slope A_sq).
    """
    import pandas as pd
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if exclude_yaml is not None and not Path(exclude_yaml).exists():
        raise FileNotFoundError(f"{exclude_yaml} not found")
    gd = Path(cfg["paths"]["scratch_processed"]) / "gas_discs"
    key = ["model", "z", "subID"]
    df = (pd.read_csv(gd / "residuals_pv_all.csv")
          .merge(pd.read_csv(gd / "residuals_cube_all.csv"), on=key)
          .merge(pd.read_csv(gd / "asymmetry_3d.csv")[key + ["A_sq"]], on=key, how="left"))

    cls_of = calibration_classes(calibration_yaml)
    order = ["discs", "unsure_discs", "perturbed"]
    cls = pd.Series([cls_of.get(k) for k in zip(df["model"], df["z"], df["subID"])], index=df.index)
    fit = df[cls.notna()].dropna(subset=["pv_res1", "cube_res1", "pv_res2", "cube_res2"])
    y = cls[fit.index].map(order.index).to_numpy()

    # LDA mix of the res2 residuals, weights rescaled to add up to 1
    w2 = LinearDiscriminantAnalysis(priors=[1 / 3] * 3).fit(
        fit[["pv_res2", "cube_res2"]].to_numpy(), y).scalings_[:, 0]
    a = w2[0] / w2.sum()
    df["mix"] = a * df["pv_res2"] + (1 - a) * df["cube_res2"]

    # line between the perturbed calibration galaxies and the rest
    use = (fit["A_sq"].notna() & ~fit["subID"].isin(LINE_LEAVE_OUT)).to_numpy()
    Z = np.column_stack([fit["A_sq"].to_numpy(), df.loc[fit.index, "mix"].to_numpy()])
    pert = (y == order.index("perturbed")).astype(int)
    svm = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1e6)).fit(Z[use], pert[use])
    w = svm[1].coef_[0] / svm[0].scale_ # the line in A_sq and mix: w . (A_sq, mix) + b = 0
    b = svm[1].intercept_[0] - w @ svm[0].mean_

    placed = (df["A_sq"].notna() & df["mix"].notna()).to_numpy()
    df["perturbed"] = False
    df.loc[placed, "perturbed"] = svm.predict(df.loc[placed, ["A_sq", "mix"]].to_numpy()) == 1
    excluded = load_excluded(exclude_yaml)
    df["flagged"] = [int(s) in excluded.get(m, {}).get(f"z{z:g}", set())
                     for m, z, s in zip(df["model"], df["z"], df["subID"])]
    df["in_sample"] = placed & ~df["perturbed"] & ~df["flagged"]
    df.attrs.update(a_pv=a, line_intercept=-b / w[1], line_slope=-w[0] / w[1])
    return df


def residual_asymmetry_sample(cfg):
    """{(model, snap): [subIDs]} of the earlier sample, built now: the gas discs of
    gas_disc_sample (disc side of the residual-asymmetry line, not hand-flagged) that are
    MORDOR discs (IsDisc == 1); and the dividing line."""
    gal = gas_disc_sample(cfg)
    line = (f"res2 mix = {gal.attrs['line_intercept']:.3f} + {gal.attrs['line_slope']:.3f} A_sq, "
            f"mix = {gal.attrs['a_pv']:.3f} PV + {1 - gal.attrs['a_pv']:.3f} cube res2")
    gal = gal[gal["in_sample"]]
    snap_of = {float(z): int(snap) for snap, z in cfg["snap_z"].items()}
    mordor = Path(cfg["paths"]["scratch_mordor"]) / "samples"
    groups = {}
    for (model, z), subs in gal.groupby(["model", "z"])["subID"]:
        snap = snap_of[float(z)]
        s, _ = load_flat(mordor / f"mordor_sample_{model}_{snap:03d}.hdf5")
        is_disc = dict(zip(np.asarray(s["halo_ids"], dtype=np.int64), np.asarray(s["IsDisc"])))
        missing = [int(x) for x in subs if int(x) not in is_disc]
        if missing:
            raise KeyError(f"{model} snap {snap}: not in the MORDOR sample: {missing}")
        groups[(model, snap)] = [int(x) for x in subs if is_disc[int(x)] == 1]
    return groups, line


def kinematic_sample(cfg, kappa_min, kappa_key="kappa_co_r99", z_cut_max=None, angle_max=None):
    """{(model, snap): [subIDs]} of the MORDOR discs (IsDisc == 1) whose neutral gas
    (processed/gas_kinematics, scripts/data/compute_gas_kinematics.py) has kappa_key >= kappa_min
    and, when given, z_cut < z_cut_max (kpc) and gas_star_deg and warp_deg <= angle_max (deg),
    built now; and a one-line description of the rule. A galaxy with nan in one of these fails
    the rule. DISC_RULE is the rule of the sample. Snapshots without a MORDOR sample (e.g. SIDM1
    z = 0) are not part of it."""
    proc = Path(cfg["paths"]["scratch_processed"])
    mordor = Path(cfg["paths"]["scratch_mordor"]) / "samples"
    groups = {}
    for model in MODELS:
        for snap in SNAPS:
            sp = mordor / f"mordor_sample_{model}_{snap:03d}.hdf5"
            if not sp.exists():
                continue
            s, _ = load_flat(sp)
            is_disc = dict(zip(np.asarray(s["halo_ids"], dtype=np.int64), np.asarray(s["IsDisc"])))
            k, _ = load_flat(proc / "gas_kinematics" / f"gas_kinematics_{model}_{snap:03d}.hdf5")
            subs = np.asarray(k["halo_ids"], dtype=np.int64)
            missing = [int(x) for x in subs if int(x) not in is_disc]
            if missing:
                raise KeyError(f"{model} snap {snap}: not in the MORDOR sample: {missing}")
            keep = (k[kappa_key] >= kappa_min) & np.array([is_disc[int(x)] == 1 for x in subs], dtype=bool)
            if z_cut_max is not None:
                keep &= k["z_cut"] < z_cut_max
            if angle_max is not None:
                keep &= (k["gas_star_deg"] <= angle_max) & (k["warp_deg"] <= angle_max)
            groups[(model, snap)] = [int(x) for x in subs[keep]]
    rule = [f"{kappa_key} >= {kappa_min:g}"]
    if z_cut_max is not None:
        rule.append(f"z_cut < {z_cut_max:g} kpc")
    if angle_max is not None:
        rule.append(f"gas-star and warp angles <= {angle_max:g} deg")
    return groups, ", ".join(rule) + " of the neutral gas, MORDOR discs"


def final_sample(cfg):
    """{(model, snap): [subIDs]} of the final sample, built now: the MORDOR discs whose neutral
    gas passes DISC_RULE (kinematic_sample); and a one-line description of the rule."""
    return kinematic_sample(cfg, **DISC_RULE)


def assemble(cfg, models=MODELS, snaps=SNAPS, exclude_yaml=None,
             flat_tol=0.05, discs_only=True, drop_failed=True, gas_discs_only=True):
    """Build the usable-disc table.

    Args:
        cfg: loaded project config (for snap_z + scratch paths).
        models, snaps: which (model, snap) to include.
        exclude_yaml: hand flags of the earlier selection to drop as well
            (config/problematic_discs.yaml); None: none.
        flat_tol: flatness tolerance of the SPARC v_flat rule (fraction).
        discs_only: keep only MORDOR IsDisc==1.
        drop_failed: drop galaxies with bbarolo_rc != 0 (failed/non-detection).
        gas_discs_only: keep only the final sample (final_sample).

    Returns:
        list of dicts with keys: model, snap, z, sub_id, Mstar, M200c,
        M_neutral (all the neutral hydrogen of the subhalo), M_cut (the neutral
        hydrogen inside the gas cut; nan without a gas_kinematics entry), v_flat,
        IsDisc, bbarolo_rc.
    """
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    mart = Path(cfg["paths"]["scratch_processed"]).parent / "martini"
    mdir = Path(cfg["paths"]["scratch_mordor"]) / "samples"
    kdir = Path(cfg["paths"]["scratch_processed"]) / "gas_kinematics"
    excl = load_excluded(exclude_yaml)
    gas = None
    if gas_discs_only:
        groups, _ = final_sample(cfg)
        gas = {(m, s, sub) for (m, s), subs in groups.items() for sub in subs}

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
            m_cut = {}
            kp = kdir / f"gas_kinematics_{model}_{snap:03d}.hdf5"
            if kp.exists():
                k, _ = load_flat(kp)
                m_cut = dict(zip(np.asarray(k["halo_ids"], np.int64).tolist(), np.asarray(k["M_cut"], float)))
            zkey = f"z{z:g}"
            ex = excl.get(model, {}).get(zkey, set())
            for i in range(len(ids)):
                sub = int(ids[i])
                if discs_only and isd[i] != 1:
                    continue
                if sub in ex:
                    continue
                if gas is not None and (model, snap, sub) not in gas:
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
                    M_neutral=m_neu, M_cut=m_cut.get(sub, float("nan")),
                    v_flat=v_flat(gdir / "bbarolo", flat_tol),
                    IsDisc=int(isd[i]), bbarolo_rc=rc))
    return rows
