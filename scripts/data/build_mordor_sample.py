"""Build a MORDOR sample HDF5 for one (model, snap).

Loads the parsed MORDOR ASCII table for a (model, snap), joins it with
the halo catalog (M200c, R200c, ...), and computes the
**interacting-galaxy flag**:

    M*_sat / M*_cen >= M_RATIO_MIN  AND
    r_sep = d / (R_half*,host + R_half*,sat) < R_SEP_MAX

over satellites in the same FoF. Positions and half-mass radii are read
in COMOVING ckpc via `codeLengthToComovingKpc`; periodic minimum-image
wrap uses `sim.boxSize` converted to ckpc.

The default `--r-sep-max` is +inf (disabled); pass an explicit value to
enable the cut.

Output (under cfg['paths']['scratch_mordor']/samples/):
    mordor_sample_<model>_<snap:03d>.hdf5

Datasets (flat layout, all length N_classified):
    halo_ids, Mstar, Munbound, Mthin, Mthick, Mbulge, Mpbulge, Mhalo,
    IsDisc, Ethin, Ethick, Ebulge, Epbulge, Ehalo,
    Cthin, Cthick, Cbulge, Cpbulge, Chalo,
    M200c, R200c, sat_mass_ratio, sat_r_sep
attrs:
    metadata: model, snap, redshift, h, source_mordor_txt, created_at
    cuts:     m_ratio_min, r_sep_max, n_star_min_mordor
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, save_flat
from galaxy_sidm.morphology import parse_mordor_output, MORDOR_COLS
from galaxy_sidm.data.aida_tng import build_central_subhalo_catalog


def compute_interacting_flag(sim, df, m_ratio_min, r_sep_max):
    """Compute sat_mass_ratio and sat_r_sep for each row in df.

    For each central subhalo `sid` in the MORDOR table, look at the
    satellites sharing its FoF group,
    find satellites that satisfy BOTH M*_sat/M*_cen >= m_ratio_min AND
    r_sep < r_sep_max, and record the most massive one. Defaults to
    (0, +inf) if no satellite qualifies.

    Returns:
        (sat_mass_ratio, sat_r_sep), each a 1D float array of length N.
    """
    sub_mstar = sim.units.codeMassToMsun(
        sim.subhalos("SubhaloMassType")[:, 4])
    sub_grnr = sim.subhalos("SubhaloGrNr")
    sub_pos = sim.units.codeLengthToComovingKpc(sim.subhalos("SubhaloPos"))
    sub_hmrs = sim.units.codeLengthToComovingKpc(
        sim.subhalos("SubhaloHalfmassRadType")[:, 4])
    box = sim.units.codeLengthToComovingKpc(sim.boxSize)

    n = len(df)
    sat_ratio = np.zeros(n)
    sat_rsep = np.full(n, np.inf)
    for i, sid in enumerate(df.index):
        if sid < 0 or sid >= len(sub_mstar):
            continue
        if sub_mstar[sid] <= 0 or sub_hmrs[sid] <= 0:
            continue
        grnr = sub_grnr[sid]
        same_fof = np.where(sub_grnr == grnr)[0]
        other = same_fof[same_fof != sid]
        if len(other) == 0:
            continue
        mratio = sub_mstar[other] / sub_mstar[sid]
        dx = sub_pos[other] - sub_pos[sid]
        dx -= box * np.round(dx / box)
        d = np.linalg.norm(dx, axis=1)
        denom = sub_hmrs[sid] + sub_hmrs[other]
        rsep = np.where(denom > 0, d / np.maximum(denom, 1e-30), np.inf)
        close_massive = (mratio >= m_ratio_min) & (rsep < r_sep_max)
        if close_massive.any():
            k = int(np.argmax(np.where(close_massive, mratio, -1.0)))
            sat_ratio[i] = float(mratio[k])
            sat_rsep[i] = float(rsep[k])
    return sat_ratio, sat_rsep


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="Path to config/scripts.yaml")
    p.add_argument("--model", required=True,
                   choices=["CDM", "SIDM1", "vSIDM", "WDM3", "WDM5"])
    p.add_argument("--snap", required=True, type=int)
    p.add_argument("--mordor-txt", type=Path, default=None,
                   help="MORDOR ASCII output (defaults to "
                        "<scratch_mordor>/outputs/snap_<NNN>/morphology_<MODEL>.txt)")
    p.add_argument("--m-ratio-min", type=float, default=0.2,
                   help="Mass-ratio threshold for interacting flag (default 0.2)")
    p.add_argument("--r-sep-max", type=float, default=float("inf"),
                   help="r_sep threshold for interacting flag "
                        "(default +inf, i.e. disabled)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output HDF5 path (overrides default location)")
    p.add_argument("--res", type=int, default=1080)
    args = p.parse_args()

    cfg = load_config(args.config)
    scratch_mordor = Path(cfg["paths"]["scratch_mordor"])

    mordor_txt = args.mordor_txt or (
        scratch_mordor / "outputs" / f"snap_{args.snap:03d}"
        / f"morphology_{args.model}.txt")
    if not mordor_txt.exists():
        sys.exit(f"MORDOR table not found: {mordor_txt}")

    import temet
    sim = temet.sim(run="aida", variant=args.model,
                    res=args.res, snap=args.snap)
    catalog = build_central_subhalo_catalog(sim)

    df = parse_mordor_output(mordor_txt, catalog=catalog)
    if len(df) == 0:
        sys.exit(f"Parsed MORDOR table is empty: {mordor_txt}")

    sat_ratio, sat_rsep = compute_interacting_flag(
        sim, df, m_ratio_min=args.m_ratio_min, r_sep_max=args.r_sep_max,
    )
    df["sat_mass_ratio"] = sat_ratio
    df["sat_r_sep"] = sat_rsep

    halo_ids = df.index.to_numpy(dtype=np.int64)
    arrays = {"halo_ids": halo_ids}
    for c in MORDOR_COLS[1:]:
        if c in df.columns:
            arrays[c] = df[c].to_numpy()
    for c in ("M200c", "R200c"):
        if c in df.columns:
            arrays[c] = df[c].to_numpy(dtype=np.float64)
    arrays["sat_mass_ratio"] = df["sat_mass_ratio"].to_numpy(dtype=np.float64)
    arrays["sat_r_sep"] = df["sat_r_sep"].to_numpy(dtype=np.float64)

    out_path = args.out or (
        scratch_mordor / "samples"
        / f"mordor_sample_{args.model}_{args.snap:03d}.hdf5")
    metadata = {
        "model": args.model,
        "snap": int(args.snap),
        "redshift": float(sim.redshift),
        "h": float(cfg["cosmology"]["h"]),
        "source_mordor_txt": str(mordor_txt),
    }
    cuts = {
        "m_ratio_min": float(args.m_ratio_min),
        "r_sep_max": float(args.r_sep_max),
        "n_star_min_mordor": float(
            cfg["defaults"].get("n_star_min_mordor", 1e4)),
    }
    save_flat(out_path, arrays, cuts=cuts, metadata=metadata)
    n_interacting = int((arrays["sat_mass_ratio"] >= args.m_ratio_min).sum())
    print(f"[build_mordor_sample] {args.model} snap {args.snap}: "
          f"{len(halo_ids)} galaxies "
          f"({n_interacting} interacting at m_ratio>={args.m_ratio_min}) "
          f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
