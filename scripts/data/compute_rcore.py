"""Fit a cored-NFW profile per halo and save r_core, r_s, rho_s, chi2.

Reads a catalog HDF5 (from build_catalog.py) and the matching per-halo
profiles HDF5 (from compute_profiles.py), runs `compute_r_core_dm` over
the halos, and saves the per-halo fit results in a flat HDF5 with one
row per surviving halo.

Output (under cfg['paths']['scratch_processed']/rcore/):
    rcore_<model>_<snap:03d>_fitmin<X>.hdf5

Datasets:
    halo_ids, M200c (Msun), R200c (ckpc), r_core (ckpc), r_s (ckpc),
    rho_s (Msun/ckpc^3), chi2
attrs:
    metadata: model, snap, redshift, source_catalog, source_profiles
    cuts:     copied from input catalog
    variants: fit_min, fit_max_factor
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat, load_per_halo, save_flat
from galaxy_sidm.observables import compute_r_core_dm


def _catalogs_dict_from_flat(cat_arrays):
    """Build an indexable catalog dict from a row-per-halo flat catalog."""
    halo_ids = np.asarray(cat_arrays["halo_ids"], dtype=np.int64)
    if len(halo_ids) == 0:
        n = 0
    else:
        n = int(halo_ids.max()) + 1
    M200c = np.zeros(n, dtype=np.float64)
    R200c = np.zeros(n, dtype=np.float64)
    N_dm = np.zeros(n, dtype=np.int64)
    Mstar = np.zeros(n, dtype=np.float64)
    M200c[halo_ids] = cat_arrays["M200c"]
    R200c[halo_ids] = cat_arrays["R200c"]
    N_dm[halo_ids] = cat_arrays["N_dm"]
    if "Mstar" in cat_arrays:
        Mstar[halo_ids] = cat_arrays["Mstar"]
    return {
        "M200c": M200c,
        "R200c": R200c,
        "N_dm":  N_dm,
        "Mstar": Mstar,
    }


def output_name(model, snap, fit_min, disc_only=False):
    disc_tag = "_disc" if disc_only else ""
    return f"rcore_{model}_{snap:03d}_fitmin{fit_min:g}{disc_tag}.hdf5"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--catalog", type=Path, required=True,
                   help="catalog_*.hdf5 produced by build_catalog.py")
    p.add_argument("--profiles", type=Path, required=True,
                   help="profiles_*.hdf5 produced by compute_profiles.py")
    p.add_argument("--fit-min", type=float, default=1.0,
                   help="Inner radius for the cored-NFW fit, ckpc")
    p.add_argument("--fit-max-factor", type=float, default=1.0,
                   help="Outer fit radius = factor * R200c")
    p.add_argument("--out", type=Path, default=None,
                   help="Output HDF5 path (default under scratch_processed/rcore/)")
    p.add_argument("--out-root", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)

    cat_arrays, cat_attrs = load_flat(args.catalog)
    meta = cat_attrs["metadata"]
    cuts = cat_attrs["cuts"]
    model = meta["model"]
    snap = int(meta["snap"])

    profs_raw, _prof_attrs = load_per_halo(args.profiles)
    profiles = {
        int(hid): {
            "r_outer": np.asarray(d["r_outer"]),
            "prof_dm": np.asarray(d["prof_dm"]) if d["prof_dm"].size > 0 else None,
        }
        for hid, d in profs_raw.items()
    }

    catalogs = {model: _catalogs_dict_from_flat(cat_arrays)}
    model_profiles = {model: profiles}

    results = compute_r_core_dm(
        catalogs, model_profiles, models=[model],
        fit_min=args.fit_min,
        fit_max_factor=args.fit_max_factor,
        # All other cuts already applied upstream; pass through.
        min_ndm=0,
    )
    res = results[model]

    arrays = {
        "halo_ids": res["halo_ids"].astype(np.int64),
        "M200c":    res["M200c"].astype(np.float64),
        "R200c":    res["R200c"].astype(np.float64),
        "r_core":   res["r_core"].astype(np.float64),
        "r_s":      res["r_s"].astype(np.float64),
        "rho_s":    res["rho_s"].astype(np.float64),
        "chi2":     res["chi2"].astype(np.float64),
    }

    metadata = dict(meta)
    metadata["source_catalog"] = str(args.catalog)
    metadata["source_profiles"] = str(args.profiles)

    variants = {
        "fit_min":        float(args.fit_min),
        "fit_max_factor": float(args.fit_max_factor),
    }

    if args.out is not None:
        out_path = args.out
    else:
        out_root = Path(args.out_root or cfg["paths"]["scratch_processed"])
        out_path = (out_root / "rcore"
                    / output_name(model, snap, args.fit_min,
                                   disc_only=bool(cuts.get("disc_only", False))))

    save_flat(out_path, arrays, cuts=cuts, variants=variants, metadata=metadata)
    print(f"[compute_rcore] {model} snap {snap}: "
          f"{len(arrays['halo_ids'])} halos -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
