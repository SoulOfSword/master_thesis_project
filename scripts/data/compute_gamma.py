"""Compute inner DM logarithmic slope gamma_DM for a halo catalog.

Reads a catalog HDF5 (from build_catalog.py) and the matching per-halo
profiles HDF5 (from compute_profiles.py), runs
`compute_gamma_dm` over the halos, and saves the per-halo gamma_DM in a
flat HDF5 with one row per surviving halo.

Output (under cfg['paths']['scratch_processed']/gamma/):
    gamma_<model>_<snap:03d>_<r_outer_kind>_factor<X>.hdf5

Datasets:
    halo_ids, M200c (Msun), R200c (ckpc), r_s (ckpc), gamma_dm
attrs:
    metadata: model, snap, redshift, source_catalog, source_profiles
    cuts:     copied from input catalog
    variants: r_inner, r_outer_kind, r_outer_factor, r_outer_floor,
              nfw_fit_min, nfw_fit_max_factor
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat, load_per_halo, save_flat
from galaxy_sidm.observables import compute_gamma_dm


def _catalogs_dict_from_flat(cat_arrays):
    """Re-expand a flat-catalog dict into a {model: cat_like_dict}.

    `compute_gamma_dm` expects per-model catalogs whose arrays are
    indexed by FoF id; here we already have a 1:1 row-per-halo
    representation. The halo_ids in the catalog are the FoF indices.
    We build a dict whose array length equals max(halo_id)+1 so that
    indexing by hid works inside compute_gamma_dm.
    """
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


def output_name(model, snap, r_outer_kind, r_outer_factor, disc_only=False):
    disc_tag = "_disc" if disc_only else ""
    return (f"gamma_{model}_{snap:03d}"
            f"_{r_outer_kind}_factor{r_outer_factor:g}{disc_tag}.hdf5")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--catalog", type=Path, required=True,
                   help="catalog_*.hdf5 produced by build_catalog.py")
    p.add_argument("--profiles", type=Path, required=True,
                   help="profiles_*.hdf5 produced by compute_profiles.py")
    p.add_argument("--r-inner", type=float, default=1.0,
                   help="Inner radius for the slope fit, ckpc")
    p.add_argument("--r-outer-kind", choices=["nfw_rs", "r200c"],
                   default="nfw_rs")
    p.add_argument("--r-outer-factor", type=float, default=None,
                   help="Factor for r_outer (default 0.3 for nfw_rs, 0.03 for r200c)")
    p.add_argument("--r-outer-floor", type=float, default=10.0,
                   help="Floor on r_outer in ckpc (only used for r200c)")
    p.add_argument("--nfw-fit-min", type=float, default=5.0,
                   help="Inner radius for the NFW fit when r_outer_kind=nfw_rs")
    p.add_argument("--nfw-fit-max-factor", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=None,
                   help="Output HDF5 path (default under scratch_processed/gamma/)")
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

    r_outer_factor = args.r_outer_factor
    if r_outer_factor is None:
        r_outer_factor = 0.3 if args.r_outer_kind == "nfw_rs" else 0.03

    results = compute_gamma_dm(
        catalogs, model_profiles, models=[model],
        r_inner=args.r_inner,
        r_outer_kind=args.r_outer_kind,
        r_outer_factor=r_outer_factor,
        r_outer_floor=args.r_outer_floor,
        nfw_fit_min=args.nfw_fit_min,
        nfw_fit_max_factor=args.nfw_fit_max_factor,
        # All other cuts already applied in build_catalog.py; pass through
        # min_ndm=0 so we don't double-reject anything in this stage.
        min_ndm=0,
    )
    res = results[model]

    arrays = {
        "halo_ids": res["halo_ids"].astype(np.int64),
        "M200c":    res["M200c"].astype(np.float64),
        "R200c":    res["R200c"].astype(np.float64),
        "r_s":      res["r_s"].astype(np.float64),
        "gamma_dm": res["gamma_dm"].astype(np.float64),
    }

    metadata = dict(meta)
    metadata["source_catalog"] = str(args.catalog)
    metadata["source_profiles"] = str(args.profiles)

    variants = {
        "r_inner":            float(args.r_inner),
        "r_outer_kind":       args.r_outer_kind,
        "r_outer_factor":     float(r_outer_factor),
        "r_outer_floor":      float(args.r_outer_floor),
        "nfw_fit_min":        float(args.nfw_fit_min),
        "nfw_fit_max_factor": float(args.nfw_fit_max_factor),
    }

    if args.out is not None:
        out_path = args.out
    else:
        out_root = Path(args.out_root or cfg["paths"]["scratch_processed"])
        out_path = (out_root / "gamma"
                    / output_name(model, snap, args.r_outer_kind, r_outer_factor,
                                   disc_only=bool(cuts.get("disc_only", False))))

    save_flat(out_path, arrays, cuts=cuts, variants=variants, metadata=metadata)
    print(f"[compute_gamma] {model} snap {snap}: "
          f"{len(arrays['halo_ids'])} halos -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
