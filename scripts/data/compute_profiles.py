"""Slice Giulia's precomputed density-profile catalog to a halo sample.

Reads a catalog.hdf5 (one row per qualifying halo) and Giulia's
`cat_halo_profiles_<snap>.hdf5` (or _test variant), extracts the
profiles for each catalog halo, and saves them in a per-halo HDF5.

Output (under cfg['paths']['scratch_processed']/profiles/):
    profiles_<model>_<snap:03d>_<sample_key>.hdf5

Groups:
    halos/fof_<id>/{r_edges, r_outer, prof_dm, prof_gas, prof_stars}
attrs:
    metadata: model, snap, redshift, h, source_catalog, created_at
    cuts:     copied from input catalog
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat, save_per_halo
from galaxy_sidm.data.aida_tng import load_precomputed_profiles


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--catalog", type=Path, required=True,
                   help="catalog_*.hdf5 produced by build_catalog.py")
    p.add_argument("--use-test", action="store_true",
                   help="Force use of the _test variant of the profile catalog "
                        "(equivalent to --suffix _test)")
    p.add_argument("--no-test", action="store_true",
                   help="Force use of the non-_test (full) variant "
                        "(equivalent to --suffix '')")
    p.add_argument("--suffix", type=str, default=None,
                   help="Explicit suffix to look for, e.g. '_new', '_test', "
                        "or '' (the empty string). Overrides --use-test/"
                        "--no-test. When unset, tries '_new' -> '_test' -> ''.")
    p.add_argument("--out-root", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)

    cat, cat_attrs = load_flat(args.catalog)
    meta = cat_attrs["metadata"]
    cuts = cat_attrs["cuts"]
    model = meta["model"]
    snap = int(meta["snap"])

    if args.use_test and args.no_test:
        sys.exit("--use-test and --no-test are mutually exclusive")
    if args.suffix is not None:
        suffix_attempts = [args.suffix]
    elif args.use_test:
        suffix_attempts = ["_test"]
    elif args.no_test:
        suffix_attempts = [""]
    else:
        suffix_attempts = ["_new", "_test", ""]

    aida_root = Path(cfg["paths"]["aida_root"])
    # AIDA stores vSIDM under L35n1080_vSIDM_correa; everything else matches "L35n1080_<model>".
    # DMO models carry a "-Dark" suffix that must come AFTER the _correa tag.
    is_dark = model.endswith("-Dark")
    base_model = model[:-len("-Dark")] if is_dark else model
    dark_suffix = "-Dark" if is_dark else ""
    if base_model == "vSIDM":
        variant_dir = f"L35n1080_vSIDM_correa{dark_suffix}"
    else:
        variant_dir = f"L35n1080_{base_model}{dark_suffix}"
    run_path = aida_root / variant_dir

    profs = None
    last_err = None
    chosen_suffix = None
    for attempt in suffix_attempts:
        try:
            profs = load_precomputed_profiles(
                run_path=run_path, snap=snap, h=cfg["cosmology"]["h"],
                suffix=attempt, halo_ids=cat["halo_ids"], redshift=meta["redshift"],
            )
            chosen_suffix = attempt
            break
        except FileNotFoundError as e:
            last_err = e
    if profs is None:
        print(f"[compute_profiles] MISSING: no profile catalog for "
              f"{model} snap {snap} (tried suffixes={suffix_attempts}): {last_err}")
        return 0

    halo_data = {
        int(hid): {
            "r_edges":    np.asarray(p["r_edges"]),
            "r_outer":    np.asarray(p["r_outer"]),
            "prof_dm":    np.asarray(p["prof_dm"]) if p.get("prof_dm") is not None else np.zeros(0),
            "prof_gas":   np.asarray(p["prof_gas"]) if p.get("prof_gas") is not None else np.zeros(0),
            "prof_stars": np.asarray(p["prof_stars"]) if p.get("prof_stars") is not None else np.zeros(0),
        }
        for hid, p in profs.items()
    }

    # Bake the actual suffix used into the output filename so a re-run with
    # a different fallback target produces a different file rather than
    # silently overwriting. Disc-filtered catalogs also tag their downstream
    # outputs so all-galaxy and disc-only outputs coexist.
    suffix_tag = chosen_suffix if chosen_suffix else ""
    disc_tag = "_disc" if cuts.get("disc_only", False) else ""
    sample_key = (f"mstar{cuts.get('mstar_min', '?'):.0e}"
                  f"_ndm{int(cuts.get('n_dm_min', 0))}"
                  f"{suffix_tag}"
                  f"{disc_tag}")
    out_root = Path(args.out_root or cfg["paths"]["scratch_processed"])
    out_path = out_root / "profiles" / f"profiles_{model}_{snap:03d}_{sample_key}.hdf5"

    save_per_halo(
        out_path, halo_data,
        cuts=cuts,
        metadata={**meta, "source_catalog": str(args.catalog),
                  "suffix": chosen_suffix},
    )
    print(f"[compute_profiles] {model} snap {snap}: "
          f"{len(halo_data)} halos -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
