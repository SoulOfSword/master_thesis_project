"""Single-galaxy MORDOR (eta, E) phase-space diagnostic.

  Left  : (eta, E/|E|_max) hexbin density, log-coloured, with the
          eta=0.7 disc threshold and the data-driven Ecut as dashed lines.
  Right : (eta, E/|E|_max) scatter coloured by MORDOR component
          (thin/thick disc, pseudo-bulge, classical bulge, halo, plus the
          unbound `morph==0` particles for context).

Workflow:
  1. If `<scratch_mordor>/<MODEL>/Gal_<subhalo_id:06d>.hdf5` is missing,
     extract it on-the-fly via `extract_galaxy_hdf5`.
  2. Decompose in-process via `run_mordor_single` (cosmo_sim mode).
  3. Plot.

Usage:
    python plot_eta_E_diagnostic.py --model CDM --snap 67 \\
        --subhalo-id 0
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config
from galaxy_sidm.morphology import extract_galaxy_hdf5, run_mordor_single


COMP_NAMES = {
    0: "Excluded particles",
    1: "Thin disc",
    2: "Thick disc",
    3: "Pseudo-bulge",
    4: "Bulge",
    5: "Halo",
}
COMP_COLORS = {
    0: "black",
    1: "tab:blue",
    2: "tab:green",
    3: "tab:cyan",
    4: "tab:red",
    5: "gold",
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", required=True,
                   choices=["CDM", "SIDM1", "vSIDM", "WDM3", "WDM5"])
    p.add_argument("--snap", required=True, type=int)
    p.add_argument("--subhalo-id", required=True, type=int)
    p.add_argument("--res", type=int, default=1080)
    p.add_argument("--gridsize", type=int, default=80,
                   help="Hexbin gridsize (default 80)")
    p.add_argument("--scatter-size", type=float, default=2.0,
                   help="Per-particle marker size on the scatter panel")
    p.add_argument("--scatter-max", type=int, default=50000,
                   help="Random subsample size for the scatter panel "
                        "(default 50000)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the scatter subsample (default 0)")
    p.add_argument("--base-path", type=Path, default=None,
                   help="Override snapshot basePath (e.g. SCRATCH shadow "
                        "tree when snapshot data is not in $WORK)")
    p.add_argument("--out-fig", type=Path, default=None,
                   help="Output figure path; default "
                        "<fig_root>/morphology/eta_E_<model>_<snap>_<sub>.pdf")
    p.add_argument("--overwrite-extract", action="store_true",
                   help="Re-extract the per-galaxy HDF5 even if it exists")
    args = p.parse_args()

    cfg = load_config(args.config)
    fig_root = Path(cfg["paths"]["fig_root"])
    scratch_mordor = Path(cfg["paths"]["scratch_mordor"])
    h = float(cfg["cosmology"]["h"])
    soft_phys_kpc = float(cfg["softening"]["phys_kpc"])

    out_fig = args.out_fig or (
        fig_root / "morphology"
        / f"eta_E_{args.model}_snap{args.snap:03d}"
          f"_sub{args.subhalo_id:06d}.pdf")
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    hdf5_path = (scratch_mordor / args.model / f"snap_{args.snap:03d}"
                 / f"Gal_{args.subhalo_id:06d}.hdf5")

    # ---- extract if missing ------------------------------------------
    Mstar_msun = None
    redshift = None
    if not hdf5_path.exists() or args.overwrite_extract:
        import temet
        from galaxy_sidm.data.aida_tng import build_central_subhalo_catalog
        sim = temet.sim(run="aida", variant=args.model,
                        res=args.res, snap=args.snap)
        cat = build_central_subhalo_catalog(sim)
        if args.base_path is not None:
            cat["basePath"] = str(args.base_path).rstrip("/") + "/"
            print(f"[plot_eta_E_diagnostic] basePath overridden -> "
                  f"{cat['basePath']}")
        Mstar_msun = float(cat["Mstar"][args.subhalo_id])
        redshift = float(cat["redshift"])
        print(f"[plot_eta_E_diagnostic] extracting -> {hdf5_path}")
        extract_galaxy_hdf5(
            base_path=cat["basePath"], snap=int(cat["snap"]),
            subhalo_id=int(args.subhalo_id), out_path=hdf5_path,
            h=h, soft_phys_kpc=soft_phys_kpc,
            overwrite=args.overwrite_extract,
        )
    else:
        print(f"[plot_eta_E_diagnostic] using existing {hdf5_path}")

    # ---- decompose ---------------------------------------------------
    print(f"[plot_eta_E_diagnostic] running MORDOR (cosmo_sim) ...")
    gal = run_mordor_single(hdf5_path, mode="cosmo_sim",
                            soft_phys_kpc=soft_phys_kpc)

    morph = np.asarray(gal.s["morph"])
    eta   = np.asarray(gal.s["jz_by_jzcirc"])
    te    = np.asarray(gal.s["te"])
    mass  = np.asarray(gal.s["mass"])

    classified = morph != 0
    Emax = float(np.abs(te[classified]).max()) if classified.any() else 1.0
    te_n = te / Emax
    low_E = (morph == 3) | (morph == 4)
    Ecut = float(np.max(te_n[low_E])) if low_E.any() else 0.0

    print("\nComponent masses [Msun]:")
    for k, label in COMP_NAMES.items():
        m_k = float(mass[morph == k].sum())
        print(f"  {label:18s}  M={m_k:.3e}  N={(morph == k).sum()}")
    mthin   = float(mass[morph == 1].sum())
    mthick  = float(mass[morph == 2].sum())
    mpbulge = float(mass[morph == 3].sum())
    mbulge  = float(mass[morph == 4].sum())
    mhalo   = float(mass[morph == 5].sum())
    mbound  = mthin + mthick + mpbulge + mbulge + mhalo
    disc_frac = (mthin + mthick + mpbulge) / mbound if mbound > 0 else 0.0
    print(f"\nIsDisc (Zana) = {disc_frac > 0.5}  (frac={disc_frac:.3f})")
    print(f"Ecut = {Ecut:.3f}")

    # If we used a cached HDF5, get Mstar/z from the snapshot for the title
    if Mstar_msun is None:
        Mstar_msun = float(mass.sum())
        try:
            redshift = float(gal.properties.get("z", np.nan))
        except Exception:
            redshift = float("nan")

    # ---- subsample for scatter panel ---------------------------------
    if len(eta) > args.scatter_max:
        rng = np.random.default_rng(args.seed)
        sub = rng.choice(len(eta), args.scatter_max, replace=False)
    else:
        sub = np.arange(len(eta))

    # ---- plot --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             sharex=True, sharey=True,
                             constrained_layout=True)

    ax = axes[0]
    ax.set_facecolor("black")
    hb = ax.hexbin(eta[classified], te_n[classified],
                   gridsize=args.gridsize, mincnt=1,
                   cmap="gist_stern", norm=LogNorm())
    ax.axvline(0.7, color="white", ls="--", lw=1.2)
    ax.axhline(Ecut, color="white", ls="--", lw=1.2)
    ax.set_xlabel(r"$\eta = j_z / j_{\rm circ}(E)$", fontsize=16)
    ax.set_ylabel(r"$E\,/\,|E|_{\rm max}$", fontsize=16)
    ax.tick_params(labelsize=13)
    fig.colorbar(hb, ax=ax, label="N stars per cell")

    ax = axes[1]
    for k in [0, 5, 4, 3, 2, 1]:
        sel = (morph[sub] == k)
        if not sel.any():
            continue
        ax.scatter(eta[sub][sel], te_n[sub][sel],
                   s=args.scatter_size, color=COMP_COLORS[k],
                   label=COMP_NAMES[k])
    ax.axvline(0.7, color="black", ls="--", lw=1.0)
    ax.axhline(Ecut, color="black", ls="--", lw=1.0)
    ax.set_xlabel(r"$\eta = j_z / j_{\rm circ}(E)$", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(loc="upper left", markerscale=6, framealpha=0.85, fontsize=12)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.05, 0.05)
    z_str = f"{redshift:.2f}" if redshift is not None and np.isfinite(redshift) else "?"
    fig.suptitle(
        f"{args.model} subhalo {args.subhalo_id} - "
        rf"$M_\star$={Mstar_msun:.2e} $M_\odot$, $z$={z_str}",
        fontsize=16
    )
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\n[plot_eta_E_diagnostic] saved {out_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
