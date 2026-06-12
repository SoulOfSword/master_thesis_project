"""Interacting-bias diagnostic: D/T vs M_star, full vs (non-)interacting.

For one (model, snap), splits the
mordor_sample HDF5 into FULL / NON-INTERACTING / INTERACTING subsets
using `sat_mass_ratio >= --m-ratio-min`, then plots binned D/T
(IsDisc fraction) with Wilson 95% CI errorbars.

Bins are shared across the three subsets within the panel (pooled
log M_star across subsets defines the common grid); errorbars are
clipped to non-negative.

Output: <fig_root>/morphology/<model>_snap<snap>_disc_fraction_interacting_bias.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat
from galaxy_sidm.morphology import disc_fraction_binned


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", default="CDM",
                   choices=["CDM", "SIDM1", "vSIDM", "WDM3", "WDM5"])
    p.add_argument("--snap", type=int, default=67)
    p.add_argument("--m-ratio-min", type=float, default=0.2,
                   help="Threshold on sat_mass_ratio for 'interacting'")
    p.add_argument("--bin-step", type=float, default=0.2)
    p.add_argument("--n-min", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    scratch_mordor = Path(cfg["paths"]["scratch_mordor"])
    fig_root = Path(cfg["paths"]["fig_root"])

    path = (scratch_mordor / "samples"
            / f"mordor_sample_{args.model}_{args.snap:03d}.hdf5")
    if not path.exists():
        sys.exit(f"missing sample: {path}")
    arrays, _ = load_flat(path)

    mstar = arrays["Mstar"]
    good = mstar > 0
    log_m = np.log10(mstar[good])
    is_disc = arrays["IsDisc"][good]
    sat_ratio = arrays["sat_mass_ratio"][good]

    interacting = sat_ratio >= args.m_ratio_min
    subsets = {
        "full":
            (np.ones_like(interacting, dtype=bool), "black", "o"),
        "non-interacting":
            (~interacting, "tab:blue", "s"),
        f"interacting (>={args.m_ratio_min:g})":
            (interacting, "tab:red", "^"),
    }

    # Shared bin grid pooled over all log_m present
    bins = np.arange(log_m.min(),
                      log_m.max() + args.bin_step, args.bin_step)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (mask, color, marker) in subsets.items():
        n_sub = int(mask.sum())
        if n_sub == 0:
            continue
        centres, fracs, lo, hi = disc_fraction_binned(
            log_m[mask], is_disc[mask], bins, n_min=args.n_min)
        if len(centres) == 0:
            continue
        ax.plot(centres, fracs, "-" + marker, color=color,
                label=f"{label} (N={n_sub})",
                ms=10, markeredgecolor="grey", markeredgewidth=2)
        ax.errorbar(centres, fracs, yerr=np.array([lo, hi]),
                     fmt="none", color=color, alpha=0.4,
                     capsize=3, capthick=1.5, zorder=0)

    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel(r"$\log_{10}(M_\star\;[M_\odot])$", fontsize=16)
    ax.set_ylabel("disc fraction", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12)
    plt.tight_layout()

    out_path = args.out or (
        fig_root / "morphology"
        / f"{args.model.lower()}_snap{args.snap:03d}"
          f"_disc_fraction_interacting_bias.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot_interacting_diagnostic] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
