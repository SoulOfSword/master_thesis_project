"""Per-galaxy diagnostic images from a MORDOR-decomposed pynbody snapshot.

Three PNGs, meant to be written *during* a MORDOR run (run_mordor_all.py), so
they reuse the decomposition already computed and cost almost nothing extra:

  1. energy histogram  (energy_dists/)  -- stellar binding-energy distribution,
        normalised to [-1, 0] and binned at MORDOR's FINEST resolution (the
        single finest version of the histogram MORDOR refines while hunting Ecut).
  2. circularity       (circularities/) -- the jz/jcirc distribution, stacked by
        morphological component (thin/thick/pbulge/bulge/halo).
  3. component map      (particle_maps/) -- MORDOR's `--ShowPlots` figure: the
        R-vs-potential profile (left) and the face-on X-Y map colour-coded by
        component (right), using the same colours MORDOR uses.

All three take the decomposed `gal` (needs gal.s['te'], gal.s['morph'],
gal.s['jz_by_jzcirc'], gal.s positions, gal['rxy'], gal['phi']) and the
`profiles` object that `decomposition.morph()` returns (for the left panel of
image 3; if None, that panel just shows the particle cloud).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)           # headless; override decomposition's WebAgg
import matplotlib.pyplot as plt
import numpy as np

# MORDOR component integer -> (label, colour), matching mordor.py --ShowPlots
_COMP = [(1, "thin", "blue"), (2, "thick", "green"), (3, "pbulge", "gold"),
         (4, "bulge", "red"), (5, "halo", "orange")]


def _finest_nbins(n):
    """MORDOR's maximum histogram bin count: max(min(int(0.5*sqrt(N)),400),80)."""
    return int(max(min(int(0.5 * np.sqrt(max(n, 1))), 400), 80))


def _save(fig, out_png):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def save_energy_histogram(gal, out_png, title=""):
    """Image 1: normalised binding-energy histogram at MORDOR's finest binning.

    te is offset by the least-bound star and scaled by the most-bound -> [-1, 0]
    (the same affine MORDOR uses internally), then binned with `_finest_nbins`.
    """
    te = np.asarray(gal.s["te"], dtype=np.float64)
    te = te - te.max()
    scale = np.abs(te).max()
    if scale > 0:
        te = te / scale
    nb = _finest_nbins(len(te))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(te, bins=nb, range=(float(te.min()), float(te.max())),
            color="steelblue", alpha=0.85)
    ax.set_xlabel(r"normalised binding energy $\tilde{e}$ "
                  r"($-1$ = most bound, $0$ = least)")
    ax.set_ylabel("star count")
    ax.set_title(f"{title}   energy distribution (nbins={nb})")
    _save(fig, out_png)


def save_circularity(gal, out_png, title=""):
    """Image 2: circularity eta = jz/jcirc, stacked by component.

    eta is clipped to [-1.5, 1.5] (MORDOR's bound range) so a few particles with
    jcirc -> 0 don't blow up the axis. The dashed line marks the thin-disc
    threshold eta = 0.7.
    """
    eta = np.asarray(gal.s["jz_by_jzcirc"], dtype=np.float64)
    morph = np.asarray(gal.s["morph"])
    bins = np.linspace(-1.5, 1.5, 121)
    data, colors, labels = [], [], []
    for c, nm, col in _COMP:
        sel = (morph == c) & np.isfinite(eta)
        data.append(np.clip(eta[sel], -1.5, 1.5))
        colors.append(col)
        labels.append(nm)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(data, bins=bins, stacked=True, color=colors, label=labels)
    ax.axvline(0.7, ls="--", c="k", lw=0.8)
    ax.set_xlabel(r"circularity $\eta = j_z / j_{\rm circ}$")
    ax.set_ylabel("star count")
    ax.set_title(f"{title}   circularity")
    ax.legend(fontsize=8)
    _save(fig, out_png)


def save_component_map(gal, profiles, out_png, title=""):
    """Image 3: MORDOR --ShowPlots figure (R-potential profile + face-on map)."""
    morph = np.asarray(gal.s["morph"])
    x = np.asarray(gal.s["x"], dtype=np.float64)
    y = np.asarray(gal.s["y"], dtype=np.float64)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    # left: cylindrical radius vs potential (particles + disc profile)
    axs[0].plot(np.asarray(gal["rxy"], dtype=np.float64),
                np.asarray(gal["phi"], dtype=np.float64),
                ",", c="orange", alpha=0.4)
    try:
        axs[0].plot(profiles["rbins"], profiles["phi"], c="blue", label="profile")
        axs[0].plot(profiles["rbins"], profiles["pot"], c="red", label="midplane")
        axs[0].legend(fontsize=8)
    except Exception:
        pass
    axs[0].set_xlabel("R [kpc]")
    axs[0].set_ylabel(r"$\Phi$ [km$^2$ s$^{-2}$]")

    # right: face-on X-Y map, coloured by component
    axs[1].set_aspect("equal")
    for c, nm, col in _COMP:
        sel = morph == c
        if sel.any():
            axs[1].plot(x[sel], y[sel], ",", c=col, alpha=0.3, label=nm)
    axs[1].set_xlabel("X [kpc]")
    axs[1].set_ylabel("Y [kpc]")
    axs[1].set_title("face-on, by component")
    axs[1].legend(fontsize=8, markerscale=20, loc="upper right")
    fig.suptitle(title)
    _save(fig, out_png)


def save_all(gal, profiles, edist_png, circ_png, map_png, title=""):
    """Write all three diagnostic PNGs for one decomposed galaxy.

    Each image is guarded independently so a failure in one (e.g. an empty
    component) cannot lose the others or break the MORDOR run.
    """
    for fn, out in ((lambda: save_energy_histogram(gal, edist_png, title), edist_png),
                    (lambda: save_circularity(gal, circ_png, title), circ_png),
                    (lambda: save_component_map(gal, profiles, map_png, title), map_png)):
        try:
            fn()
        except Exception as exc:
            print(f"[diagnostics] {Path(out).name} failed: {exc!r}", flush=True)
