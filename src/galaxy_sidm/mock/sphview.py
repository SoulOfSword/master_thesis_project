"""Face-on / edge-on surface-density maps with sphviewer2.

2x2 grid: top face-on, bottom edge-on (a thin horizontal strip); left
neutral gas, right stars. Disc plane comes from the gas angular momentum
(L_hat); stars get a neighbour-based smoothing length from estimate_h.
Surface density is shown as log10(Sigma) in Msun/pkpc^2.
"""

from pathlib import Path

import numpy as np
import astropy.units as U
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import sphviewer2

from .gas import GalaxyGas

# Computer Modern (the paper's LaTeX font) without needing a TeX install
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})


def _safe_log10(s):
    out = np.full(s.shape, np.nan)
    pos = s > 0
    out[pos] = np.log10(s[pos])
    return out


def _disc_frame_rotation(L_hat):
    """Rotation matrix (world->disc) mapping L_hat onto +z."""
    L_hat = np.asarray(L_hat, float)
    L_hat = L_hat / (np.linalg.norm(L_hat) + 1e-30)
    a = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(L_hat, a)) > 0.99:
        a = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, L_hat); e1 /= np.linalg.norm(e1) + 1e-30
    e2 = np.cross(L_hat, e1)
    return np.vstack([e1, e2, L_hat])


def _sigma_map(x, y, z, h, m, half, npix, num_threads):
    """Surface density [Msun/pkpc^2] over +/-half (pkpc). render's 1st
    coord arg maps to the horizontal axis, so no transpose is applied."""
    r_max = int(round(np.log2(max(npix, 32))))
    out = sphviewer2.render(x, y, z, h, m, 4.0 * half, extent=half,
                            xc=0.0, yc=0.0, zc=0.0, periodic=False,
                            r_max=r_max, target_cells_per_h=4,
                            num_threads=num_threads)
    img = np.asarray(out[0] if isinstance(out, (tuple, list)) else out)
    pix_area = (2.0 * half / img.shape[0]) ** 2
    return img / pix_area


def render_face_edge(gas: GalaxyGas, out_path, half_kpc=None, npix=512,
                     num_threads=4, star_neighbours=32, label=None):
    R = _disc_frame_rotation(gas.L_hat)
    xg = gas.xyz_g.to_value(U.kpc) @ R.T
    mHI = gas.mHI_g.to_value(U.Msun)
    hg = gas.hsm_g.to_value(U.kpc)
    xs = gas.xyz_s.to_value(U.kpc) @ R.T
    ms = gas.m_s.to_value(U.Msun)

    if half_kpc is None:
        if len(xs):
            rr = np.linalg.norm(xs[:, :2], axis=1); ww = ms
        else:
            rr = np.linalg.norm(xg[:, :2], axis=1); ww = mHI
        order = np.argsort(rr); cum = np.cumsum(ww[order])
        r50 = rr[order][np.searchsorted(cum, 0.5 * cum[-1])] if len(rr) else 5.0 # r50 is the half-mass radius in the disc plane; use it to set the map's spatial scale. We clip it to 5-15 pkpc, which is a reasonable range for our sample and prevents extreme outliers from dominating the figure.
        half_kpc = float(np.clip(3.5 * r50, 5.0, 15.0)) # 5-15 pkpc, typically ~10 pkpc for our sample
    edge_frac = 0.45 # edge-on map is a thin horizontal strip; this is its height / width ratio
    Lbox = 4.0 * half_kpc
    if len(xs):
        hs = sphviewer2.estimate_h(xs[:, 0], xs[:, 1], xs[:, 2], Lbox,
                                   k=star_neighbours, num_threads=num_threads)
    else:
        hs = np.zeros(0)

    # face-on projects (x, y); edge-on projects (x, z) -> disc lies horizontal
    panels = {
        ("face", "gas"): (xg[:, 0], xg[:, 1], xg[:, 2], hg, mHI),
        ("edge", "gas"): (xg[:, 0], xg[:, 2], xg[:, 1], hg, mHI),
        ("face", "star"): (xs[:, 0], xs[:, 1], xs[:, 2], hs, ms),
        ("edge", "star"): (xs[:, 0], xs[:, 2], xs[:, 1], hs, ms),
    }
    cols = [("gas", "magma"), ("star", "bone")]

    fig = plt.figure(figsize=(8.2, 7.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.05, 1.0, edge_frac],
                          hspace=0.03, wspace=0.13,
                          left=0.09, right=0.97, top=0.95, bottom=0.07)

    for j, (key, cmap) in enumerate(cols):
        sig = {r: _sigma_map(*panels[(r, key)][:3], panels[(r, key)][3],
                             panels[(r, key)][4], half_kpc, npix, num_threads)
               for r in ("face", "edge")}
        logs = {r: _safe_log10(s) for r, s in sig.items()}
        vmax = np.nanpercentile(logs["face"], 99.8)
        vmin = vmax - 3.5
        ext = [-half_kpc, half_kpc, -half_kpc, half_kpc]

        cax = fig.add_subplot(gs[0, j])
        axf = fig.add_subplot(gs[1, j])
        axe = fig.add_subplot(gs[2, j])

        im = axf.imshow(logs["face"], origin="lower", extent=ext, cmap=cmap,
                        vmin=vmin, vmax=vmax, aspect="auto")
        axf.set_xlim(-half_kpc, half_kpc); axf.set_ylim(-half_kpc, half_kpc)
        axe.imshow(logs["edge"], origin="lower", extent=ext, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="auto")
        axe.set_xlim(-half_kpc, half_kpc)
        axe.set_ylim(-half_kpc * edge_frac, half_kpc * edge_frac)

        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
        cb.set_label(rf"$\log_{{10}}\,\Sigma_{{\rm {key}}}$ [M$_\odot$ pkpc$^{{-2}}$]",
                     fontsize=12)
        cb.ax.tick_params(labelsize=10)

        for ax in (axf, axe):
            ax.tick_params(labelsize=11, direction="in", color="white",
                           top=True, right=True)
        axe.set_xlabel("x [pkpc]", fontsize=13)
        if j == 0:
            axf.set_ylabel("face-on\ny [pkpc]", fontsize=13)
            axe.set_ylabel("edge-on\nz [pkpc]", fontsize=13)
            txt = axf.text(0.04, 0.96, label or
                           f"subID {gas.subhalo_id}\nz={gas.redshift:.2f}",
                           transform=axf.transAxes, va="top", ha="left",
                           color="white", fontsize=16, linespacing=1.4)
            txt.set_path_effects([pe.withStroke(linewidth=1.4,
                                                foreground="black")])
        else:
            axf.set_yticklabels([]); axe.set_yticklabels([])
        axf.set_xticklabels([])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
