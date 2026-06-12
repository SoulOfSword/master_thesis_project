"""Kinematics figure from BBarolo's output (4 panels).

moment 0 (intensity, jet) with contours, scale bar and centre cross;
moment 1 (V_LOS, jet) with iso-velocity contours, major/minor axis lines,
centre cross and beam; and the major/minor PV slices with the data and
best-fit model contours, the projected rotation curve, and zero lines.
All in physical units, masked and cropped to the galaxy.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import BoundaryNorm
from scipy import ndimage
from astropy.io import fits

# Computer Modern (the paper's LaTeX font) without needing a TeX install
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

_ARCSEC_PER_RAD = 206264.806


def _data(p):
    return np.squeeze(fits.getdata(str(p))).astype(float) # squeeze to drop degenerate axes from moment maps


def _find(folder, *pats, model=False):
    for pat in pats:
        for p in sorted(Path(folder).rglob(pat)):
            if "mask" in p.stem:
                continue
            if ("mod" in p.stem) == model:
                return p
    return None


def _kpp(cdelt_deg, dist_mpc):
    # Convert pixel scale to kpc/pixel: CDELT [deg] * 3600 [arcsec/deg] / 206265 [arcsec/rad] * distance [Mpc] * 1e3 [kpc/Mpc]
    return abs(cdelt_deg) * 3600.0 / _ARCSEC_PER_RAD * dist_mpc * 1e3


def _galaxy_mask(mom0, frac=0.03):
    """Largest connected blob above frac*peak (drops isolated noise specks)."""
    m = np.nan_to_num(mom0) > frac * np.nanmax(mom0)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    biggest = 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
    return lab == biggest


def _rings(bb):
    """(rad_kpc, vrot, vdisp) from rings_final1.txt (cols 0, 2, 3)."""
    f = bb / "rings_final1.txt"
    if not f.exists():
        return (np.zeros(0),) * 3
    rows = [r.split() for r in f.read_text().splitlines()
            if r.strip() and not r.startswith("#")]
    a = np.array([[float(c) for c in r[:4]] for r in rows if len(r) >= 4])
    if a.size == 0:
        return (np.zeros(0),) * 3
    return a[:, 0], a[:, 2], a[:, 3]


def _v_over_sigma(vrot, vdisp):
    if not len(vrot):
        return float("nan")
    vflat = (vrot[1:][np.argmin(np.abs(np.diff(vrot)))]
             if len(vrot) > 1 else vrot[-1])
    V = 0.5 * (float(np.max(vrot)) + float(vflat))
    sig = float(np.mean(vdisp))
    return V / sig if sig > 0 else float("nan")


def _emission_radius(bb, distance_mpc=5.0):
    """HI emission radius [kpc] from the moment-0 map (3%-of-peak mask)."""
    f = _find(bb, "*_0mom.fits")
    if f is None:
        return float("inf")
    mom0 = _data(f); hdr = fits.getheader(str(f))
    kpp = _kpp(hdr["CDELT1"], distance_mpc)
    ys, xs = np.where(_galaxy_mask(mom0))
    if not len(xs):
        return float("inf")
    ny, nx = mom0.shape
    return float(np.sqrt((xs - nx / 2) ** 2 + (ys - ny / 2) ** 2).max()) * kpp


def _rings_trimmed(bb, distance_mpc=5.0):
    """Rings cut to the HI emission radius. BBarolo fits data-free outer
    rings that otherwise inflate V (Vflat) and deflate sigma."""
    bb = Path(bb)
    rad, vrot, vdisp = _rings(bb)
    if len(rad):
        keep = rad <= _emission_radius(bb, distance_mpc)
        rad, vrot, vdisp = rad[keep], vrot[keep], vdisp[keep]
    return rad, vrot, vdisp


def ring_kinematics(bbarolo_dir, distance_mpc=5.0):
    """(V, sigma, V/sigma) from rings trimmed to the HI emission."""
    _, vrot, vdisp = _rings_trimmed(bbarolo_dir, distance_mpc)
    if not len(vrot):
        return float("nan"), float("nan"), float("nan")
    V = 0.5 * (float(np.max(vrot)) + _vflat(vrot))
    sig = float(np.mean(vdisp))
    return V, sig, (V / sig if sig > 0 else float("nan"))


def _vflat(vrot):
    return (float(vrot[1:][np.argmin(np.abs(np.diff(vrot)))])
            if len(vrot) > 1 else float(vrot[-1]) if len(vrot) else float("nan"))


def ring_v_over_sigma(bbarolo_dir, distance_mpc=5.0):
    """V/sigma from a BBarolo run, trimmed to the HI emission."""
    return ring_kinematics(bbarolo_dir, distance_mpc)[2]


def _cbar(fig, ax, im, label, ticks=None):
    cax = ax.inset_axes([0.0, -0.11, 1.0, 0.05])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cb.set_label(label, fontsize=14); cb.ax.tick_params(labelsize=13)


def _pv_vlim(pv_p, frac=0.02, margin=1.15):
    """Velocity half-range from where the PV actually has signal (no cap)."""
    pv = _data(pv_p); h = fits.getheader(str(pv_p))
    dv = (np.arange(pv.shape[0]) + 1 - h["CRPIX2"]) * h["CDELT2"] / 1e3
    vprof = np.nansum(np.clip(pv, 0.0, None), axis=1)
    if vprof.max() > 0:
        return margin * float(np.abs(dv[vprof > frac * vprof.max()]).max())
    return margin * float(np.abs(dv).max())

def _pv_olim(pv_p, kpp, frac=0.02, margin=1.15):
    """Offset half-range from where the PV actually has signal (no cap)."""
    pv = _data(pv_p); h = fits.getheader(str(pv_p))
    off = (np.arange(pv.shape[1]) - (pv.shape[1] - 1) / 2.0) * kpp
    oprof = np.nansum(np.clip(pv, 0.0, None), axis=0)
    if oprof.max() > 0:
        return margin * float(np.abs(off[oprof > frac * oprof.max()]).max())
    return margin * float(np.abs(off).max())

def _pv_panel(ax, pv_p, mod_p, kpp, zoom, name, rc=None, vlim=None):
    """Plot a PV slice with data in blue, model contours in red, and the rotation curve as yellow points.
    Explaination of all steps in the code:
1. Load the PV data from the FITS file using the `_data` function, which reads the data and squeezes it to remove any degenerate axes. Also, get the header of the FITS file to extract necessary information for plotting.
2. Determine the number of velocity channels (`nv`) and the number of spatial offsets (noff) from the shape of the PV data.
3. Create an array of spatial offsets (`off`) centered on the middle of the array, using the kpc/pixel scale (`kpp`) to convert from pixels to physical units.
4. Create an array of velocity channels (`dv`) based on the header information, converting from the pixel scale to km/s.
5. If the velocity channels are in descending order, reverse both the PV data and the velocity array to ensure they are in ascending order.
6. If a model PV file is provided, load the model data as well.
7. Plot the PV data using `imshow`, setting the origin to "lower", aspect ratio to "auto", and using a blue colormap. The extent of the plot is set based on the spatial offsets and velocity channels. The color limits are set to the 99th percentile of the finite values in the PV data to enhance contrast.
8. Add contour lines to the PV plot at the 85th, 95th, and 99th percentiles of the finite values in the PV data, using black contours.
9. If model data is available and has finite values, add red contour lines for the model at the 85th, 95th, and 99th percentiles of the finite values in the model data.
10. If rotation curve data (`rc`) is provided and the panel is for the major axis, plot the rotation curve points as yellow circles with black edges. The points are plotted symmetrically on both sides of the center, and their orientation is determined by the sign of the intensity-weighted offset-velocity covariance to ensure they align with the velocity gradient of the data.
11. Add horizontal and vertical lines at zero velocity and zero offset to indicate the center of the plot.
12. Set the limits of the plot based on the zoom level and the maximum velocity range.
13. Label the axes with appropriate units and set the tick parameters for better visibility.
14. Add a text label to indicate whether the panel is for the major or minor axis, using a bold font for emphasis.
    
    """
    pv = _data(pv_p); h = fits.getheader(str(pv_p))
    nv, noff = pv.shape
    # BBarolo can write a garbage offset WCS on the minor-axis PV, so centre
    # the offset on the array middle and use the map's kpc/pixel scale instead
    off = (np.arange(noff) - (noff - 1) / 2.0) * kpp
    dv = (np.arange(nv) + 1 - h["CRPIX2"]) * h["CDELT2"] / 1e3
    mod = _data(mod_p) if mod_p else None
    if dv[0] > dv[-1]:
        pv, dv = pv[::-1], dv[::-1]
        if mod is not None:
            mod = mod[::-1]
    fin = pv[np.isfinite(pv)]
    ax.imshow(pv, origin="lower", aspect="auto",
              extent=[off[0], off[-1], dv[0], dv[-1]], cmap="Blues",
              vmin=0.0, vmax=np.nanpercentile(fin, 99))
    ax.contour(off, dv, pv, levels=np.nanpercentile(fin, [85, 95, 99]),
               colors="k", linewidths=0.5)
    if mod is not None and np.isfinite(mod).any():
        ml = np.nanpercentile(mod[np.isfinite(mod)], [85, 95, 99])
        ax.contour(off, dv, mod, levels=ml, colors="red", linewidths=1.0)
    if rc is not None and name == "major" and len(rc[0]):
        r, v = rc
        # orient the points to the data's velocity gradient (sign of the
        # intensity-weighted offset-velocity covariance)
        OFF, DV = np.meshgrid(off, dv)
        w = np.clip(np.nan_to_num(pv), 0.0, None)
        sgn = 1.0 if np.sum(w * OFF * DV) >= 0 else -1.0
        ax.scatter(np.r_[r, -r], sgn * np.r_[v, -v], s=26, c="yellow",
                   edgecolors="k", linewidths=0.7, zorder=5)
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    if vlim is None:
        vlim = _pv_vlim(pv_p)
    ax.set_xlim(-zoom, zoom); ax.set_ylim(-vlim, vlim)
    ax.set_xlabel("Offset [pkpc]", fontsize=13)
    ax.set_ylabel(r"$\Delta V_{\rm LOS}$ [km s$^{-1}$]", fontsize=13)
    ax.tick_params(labelsize=11, direction="in", top=True, right=True)
    ax.text(0.95, 0.94, rf"$\phi={90 if name=='major' else 180}^\circ$",
            transform=ax.transAxes, ha="right", va="top", fontsize=13,
            fontweight="bold")


def plot_kinematics(bbarolo_dir, out_path, distance_mpc=5.0, inc_deg=60.0,
                    suptitle=None):
    bb = Path(bbarolo_dir)
    mom0 = _data(_find(bb, "*_0mom.fits"))
    mom1 = _data(_find(bb, "*_1mom.fits"))
    hdr = fits.getheader(str(_find(bb, "*_0mom.fits")))
    kpp = _kpp(hdr["CDELT1"], distance_mpc)
    beam = _kpp(hdr.get("BMAJ", 0.008333), distance_mpc)

    ny, nx = mom0.shape
    ext = [-nx / 2 * kpp, nx / 2 * kpp, -ny / 2 * kpp, ny / 2 * kpp]
    mask = _galaxy_mask(mom0)
    ys, xs = np.where(mask)
    rr = np.sqrt((xs - nx / 2) ** 2 + (ys - ny / 2) ** 2).max() if len(xs) else 40
    zoom = float(rr * 1.3 * kpp)

    # BBarolo fits rings out past the HI emission; those data-free outer rings
    # spuriously inflate V (Vflat) and deflate sigma. Cut at the emission edge.
    r_emit = rr * kpp
    rad, vrot, vdisp = _rings(bb)
    if len(rad):
        keep = rad <= r_emit
        rad, vrot, vdisp = rad[keep], vrot[keep], vdisp[keep]
    rc = (rad, vrot * np.sin(np.deg2rad(inc_deg)))
    vsig = _v_over_sigma(vrot, vdisp)

    m0 = np.where(mask, mom0, np.nan)
    vsys = np.nanmedian(mom1[mask]) if mask.any() else 0.0
    m1 = np.where(mask, mom1 - vsys, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2),
                             gridspec_kw={"wspace": 0.3})

    ax = axes[0]
    im = ax.imshow(m0, origin="lower", extent=ext, cmap="Spectral_r")
    ax.contour(m0, levels=np.nanpercentile(m0[mask], [50, 75, 90, 97]),
               extent=ext, colors="k", linewidths=0.5)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.plot(0, 0, "x", color="k", ms=8, mew=2)
    ax.plot([-0.8 * zoom, -0.8 * zoom + 1.0], [-0.85 * zoom, -0.85 * zoom],
            color="k", lw=3)
    ax.text(-0.8 * zoom + 0.5, -0.78 * zoom, "1 pkpc", ha="center",
            fontsize=11, fontweight="bold")
    ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_title("moment 0", fontsize=16)
    _cbar(fig, ax, im, r"$I$ [Jy beam$^{-1}$ km s$^{-1}$]")

    ax = axes[1]
    vlim = np.ceil(np.nanpercentile(np.abs(m1), 97) / 50.0) * 50.0
    levels = np.linspace(-vlim, vlim, 26)
    cmap1 = plt.get_cmap("seismic", 25)
    im = ax.imshow(m1, origin="lower", extent=ext, cmap=cmap1,
                   norm=BoundaryNorm(levels, cmap1.N))
    ax.contour(m1, levels=[0.0], extent=ext, colors="green", linewidths=2.0)
    ax.contour(m1, levels=np.nanpercentile(np.abs(m1[mask]), [25, 50, 75]),
               extent=ext, colors="grey", linewidths=1.0)
    ax.contour(m1, levels=-np.nanpercentile(np.abs(m1[mask]), [25, 50, 75])[::-1],
               extent=ext, colors="grey", linewidths=1.0)
    ax.axhline(0, color="k", lw=0.9); ax.axvline(0, color="k", lw=0.9)
    ax.plot(0, 0, "x", color="k", ms=8, mew=2)
    ax.add_patch(Ellipse((-0.8 * zoom, -0.8 * zoom), beam, beam,
                         facecolor="0.4", edgecolor="k"))
    ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_title("moment 1", fontsize=16)
    vt = np.arange(0.0, vlim + 1, max(50.0, round(vlim / 3 / 50) * 50))
    vt = np.unique(np.r_[-vt, vt])
    _cbar(fig, ax, im, r"$V_{\rm LOS}$ [km s$^{-1}$]", ticks=vt)

    pv_major = _find(bb, "*_pv_a.fits")
    vlim_pv = _pv_vlim(pv_major)   # shared y-range from the major axis
    olim_pv = _pv_olim(pv_major, kpp) # shared x-range from the major axis
    _pv_panel(axes[2], pv_major, _find(bb, "*pv_a*.fits", model=True),
              kpp, olim_pv, "major", rc, vlim_pv)
    _pv_panel(axes[3], _find(bb, "*_pv_b.fits"),
              _find(bb, "*pv_b*.fits", model=True), kpp, olim_pv, "minor",
              vlim=vlim_pv)

    title = (suptitle + r"  $\vert$  " if suptitle else "") + rf"$V/\sigma = {vsig:.2f}$"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
