"""3D asymmetry of a mock HI cube (He et al. 2026).

    A = sum |I(i,j,k) - I(-i,-j,-k)| / sum |I(i,j,k) + I(-i,-j,-k)|

summed over the voxels of a symmetric mask, with (i, j) the pixel offsets from the
galaxy centre and k the channel offset from the central channel: every voxel is
compared with its point reflection, so the approaching and receding "wings" of the
cube are compared with each other. A = 0 for a perfectly symmetric galaxy, 1 for
emission on one side only.

  * centre: the cube centre, (n-1)/2 on both axes, where the BBarolo fit fixes
    XPOS/YPOS (checked against bbarolo.par).
  * central channel (He+2026): the channel nearest to BBarolo's VSYS (rings file) if
    VSYS lies within 1/3 of a channel width of it; otherwise both nearest channels
    are central, and the reflection is about the point between them.
  * mask: the mask.fits of the 3-ring fit (bbarolo_3rings; 1 = emission, from BBarolo's
    SMOOTH&SEARCH), made symmetric: a pair is in when either of its voxels is (He+2026).
    With the mask as it is, a feature on one side only would enter the sum without its
    empty mirror. The 3-ring fit's mask, not the final fit's: the final fit's mask has
    the hole rings cut out, and the gas in them should count.

Noise: where the galaxy is symmetric, I - I' is pure noise, and |noise1 - noise2|
averages 2 sigma / sqrt(pi) (sigma: the cube noise, residuals.cube_noise). Summed
over the N voxels of the mask, a perfectly symmetric galaxy therefore still gets

    A_noise = (2 / sqrt(pi)) sigma N / sum |I + I'|        (0.09-0.21 in our cubes)

and A_corr = A - A_noise removes it. That is exact for symmetric emission and
slightly over-corrects strongly asymmetric voxels, whose |I - I'| has no noise bias.
He+2026 call a galaxy asymmetric above A = 0.35.

Squared version (Deg et al. 2023, MNRAS 523, 4340), from the same voxel pairs:

    A_sq = (sum (I - I')^2 - 2 N sigma^2) / (sum (I + I')^2 - 2 N sigma^2)

With I = S + n, the noise adds exactly 2 sigma^2 to the mean of every (I - I')^2 and
(I + I')^2, whatever the galaxy does there (the cross term 2 (S - S')(n - n') averages
to zero), so subtracting 2 N sigma^2 is right at any asymmetry and S/N. |I - I'| has
no such split: the noise adds 2 sigma / sqrt(pi) to it where S = S', but nothing
where |S - S'| >> sigma, so the constant A_noise is right only for symmetric
emission. Deg+2023 quote sqrt(A_sq), which is on the scale of A. A_sq < 0 when the
noise accounts for all of sum (I - I')^2; Deg+2023 set A = -1 there.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import astropy.units as U
from astropy.io import fits

from .barolo import read_par, rings_file
from .residuals import cube_noise

HE26_THRESHOLD = 0.35     # He+2026: asymmetric above this A


@dataclass
class Asymmetry:
    A: float                 # 3D asymmetry
    A_noise: float           # part of A a perfectly symmetric galaxy gets from the noise
    A_corr: float            # A - A_noise
    A_sq: float              # squared asymmetry with the noise taken out (Deg+2023)
    n_voxels: int            # voxels in the symmetric mask
    central_channel: float   # 0-based; x.5 = two central channels
    vsys: float              # km/s, from the BBarolo fit


def fit_vsys(bbarolo_dir):
    """BBarolo's VSYS [km/s]: the median of the VSYS column of the fit's rings file
    (stage 2 uses one VSYS for all rings)."""
    rf = rings_file(bbarolo_dir)
    lines = rf.read_text().splitlines()
    header = next((l for l in lines if l.startswith("#") and "VSYS(km/s)" in l), None)
    if header is None:
        raise ValueError(f"no VSYS(km/s) column in {rf}")
    col = header.lstrip("#").split().index("VSYS(km/s)")
    v = [float(l.split()[col]) for l in lines if l.strip() and not l.startswith("#")]
    if not v:
        raise ValueError(f"no rings in {rf}")
    return float(np.median(v))


def central_channel(vsys, header):
    """0-based central channel for VSYS [km/s] (He+2026): the nearest channel if VSYS
    is within 1/3 of a channel width of it, else the point between the two nearest (x.5)."""
    if not np.isfinite(vsys):
        raise ValueError(f"VSYS is {vsys}")
    unit = U.Unit(header["CUNIT3"])
    crval = (header["CRVAL3"] * unit).to_value(U.km / U.s)
    cdelt = (header["CDELT3"] * unit).to_value(U.km / U.s)
    c = header["CRPIX3"] - 1 + (vsys - crval) / cdelt      # fractional channel of VSYS
    lo = np.floor(c)
    if c - lo < 1 / 3:
        return float(lo)
    if c - lo > 2 / 3:
        return float(lo + 1)
    return float(lo + 0.5)


def asymmetry_3d(gal_dir, cube=None, sigma=None):
    """He+2026 3D asymmetry of <gal>/cube.fits in the 3-ring fit's symmetrised mask (see module doc).

    cube: measure this cube instead, a FITS file or an array on the same grid (e.g. a
    noiseless twin of cube.fits), in the same mask and about the same centre and central
    channel. sigma: its noise (0 for a noiseless cube); default: the noise of cube.fits.
    """
    g = Path(gal_dir)
    bb = g / "bbarolo"
    header = fits.getheader(g / "cube.fits")
    if cube is None or isinstance(cube, (str, Path)):
        cube = fits.getdata(cube or g / "cube.fits")
    cube = np.nan_to_num(np.array(cube, dtype=float), copy=False)
    mask = fits.getdata(g / "bbarolo_3rings" / "mask.fits") != 0
    if mask.shape != cube.shape:
        raise ValueError(f"{g}: mask {mask.shape} and cube {cube.shape} differ")
    nch, ny, nx = cube.shape
    par = read_par(bb)
    centre = ((nx - 1) / 2, (ny - 1) / 2)
    fit_centre = tuple(float(par[k]) if k in par else None for k in ("XPOS", "YPOS"))
    if fit_centre != centre:
        raise ValueError(f"{g}: the BBarolo centre {fit_centre} is not the cube centre {centre}")
    vsys = fit_vsys(bb)
    kc = central_channel(vsys, header)
    K = int(round(2 * kc))      # voxel (k, y, x) pairs with (K - k, ny-1-y, nx-1-x)

    k, y, x = np.nonzero(mask)
    if len(k) == 0:
        return Asymmetry(np.nan, np.nan, np.nan, np.nan, 0, kc, vsys)
    if ((K - k < 0) | (K - k >= nch)).any():
        raise ValueError(f"{g}: emission whose mirror channel about channel {kc} is outside the cube")
    sym = mask.copy()
    sym[K - k, ny - 1 - y, nx - 1 - x] = True       # a pair is in if either voxel is
    k, y, x = np.nonzero(sym)
    I = cube[k, y, x]
    I_mirror = cube[K - k, ny - 1 - y, nx - 1 - x]
    diff, tot = I - I_mirror, I + I_mirror
    sig = cube_noise(g) if sigma is None else float(sigma)
    den = float(np.abs(tot).sum())
    A = float(np.abs(diff).sum()) / den
    A_noise = 2 / np.sqrt(np.pi) * sig * len(I) / den
    B = 2 * len(I) * sig ** 2       # what the noise adds to each sum of squares
    A_sq = (float(np.sum(diff ** 2)) - B) / (float(np.sum(tot ** 2)) - B)
    return Asymmetry(A, A_noise, A - A_noise, A_sq, len(I), kc, vsys)
