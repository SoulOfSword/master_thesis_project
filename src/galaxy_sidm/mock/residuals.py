"""Global data-model residuals of the BBarolo fits.

Three statistics per galaxy (i, j = pixel and
channel; element-wise data - model, then sum):

    res1 = sum_ij (D_ij - M_ij)^2 / sigma^2      (chi^2)
    res2 = sum_ij |D_ij - M_ij|   / sigma        (chi)
    res3 = sum_ij |D_ij - M_ij| / sum_ij |D_ij|  (fraction of signal unexplained)

Computed for two data/model pairs:

  * CUBE:  <gal>/cube.fits  vs  <gal>/bbarolo/MOCKmod_azim.fits
  * PV  :  <gal>/bbarolo/pvs/MOCK_pv_a.fits  vs  MOCKmod_pv_a_azim.fits
           (major-axis position-velocity slice, the one judged by eye)

sigma is the noise of the cube, the standard deviation of its first and last 3
channels (no emission there), for the PV too: the PV noise is the cube noise, and
the PV's own 6 end rows (6 x npix pixels) give a sigma off by ~5% per galaxy (up
to 14%), against 1.5% (138-px cubes) to 0.35% (>= 400 px) from the cube.

Noise floor. Where the model is right -- which includes everything outside the
galaxy, where the model is 0 -- D - M is Gaussian noise: |noise| averages
sqrt(2/pi) sigma = 0.798 sigma and noise^2 averages sigma^2. Summed over the N
elements of the array (PV: 64 channels x npix positions; cube: 64 x npix x npix),
a perfect fit still gives res1 = N and res2 = 0.798 N, set by the cube size alone.
So each function returns the floor-subtracted versions as well:

    res1 - N
    res2 - 0.798 N
    (sum |D-M| - 0.798 sigma N) / (sum |D| - 0.798 sigma N)   (floor out of both sums)

i.e. the residual in excess of pure noise: 0 for a perfect fit (res3 still 1 for
an empty model). The noise itself scatters them around that (res2 by ~ +-150 for
a 138-px PV), so a near-perfect fit can come out slightly negative.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits

N_EDGE = 3                       # line-free channels at each end of the velocity axis
MEAN_ABS = np.sqrt(2 / np.pi)    # <|noise|> / sigma of Gaussian noise
NAN3 = (float("nan"), float("nan"), float("nan"))


def _load(path):
    """(FITS data as a float array with NaNs zeroed, number of non-NaN values);
    (None, 0) if unreadable."""
    try:
        a = np.array(fits.getdata(path), dtype=float)
    except Exception:
        return None, 0
    n = int(np.isfinite(a).sum())
    return np.nan_to_num(a, copy=False), n


def _edge_std(cube):
    """Standard deviation of the N_EDGE first and last channels of a cube."""
    edge = np.concatenate([np.ravel(cube[:N_EDGE]), np.ravel(cube[-N_EDGE:])])
    return float(np.std(np.nan_to_num(edge.astype(float))))


def cube_noise(gal_dir):
    """sigma of <gal>/cube.fits from its end channels (only those are read); nan if unreadable."""
    try:
        with fits.open(Path(gal_dir) / "cube.fits", memmap=True) as h:
            return _edge_std(h[0].data)
    except Exception:
        return float("nan")


def _residuals(data, model, sigma, n):
    """(raw, floor-subtracted), each (res1, res2, res3), over the full arrays.

    n: number of data elements holding noise (the non-NaN ones).
    """
    if data is None or model is None or data.shape != model.shape:
        return NAN3, NAN3
    if not np.isfinite(sigma) or sigma <= 0:
        return NAN3, NAN3
    diff = data - model
    abs_sum = float(np.sum(np.abs(diff)))
    sq_sum = float(np.sum(diff ** 2))
    flux = float(np.sum(np.abs(data)))
    floor = MEAN_ABS * sigma * n         # sum of |noise| over the array
    raw = (sq_sum / sigma ** 2,
           abs_sum / sigma,
           abs_sum / flux if flux > 0 else float("nan"))
    sub = (sq_sum / sigma ** 2 - n,
           (abs_sum - floor) / sigma,
           (abs_sum - floor) / (flux - floor) if flux > floor else float("nan"))
    return raw, sub


def cube_residuals(gal_dir):
    """MARTINI cube vs the BBarolo model cube: (raw, floor-subtracted), each (res1, res2, res3)."""
    g = Path(gal_dir)
    cube, n = _load(g / "cube.fits")
    model, _ = _load(g / "bbarolo" / "MOCKmod_azim.fits")
    sigma = _edge_std(cube) if cube is not None else float("nan")
    return _residuals(cube, model, sigma, n)


def pv_residuals(gal_dir):
    """Major-axis PV, data vs BBarolo model, with the cube's sigma:
    (raw, floor-subtracted), each (res1, res2, res3)."""
    g = Path(gal_dir)
    pvs = g / "bbarolo" / "pvs"
    data, n = _load(pvs / "MOCK_pv_a.fits")
    model, _ = _load(pvs / "MOCKmod_pv_a_azim.fits")
    return _residuals(data, model, cube_noise(g), n)
