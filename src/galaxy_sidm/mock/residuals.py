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

"""

from pathlib import Path

import numpy as np
from astropy.io import fits

NAN3 = (float("nan"), float("nan"), float("nan"))


def _load(path):
    """FITS data as a float array with NaNs zeroed; None if unreadable."""
    try:
        return np.nan_to_num(np.asarray(fits.getdata(path), float))
    except Exception:
        return None


def _noise(data, n_edge=3):
    """Noise from the n_edge line-free channels at each end of the velocity axis."""
    if data.shape[0] <= 2 * n_edge:
        return float(np.std(data))
    return float(np.std(np.concatenate([data[:n_edge].ravel(),
                                        data[-n_edge:].ravel()])))


def _residuals(data, model):
    """(res1, res2, res3) over the full arrays."""
    if data is None or model is None or data.shape != model.shape:
        return NAN3
    sigma = _noise(data)
    if not np.isfinite(sigma) or sigma <= 0:
        return NAN3
    diff = data - model
    abs_sum = float(np.sum(np.abs(diff)))
    res1 = float(np.sum(diff ** 2) / sigma ** 2)
    res2 = abs_sum / sigma
    flux = float(np.sum(np.abs(data)))
    res3 = abs_sum / flux if flux > 0 else float("nan")
    return res1, res2, res3


def cube_residuals(gal_dir):
    """(res1, res2, res3) for the MARTINI cube vs the BBarolo model cube."""
    g = Path(gal_dir)
    return _residuals(_load(g / "cube.fits"),
                      _load(g / "bbarolo" / "MOCKmod_azim.fits"))


def pv_residuals(gal_dir):
    """(res1, res2, res3) for the major-axis PV: data vs BBarolo model."""
    pvs = Path(gal_dir) / "bbarolo" / "pvs"
    return _residuals(_load(pvs / "MOCK_pv_a.fits"),
                      _load(pvs / "MOCKmod_pv_a_azim.fits"))