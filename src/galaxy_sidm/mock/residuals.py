"""Data-model residuals of the BBarolo final fits.

Two statistics per galaxy, per voxel and with the noise taken out (i = voxel):

    res1 = sum_V (D_i - M_i)^2 / sigma^2 / N_V  -  1
    res2 = sum_V |D_i - M_i|   / sigma   / N_V  -  sqrt(2/pi)

summed over the voxels V where there is gas or model, N_V of them:

  * inside the final fit's mask (bbarolo/mask.fits, the 3-ring fit's mask with the
    hole rings cut out, i.e. the voxels the final fit itself used); it still includes
    the gas beyond the last ring, where the final model is empty;
  * or where the final model is brighter than MODEL_CUT sigma: model gas where the
    data has none counts too.

Noise: where the model is right, D - M is Gaussian noise, whose square averages
sigma^2 and whose absolute value averages sqrt(2/pi) sigma = 0.798 sigma. So a
perfect fit gives res1 = res2 = 0 (slightly below or above, from the noise itself),
whatever the size of the cube or of the galaxy.

Computed for two data/model pairs:

  * CUBE: <gal>/cube.fits vs <gal>/bbarolo/MOCKmod_azim.fits,
          mask <gal>/bbarolo/mask.fits
  * PV:   <gal>/bbarolo/pvs/MOCK_pv_a.fits vs MOCKmod_pv_a_azim.fits (the major-axis
          position-velocity slice), mask <gal>/bbarolo/pvs/MOCKmask_pv_a.fits

sigma is the noise of the cube, the standard deviation of its first and last 3
channels (no emission there), for the PV too: the PV noise is the cube noise, and
the PV's own 6 end rows (6 x npix pixels) give a sigma off by ~5% per galaxy (up
to 14%), against 1.5% (138-px cubes) to 0.35% (>= 400 px) from the cube.

Galaxies without a final fit (no counted emission along the major axis, see
barolo.pv_rings) get NaN.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits

N_EDGE = 3                       # line-free channels at each end of the velocity axis
MEAN_ABS = np.sqrt(2 / np.pi)    # <|noise|> / sigma of Gaussian noise
MODEL_CUT = 1.0                  # model voxels above this many sigma count, also outside the mask
NAN2 = (float("nan"), float("nan"))


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


def has_final_fit(gal_dir):
    """True if the galaxy has a final BBarolo model. Without one, the final fit found
    no counted emission along the major axis and was not run."""
    return (Path(gal_dir) / "bbarolo" / "MOCKmod_azim.fits").exists()


def _residuals(data_file, model_file, mask_file, sigma):
    """(res1, res2) of data vs model over the voxels inside the mask or where the model
    is above MODEL_CUT sigma (see the module doc).

    The files are read one channel (cube) or one velocity row (PV) at a time, so a
    large cube never sits in memory several times over.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"no usable noise for {data_file}: sigma = {sigma}")
    with fits.open(data_file, memmap=True) as hd, fits.open(model_file, memmap=True) as hm, \
            fits.open(mask_file, memmap=True) as hk:
        data, model, mask = (np.squeeze(h[0].data) for h in (hd, hm, hk))
        if not data.shape == model.shape == mask.shape:
            raise ValueError(f"data {data.shape}, model {model.shape} and mask {mask.shape} "
                             f"differ ({data_file})")
        n, sq_sum, abs_sum = 0, 0.0, 0.0
        for k in range(data.shape[0]):
            D = np.asarray(data[k], dtype=float)
            M = np.nan_to_num(np.asarray(model[k], dtype=float))
            K = np.nan_to_num(np.asarray(mask[k], dtype=float))
            use = np.isfinite(D) & ((K != 0) | (M > MODEL_CUT * sigma))
            d = (D[use] - M[use]) / sigma
            n += int(use.sum())
            sq_sum += float(np.sum(d ** 2))
            abs_sum += float(np.sum(np.abs(d)))
    if n == 0:
        return NAN2
    return float(sq_sum / n - 1), float(abs_sum / n - MEAN_ABS)


def cube_residuals(gal_dir):
    """(res1, res2) of the MARTINI cube vs the final BBarolo model cube; NaN without a final fit."""
    g = Path(gal_dir)
    if not has_final_fit(g):
        return NAN2
    return _residuals(g / "cube.fits", g / "bbarolo" / "MOCKmod_azim.fits",
                      g / "bbarolo" / "mask.fits", cube_noise(g))


def pv_residuals(gal_dir):
    """(res1, res2) of the major-axis PV, data vs final BBarolo model, with the cube's sigma;
    NaN without a final fit."""
    g = Path(gal_dir)
    if not has_final_fit(g):
        return NAN2
    pvs = g / "bbarolo" / "pvs"
    return _residuals(pvs / "MOCK_pv_a.fits", pvs / "MOCKmod_pv_a_azim.fits",
                      pvs / "MOCKmask_pv_a.fits", cube_noise(g))
