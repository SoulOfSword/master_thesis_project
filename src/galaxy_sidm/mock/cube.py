"""MARTINI datacube from a galaxy's neutral gas.

Galaxy at 5 Mpc, 5" pixels, 30" beam (~720 pc), 30 km/s channels x64,
inclined 60 deg at PA 90 (set from the gas angular momentum). Noise off
by default.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import astropy.units as U

from .gas import GalaxyGas

_ARCSEC_PER_RAD = 206264.806

@dataclass
class CubeParams:
    distance: U.Quantity = 5.0 * U.Mpc
    px_size: U.Quantity = 5.0 * U.arcsec
    beam_fwhm: U.Quantity = 30.0 * U.arcsec
    channel_width: U.Quantity = 30.0 * U.km / U.s
    n_channels: int = 64
    inclination: U.Quantity = 60.0 * U.deg
    position_angle: U.Quantity = 90.0 * U.deg
    fov_factor: float = 4.0 # FOV = fov_factor * stellar half-mass radius
    max_npix: int = 400
    add_noise: bool = True
    noise_rms: U.Quantity = 1.0e-5 * U.Jy / U.arcsec ** 2

def hi_radius_kpc(gas, sigma_thresh=1.0, dr_kpc=0.5):
    """Face-on neutral-gas size R_HI (kpc).

    Outermost radius at which the azimuthally-averaged neutral-hydrogen
    surface density falls to `sigma_thresh` (Msun/pc^2; 1.0 is the standard
    R_HI definition). Uses the same neutral field fed to MARTINI
    (`mH_neutral_g`) and the disc plane already measured for the galaxy
    (`gas.L_hat`, the inner-gas angular momentum) -> the size is consistent
    with the cube. Built from the 3-D particles, not the beam-smeared,
    inclined datacube.

    Args:
        gas: GalaxyGas (needs xyz_g, mH_neutral_g, L_hat).
        sigma_thresh: surface-density threshold in Msun/pc^2.
        dr_kpc: radial bin width in kpc.

    Returns:
        R_HI in kpc (outermost crossing, linearly interpolated between the
        bracketing bins), or 0.0 if the profile never reaches the threshold.
    """
    x = gas.xyz_g.to_value(U.kpc)
    m = gas.mH_neutral_g.to_value(U.Msun)
    zhat = np.asarray(gas.L_hat, dtype=float)                # inner-disc normal
    R = np.linalg.norm(x - np.outer(x @ zhat, zhat), axis=1)  # face-on radius
    edges = np.arange(0.0, R.max() + dr_kpc, dr_kpc) # radial bins
    msum, _ = np.histogram(R, bins=edges, weights=m) # Msun in each ring
    sigma = msum / (np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * 1e6) # Msun/pc^2
    cent = 0.5 * (edges[1:] + edges[:-1]) # ring-centre radii in kpc
    above = np.where(sigma >= sigma_thresh)[0] #indices of rings above the threshold
    if len(above) == 0:
        return 0.0
    k = int(above[-1]) # outermost ring above thresh
    if k == len(sigma) - 1:
        return float(cent[k]) # never drops within the data
    s0, s1 = sigma[k], sigma[k + 1]
    return float(cent[k] + (cent[k + 1] - cent[k]) * (s0 - sigma_thresh) / (s0 - s1))

def _n_px(gas: GalaxyGas, p: CubeParams):
    """Pixels per side: FOV = fov_factor x stellar half-mass radius.

    Stars trace the disc; the neutral-gas half-mass radius is biased high
    by a diffuse envelope, which over-sizes the cube.
    """
    if len(gas.m_s):
        r = np.linalg.norm(gas.xyz_s.to_value(U.kpc)[:, :2], axis=1)
        w = gas.m_s.to_value(U.Msun)
    else:
        r = np.linalg.norm(gas.xyz_g.to_value(U.kpc)[:, :2], axis=1)
        w = gas.mH_neutral_g.to_value(U.Msun)
    order = np.argsort(r)
    cum = np.cumsum(w[order])
    r50 = r[order][np.searchsorted(cum, 0.5 * cum[-1])] if len(r) else 4.0
    half_kpc = float(np.clip(p.fov_factor * r50, 8.0, 20.0))
    half_ang = (half_kpc * U.kpc / p.distance).to_value(
        U.dimensionless_unscaled) * U.rad
    npix = int(2 * np.ceil((half_ang / p.px_size).to_value(
        U.dimensionless_unscaled)))
    return int(np.clip(npix + 4, 32, p.max_npix)), half_kpc


def build_cube(gas: GalaxyGas, out_fits, params: CubeParams = None, ncpu=1):
    """Generate and write the MARTINI datacube. Returns the FITS path."""
    from martini import Martini, DataCube
    from martini.sources.sph_source import SPHSource
    from martini.sph_kernels import CubicSplineKernel
    from martini.spectral_models import GaussianSpectrum
    from martini.beams import GaussianBeam
    from martini.noise import GaussianNoise
    from martini import L_coords

    p = params or CubeParams()
    npix, half_kpc = _n_px(gas, p)

    # only gas near the disc matters for the cube; drop the far filaments
    keep = np.linalg.norm(gas.xyz_g.to_value(U.kpc), axis=1) < 1.5 * half_kpc

    # the source is defined by the gas properties, but the cube is defined by the parameters; the source is centred on the galaxy and oriented by the
    # gas angular momentum, but the cube is oriented by the position angle parameter; the source has a systemic velocity set by the gas, but the cube is centred on zero velocity
    source = SPHSource(
        distance=p.distance,
        mHI_g=gas.mH_neutral_g[keep], #MARTINI wants mHI_g
        xyz_g=gas.xyz_g[keep],
        vxyz_g=gas.vxyz_g[keep],
        T_g=gas.T_g[keep],
        hsm_g=gas.hsm_g[keep],
        h=gas.h,
        L_coords=L_coords(incl=p.inclination, pa=p.position_angle),
    )

    datacube = DataCube(
        n_px_x=npix, n_px_y=npix, n_channels=p.n_channels,
        px_size=p.px_size, channel_width=p.channel_width,
        spectral_centre=source.vsys,
    )
    beam = GaussianBeam(bmaj=p.beam_fwhm, bmin=p.beam_fwhm, bpa=0.0 * U.deg)
    noise = GaussianNoise(rms=p.noise_rms) if p.add_noise else None
    spectral_model = GaussianSpectrum(sigma="thermal")
    sph_kernel = CubicSplineKernel()

    M = Martini(source=source, datacube=datacube, beam=beam, noise=noise,
                sph_kernel=sph_kernel, spectral_model=spectral_model,
                quiet=True)
    M.init_spectra()
    M.insert_source_in_cube(ncpu=ncpu)
    if noise is not None:
        M.add_noise()
    M.convolve_beam()

    out_fits = Path(out_fits)
    out_fits.parent.mkdir(parents=True, exist_ok=True)
    M.write_fits(str(out_fits), overwrite=True)
    return out_fits, npix
