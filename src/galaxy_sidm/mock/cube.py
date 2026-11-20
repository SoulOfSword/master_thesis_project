"""MARTINI datacube from a galaxy's neutral gas.

Galaxy at 5 Mpc, 5" pixels, 30" beam (~720 pc), 30 km/s channels x64,
inclined 60 deg at PA 90 (set from the gas angular momentum).

Gas cut (CubeParams.disc_normal, z_cut_kpc, R_cut_kpc; disc_cut): MARTINI gets only the cells
with |z| < z_cut_kpc and R < R_cut_kpc about the plane of unit normal disc_normal, and the
cube is inclined about that normal instead of the one MARTINI's L_align would find from the
cells it gets (_disc_rotation).

Noise: the same `noise_rms` for every galaxy, or -- with `reference_snr` set --
the same signal-to-noise for every galaxy: noise_rms = signal / reference_snr,
with `signal` (measure_signal) taken from the galaxy's own noiseless,
beam-convolved cube. `reference_snr` comes from scripts/mock/snr_reference.py.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import astropy.units as U
from scipy.spatial.transform import Rotation

from .gas import GalaxyGas

_ARCSEC_PER_RAD = 206264.806
SOURCE_RADIUS_FACTOR = 1.5 # gas cells given to MARTINI

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
    half_kpc: float = None
    add_noise: bool = True
    noise_rms: U.Quantity = 1.0e-5 * U.Jy / U.arcsec ** 2
    reference_snr: float = None
    signal_peak_fraction: float = 0.1  # signal = mean of voxels >= this x brightest voxel
    noise_seed: int = 0  # MARTINI noise RNG seed (0 = MARTINI's default)
    # MARTINI 3 only: memory [GB] it may use to fill the cube before it switches to batches
    # (insert_source_in_cube's mem_lim_GB). None = not passed (MARTINI 2 has no such argument)
    mem_lim_GB: float = None
    # gas cut, all three or none
    disc_normal: tuple = None
    z_cut_kpc: float = None
    R_cut_kpc: float = None


@dataclass
class CubeResult:
    fits: Path
    npix: int
    half_kpc: float # FOV half-width used, kpc
    signal: float # Jy/beam, from the noiseless cube (nan if not measured)
    noise_rms: float # Jy/arcsec^2 given to MARTINI (0 if no noise)


def measure_signal(cube, peak_fraction=0.1):
    """Mean brightness of the voxels >= peak_fraction x the brightest voxel.

    Meant for a NOISELESS cube. Only voxels with emission enter the mean, so the
    value does not depend on the size of the image (an average over all pixels
    would also count empty sky). Returns nan if the cube has no emission.
    """
    c = np.nan_to_num(np.asarray(cube, dtype=float))
    peak = c.max()
    if peak <= 0:
        return float("nan")
    return float(c[c >= peak_fraction * peak].mean())

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

    The half-mass radius is measured face-on, in the plane of the gas disc
    (normal gas.L_hat, the axis MARTINI inclines), so it does not depend on
    how the galaxy happens to be oriented in the simulation box.
    """
    if len(gas.m_s):
        xyz = gas.xyz_s.to_value(U.kpc)
        w = gas.m_s.to_value(U.Msun)
    else:
        xyz = gas.xyz_g.to_value(U.kpc)
        w = gas.mH_neutral_g.to_value(U.Msun)
    zhat = np.asarray(gas.L_hat, dtype=float)
    r = np.linalg.norm(xyz - np.outer(xyz @ zhat, zhat), axis=1) # face-on radius
    order = np.argsort(r) # sort by radius
    cum = np.cumsum(w[order]) # cumulative mass profile
    r50 = r[order][np.searchsorted(cum, 0.5 * cum[-1])] if len(r) else 4.0
    half_kpc = float(np.clip(p.fov_factor * r50, 8.0, 20.0))
    max_npix = p.max_npix
    if p.half_kpc is not None:   # explicit FOV (CubeParams.half_kpc)
        half_kpc, max_npix = float(p.half_kpc), np.inf
    half_ang = (half_kpc * U.kpc / p.distance).to_value(
        U.dimensionless_unscaled) * U.rad
    npix = int(2 * np.ceil((half_ang / p.px_size).to_value(
        U.dimensionless_unscaled)))
    return int(np.clip(npix + 4, 32, max_npix)), half_kpc


def source_mask(gas, half_kpc):
    """The gas cells MARTINI receives: those within SOURCE_RADIUS_FACTOR x half_kpc (kpc)."""
    return np.linalg.norm(gas.xyz_g.to_value(U.kpc), axis=1) < SOURCE_RADIUS_FACTOR * half_kpc


def rule_half_kpc(gas):
    """FOV half-width (kpc) of the cube-size rule: _n_px with CubeParams' defaults, i.e.
    fov_factor x the face-on stellar half-mass radius, clipped to 8-20 kpc."""
    return _n_px(gas, CubeParams())[1]


def disc_normal(gas):
    """Unit normal of the disc plane, found as MARTINI does (L_align: angular momentum of the
    innermost cells holding 30% of the neutral mass, at least 100) on the cells a cube of
    the rule's half-width receives (source_mask, rule_half_kpc). Needs no cube."""
    from martini.sources._L_align import L_align
    keep = source_mask(gas, rule_half_kpc(gas))
    return L_align(gas.xyz_g[keep].T, gas.vxyz_g[keep].T, gas.mH_neutral_g[keep])[2]


def disc_coords(xyz, zhat):
    """(height z, cylindrical radius R) of positions xyz about the plane of unit normal zhat."""
    z = xyz @ zhat
    return z, np.linalg.norm(xyz - np.outer(z, zhat), axis=1)


def disc_velocities(xyz, vel, zhat):
    """(v_R, v_phi, v_z) of velocities vel at positions xyz about the plane of unit normal
    zhat; v_phi > 0 for rotation in the sense of zhat."""
    z, R = disc_coords(xyz, zhat)
    e_R = (xyz - np.outer(z, zhat)) / R[:, None]
    return np.sum(vel * e_R, axis=1), np.sum(vel * np.cross(zhat, e_R), axis=1), vel @ zhat


def _weighted_quantile(x, w, q):
    """Value of x below which the fraction q of the total weight w lies."""
    pos = w > 0
    if not pos.any():
        raise ValueError("no weight to take a quantile of")
    order = np.argsort(x[pos])
    cum = np.cumsum(w[pos][order])
    return float(np.interp(q * cum[-1], cum, x[pos][order]))


def layer_radii(gas, zhat, z_cut, qs):
    """Radii (kpc) holding the fractions qs of the neutral hydrogen with |z| < z_cut about the
    plane of unit normal zhat."""
    z, R = disc_coords(gas.xyz_g.to_value(U.kpc), zhat)
    layer = np.abs(z) < z_cut
    m = gas.mH_neutral_g.to_value(U.Msun)[layer]
    return [_weighted_quantile(R[layer], m, q) for q in qs]


def disc_cut(gas, zhat, r_half_star, k_z=5.0, q_R=0.99):
    """Gas cells near the disc plane of unit normal zhat: |z| < k_z z_half and R < R_cut.

    z_half is the half-mass height of the neutral hydrogen within R < 2 r_half_star (kpc);
    R_cut is the radius holding the fraction q_R of the neutral hydrogen with
    |z| < k_z z_half. Returns (mask of the kept cells, z_half, z_cut, R_cut), lengths in kpc.
    """
    z, R = disc_coords(gas.xyz_g.to_value(U.kpc), zhat)
    m = gas.mH_neutral_g.to_value(U.Msun)
    inner = R < 2 * r_half_star
    z_half = _weighted_quantile(np.abs(z[inner]), m[inner], 0.5)
    z_cut = k_z * z_half
    R_cut = layer_radii(gas, zhat, z_cut, [q_R])[0]
    return (np.abs(z) < z_cut) & (R < R_cut), z_half, z_cut, R_cut


def _disc_rotation(zhat, incl, pa):
    """The rotation MARTINI's SPHSource.rotate makes for L_coords(incl=incl, pa=pa), with the unit
    normal zhat in place of the axis its L_align finds: the same face-on frame built around
    zhat (L_align, Laxis="x"), then the inclination about y and the position angle about x."""
    xaxis = np.ones(3) / np.sqrt(3)
    xhat = xaxis - (xaxis @ zhat) * zhat
    xhat /= np.linalg.norm(xhat)
    face_on = np.vstack([zhat, xhat, np.cross(zhat, xhat)])
    pa_rot = pa - (90 if incl >= 0 else 270) * U.deg
    return (Rotation.from_euler("x", pa_rot.to_value(U.rad))
            * Rotation.from_euler("y", incl.to_value(U.rad))
            * Rotation.from_matrix(face_on))


def build_cube(gas: GalaxyGas, out_fits, params: CubeParams = None, ncpu=1):
    """Generate and write the MARTINI datacube. Returns a CubeResult."""
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
    keep = source_mask(gas, half_kpc)
    cut = (p.disc_normal, p.z_cut_kpc, p.R_cut_kpc)
    if any(v is None for v in cut) and any(v is not None for v in cut):
        raise ValueError(f"disc_normal, z_cut_kpc and R_cut_kpc go together, got {cut}")
    if p.disc_normal is None:
        orientation = {"L_coords": L_coords(incl=p.inclination, pa=p.position_angle)}
    else:
        zhat = np.asarray(p.disc_normal, dtype=float)
        zhat = zhat / np.linalg.norm(zhat)
        z, R = disc_coords(gas.xyz_g.to_value(U.kpc), zhat)
        keep &= (np.abs(z) < p.z_cut_kpc) & (R < p.R_cut_kpc)
        orientation = {"rotation": _disc_rotation(zhat, p.inclination, p.position_angle)}

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
        **orientation,
    )

    datacube = DataCube(
        n_px_x=npix, n_px_y=npix, n_channels=p.n_channels,
        px_size=p.px_size, channel_width=p.channel_width,
        spectral_centre=source.vsys,
    )
    beam = GaussianBeam(bmaj=p.beam_fwhm, bmin=p.beam_fwhm, bpa=0.0 * U.deg)
    spectral_model = GaussianSpectrum(sigma="thermal")
    sph_kernel = CubicSplineKernel()

    # noise is attached after insertion: in constant-S/N mode its rms depends
    # on the galaxy's own noiseless cube
    M = Martini(source=source, datacube=datacube, beam=beam, noise=None,
                sph_kernel=sph_kernel, spectral_model=spectral_model,
                quiet=True)
    if p.mem_lim_GB is None:
        M.insert_source_in_cube(ncpu=ncpu)
    else:
        M.insert_source_in_cube(ncpu=ncpu, mem_lim_GB=p.mem_lim_GB)

    signal = float("nan")
    rms = p.noise_rms if p.add_noise else None
    if p.add_noise and p.reference_snr is not None:
        # measure the signal on a beam-convolved copy of the noiseless cube,
        # then go back to the unconvolved cube to add the matching noise
        noiseless = M._datacube.copy()
        M.convolve_beam()
        signal = measure_signal(M._datacube._array.to_value(U.Jy / U.beam),
                                p.signal_peak_fraction)
        if not np.isfinite(signal):
            raise RuntimeError("noiseless cube has no emission: cannot set the "
                               "constant-S/N noise")
        M._datacube = noiseless
        rms = signal / p.reference_snr * U.Jy / U.arcsec ** 2
    if rms is not None:
        # MARTINI takes the rms in Jy/beam (MARTINI 3 refuses Jy/arcsec^2); MARTINI 2 made
        # this same conversion itself, so the noise is unchanged
        M.noise = GaussianNoise(rms=rms.to(U.Jy / U.beam, equivalencies=U.beam_angular_area(beam.area)),
                                seed=p.noise_seed)
        M.add_noise()
    M.convolve_beam()
    if not p.add_noise:
        signal = measure_signal(M._datacube._array.to_value(U.Jy / U.beam),
                                p.signal_peak_fraction)

    out_fits = Path(out_fits)
    out_fits.parent.mkdir(parents=True, exist_ok=True)
    M.write_fits(str(out_fits), overwrite=True)
    return CubeResult(fits=out_fits, npix=npix, half_kpc=half_kpc, signal=signal,
                      noise_rms=float(rms.value) if rms is not None else 0.0)