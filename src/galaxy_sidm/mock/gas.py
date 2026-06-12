"""Load one galaxy's gas + stars for mock-observation work.

Returns particle arrays in **physical** units (proper kpc, peculiar km/s,
Msun), centred on the subhalo (length x a/h, velocity x sqrt(a)).

The neutral-hydrogen mass per gas cell (H I + H2) is computed with the
`Hdecompose` package: the total neutral gas mass is fed to MARTINI, not just H I.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import astropy.units as U
import astropy.constants as C
import illustris_python as il

from ..data.aida_tng import get_snap_scale_factor

GAMMA = 5.0 / 3.0
X_H_DEFAULT = 0.76 # default hydrogen mass fraction


@dataclass
class GalaxyGas:
    """Physical-unit, subhalo-centred particle arrays for one galaxy."""
    # gas (astropy Quantity arrays)
    xyz_g: U.Quantity        # (N,3) kpc, centred
    vxyz_g: U.Quantity       # (N,3) km/s, peculiar, centred
    mH_neutral_g: U.Quantity # (N,) Msun, total neutral H (HI+H2)
    mgas_g: U.Quantity       # (N,) Msun, total cell gas mass
    T_g: U.Quantity          # (N,) K
    hsm_g: U.Quantity        # (N,) kpc, cell size (kernel smoothing)
    # stars
    xyz_s: U.Quantity        # (N_s,3) kpc, centred
    m_s: U.Quantity          # (N_s,) Msun
    # geometry / meta
    L_hat: np.ndarray        # (3,) unit angular-momentum vector (neutral gas)
    redshift: float
    a: float
    h: float
    subhalo_id: int
    snap: int


def _periodic_center(coords, center, box):
    dx = coords - center
    dx -= box * np.round(dx / box)
    return dx


def load_galaxy_gas(base_path, snap, subhalo_id, h=0.6774,
                    orient_radius_ckpc=10.0):
    """Load gas + stars of one subhalo in physical units, centred.

    Args:
        base_path: simulation `output/` dir (str/Path with trailing-safe).
        snap: snapshot number.
        subhalo_id: central subhalo id (GroupFirstSub for FoF centrals).
        h: little-h.
        orient_radius_ckpc: radius (comoving kpc) within which the gas
            angular momentum is measured to define the disc plane.

    Returns:
        GalaxyGas. mH_neutral_g is the total neutral-H (HI+H2) cell mass.
    """
    base_path = str(base_path)
    run_path = Path(base_path).parent
    z, a = get_snap_scale_factor(run_path, snap)

    gfields = ["Coordinates", "Velocities", "Masses", "Density",
               "InternalEnergy", "ElectronAbundance", "StarFormationRate",
               "GFM_Metals", "NeutralHydrogenAbundance"]
    g = il.snapshot.loadSubhalo(base_path, snap, subhalo_id, 0, fields=gfields)
    if not isinstance(g, dict) or g.get("count", 0) == 0:
        raise RuntimeError(f"subhalo {subhalo_id} snap {snap}: no gas cells")
    s = il.snapshot.loadSubhalo(base_path, snap, subhalo_id, 4,
                                fields=["Coordinates", "Masses"])

    sub = il.groupcat.loadSingle(base_path, snap, subhaloID=subhalo_id)
    pos0 = np.asarray(sub["SubhaloPos"], dtype=np.float64) # ckpc/h
    vel0 = np.asarray(sub["SubhaloVel"], dtype=np.float64) # km/s (phys)
    with __import__("h5py").File(
            sorted((run_path / "output" / f"snapdir_{snap:03d}").glob(
                f"snap_{snap:03d}.*.hdf5"))[0], "r") as f:
        box = float(f["Header"].attrs["BoxSize"]) # ckpc/h

    # --- gas: physical, centred ---
    dx = _periodic_center(np.asarray(g["Coordinates"], np.float64), pos0, box)
    xyz_g = (dx * a / h) * U.kpc
    vxyz_g = (np.asarray(g["Velocities"], np.float64) * np.sqrt(a)
              - vel0) * (U.km / U.s)
    mgas = np.asarray(g["Masses"], np.float64) * 1e10 / h          # Msun
    rho_code = np.asarray(g["Density"], np.float64)               # 1e10 Msun/h / (ckpc/h)^3
    u_int = np.asarray(g["InternalEnergy"], np.float64)
    x_e = np.asarray(g["ElectronAbundance"], np.float64)
    sfr = np.asarray(g["StarFormationRate"], np.float64) * (U.Msun / U.yr)
    metals = np.asarray(g["GFM_Metals"], np.float64)
    X_H = metals[:, 0] if metals.ndim == 2 else np.full(len(mgas), X_H_DEFAULT)

    # temperature
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * x_e)
    u_cgs = u_int * (U.km / U.s) ** 2
    T = ((GAMMA - 1.0) * u_cgs * mu * C.m_p / C.k_B).to(U.K)

    # physical mass density and hydrogen number density
    rho_phys = (rho_code * 1e10 * U.Msun / h
                / (U.kpc / h * a) ** 3).to(U.g / U.cm ** 3)
    nH = (X_H * rho_phys / C.m_p).to(U.cm ** -3)

    # neutral fraction (HI+H2), TNG correction on
    from Hdecompose.atomic_frac import neutral_frac
    f_neutral = neutral_frac(
        z, nH, T, rho=rho_phys, Habundance=X_H * U.dimensionless_unscaled,
        SFR=sfr, mu=mu, gamma=GAMMA, fH=np.median(X_H),
        TNG_corrections=True, local=False)
    f_neutral = np.clip(np.asarray(f_neutral), 0.0, 1.0)
    mH_neutral = (f_neutral * X_H * mgas) * U.Msun # Msun of neutral H (HI+H2) per cell

    # smoothing length = equal-volume sphere radius of the cell
    vol = (mgas * U.Msun / rho_phys).to(U.kpc ** 3)
    hsm = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)

    # disc normal: angular momentum of the inner neutral gas
    r = np.linalg.norm(xyz_g.to_value(U.kpc), axis=1)
    inner = r < (orient_radius_ckpc * a / h)
    w = np.asarray(mH_neutral.to_value(U.Msun))[inner][:, None]
    Lvec = np.sum(np.cross(xyz_g.to_value(U.kpc)[inner],
                           vxyz_g.to_value(U.km / U.s)[inner]) * w, axis=0)
    L_hat = Lvec / (np.linalg.norm(Lvec) + 1e-30)

    # --- stars ---
    if isinstance(s, dict) and s.get("count", 0) > 0:
        dxs = _periodic_center(np.asarray(s["Coordinates"], np.float64), pos0, box)
        xyz_s = (dxs * a / h) * U.kpc
        m_s = (np.asarray(s["Masses"], np.float64) * 1e10 / h) * U.Msun
    else:
        xyz_s = np.zeros((0, 3)) * U.kpc
        m_s = np.zeros(0) * U.Msun

    return GalaxyGas(
        xyz_g=xyz_g, vxyz_g=vxyz_g, mH_neutral_g=mH_neutral, mgas_g=mgas * U.Msun,
        T_g=T, hsm_g=hsm, xyz_s=xyz_s, m_s=m_s, L_hat=L_hat,
        redshift=float(z), a=float(a), h=float(h),
        subhalo_id=int(subhalo_id), snap=int(snap))
