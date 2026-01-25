"""Halo and subhalo data structures."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Halo:
    """Container for halo properties.

    Attributes
    ----------
    halo_id : int
        Unique identifier for this halo.
    M200 : float
        Virial mass in Msun (mass within R200).
    R200 : float
        Virial radius in kpc.
    Vmax : float
        Maximum circular velocity in km/s.
    M_star : float
        Stellar mass in Msun.
    R_half : float
        Stellar half-mass radius in kpc.
    V_rot : float, optional
        Rotation velocity at R_half in km/s.
    sigma_star : float, optional
        Stellar velocity dispersion in km/s.
    j_star : float, optional
        Specific stellar angular momentum in kpc km/s.
    """

    halo_id: int
    M200: float
    R200: float
    Vmax: float
    M_star: float
    R_half: float
    V_rot: float | None = None
    sigma_star: float | None = None
    j_star: float | None = None
    simulation_type: str = "sidm"  # "sidm" or "cdm"


@dataclass
class HaloCatalog:
    """Collection of halos with array-based access.

    Provides both object-oriented access via `halos` list and
    array-based access for vectorized operations.
    """

    halos: list[Halo] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.halos)

    def __getitem__(self, idx: int) -> Halo:
        return self.halos[idx]

    @property
    def M_star(self):
        """Array of stellar masses."""
        return np.array([h.M_star for h in self.halos])

    @property
    def V_rot(self):
        """Array of rotation velocities."""
        return np.array([h.V_rot for h in self.halos])

    @property
    def R_half(self):
        """Array of half-mass radii."""
        return np.array([h.R_half for h in self.halos])

    @property
    def j_star(self):
        """Array of specific angular momenta."""
        return np.array([h.j_star for h in self.halos])

    def select(self, mask):
        """Return a new catalog with only halos where mask is True."""
        return HaloCatalog(halos=[h for h, m in zip(self.halos, mask) if m])
