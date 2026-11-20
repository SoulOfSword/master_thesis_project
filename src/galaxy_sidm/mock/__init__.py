"""Mock observations: sphviewer2 + MARTINI + BBarolo.

Per galaxy: load gas/stars (gas), render face/edge maps (sphview), build a
MARTINI datacube (cube), fit it with BBarolo (barolo), draw the moment/PV
figure (kinematics).
"""

from .gas import GalaxyGas, load_galaxy_gas
from .sphview import render_face_edge
from .cube import (CubeParams, build_cube, disc_coords, disc_cut, disc_normal, disc_velocities,
                   hi_radius_kpc, layer_radii, rule_half_kpc, source_mask)
from .barolo import BaroloResult, run_bbarolo
from .kinematics import plot_kinematics
from .tables import sparc_mod_vflat, sparc_rings, vcirc_at, vcirc_curves

__all__ = [
    "GalaxyGas", "load_galaxy_gas", "render_face_edge", "hi_radius_kpc",
    "CubeParams", "build_cube", "disc_coords", "disc_cut", "disc_normal", "disc_velocities",
    "rule_half_kpc",
    "source_mask", "layer_radii", "BaroloResult", "run_bbarolo",
    "plot_kinematics", "sparc_mod_vflat", "sparc_rings", "vcirc_at", "vcirc_curves",
]
