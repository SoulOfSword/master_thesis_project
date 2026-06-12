"""Mock observations: sphviewer2 + MARTINI + BBarolo.

Per galaxy: load gas/stars (gas), render face/edge maps (sphview), build a
MARTINI datacube (cube), fit it with BBarolo (barolo), draw the moment/PV
figure (kinematics).
"""

from .gas import GalaxyGas, load_galaxy_gas
from .sphview import render_face_edge
from .cube import CubeParams, build_cube
from .barolo import BaroloResult, run_bbarolo
from .kinematics import plot_kinematics

__all__ = [
    "GalaxyGas", "load_galaxy_gas", "render_face_edge",
    "CubeParams", "build_cube", "BaroloResult", "run_bbarolo",
    "plot_kinematics",
]
