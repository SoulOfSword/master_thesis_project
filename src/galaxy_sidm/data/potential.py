"""Gravitational potential of one subhalo's particles, for circular velocities.

Positions in proper kpc centred on the subhalo (SubhaloPos, the potential
minimum; periodic box), potential in physical (km/s)^2. TNG stores Potential in
(km/s)^2/a, so it is divided by the scale factor (as temet's
codePotentialToEscapeVelKms does).
"""

from pathlib import Path

import numpy as np
import illustris_python as il

from .aida_tng import get_snap_scale_factor


def load_subhalo_potential(base_path, snap, subhalo_id, h=0.6774,
                           part_types=(0, 1, 4, 5)):
    """Positions and potential of the SUBFIND-bound particles of a subhalo.

    Args:
        base_path: simulation `output/` directory.
        snap: snapshot number.
        subhalo_id: subhalo index.
        h: little-h.
        part_types: particle types to include (gas, DM, stars, black holes).

    Returns:
        (xyz, phi): (N, 3) positions in kpc and (N,) potential in (km/s)^2.
    """
    base_path = str(base_path)
    _, a = get_snap_scale_factor(Path(base_path).parent, snap)
    box = float(il.groupcat.loadHeader(base_path, snap)["BoxSize"])  # ckpc/h
    pos0 = np.asarray(il.groupcat.loadSingle(base_path, snap, subhaloID=subhalo_id)["SubhaloPos"],
                      dtype=np.float64)                              # ckpc/h

    xyz, phi = [], []
    for pt in part_types:
        d = il.snapshot.loadSubhalo(base_path, snap, subhalo_id, pt,
                                    fields=["Coordinates", "Potential"])
        if not isinstance(d, dict) or d.get("count", 0) == 0:
            continue
        dx = np.asarray(d["Coordinates"], dtype=np.float64) - pos0
        dx -= box * np.round(dx / box)
        xyz.append(dx * a / h)
        phi.append(np.asarray(d["Potential"], dtype=np.float64) / a)
    if not xyz:
        raise RuntimeError(f"subhalo {subhalo_id} snap {snap}: no particles")
    return np.concatenate(xyz), np.concatenate(phi)
