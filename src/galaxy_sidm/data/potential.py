"""Gravity of one subhalo's particles, for circular velocities.

load_subhalo_potential gives the snapshot's stored Potential at the particles.
Positions in proper kpc centred on the subhalo (SubhaloPos, the potential
minimum; periodic box), potential in physical (km/s)^2. TNG stores Potential in
(km/s)^2/a, so it is divided by the scale factor (as temet's
codePotentialToEscapeVelKms does).

load_subhalo_particles gives the particles as sources of gravity, with their
masses and AREPO's softening (for observables.vcirc2_components). The
softening rules follow the public AREPO source with the run's parameters
(read_softening). All softening lengths here are the spline support h = 2.8 x
the Plummer-equivalent softening eps; gravity is Newtonian beyond h.
    DM, stars, wind: eps = SofteningComovingType (comoving), capped at
        SofteningMaxPhysType (physical), of the type SofteningTypeOfPartType.
    Gas: h = GasSoftFactor x r_cell, r_cell = (3 V / 4 pi)^(1/3) with
        V = Masses / Density, rounded in log to the nearest step of the ladder
        2.8 MinimumComovingHydroSoftening x AdaptiveHydroSofteningSpacing^k
        (comoving, no physical cap). GasSoftFactor x r_cell is h, not eps, in
        AREPO's code (grav_softening.c), which the stored Potential of gas cells
        confirms in all three runs.
    Black holes (INDIVIDUAL_GRAVITY_SOFTENING): the softening type 1-5 closest
        in log to SofteningComovingType1 x (m / m_DM)^(1/3).
"""

from pathlib import Path

import h5py
import numpy as np
import illustris_python as il

from .aida_tng import get_snap_scale_factor

# AREPO: spline support h = 2.8 x Plummer-equivalent softening
H_OVER_EPS = 2.8

# Components of load_subhalo_particles, in the order of their index
COMPONENTS = ("dm", "stars", "gas", "wind", "bh")

# Code units the softening rules and observables.G_AREPO assume
# (kpc/h, 1e10 Msun/h, km/s)
_UNITS = {"UnitLength_in_cm": 3.085678e21, "UnitMass_in_g": 1.989e43,
          "UnitVelocity_in_cm_per_s": 1e5}


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


def read_softening(base_path, snap):
    """Gravitational softening settings of a snapshot, from the Header,
    Parameters and Config of its first file.

    Args:
        base_path: simulation `output/` directory.
        snap: snapshot number.

    Returns:
        dict with a, redshift, hubble, box (ckpc/h), mass_table (code units),
        params (the softening parameters), n_gas_soft (steps of the gas
        softening ladder), bh_mass_dependent (INDIVIDUAL_GRAVITY_SOFTENING
        covers black holes) and gas_position, the field where gas cells sit
        for gravity ('CenterOfMass' with CELL_CENTER_GRAVITY, else
        'Coordinates').

    Raises:
        KeyError if a header entry or parameter is missing; ValueError if the
        run uses settings these rules do not reproduce.
    """
    path = il.snapshot.snapPath(str(base_path), snap)
    with h5py.File(path, "r") as f:
        header = dict(f["Header"].attrs)
        params = dict(f["Parameters"].attrs)
        config = dict(f["Config"].attrs)

    keys = (["GasSoftFactor", "MinimumComovingHydroSoftening", "AdaptiveHydroSofteningSpacing"]
            + [f"Softening{kind}Type{t}" for kind in ("Comoving", "MaxPhys") for t in range(6)]
            + [f"SofteningTypeOfPartType{t}" for t in range(6)])
    missing = [k for k in keys + ["ComovingIntegrationOn", "GravityConstantInternal"]
               if k not in params]
    missing += [k for k in ["Time", "Redshift", "HubbleParam", "BoxSize", "MassTable", *_UNITS]
                if k not in header]
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    for k, v in _UNITS.items():
        if not np.isclose(header[k], v, rtol=1e-6):
            raise ValueError(f"{path}: {k} = {header[k]}, the softening rules assume {v}")
    if int(params["ComovingIntegrationOn"]) != 1 or float(params["GravityConstantInternal"]) != 0.0:
        raise ValueError(f"{path}: needs comoving integration and AREPO's own G")
    for flag in ("ADAPTIVE_HYDRO_SOFTENING", "MULTIPLE_NODE_SOFTENING"):
        if flag not in config:
            raise ValueError(f"{path}: Config lacks {flag}, which the gas and BH rules assume")
    if "NSOFTTYPES" in config and int(config["NSOFTTYPES"]) != 6:
        raise ValueError(f"{path}: NSOFTTYPES = {config['NSOFTTYPES']}, the rules assume 6")
    individual = int(config["INDIVIDUAL_GRAVITY_SOFTENING"]) if "INDIVIDUAL_GRAVITY_SOFTENING" in config else 0
    if individual not in (0, 32):
        raise ValueError(f"{path}: INDIVIDUAL_GRAVITY_SOFTENING = {individual}; "
                         f"only black holes (32) are handled")
    return dict(
        a=float(header["Time"]), redshift=float(header["Redshift"]),
        hubble=float(header["HubbleParam"]), box=float(header["BoxSize"]),
        mass_table=np.asarray(header["MassTable"], dtype=np.float64),
        params={k: float(params[k]) for k in keys},
        # AREPO's default length of the ladder when Config does not set it
        n_gas_soft=int(config["NSOFTTYPES_HYDRO"]) if "NSOFTTYPES_HYDRO" in config else 64,
        bh_mass_dependent=individual == 32,
        gas_position="CenterOfMass" if "CELL_CENTER_GRAVITY" in config else "Coordinates",
    )


def softening_eps(soft, soft_type):
    """Plummer-equivalent softening [kpc] of a fixed softening type: comoving
    SofteningComovingType, capped at SofteningMaxPhysType physical."""
    p = soft["params"]
    return min(p[f"SofteningComovingType{soft_type}"] * soft["a"],
               p[f"SofteningMaxPhysType{soft_type}"]) / soft["hubble"]


def h_collisionless(soft, part_type):
    """Spline support h [kpc] of a particle type with a fixed softening (DM,
    stars and wind)."""
    return H_OVER_EPS * softening_eps(soft, int(soft["params"][f"SofteningTypeOfPartType{part_type}"]))


def h_gas(soft, mass, density):
    """Spline support h [kpc] of gas cells (adaptive softening).

    Args:
        soft: read_softening output.
        mass, density: Masses and Density of the cells (code units).
    """
    p = soft["params"]
    r_cell = (3.0 * np.asarray(mass, dtype=np.float64)
              / (4.0 * np.pi * np.asarray(density, dtype=np.float64))) ** (1.0 / 3.0)  # ckpc/h
    h0 = H_OVER_EPS * p["MinimumComovingHydroSoftening"] # first step of the ladder
    step = p["AdaptiveHydroSofteningSpacing"]
    # nearest step in log of GasSoftFactor x r_cell, the first step if below it
    k = np.floor(0.5 + np.log(np.maximum(p["GasSoftFactor"] * r_cell, h0) / h0) / np.log(step))
    k = np.minimum(k, soft["n_gas_soft"] - 1)
    return h0 * step ** k * soft["a"] / soft["hubble"]


def h_bh(soft, mass):
    """Spline support h [kpc] of black holes with mass-dependent softening.

    The softening type 1-5 closest in log to SofteningComovingType1 x
    max(1, (m / m_DM)^(1/3)), the first one on a tie, as AREPO's
    get_softening_type_from_mass (type 0 is left to the gas).

    Args:
        soft: read_softening output.
        mass: dynamical Masses of the black holes (code units).
    """
    p, a = soft["params"], soft["a"]
    m_dm = soft["mass_table"][1]
    if m_dm <= 0:
        raise ValueError("MassTable[1] = 0: the BH softening scales with the DM particle mass")
    table = np.array([min(p[f"SofteningComovingType{t}"], p[f"SofteningMaxPhysType{t}"] / a)
                      for t in range(1, 6)])  # current comoving softening of types 1-5
    want = p["SofteningComovingType1"] * np.maximum(
        1.0, (np.asarray(mass, dtype=np.float64) / m_dm) ** (1.0 / 3.0))
    best = np.argmin(np.abs(np.log(want)[:, None] - np.log(table)[None, :]), axis=1)
    return H_OVER_EPS * table[best] * a / soft["hubble"]


def load_subhalo_particles(base_path, snap, subhalo_id, soft):
    """The SUBFIND-bound particles of a subhalo as sources of gravity.

    Components, by index into COMPONENTS: dark matter, stars, gas (all
    cells), wind particles (PartType4 with GFM_StellarFormationTime <= 0) and
    black holes (with their dynamical Masses). Gas cells sit at
    soft['gas_position']. A component the subhalo lacks has no particles.

    Args:
        base_path: simulation `output/` directory.
        snap: snapshot number.
        subhalo_id: subhalo index.
        soft: read_softening(base_path, snap).

    Returns:
        dict with xyz (N, 3) positions in kpc centred on SubhaloPos (periodic
        box), mass (N,) in Msun, h (N,) spline support in kpc and comp (N,)
        the component index.
    """
    base_path = str(base_path)
    a, hub, box = soft["a"], soft["hubble"], soft["box"]
    pos0 = np.asarray(il.groupcat.loadSingle(base_path, snap, subhaloID=subhalo_id)["SubhaloPos"],
                      dtype=np.float64)                              # ckpc/h
    out = {"xyz": [], "mass": [], "h": [], "comp": []}

    def load(part_type, fields):
        d = il.snapshot.loadSubhalo(base_path, snap, subhalo_id, part_type, fields=fields)
        if isinstance(d, np.ndarray): # illustris_python returns a bare array for one field
            d = {fields[0]: d, "count": len(d)}
        return d

    def add(name, coords, mass, h):
        dx = np.asarray(coords, dtype=np.float64) - pos0
        dx -= box * np.round(dx / box)
        out["xyz"].append(dx * a / hub)
        out["mass"].append(np.asarray(mass, dtype=np.float64) * 1e10 / hub)
        out["h"].append(np.broadcast_to(h, len(dx)).astype(np.float64))
        out["comp"].append(np.full(len(dx), COMPONENTS.index(name)))

    d = load(1, ["Coordinates"])
    if d["count"]:
        if soft["mass_table"][1] <= 0:
            raise ValueError("MassTable[1] = 0: no DM particle mass in the header")
        add("dm", d["Coordinates"], np.full(d["count"], soft["mass_table"][1]), h_collisionless(soft, 1))
    d = load(4, ["Coordinates", "Masses", "GFM_StellarFormationTime"])
    if d["count"]:
        wind = d["GFM_StellarFormationTime"] <= 0
        for name, sel in (("stars", ~wind), ("wind", wind)):
            add(name, d["Coordinates"][sel], d["Masses"][sel], h_collisionless(soft, 4))
    d = load(0, [soft["gas_position"], "Masses", "Density"])
    if d["count"]:
        add("gas", d[soft["gas_position"]], d["Masses"], h_gas(soft, d["Masses"], d["Density"]))
    d = load(5, ["Coordinates", "Masses"])
    if d["count"]:
        add("bh", d["Coordinates"], d["Masses"],
            h_bh(soft, d["Masses"]) if soft["bh_mass_dependent"] else h_collisionless(soft, 5))
    if not out["xyz"]:
        raise RuntimeError(f"subhalo {subhalo_id} snap {snap}: no particles")
    return {k: np.concatenate(v) for k, v in out.items()}
