"""Extract one subhalo's bound particles into a standalone Gadget-HDF5
file that pynbody (and therefore MORDOR) can load directly."""

from pathlib import Path
import numpy as np
import h5py
import illustris_python as il

from ..data.aida_tng import get_snap_scale_factor


_FIELDS_BY_TYPE = {
    0: ["Coordinates", "Velocities", "Masses", "ParticleIDs", "Potential"],
    1: ["Coordinates", "Velocities", "ParticleIDs", "Potential"],
    4: ["Coordinates", "Velocities", "Masses", "ParticleIDs", "Potential",
        "GFM_InitialMass", "GFM_StellarFormationTime"],
    5: ["Coordinates", "Velocities", "Masses", "ParticleIDs", "Potential"],
}


def _read_header(run_path, snap):
    run_path = Path(run_path)
    snapdir = run_path / "output" / f"snapdir_{snap:03d}"
    files = sorted(snapdir.glob(f"snap_{snap:03d}.*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No snapshot files under {snapdir}")
    with h5py.File(files[0], "r") as f:
        return dict(f["Header"].attrs)


def extract_galaxy_hdf5(base_path, snap, subhalo_id, out_path,
                        h=0.6774, soft_phys_kpc=0.57,
                        part_types=(0, 1, 4, 5), overwrite=False):
    """Write a single subhalo's bound particles as a Gadget-HDF5 file.

    Units are kept in TNG code units so pynbody applies its standard
    conversions via the header attributes. MORDOR's `cosmo_sim` mode
    uses the per-particle `Potential` field stored here.

    Args:
        base_path: Path to the simulation `output/` directory.
        snap: Snapshot number.
        subhalo_id: Subhalo index. For FoF central galaxies, pass
            `GroupFirstSub` of the FoF group. Particles loaded are those
            bound to this subhalo by SUBFIND, excluding satellites and
            intra-FoF unbound stars.
        out_path: Destination HDF5 path.
        h: Little-h. Used for logging only; values stay in code units.
        soft_phys_kpc: Plummer-equivalent physical softening in kpc,
            written as a per-particle `SofteningLength` dataset for
            pynbody. For AIDA 50/A at z <= 1 use 0.57.
        part_types: Particle types to include.
        overwrite: Re-extract even if out_path exists.

    Returns:
        Path to the written HDF5 file.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_path = Path(base_path).parent
    hdr = _read_header(run_path, snap)
    z, a = get_snap_scale_factor(run_path, snap)
    a_sqrt = np.sqrt(a)

    type_data = {}
    for pt in part_types:
        requested = [f for f in _FIELDS_BY_TYPE[pt]]
        data = il.snapshot.loadSubhalo(base_path, snap, subhalo_id, pt, fields=requested)
        if not isinstance(data, dict) or data.get("count", 0) == 0:
            continue
        n = data["count"]
        if pt == 1 and "Masses" not in data:
            m_dm = float(hdr["MassTable"][1])
            data["Masses"] = np.full(n, m_dm, dtype=np.float32)
        type_data[pt] = data

    if not type_data:
        raise RuntimeError(f"No particles in subhalo {subhalo_id} at snap {snap}")

    num_part = np.zeros(6, dtype=np.int64)
    for pt, d in type_data.items():
        num_part[pt] = int(d["count"])

    with h5py.File(out_path, "w") as f:
        H = f.create_group("Header")
        H.attrs["BoxSize"] = float(hdr["BoxSize"])
        H.attrs["HubbleParam"] = float(hdr["HubbleParam"])
        H.attrs["Omega0"] = float(hdr["Omega0"])
        H.attrs["OmegaLambda"] = float(hdr["OmegaLambda"])
        H.attrs["OmegaBaryon"] = float(hdr.get("OmegaBaryon", 0.0486))
        H.attrs["Redshift"] = float(hdr["Redshift"])
        H.attrs["Time"] = float(hdr["Time"])
        H.attrs["NumFilesPerSnapshot"] = 1
        H.attrs["NumPart_Total"] = num_part.astype(np.uint32)
        H.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        H.attrs["NumPart_ThisFile"] = num_part.astype(np.int32)
        H.attrs["MassTable"] = np.asarray(hdr["MassTable"], dtype=np.float64)
        H.attrs["UnitLength_in_cm"] = float(hdr.get("UnitLength_in_cm", 3.085678e21))
        H.attrs["UnitMass_in_g"] = float(hdr.get("UnitMass_in_g", 1.989e43))
        H.attrs["UnitVelocity_in_cm_per_s"] = float(hdr.get("UnitVelocity_in_cm_per_s", 1e5))
        H.attrs["Flag_DoublePrecision"] = 0
        H.attrs["Flag_Sfr"] = 1
        H.attrs["Flag_Cooling"] = 1
        H.attrs["Flag_StellarAge"] = 1
        H.attrs["Flag_Metals"] = 1
        H.attrs["Flag_Feedback"] = 1

        soft_code = soft_phys_kpc * h / max(a, 1e-8)
        for pt, d in type_data.items():
            g = f.create_group(f"PartType{pt}")
            for field in _FIELDS_BY_TYPE[pt]:
                if field not in d:
                    continue
                g.create_dataset(field, data=np.asarray(d[field]))
            g.create_dataset(
                "SofteningLength",
                data=np.full(int(d["count"]), soft_code, dtype=np.float32),
            )
    return out_path
