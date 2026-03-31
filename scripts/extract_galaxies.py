"""Extract individual galaxy HDF5 files from TNG snapshots for MORDOR.

Writes one Gadget-format HDF5 file per subhalo, containing all particle types
(gas, DM, stars, BH) belonging to that subhalo. The output files are readable
by pynbody and compatible with MORDOR's morphological decomposition pipeline.

Usage:
    python scripts/extract_galaxies.py --basepath data/TNG50-4/output --snap 99 \
        --outdir data/TNG50-4/galaxies --min-stars 10000

    # Or specify subhalo IDs directly:
    python scripts/extract_galaxies.py --basepath data/TNG50-4/output --snap 99 \
        --outdir data/TNG50-4/galaxies --subhalo-ids 0 308 503
"""

import argparse
import os
from pathlib import Path

import h5py
import illustris_python as il
import numpy as np


# Particle types to extract and their TNG field names.
# These are the fields MORDOR/pynbody need at minimum.
PARTICLE_FIELDS = {
    "gas": {
        "type_idx": 0,
        "fields": [
            "Coordinates", "Velocities", "Masses", "ParticleIDs",
            "Potential", "Density", "InternalEnergy",
        ],
    },
    "dm": {
        "type_idx": 1,
        "fields": [
            "Coordinates", "Velocities", "ParticleIDs", "Potential",
        ],
    },
    "stars": {
        "type_idx": 4,
        "fields": [
            "Coordinates", "Velocities", "Masses", "ParticleIDs",
            "Potential", "GFM_StellarFormationTime", "GFM_Metallicity",
        ],
    },
    "bh": {
        "type_idx": 5,
        "fields": [
            "Coordinates", "Velocities", "Masses", "ParticleIDs",
            "Potential",
        ],
    },
}


def read_snapshot_metadata(basepath, snap):
    """Read the header and per-field unit attributes from the first snapshot chunk.

    Args:
        basepath: Path to the TNG output directory.
        snap: Snapshot number.

    Returns:
        Tuple of (header_dict, field_attrs_dict). field_attrs_dict maps
        'PartTypeX/FieldName' to a dict of HDF5 dataset attributes (used by
        pynbody for unit inference).
    """
    snap_dir = Path(basepath) / f"snapdir_{snap:03d}"
    first_chunk = sorted(snap_dir.glob(f"snap_{snap:03d}.*.hdf5"))[0]
    field_attrs = {}
    with h5py.File(first_chunk, "r") as f:
        header = dict(f["Header"].attrs)
        for grp_name in f:
            if not grp_name.startswith("PartType"):
                continue
            for ds_name in f[grp_name]:
                ds = f[grp_name][ds_name]
                if ds.attrs:
                    field_attrs[f"{grp_name}/{ds_name}"] = dict(ds.attrs)
    return header, field_attrs


def load_subhalo_particles(basepath, snap, sub_id):
    """Load all particle data for a subhalo.

    Args:
        basepath: Path to the TNG output directory.
        snap: Snapshot number.
        sub_id: Subhalo index.

    Returns:
        Dict mapping particle type name to dict of field arrays, plus 'count'
        per type. Returns None for types with zero particles.
    """
    particles = {}
    for ptype_name, ptype_info in PARTICLE_FIELDS.items():
        fields = ptype_info["fields"]
        # Load with 2+ fields to ensure dict return from illustris_python
        try:
            data = il.snapshot.loadSubhalo(
                basepath, snap, sub_id, ptype_name, fields=fields,
            )
        except Exception:
            data = {"count": 0}

        if isinstance(data, np.ndarray):
            # Single-field fallback: illustris_python returns raw array
            data = {fields[0]: data, "count": len(data)}

        if data["count"] > 0:
            particles[ptype_name] = data
        else:
            particles[ptype_name] = None

    return particles


def write_galaxy_hdf5(filepath, particles, header, field_attrs, sub_id,
                      dm_particle_mass):
    """Write a single-galaxy Gadget-format HDF5 file readable by pynbody.

    Args:
        filepath: Output file path.
        particles: Dict from load_subhalo_particles.
        header: Original snapshot header dict.
        field_attrs: Per-field HDF5 attributes from the original snapshot,
            used by pynbody for unit inference.
        sub_id: Subhalo ID (stored in header for reference).
        dm_particle_mass: DM particle mass from the MassTable.
    """
    num_part = np.zeros(6, dtype=np.int64)
    for ptype_name, ptype_info in PARTICLE_FIELDS.items():
        if particles[ptype_name] is not None:
            num_part[ptype_info["type_idx"]] = particles[ptype_name]["count"]

    with h5py.File(filepath, "w") as f:
        # Write header matching Gadget HDF5 conventions
        hdr = f.create_group("Header")
        hdr.attrs["NumPart_ThisFile"] = num_part.astype(np.int32)
        hdr.attrs["NumPart_Total"] = num_part.astype(np.uint32)
        hdr.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        hdr.attrs["NumFilesPerSnapshot"] = 1

        # Copy cosmological parameters from the original snapshot
        for key in [
            "BoxSize", "HubbleParam", "Omega0", "OmegaBaryon",
            "OmegaLambda", "Redshift", "Time",
            "Flag_Cooling", "Flag_DoublePrecision", "Flag_Feedback",
            "Flag_Metals", "Flag_Sfr", "Flag_StellarAge",
            "UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s",
        ]:
            if key in header:
                hdr.attrs[key] = header[key]

        # MassTable: DM mass is fixed, others vary per-particle
        mass_table = np.zeros(6, dtype=np.float64)
        mass_table[1] = dm_particle_mass
        hdr.attrs["MassTable"] = mass_table

        # Store subhalo ID for provenance
        hdr.attrs["SubhaloID"] = sub_id

        # Write particle data
        for ptype_name, ptype_info in PARTICLE_FIELDS.items():
            if particles[ptype_name] is None:
                continue

            type_idx = ptype_info["type_idx"]
            grp = f.create_group(f"PartType{type_idx}")

            for field in ptype_info["fields"]:
                if field in particles[ptype_name]:
                    ds = grp.create_dataset(
                        field, data=particles[ptype_name][field],
                    )
                    # Copy unit attributes so pynbody can infer units
                    attr_key = f"PartType{type_idx}/{field}"
                    if attr_key in field_attrs:
                        for ak, av in field_attrs[attr_key].items():
                            ds.attrs[ak] = av


def extract_galaxies(basepath, snap, outdir, subhalo_ids):
    """Extract individual galaxy files for a list of subhalos.

    Args:
        basepath: Path to the TNG output directory.
        snap: Snapshot number.
        outdir: Output directory for galaxy files.
        subhalo_ids: Array of subhalo indices to extract.
    """
    os.makedirs(outdir, exist_ok=True)
    header, field_attrs = read_snapshot_metadata(basepath, snap)
    dm_mass = header["MassTable"][1]

    print(f"Extracting {len(subhalo_ids)} galaxies from {basepath}, snap {snap}")
    print(f"Output: {outdir}/Gal_XXXXXX.hdf5\n")

    for i, sub_id in enumerate(subhalo_ids):
        filename = f"Gal_{sub_id:06d}.hdf5"
        filepath = os.path.join(outdir, filename)

        particles = load_subhalo_particles(basepath, snap, sub_id)

        n_stars = (
            particles["stars"]["count"] if particles["stars"] is not None else 0
        )

        write_galaxy_hdf5(filepath, particles, header, field_attrs, sub_id,
                          dm_mass)

        print(
            f"  [{i+1}/{len(subhalo_ids)}] {filename}  "
            f"(N_star={n_stars:,}, "
            f"N_dm={particles['dm']['count'] if particles['dm'] else 0:,}, "
            f"N_gas={particles['gas']['count'] if particles['gas'] else 0:,})"
        )

    print(f"\nDone. {len(subhalo_ids)} files written to {outdir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Extract individual galaxy HDF5 files from TNG snapshots.",
    )
    parser.add_argument(
        "--basepath", required=True,
        help="Path to TNG output directory (e.g. data/TNG50-4/output)",
    )
    parser.add_argument("--snap", type=int, required=True, help="Snapshot number")
    parser.add_argument(
        "--outdir", required=True,
        help="Output directory for galaxy files",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subhalo-ids", type=int, nargs="+",
        help="Specific subhalo IDs to extract",
    )
    group.add_argument(
        "--min-stars", type=int,
        help="Extract all flagged subhalos with at least this many star particles",
    )

    args = parser.parse_args()

    if args.subhalo_ids is not None:
        subhalo_ids = np.array(args.subhalo_ids)
    else:
        subs = il.groupcat.loadSubhalos(
            args.basepath, args.snap,
            fields=["SubhaloLenType", "SubhaloFlag"],
        )
        n_star = subs["SubhaloLenType"][:, 4]
        flag = subs["SubhaloFlag"]
        mask = flag & (n_star >= args.min_stars)
        subhalo_ids = np.where(mask)[0]
        print(f"Found {len(subhalo_ids)} subhalos with N_star >= {args.min_stars}")

    extract_galaxies(args.basepath, args.snap, args.outdir, subhalo_ids)


if __name__ == "__main__":
    main()
