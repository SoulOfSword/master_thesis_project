"""Invoke MORDOR on per-galaxy HDF5 files."""

from pathlib import Path
import os
import shutil
import subprocess

import numpy as np


DEFAULT_MORDOR_DIR = Path.home() / "software" / "mordor"


def format_mordor_row(gal, filename, jcut=0.5):
    """Return MORDOR's ASCII output row for a decomposed pynbody snap.

    Mirrors the column layout written by `mordor.py`'s CLI in list mode:
    `filename Mstar Munbound Mthin Mthick Mbulge Mpbulge Mhalo IsDisc
     Ethin Ethick Ebulge Epbulge Ehalo Cthin Cthick Cbulge Cpbulge Chalo`.

    Args:
        gal: Decomposed pynbody snapshot from `run_mordor_single`. Must
            have `gal.s['morph']`, `gal.s['te']`, `gal.s['jz_by_jzcirc']`,
            and `gal.s['mass']` populated.
        filename: Filename string written as the first column.
        jcut: Disc threshold, default 0.5 (matches Zana+22 + mordor.py).

    Returns:
        Single line as a string (no trailing newline).
    """
    morph = np.asarray(gal.s["morph"])
    te    = np.asarray(gal.s["te"])
    eta   = np.asarray(gal.s["jz_by_jzcirc"])
    mass  = np.asarray(gal.s["mass"])

    Mstar    = float(mass.sum())
    Munbound = float(mass[morph == 0].sum())
    Mthin    = float(mass[morph == 1].sum())
    Mthick   = float(mass[morph == 2].sum())
    Mpbulge  = float(mass[morph == 3].sum())
    Mbulge   = float(mass[morph == 4].sum())
    Mhalo    = float(mass[morph == 5].sum())

    Mbound = Mthin + Mthick + Mpbulge + Mbulge + Mhalo
    flag = int((Mthin + Mthick + Mpbulge) / Mbound > jcut) if Mbound > 0 else 0

    classified = morph != 0
    Emax = float(np.abs(te[classified]).max()) if classified.any() else 1.0

    def _mean(arr, sel):
        return float(arr[sel].mean()) if sel.any() else 0.0

    Ethin   = _mean(te / Emax, morph == 1)
    Ethick  = _mean(te / Emax, morph == 2)
    Ebulge  = _mean(te / Emax, morph == 4)
    Epbulge = _mean(te / Emax, morph == 3)
    Ehalo   = _mean(te / Emax, morph == 5)

    Cthin   = _mean(eta, morph == 1)
    Cthick  = _mean(eta, morph == 2)
    Cbulge  = _mean(eta, morph == 4)
    Cpbulge = _mean(eta, morph == 3)
    Chalo   = _mean(eta, morph == 5)

    return (
        f"{filename} {Mstar:g} {Munbound:g} "
        f"{Mthin:g} {Mthick:g} {Mbulge:g} {Mpbulge:g} {Mhalo:g} {flag:d} "
        f"{Ethin:f} {Ethick:f} {Ebulge:f} {Epbulge:f} {Ehalo:f} "
        f"{Cthin:f} {Cthick:f} {Cbulge:f} {Cpbulge:f} {Chalo:f}"
    )


def write_filelist(hdf5_paths, list_path):
    """Write one HDF5 path per line for MORDOR's list mode."""
    list_path = Path(list_path)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(list_path, "w") as f:
        for p in hdf5_paths:
            f.write(f"{Path(p).resolve()}\n")
    return list_path


def _gsoft_wrapper_script(mordor_dir, soft_kpc):
    """Write a small driver that monkey-patches MORDOR's gsoft() for AIDA.

    Returns the path to the driver. Safe to call repeatedly.
    """
    mordor_dir = Path(mordor_dir).expanduser().resolve()
    driver = mordor_dir / "mordor_aida_driver.py"
    driver.write_text(f"""import sys, os
sys.path.insert(0, {str(mordor_dir)!r})
import mordor as _m
_soft = {float(soft_kpc)!r}
def _gsoft(z, box=50):
    return _soft if z <= 1 else _soft * 2.0 / (1.0 + z)
_m.gsoft = _gsoft
sys.argv = ['mordor.py'] + sys.argv[1:]
exec(open(os.path.join({str(mordor_dir)!r}, 'mordor.py')).read(), {{'__name__': '__main__'}})
""")
    return driver


def run_mordor_batch(filelist_path, mode="cosmo_sim",
                     out_prefix="morphology_aida",
                     mordor_dir=DEFAULT_MORDOR_DIR,
                     python_exe=None, timeout=None,
                     soft_phys_kpc=0.57, output_dir=None):
    """Run MORDOR in list mode. Returns the path to the ASCII output file.

    Args:
        filelist_path: Text file with one galaxy HDF5 path per line.
        mode: MORDOR potential mode ('cosmo_sim', 'tree', 'direct', ...).
        out_prefix: Prefix for MORDOR's output file.
        mordor_dir: Directory containing mordor.py.
        python_exe: Python interpreter to use. Defaults to sys.executable.
        timeout: Seconds; None disables.
        soft_phys_kpc: Physical softening in kpc for the gsoft() patch.
        output_dir: Where to move MORDOR's output ASCII (defaults next to
            the filelist).
    """
    filelist_path = Path(filelist_path).resolve()
    mordor_dir = Path(mordor_dir).expanduser().resolve()
    if python_exe is None:
        import sys as _sys
        python_exe = _sys.executable

    driver = _gsoft_wrapper_script(mordor_dir, soft_phys_kpc)
    cmd = [str(python_exe), str(driver),
           str(filelist_path), "-l",
           "--mode", mode,
           "--OutPrefix", out_prefix]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    subprocess.run(cmd, cwd=str(mordor_dir), env=env,
                   check=True, timeout=timeout)

    generated = mordor_dir / f"{out_prefix}_{filelist_path.name}"
    if not generated.exists():
        alt = mordor_dir / f"{out_prefix}_{filelist_path.stem}"
        if alt.exists():
            generated = alt
        else:
            raise FileNotFoundError(
                f"MORDOR output not found at {generated}; check stdout."
            )

    dest_dir = Path(output_dir) if output_dir else filelist_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / generated.name
    shutil.move(str(generated), str(dest))
    return dest


def run_mordor_single(hdf5_path, mode="cosmo_sim", soft_phys_kpc=0.57,
                      mordor_dir=DEFAULT_MORDOR_DIR):
    """Decompose one galaxy and return the pynbody snap with morph set.

    Imports MORDOR's `decomposition` module and runs it in this Python
    process (no subprocess). The returned pynbody snapshot has
    `gal.s['morph']`, `gal.s['te']`, and `gal.s['jz_by_jzcirc']`
    populated — suitable for diagnostic plots in the (eta, E) plane.

    Args:
        hdf5_path: Per-galaxy HDF5 file (as produced by
            `extract_galaxy_hdf5`).
        mode: MORDOR potential mode. 'cosmo_sim' uses the snapshot's
            stored Potential field with a 1/a^2 fix.
        soft_phys_kpc: Plummer-equivalent softening in kpc; sets
            `gal['eps'] = 2.8 * soft_phys_kpc` per particle if missing.
        mordor_dir: Directory containing `decomposition.py`.

    Returns:
        Decomposed pynbody simulation snapshot.
    """
    import sys
    import numpy as np
    import pynbody

    mordor_dir = Path(mordor_dir).expanduser().resolve()
    if str(mordor_dir) not in sys.path:
        sys.path.insert(0, str(mordor_dir))
    import decomposition  # noqa: E402

    gal = pynbody.load(str(Path(hdf5_path).resolve()))
    gal.physical_units()

    eps_kpc = 2.8 * float(soft_phys_kpc)
    if "eps" not in gal:
        gal["eps"] = pynbody.array.SimArray(
            eps_kpc * np.ones_like(gal["x"], dtype=gal["x"].dtype), "kpc")

    if mode in ("cosmo_sim", "iso_sim"):
        gal["phi"] /= gal.properties["a"]**2
    elif mode == "tree":
        from kdtree import KDPotential
        gal["phi"] = KDPotential(gal["pos"], gal["mass"], gal["eps"], theta=0.5)
        gal["phi"].convert_units("km^2 s^-2")
    elif mode == "direct":
        from kdtree import BruteForcePotentialTarget
        gal["phi"] = BruteForcePotentialTarget(gal["pos"], gal["pos"],
                                                gal["mass"], gal["eps"])
        gal["phi"].units = pynbody.units.G * gal["mass"].units / gal["pos"].units
        gal["phi"].convert_units("km^2 s^-2")
    elif mode == "auxiliary":
        raise NotImplementedError("auxiliary mode requires potential_*.npy")

    pynbody.analysis.halo.center(gal, wrap=True, mode="hyb")

    def _hmr(stars):
        prof = pynbody.analysis.profile.Profile(stars, ndim=3, type="log")
        return float(np.min(
            prof["rbins"][prof["mass_enc"] > 0.5*prof["mass_enc"][-1]]))

    hmr = _hmr(gal.s)
    sc = pynbody.analysis.halo.center(gal.s, retcen=True, mode="hyb")
    if np.sqrt(np.sum(np.asarray(sc)**2)) > max(0.5*hmr, eps_kpc):
        try:
            pynbody.analysis.halo.center(gal.s, mode="hyb")
        except Exception:
            pynbody.analysis.halo.center(gal.s, mode="hyb", cen_size="3 kpc")
        hmr = _hmr(gal.s)

    size_ang = max(3*hmr, eps_kpc)
    pynbody.analysis.angmom.faceon(gal.s, disk_size=f"{size_ang} kpc",
                                    already_centered=True)

    decomposition.morph(gal, j_circ_from_r=False, LogInterp=False,
                        BoundOnly=True, Ecut=None, jThinMin=0.7,
                        mode=mode, dimcell=None)
    return gal
