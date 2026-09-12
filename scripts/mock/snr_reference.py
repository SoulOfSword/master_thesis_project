"""Reference signal-to-noise for the constant-S/N mock cubes.

What it does:
  1. builds the reference galaxy's MARTINI cube WITHOUT noise, with the same
     CubeParams as the pipeline (or reuses the cached copy of it),
  2. signal = cube.measure_signal: the mean brightness of the voxels that are at
     least 10% (CubeParams.signal_peak_fraction) of the brightest voxel -- the
     SAME function build_cube uses for every galaxy. Only voxels with emission
     enter, so the image size does not matter,
  3. reference_snr = signal / noise_rms, with noise_rms the noise this galaxy was
     built with (CubeParams.noise_rms = 1e-5, the value fed to MARTINI),
  4. writes config/snr_reference.yaml. build_galaxy.py reads it and gives every
     galaxy noise_rms = its own signal / reference_snr, i.e. the same signal/noise.

Run on a compute node (MARTINI is memory-hungry), from the project root:
  srun --partition=regular --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --time=00:30:00 python3 scripts/mock/snr_reference.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
import astropy.units as U
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config
from galaxy_sidm.mock import load_galaxy_gas, CubeParams, build_cube
from galaxy_sidm.mock.cube import measure_signal

sys.path.insert(0, str(ROOT / "scripts" / "mock"))
from build_galaxy import resolve_base_path, SNR_REFERENCE

# ---------------------------------------------------------------- EDIT ME ---
SUB_ID = 29759          # subhalo id
MODEL = "CDM"           # CDM / SIDM1 / vSIDM
SNAP = 33               # 17=z5, 21=z4, 25=z3, 33=z2, 50=z1, 67=z0.5
NCPU = 8                # cores for MARTINI's source insertion
REBUILD = False         # True: rebuild the noiseless cube even if it is cached
# -----------------------------------------------------------------------------


def same_setup(hdr, p):
    """True if a cached cube has this pixel size, beam and channels."""
    return (np.isclose(abs(hdr["CDELT1"]), p.px_size.to_value(U.deg))
            and np.isclose(hdr["BMAJ"], p.beam_fwhm.to_value(U.deg))
            and np.isclose(abs(hdr["CDELT3"]), p.channel_width.to_value(U.m / U.s))
            and hdr["NAXIS3"] == p.n_channels)


def main():
    cfg = load_config(None)
    z = {int(k): float(v) for k, v in cfg["snap_z"].items()}[SNAP]
    print(f"[snr] {MODEL} snap {SNAP} (z={z:g}) sub {SUB_ID}")

    p = CubeParams(add_noise=False)
    cache = (Path(cfg["paths"]["scratch_processed"]).parent / "tmp"
             / f"ref_nonoise_{MODEL}_{SNAP:03d}_{SUB_ID:06d}.fits")
    if cache.exists() and not REBUILD and same_setup(fits.getheader(cache), p):
        print(f"[snr] reusing noiseless cube {cache}")
    else:
        base_path = resolve_base_path(MODEL, SNAP, 1080, cfg)
        gas = load_galaxy_gas(base_path, SNAP, SUB_ID, h=float(cfg["cosmology"]["h"]))
        print(f"[snr] gas cells={len(gas.mH_neutral_g)} stars={len(gas.m_s)}")
        res = build_cube(gas, cache, p, ncpu=NCPU)
        print(f"[snr] noiseless cube ({res.npix}px) -> {cache}")

    cube = np.nan_to_num(np.asarray(fits.getdata(cache), float))
    signal = measure_signal(cube, p.signal_peak_fraction)
    if not np.isfinite(signal):
        print("[snr] the cube has no flux at all -- nothing to measure")
        return 1
    noise_rms = p.noise_rms.to_value(U.Jy / U.arcsec ** 2)
    reference_snr = signal / noise_rms

    sel = cube >= p.signal_peak_fraction * cube.max()
    print(f"[snr] cube {cube.shape[2]}x{cube.shape[1]}x{cube.shape[0]}, "
          f"brightest voxel = {cube.max():.4e} Jy/beam")
    print(f"[snr] signal = mean of the {sel.sum()} voxels >= "
          f"{p.signal_peak_fraction:g} x brightest ({100 * cube[sel].sum() / cube.sum():.0f}% "
          f"of the flux) = {signal:.6e} Jy/beam")
    print(f"[snr] reference_snr = signal / {noise_rms:g} = {reference_snr:.4f}")

    SNR_REFERENCE.write_text(
        "# Written by scripts/mock/snr_reference.py -- rerun it instead of editing.\n"
        + yaml.safe_dump({
            "reference_snr": float(reference_snr),
            "signal": float(signal),                 # Jy/beam, noiseless cube
            "noise_rms": float(noise_rms),           # Jy/arcsec^2, given to MARTINI
            "signal_peak_fraction": float(p.signal_peak_fraction),
            "galaxy": {"model": MODEL, "snap": SNAP, "sub_id": SUB_ID},
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, sort_keys=False))
    print(f"[snr] wrote {SNR_REFERENCE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
