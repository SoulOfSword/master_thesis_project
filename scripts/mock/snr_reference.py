"""Intrinsic signal-to-noise of one galaxy, from a NOISELESS MARTINI cube.

What it does:
  1. loads the galaxy's gas + stars,
  2. builds its MARTINI cube with add_noise=False, i.e. pure signal, using the
     SAME CubeParams as the real pipeline (5 Mpc, 5" px, 30" beam, 30 km/s x64,
     FOV = 4 x stellar r50), so the geometry matches the science cubes,
  3. takes the SPATIAL mean of each velocity channel -> one value per channel,
  4. averages those over the channels that CONTAIN SIGNAL. In a noiseless cube
     an empty channel is exactly 0, so "has signal" is simply mean > 0 -- no
     mask and no arbitrary threshold is needed,
  5. divides by NOISE_RMS (1e-5, the value fed to MARTINI) -> the intrinsic S/N.

Run it on a compute node (MARTINI is memory-hungry), from the project root in a
terminal with the thesis venv active (~/.bashrc does that):
  srun --partition=regular --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --time=00:30:00 python3 scripts/mock/snr_reference.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config
from galaxy_sidm.mock import load_galaxy_gas, CubeParams, build_cube

sys.path.insert(0, str(ROOT / "scripts" / "mock"))
from build_galaxy import resolve_base_path

# ---------------------------------------------------------------- EDIT ME ---
SUB_ID = 29759          # subhalo id
MODEL = "CDM"           # CDM / SIDM1 / vSIDM
SNAP = 33               # 17=z5, 21=z4, 25=z3, 33=z2, 50=z1, 67=z0.5
NCPU = 8                # cores for MARTINI's source insertion
NOISE_RMS = 1.0e-5      # the rms handed to MARTINI [Jy/arcsec^2]
CHANNELS = (26, 41)     # channels holding the line, read off by eye in DS9
SAVE_NOISELESS_TO = None  # set to a path to keep the noiseless cube for DS9
# -----------------------------------------------------------------------------


def main():
    cfg = load_config(None)
    z = {int(k): float(v) for k, v in cfg["snap_z"].items()}[SNAP]
    print(f"[snr] {MODEL} snap {SNAP} (z={z:g}) sub {SUB_ID}")

    base_path = resolve_base_path(MODEL, SNAP, 1080, cfg)
    gas = load_galaxy_gas(base_path, SNAP, SUB_ID, h=float(cfg["cosmology"]["h"]))
    print(f"[snr] gas cells={len(gas.mH_neutral_g)} stars={len(gas.m_s)}")

    # noiseless cube -> a temp file, so the science cube.fits is left alone
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "cube_nonoise.fits"
        _, npix = build_cube(gas, tmp, CubeParams(add_noise=False), ncpu=NCPU)
        cube = np.nan_to_num(np.asarray(fits.getdata(tmp), float))
        bunit = fits.getheader(tmp).get("BUNIT", "?")
        if SAVE_NOISELESS_TO:                 # keep a copy to inspect in DS9
            dest = Path(SAVE_NOISELESS_TO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(tmp, dest)
            print(f"[snr] noiseless cube -> {dest}")

    chan_mean = cube.mean(axis=(1, 2))       # spatial mean, one per channel
    peak = float(chan_mean.max())
    if peak <= 0:
        print("[snr] the cube has no flux at all -- nothing to measure")
        return 1

    lo, hi = CHANNELS
    sel = slice(lo - 1, hi) # DS9 1-indexed inclusive -> numpy
    signal = float(chan_mean[sel].mean()) # spectral mean of the spatial mean
    snr = signal / NOISE_RMS

    print(f"[snr] cube {npix}x{npix}x{cube.shape[0]}  BUNIT={bunit}")
    print(f"[snr] brightest channel (spatial mean) = {peak:.6e}")
    print(f"[snr] using DS9 channels {lo}-{hi} = numpy [{lo-1}:{hi}] "
          f"({hi - lo + 1} channels)")
    print(f"[snr] signal (spectral mean of spatial mean) = {signal:.6e}")
    print(f"[snr] reference S/N = signal / {NOISE_RMS:g} = {snr:.2f}")
    print(f"\n[snr] per-channel spatial mean / peak  (* = inside CHANNELS):")
    for i, v in enumerate(chan_mean):
        mark = "*" if lo - 1 <= i < hi else " "
        print(f"  {mark} DS9 ch {i+1:2d}  {v / peak:10.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())