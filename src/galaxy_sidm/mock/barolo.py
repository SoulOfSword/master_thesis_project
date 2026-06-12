"""Run BBarolo (3D tilted-ring fit) on a MARTINI cube.

Rings one beam wide, inc and PA fixed to the injected values, zero
thickness, AZIM norm, SEARCH mask (SNRCUT=3, GROWTHCUT=2), fitting VROT
and VDISP only. Returns V, sigma, V/sigma from the rings, with V the mean
of Vmax and Vflat.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BaroloResult:
    out_dir: Path
    rings_txt: Path
    rad_kpc: np.ndarray
    vrot: np.ndarray
    vdisp: np.ndarray
    V: float
    sigma: float
    V_over_sigma: float
    returncode: int


def _vflat(vrot):
    """Vflat = velocity at the smallest change between consecutive rings."""
    if len(vrot) < 2:
        return float(vrot[-1]) if len(vrot) else np.nan
    d = np.abs(np.diff(vrot))
    return float(vrot[1:][np.argmin(d)])


def write_par(cube_fits, out_dir, inc_deg, pa_deg, beam_arcsec,
              radsep_arcsec=None, vrot0=100.0, vdisp0=25.0,
              nradii=None, threads=4, extra=None):
    """Write a BBarolo 3DFIT parameter file; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    par = out_dir / "bbarolo.par"
    radsep = radsep_arcsec or beam_arcsec
    from astropy.io import fits
    hdr = fits.getheader(str(cube_fits))
    vsys = float(hdr.get("CRVAL3", 0.0)) # systemic velocity in the cube, in km/s (if CUNIT3 is m/s, convert it)
    if str(hdr.get("CUNIT3", "")).strip().lower() in ("m s-1", "m/s", "ms-1"):
        vsys /= 1000.0 # km/s
    lines = [
        f"FITSFILE    {Path(cube_fits)}",
        "3DFIT       true",
        f"OUTFOLDER   {out_dir}/",
        "THREADS     %d" % threads,
        # fixed geometry
        f"INC         {inc_deg}",
        f"PA          {pa_deg}",
        "Z0          0",
        f"RADSEP      {radsep}",
        f"VROT        {vrot0}",
        f"VDISP       {vdisp0}",
        f"VSYS        {vsys:.1f}",
        "FREE        VROT VDISP",
        "SIDE        B",
        "NORM        AZIM",
        "TWOSTAGE    false",
        # mask
        "MASK        SEARCH",
        "SNRCUT      3",
        "GROWTHCUT   2",
        "FLAGERRORS  false",
    ]
    if nradii is not None:
        lines.append(f"NRADII      {nradii}")
    if extra:
        lines.extend(extra)
    par.write_text("\n".join(lines) + "\n")
    return par


def _parse_rings(rings_txt):
    """Read rings_final1.txt -> (rad_kpc, vrot, vdisp). Column order:
    RAD(arcs) RAD(Kpc) VROT DISP INC PA ..."""
    rows = []
    for line in Path(rings_txt).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            rows.append([float(x) for x in parts[:4]])
        except ValueError:
            continue
    arr = np.array(rows) if rows else np.zeros((0, 4))
    if arr.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return arr[:, 1], arr[:, 2], arr[:, 3]


def run_bbarolo(cube_fits, out_dir, inc_deg=60.0, pa_deg=90.0,
                beam_arcsec=30.0, bbarolo="BBarolo", timeout=5400, **par_kw):
    """Run BBarolo on a cube and return a BaroloResult (V, sigma, V/sigma)."""
    out_dir = Path(out_dir)
    par = write_par(cube_fits, out_dir, inc_deg, pa_deg, beam_arcsec, **par_kw)
    proc = subprocess.run([bbarolo, "-p", str(par)], cwd=str(out_dir),
                          capture_output=True, text=True, timeout=timeout)
    rings = sorted(out_dir.rglob("rings_final1.txt"))
    rad = vrot = vdisp = np.zeros(0)
    V = sigma = vsig = float("nan")
    rings_txt = rings[0] if rings else (out_dir / "rings_final1.txt")
    if rings:
        rad, vrot, vdisp = _parse_rings(rings[0])
    if len(vrot):
        V = 0.5 * (float(np.max(vrot)) + _vflat(vrot))
        sigma = float(np.mean(vdisp))
        vsig = V / sigma if sigma > 0 else float("nan")
    return BaroloResult(
        out_dir=out_dir, rings_txt=rings_txt, rad_kpc=rad, vrot=vrot,
        vdisp=vdisp, V=V, sigma=sigma, V_over_sigma=vsig,
        returncode=proc.returncode)
