"""Run BBarolo (3D tilted-ring fit) on a MARTINI cube.

Rings one beam wide; inclination, position angle and centre fixed to the
injected values (MARTINI puts the galaxy in the middle of the image); AZIM
norm, SMOOTH&SEARCH mask (SNRCUT=5, GROWTHCUT=3), fitting VROT, VDISP and
VSYS in two stages (stage 2 fixes VSYS to the median of the rings). Returns V,
sigma, V/sigma from the rings, with V the mean of Vmax and Vflat.
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


@dataclass
class MajorAxisExtent:
    xpos: float          # centre [pix, 0-based like XPOS/YPOS]: middle of the map
    ypos: float
    left_arcsec: float   # centre -> outer edge of the emission, on each side
    right_arcsec: float
    left_at_edge: bool   # that edge is the map border: the emission may
    right_at_edge: bool  # continue outside the cube
    hole_arcsec: float   # centre -> first emission, nearer side (0: emission at the centre)


def image_centre(nx, ny):
    """Galaxy centre in 0-based pixels, BBarolo's XPOS/YPOS convention.

    MARTINI puts the galaxy in the middle of the image. Do not take it from the
    FITS WCS: the reference pixel of these cubes is not at the galaxy.
    """
    return (nx - 1) / 2.0, (ny - 1) / 2.0


def major_axis_extent(bbarolo_dir):
    """How far the data velocity field reaches along the major axis.

    Uses BBarolo's DATA moment-1 map (maps/*_1mom.fits: the masked data, the
    same whatever the rings or centre fitted), measured from the middle of the
    map, where the galaxy is (image_centre). With PA = 90 deg the major axis is
    the map row through the centre. On each side, walk out from the centre
    along that row and stop at the first blank pixel, so gas beyond a gap does
    not count even where it joins the disc elsewhere in the map. A blank centre
    (a central hole) is stepped over: the walk starts at the first emission on
    that side, and hole_arcsec is how far that is on the nearer side.
    """
    from astropy.io import fits

    bb = Path(bbarolo_dir)
    maps = [p for p in sorted((bb / "maps").glob("*_1mom.fits")) if "mod" not in p.name]
    if not maps:
        raise FileNotFoundError(f"no data moment-1 map in {bb / 'maps'}")
    mom1 = np.squeeze(fits.getdata(maps[0])).astype(float)
    px_arcsec = abs(fits.getheader(maps[0])["CDELT1"]) * 3600.0

    xpos, ypos = image_centre(mom1.shape[1], mom1.shape[0])
    x0, y0 = int(xpos), int(ypos)

    row = np.isfinite(mom1[y0])
    edges, starts = [], []
    for path in (row[x0::-1], row[x0:]):   # centre -> left border, centre -> right border
        if not path.any():                 # no emission on this side
            edges.append((0.0, False))
            continue
        start = int(np.argmax(path))       # first emission from the centre
        gaps = np.flatnonzero(~path[start:])
        last = start + int(gaps[0]) - 1 if len(gaps) else len(path) - 1
        edges.append((last * px_arcsec, last == len(path) - 1))
        starts.append(start * px_arcsec)
    (left, left_edge), (right, right_edge) = edges
    hole = min(starts) if starts else 0.0
    return MajorAxisExtent(xpos, ypos, left, right, left_edge, right_edge, hole)


def _vflat(vrot):
    """Vflat = velocity at the smallest change between consecutive rings."""
    if len(vrot) < 2:
        return float(vrot[-1]) if len(vrot) else np.nan
    d = np.abs(np.diff(vrot))
    return float(vrot[1:][np.argmin(d)])


def write_par(cube_fits, out_dir, inc_deg, pa_deg, beam_arcsec,
              radsep_arcsec=None, vrot0=100.0, vdisp0=25.0,
              nradii=None, threads=4, distance_mpc=None, extra=None):
    """Write a BBarolo 3DFIT parameter file; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    par = out_dir / "bbarolo.par"
    radsep = radsep_arcsec or beam_arcsec
    from astropy.io import fits
    hdr = fits.getheader(str(cube_fits))
    xc, yc = image_centre(hdr["NAXIS1"], hdr["NAXIS2"])
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
        f"XPOS        {xc}",
        f"YPOS        {yc}",
        *( [f"DISTANCE    {distance_mpc}"] if distance_mpc is not None else [] ),
        # f"NRADII       {nradii}",
        f"Z0          {radsep/6}",
        f"RADSEP      {radsep}",
        f"VROT        {vrot0}",
        f"VDISP       {vdisp0}",
        #f"VSYS        {vsys:.1f}",
        "FREE        VROT VDISP VSYS",
        "SIDE        B",
        "NORM        AZIM",
        "TWOSTAGE     true",
        "PLOTMASK     true",
        "SEARCH       true",
        # mask
        "MASK        SMOOTH&SEARCH",
        "SNRCUT      5",
        "GROWTHCUT   3",
        "FLAGERRORS  false",
    ]
    if nradii is not None:
        lines.append(f"NRADII      {nradii}")
    if extra:
        lines.extend(extra)
    par.write_text("\n".join(lines) + "\n")
    return par


def read_par(bbarolo_dir):
    """<bbarolo_dir>/bbarolo.par as {KEY: value}, keys upper-case."""
    par = {}
    for line in (Path(bbarolo_dir) / "bbarolo.par").read_text().splitlines():
        parts = line.split(None, 1)
        if parts and not parts[0].startswith("#"):   # a later line overrides an earlier one
            par[parts[0].upper()] = parts[1].strip() if len(parts) > 1 else ""
    return par


def rings_file(bbarolo_dir):
    """The rings file holding the result of a BBarolo fit, as its bbarolo.par implies.

    BBarolo runs a second stage only with TWOSTAGE true and a free geometric
    parameter (here VSYS): stage 2 fixes it to the median of the rings, refits
    VROT/DISP and writes rings_final2.txt. Otherwise the result is
    rings_final1.txt. Raises FileNotFoundError if that file is missing, instead
    of reading the other one.
    """
    bb = Path(bbarolo_dir)
    par = read_par(bb)
    free_geometry = set(par.get("FREE", "").lower().split()) & {"inc", "pa", "phi", "z0", "xpos", "ypos", "vsys"}
    two_stage = par.get("TWOSTAGE", "false").lower() in ("true", "t", "yes", "1")
    rf = bb / ("rings_final2.txt" if two_stage and free_geometry else "rings_final1.txt")
    if not rf.exists():
        raise FileNotFoundError(f"{rf} is missing: BBarolo did not write the rings file this fit should have")
    return rf


def _parse_rings(rings_txt):
    """Read a rings_final*.txt -> (rad_kpc, vrot, vdisp). Column order:
    RAD(Kpc) RAD(arcs) VROT DISP INC PA ..."""
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
    return arr[:, 0], arr[:, 2], arr[:, 3]

_PLOT_SKIP = {"plot_all.py", "plot_pvs_old.py"}


def _run_plotscripts(out_dir, timeout=600):
    """Run BBarolo's generated plot_*.py sequentially; return [failed names]."""
    import os
    import sys
    scripts = sorted(p for p in Path(out_dir).rglob("plot_*.py")
                     if p.name not in _PLOT_SKIP)
    env = dict(os.environ, MPLBACKEND="Agg")
    fails = []
    for s in scripts:
        try:
            r = subprocess.run([sys.executable, str(s)], cwd=str(s.parent),
                               env=env, capture_output=True, text=True,
                               timeout=timeout)
            if r.returncode != 0:
                fails.append(s.name)
        except Exception:
            fails.append(s.name)
    return fails


def run_bbarolo(cube_fits, out_dir, inc_deg=60.0, pa_deg=90.0,
                beam_arcsec=30.0, bbarolo="BBarolo", timeout=5400,
                make_plots=True, **par_kw):
    """Run BBarolo on a cube and return a BaroloResult (V, sigma, V/sigma).
    """
    out_dir = Path(out_dir)
    par = write_par(cube_fits, out_dir, inc_deg, pa_deg, beam_arcsec, **par_kw)
    # BBarolo intermittently crashes (rc=-11); retry a few times. A non-zero
    # rc means no fresh rings, so we never parse/plot stale ones below.
    for attempt in range(1, 6):
        proc = subprocess.run([bbarolo, "-p", str(par)], cwd=str(out_dir),
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            break
        if attempt < 5:
            print(f"[run_bbarolo] BBarolo rc={proc.returncode}; "
                  f"retrying ({attempt + 1}/5)")
    ok = proc.returncode == 0
    rad = vrot = vdisp = np.zeros(0)
    V = sigma = vsig = float("nan")
    rings_txt = None
    if ok:
        rings_txt = rings_file(out_dir)   # raises if BBarolo did not write it
        rad, vrot, vdisp = _parse_rings(rings_txt)
    if len(vrot):
        V = 0.5 * (float(np.max(vrot)) + _vflat(vrot))
        sigma = float(np.mean(vdisp))
        vsig = V / sigma if sigma > 0 else float("nan")
    if make_plots and ok:
        _run_plotscripts(out_dir)
    return BaroloResult(
        out_dir=out_dir, rings_txt=rings_txt, rad_kpc=rad, vrot=vrot,
        vdisp=vdisp, V=V, sigma=sigma, V_over_sigma=vsig,
        returncode=proc.returncode)
