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
    xpos: float # centre [pix, 0-based like XPOS/YPOS]: middle of the map
    ypos: float
    left_arcsec: float # centre -> outer edge of the emission, on each side
    right_arcsec: float
    left_at_edge: bool # that edge is the map border: the emission may
    right_at_edge: bool # continue outside the cube
    hole_arcsec: float # centre -> first emission, nearer side (0: emission at the centre)

@dataclass
class PVRings:
    radii: list # ring centres, arcsec from the galaxy centre
    width: float # ring width, arcsec (one beam, or a little less after the 5" rule)
    hole: list # per ring: True if no counted emission lies within it on either side
    inner: list # [left, right]: where the counted emission starts, arcsec (None: none on that side)
    outer: list # [left, right]: where it ends, arcsec

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
    maps = [p for p in sorted((bb / "maps").glob("*_1mom.fits"))
            if "mod" not in p.name and not p.name.startswith(".")] # skip hidden files (e.g. macOS ._ copies)
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


def emission_at_border(bbarolo_dir, margin_px):
    """Whether the emission reaches the cube border along the major axis: (left, right).

    Walks along the same row of BBarolo's DATA moment-1 map as
    major_axis_extent, from the centre out to the outermost pixel with emission
    on each side, without stopping at gaps, and checks whether that pixel is
    within margin_px pixels of the border, i.e. whether any of the last
    margin_px + 1 pixels of the row has emission. True on a side means the cube
    is too small for the galaxy.

    The margin should be one beam: before searching for emission, BBarolo's
    SMOOTH&SEARCH mask blurs the cube to twice the beam, counting everything
    beyond the cube as zero, so near the border the blurred signal is weaker
    (about 45% at the border, 7% one beam in) and the mask can stop a few pixels
    short of a border that cuts through the gas.
    """
    from astropy.io import fits

    bb = Path(bbarolo_dir)
    maps = [p for p in sorted((bb / "maps").glob("*_1mom.fits"))
            if "mod" not in p.name and not p.name.startswith(".")] # skip hidden files (e.g. macOS ._ copies)
    if not maps:
        raise FileNotFoundError(f"no data moment-1 map in {bb / 'maps'}")
    mom1 = np.squeeze(fits.getdata(maps[0]))
    row = np.isfinite(mom1[int(image_centre(mom1.shape[1], mom1.shape[0])[1])])
    return bool(row[:margin_px + 1].any()), bool(row[len(row) - 1 - margin_px:].any())


def pv_rings(bbarolo_dir, sigma, ring_arcsec, margin_arcsec, level=3.0, seed=5.0):
    """The final fit's rings, from the major-axis PV of a BBarolo fit (the 3-ring one).

    1. Emission: in the DATA position-velocity slice along the major axis
       (pvs/*_pv_a.fits, a cut through the cube, the same whatever was fitted),
       pixels at or above level * sigma, grouped into patches of touching pixels.
       A patch counts if it holds a pixel at or above seed * sigma (so it is not
       just noise) and overlaps BBarolo's mask on the same slice
       (pvs/*mask_pv_a.fits; so it is part of what BBarolo found as the galaxy).
    2. Edges: on each side of the centre, where the counted emission starts and
       ends. The rings start at the larger of the two starts (this skips a
       central hole) and end at the larger of the two ends.
    3. Rings: ring_arcsec wide, centred at start + ring_arcsec/2 and then every
       ring_arcsec. A ring centred at most margin_arcsec beyond the end is kept
       and all rings are narrowed evenly, keeping their number, so that it sits
       on the end; rings farther out are dropped.
    4. Hole rings: no counted emission within the ring's radial range on
       either side.
    Radii are measured from the image centre, which lies between two pixels
    (the cubes have an even number of pixels), so pixel k from the centre on
    either side spans k to k + 1 pixels. sigma: the cube's noise. Returns None
    if no ring can be placed (no counted emission).
    """
    from astropy.io import fits
    from scipy import ndimage

    bb = Path(bbarolo_dir)
    pvs = sorted(p for p in (bb / "pvs").glob("*_pv_a.fits")
                 if not p.name.startswith(".")) # skip hidden files (e.g. macOS ._ copies)
    data_pv = [p for p in pvs if "mask" not in p.name and "mod" not in p.name]
    mask_pv = [p for p in pvs if "mask" in p.name]
    if not data_pv or not mask_pv:
        raise FileNotFoundError(f"no data or mask major-axis PV in {bb / 'pvs'}")
    pv = np.nan_to_num(np.squeeze(fits.getdata(data_pv[0])).astype(float))   # (velocity, position)
    mask = np.squeeze(fits.getdata(mask_pv[0])) > 0
    px_arcsec = round(abs(fits.getheader(data_pv[0])["CDELT1"]) * 3600.0, 6)

    # 1. emission: patches of touching pixels (diagonals included) >= level sigma; keep those
    #    with a pixel >= seed sigma that overlap BBarolo's mask
    snr = pv / sigma
    patches, _ = ndimage.label(snr >= level, structure=np.ones((3, 3)))
    counted = [i for i in np.unique(patches[snr >= seed]) if i > 0 and (mask & (patches == i)).any()]
    columns = np.isin(patches, counted).any(axis=0)   # positions with counted emission, at any velocity

    # 2. edges: each side from the centre outward
    x0 = int(image_centre(pv.shape[1], 1)[0])         # the centre lies between pixels x0 and x0 + 1
    sides = (columns[x0::-1], columns[x0 + 1:])       # left, right
    inner, outer = [], []
    for path in sides:
        k = np.flatnonzero(path)
        inner.append(float(k[0] * px_arcsec) if len(k) else None)
        outer.append(float((k[-1] + 1) * px_arcsec) if len(k) else None)
    if all(v is None for v in inner):
        return None
    start = max(v for v in inner if v is not None)
    end = max(v for v in outer if v is not None)

    # 3. rings, with the margin rule at the end
    n = int((end + margin_arcsec - start - ring_arcsec / 2) // ring_arcsec) + 1
    radii = start + ring_arcsec / 2 + ring_arcsec * np.arange(max(n, 0))
    radii = radii[radii <= end + margin_arcsec]
    if not len(radii):
        return None
    width = float(ring_arcsec)
    if radii[-1] > end:
        width = (end - start) / (len(radii) - 0.5)
        radii = start + width / 2 + width * np.arange(len(radii))

    # 4. hole rings: pixel k (either side) spans k..k+1 pixels; a ring is a hole if no
    #    counted pixel overlaps its radial range [r - width/2, r + width/2]
    k = np.unique(np.concatenate([np.flatnonzero(path) for path in sides]))
    hole = [not bool(np.any((k * px_arcsec < r + width / 2) & ((k + 1) * px_arcsec > r - width / 2)))
            for r in radii]
    return PVRings([float(r) for r in radii], width, hole, inner, outer)


def write_fit_mask(mask_in, mask_out, rings, inc_deg):
    """Write BBarolo's mask with the hole rings removed; return how many map pixels were removed.

    mask_in: BBarolo's mask (mask.fits, 1 = emission), e.g. from the 3-ring fit.
    rings: the PVRings from pv_rings. The hole mask is 1 everywhere and 0 where a
    pixel's radius in the plane of the disc falls inside a hole ring; it is the same
    in every channel and is multiplied with BBarolo's mask, so nothing is added and
    nothing changes outside the hole rings. A pixel's disc-plane radius: the disc is
    tilted by inc_deg, so a circle in the disc looks like an ellipse on the sky,
    squashed by cos(inc) along the minor axis; with PA = 90 deg (as in all our fits)
    the major axis runs along x and the minor axis along y.
    """
    from astropy.io import fits

    with fits.open(mask_in) as hdul:
        mask = hdul[0].data # (channels, y, x)
        ny, nx = mask.shape[-2:]
        px_arcsec = abs(hdul[0].header["CDELT1"]) * 3600.0
        xc, yc = image_centre(nx, ny)
        yy, xx = np.mgrid[:ny, :nx] # the y and x index of every map pixel
        radius = np.hypot((xx - xc) * px_arcsec, (yy - yc) * px_arcsec / np.cos(np.radians(inc_deg)))
        hole_mask = np.ones((ny, nx), dtype=mask.dtype)
        for r, h in zip(rings.radii, rings.hole):
            if h:
                hole_mask[(radius >= r - rings.width / 2) & (radius < r + rings.width / 2)] = 0
        hdul[0].data = mask * hole_mask # the same map in every channel
        hdul.writeto(mask_out, overwrite=True)
    return int((hole_mask == 0).sum())


def _vflat(vrot):
    """Vflat = velocity at the smallest change between consecutive rings."""
    if len(vrot) < 2:
        return float(vrot[-1]) if len(vrot) else np.nan
    d = np.abs(np.diff(vrot))
    return float(vrot[1:][np.argmin(d)])


# stellar-mass Tully-Fisher relation of Di Teodoro et al. (2021): log10 Mstar = ALPHA log10 Vflat + BETA
DT21_ALPHA, DT21_BETA = 4.25, 0.80


def vrot0_from_mstar(mstar):
    """BBarolo's starting VROT [km/s] for a galaxy of stellar mass mstar [Msun]: the Vflat that the
    stellar-mass Tully-Fisher relation of Di Teodoro et al. (2021) gives for that mass.

    A start far below the true rotation lets BBarolo settle on ring fits that match only
    one side of the galaxy, with a wrong VSYS (as with the old fixed start of 100 km/s).
    """
    if mstar is None or not mstar > 0:
        raise ValueError(f"no stellar mass to set BBarolo's starting VROT (Mstar = {mstar})")
    return round(10 ** ((np.log10(mstar) - DT21_BETA) / DT21_ALPHA), 1)


def write_par(cube_fits, out_dir, inc_deg, pa_deg, beam_arcsec,
              radsep_arcsec=None, vrot0=None, vdisp0=25.0,
              nradii=None, radii=None, mask_file=None, threads=4, distance_mpc=None, extra=None):
    """Write a BBarolo 3DFIT parameter file; return its path.

    Rings: the ring centres `radii` (arcsec) if given, else `nradii` rings from
    the centre; either way `radsep_arcsec` apart (default: the beam). BBarolo
    makes each ring reach halfway to its neighbours, and a single ring
    radsep_arcsec wide. Mask: `mask_file` (a FITS file like BBarolo's mask.fits)
    if given, else BBarolo's own SMOOTH&SEARCH mask. The disc thickness Z0 is a
    sixth of the beam, whatever the rings. vrot0: the starting VROT of every ring
    [km/s], required (e.g. vrot0_from_mstar(Mstar)).
    """
    if vrot0 is None:
        raise ValueError("write_par needs vrot0, the starting VROT (e.g. vrot0_from_mstar(Mstar))")
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
        f"Z0          {beam_arcsec/6}",
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
        f"MASK        FILE({mask_file})" if mask_file else "MASK        SMOOTH&SEARCH",
        "SNRCUT      5",
        "GROWTHCUT   3",
        "FLAGERRORS  false",
    ]
    if radii is not None:
        lines.append("RADII       " + " ".join(f"{r:.3f}" for r in radii))
    elif nradii is not None:
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
