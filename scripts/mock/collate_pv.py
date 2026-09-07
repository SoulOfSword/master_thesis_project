"""Collate BBarolo MOCK_pv_azim.pdf plots into a single review PDF.

Each galaxy's existing position-velocity plot
    <martini>/z<z>/<model>/gal_<subID>/bbarolo/MOCK_pv_azim.pdf
is stitched onto its own titled page of one multi-page PDF (via PyMuPDF),
so you can flip through the fits and choose a per-category NRADII by eye.

Two modes:

  --mode all   (default): every MORDOR disc in ONE (model, snap), one page
               per galaxy, sorted by stellar mass. Default SIDM1, snap 21
               (z=4) -- but works for any model / redshift via --model/--snap.

  --mode bins: up to TWO representative discs per (redshift, stellar-mass bin)
               for a model (default SIDM1), across all pipeline redshifts --
               the two nearest the bin centre (two most massive for the open
               top bin), for comparison. The "pick NRADII per category" view.

Only MORDOR discs (IsDisc==1) that already have a MOCK_pv_azim.pdf are
included; missing ones are skipped and tallied. Each page title carries
    model | z | subID | logMstar | NRADII  (ring count from rings_final1.txt)
so you can read off how many rings the current fit used.

Usage:
  python scripts/mock/collate_pv.py                        # SIDM1 z4, all discs
  python scripts/mock/collate_pv.py --model CDM --snap 50  # CDM z1, all discs
  python scripts/mock/collate_pv.py --mode bins            # SIDM1, one per z x mass
  python scripts/mock/collate_pv.py --mode bins --mass-edges 9.5 10 10.5 11 11.5
  python scripts/mock/collate_pv.py --out /path/review.pdf
  python scripts/mock/collate_pv.py --exclude config/problematic_discs.yaml  # gas-discs only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from galaxy_sidm.io import load_config, load_flat

DEFAULT_SNAPS = [17, 21, 25, 33, 50, 67]        # z = 5, 4, 3, 2, 1, 0.5
DEFAULT_MASS_EDGES = [9.5, 10.0, 10.5, 11.0, 11.5]  # log10(Mstar / Msun)


def _paths(cfg):
    mart = Path(cfg["paths"]["scratch_processed"]).parent / "martini"
    mdir = Path(cfg["paths"]["scratch_mordor"]) / "samples"
    return mart, mdir


def _discs(mdir, model, snap):
    """Return [(subID, Mstar), ...] for every MORDOR disc at (model, snap)."""
    p = mdir / f"mordor_sample_{model}_{snap:03d}.hdf5"
    if not p.exists():
        return []
    a, _ = load_flat(p)
    ids = np.asarray(a["halo_ids"], np.int64)
    isd = np.asarray(a["IsDisc"]).astype(int)
    mstar = np.asarray(a["Mstar"], float)
    return [(int(ids[i]), float(mstar[i])) for i in np.where(isd == 1)[0]]


def _excluded(exclude_yaml, model, z):
    """subIDs to drop for (model, z), from the problematic-discs YAML."""
    if not exclude_yaml or not Path(exclude_yaml).exists():
        return set()
    import yaml
    d = yaml.safe_load(Path(exclude_yaml).read_text()) or {}
    lst = ((d.get(model) or {}).get(f"z{z:g}")) or []
    return set(int(x) for x in lst)


def _pv_pdf(mart, model, snap, snap_z, sub_id):
    """Path to one galaxy's MOCK_pv_azim.pdf."""
    z = snap_z[snap]
    return (mart / f"z{z:g}" / model / f"gal_{sub_id:06d}"
            / "bbarolo" / "MOCK_pv_azim.pdf")


def _nrings(pv_pdf):
    """Ring count fitted, read from the sibling rings_final1.txt ('?' if absent)."""
    rf = pv_pdf.parent / "rings_final1.txt"
    if not rf.exists():
        return "?"
    n = sum(1 for ln in rf.read_text().splitlines()
            if ln.strip() and not ln.startswith("#") and len(ln.split()) > 3)
    return str(n)


def _add_page(out, pv_pdf, title):
    """Append pv_pdf's first page onto a new titled page of `out`.

    Returns False (adding nothing) if the source PDF cannot be opened, so a
    single corrupt file does not abort the whole run and lose earlier pages.
    """
    try:
        src = fitz.open(str(pv_pdf))
        r = src[0].rect
        strip = 44  # title strip height (pt) added above the plot
        page = out.new_page(width=r.width, height=r.height + strip)
        # big label, right-aligned in the strip above the plot. insert_text
        # (not insert_textbox) always draws even if wide -> no silent drop.
        fs = 22
        tw = fitz.get_text_length(title, fontname="helv", fontsize=fs)
        page.insert_text((max(8, r.width - tw - 14), 31), title,
                         fontsize=fs, fontname="helv")
        page.show_pdf_page(fitz.Rect(0, strip, r.width, r.height + strip), src, 0)
        src.close()
        return True
    except Exception as e:
        print(f"  skip (unreadable PDF) {pv_pdf}: {e}")
        return False


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["all", "bins"], default="all")
    p.add_argument("--model", default="SIDM1",
                   choices=["CDM", "SIDM1", "vSIDM", "WDM3", "WDM5"])
    p.add_argument("--snap", type=int, default=21,
                   help="Snapshot for --mode all (default 21 = z4)")
    p.add_argument("--mass-edges", type=float, nargs="+",
                   default=DEFAULT_MASS_EDGES,
                   help="log10(Mstar) bin edges for --mode bins")
    p.add_argument("--exclude", type=Path, default=None,
                   help="YAML of problematic subIDs (per model/z) to drop, e.g. "
                        "config/problematic_discs.yaml; routes output to gas_discs/.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(None)
    snap_z = {int(k): float(v) for k, v in cfg["snap_z"].items()}
    mart, mdir = _paths(cfg)

    if args.out is None:
        subdir = "gas_discs" if args.exclude else "all"
        tag = "gasdiscs" if args.exclude else "all"
        stem = (f"pv_{tag}_{args.model}_snap{args.snap:03d}"
                if args.mode == "all" else f"pv_{tag}bins_{args.model}")
        args.out = ROOT / "figures" / "pvs" / subdir / f"{stem}.pdf"

    out = fitz.open()
    included = skipped = 0

    if args.mode == "all":
        if args.snap not in snap_z:
            print(f"--snap {args.snap} not in snap_z {sorted(snap_z)}")
            return 1
        rows = sorted(_discs(mdir, args.model, args.snap), key=lambda t: t[1])
        z = snap_z[args.snap]
        excl = _excluded(args.exclude, args.model, z)
        rows = [(s, m) for s, m in rows if s not in excl]
        for sub, mstar in rows:
            pv = _pv_pdf(mart, args.model, args.snap, snap_z, sub)
            if not pv.exists():
                skipped += 1
                continue
            title = (f"{args.model} | z={z:g} | subID {sub} | "
                     f"logM*={np.log10(mstar):.2f} | NRADII={_nrings(pv)}")
            if _add_page(out, pv, title):
                included += 1
            else:
                skipped += 1
    else:  # bins: one representative per (snap, mass bin)
        edges = args.mass_edges
        # adjacent [lo,hi) bins, plus an open-ended top bin so the most
        # massive discs (>= the top edge) always get a representative
        mass_bins = list(zip(edges[:-1], edges[1:])) + [(edges[-1], float("inf"))]
        for snap in DEFAULT_SNAPS:
            if snap not in snap_z:
                continue
            z = snap_z[snap]
            excl = _excluded(args.exclude, args.model, z)
            rows = _discs(mdir, args.model, snap)
            for lo, hi in mass_bins:
                cand = [(s, m) for s, m in rows
                        if s not in excl and lo <= np.log10(m) < hi
                        and _pv_pdf(mart, args.model, snap, snap_z, s).exists()]
                if not cand:
                    skipped += 1
                    continue
                if np.isinf(hi):     # open top bin: most massive discs first
                    ordered = sorted(cand, key=lambda t: -t[1])
                    hitxt = "+"
                else:                # closed bin: discs nearest the bin centre
                    centre = 0.5 * (lo + hi)
                    ordered = sorted(cand,
                                     key=lambda t: abs(np.log10(t[1]) - centre))
                    hitxt = f"{hi:.1f}"
                for sub, mstar in ordered[:2]:   # up to 2 per (z, mass bin)
                    pv = _pv_pdf(mart, args.model, snap, snap_z, sub)
                    title = (f"{args.model} | z={z:g} | logM* [{lo:.1f},{hitxt}) | "
                             f"subID {sub} | logM*={np.log10(mstar):.2f} | "
                             f"NRADII={_nrings(pv)}")
                    if _add_page(out, pv, title):
                        included += 1
                    else:
                        skipped += 1

    if included == 0:
        print("No MOCK_pv_azim.pdf found -- has the mock pipeline run for "
              f"{args.model}?")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(args.out))
    out.close()
    print(f"wrote {args.out}  ({included} pages, {skipped} skipped/missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
