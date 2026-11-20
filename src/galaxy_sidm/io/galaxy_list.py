"""Galaxy lists: text files with one "model snap subID" per line (the mock manifest format)."""

from collections import defaultdict
from pathlib import Path


def read_galaxy_list(path):
    """{(model, snap): [subIDs]} from a galaxy list; further columns and lines starting
    with # are ignored."""
    groups = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            model, snap, sub = line.split()[:3]
            groups[(model, int(snap))].append(int(sub))
    return dict(groups)
