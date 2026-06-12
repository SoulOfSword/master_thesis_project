"""Reusable visualisation helpers for mosaic plots.

These two functions appear in every density-mosaic plot script
(gamma, r_core, r_core/r_half_*), so they live here rather than being
duplicated in scripts/plots/density/.
"""

import colorsys

import matplotlib.colors as mcolors
import numpy as np


def lighten(color, amount=0.6):
    """Increase lightness in HLS space.

    Args:
        color: any matplotlib-recognised colour spec.
        amount: 0 -> original colour, 1 -> white.

    Returns:
        (r, g, b) tuple.
    """
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l_new = l + (1 - l) * amount
    return colorsys.hls_to_rgb(h, l_new, s)


def running_median(x, y, bin_edges, min_count=5):
    """Per-bin median of y(x), keeping bins with >= min_count points.

    Args:
        x: 1D array of x-coordinates.
        y: 1D array of y-coordinates, same length as x.
        bin_edges: 1D array of bin edges in x.
        min_count: minimum number of points required per bin.

    Returns:
        (bin_centres, medians) — both 1D numpy arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    centres = []
    medians = []
    for j in range(len(bin_edges) - 1):
        in_bin = (x >= bin_edges[j]) & (x < bin_edges[j + 1])
        if in_bin.sum() >= min_count:
            centres.append(0.5 * (bin_edges[j] + bin_edges[j + 1]))
            medians.append(np.median(y[in_bin]))
    return np.asarray(centres), np.asarray(medians)
