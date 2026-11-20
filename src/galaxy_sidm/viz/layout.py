"""Panel layout of the redshift mosaics."""

import matplotlib.pyplot as plt


def panel_grid(n, panel_size, sharex=True, sharey=True, **fig_kw):
    """Figure with n panels: one row up to 3 panels, else two rows of ceil(n / 2), a shorter
    second row centred under the first (7 panels: 4 + 3, offset by half a panel).

    Shared y: tick labels on the first panel of each row. Shared x: tick labels on the second
    row, and on the first too when the rows are offset (their columns do not line up). fig_kw
    go to plt.figure (e.g. constrained_layout).

    Returns (fig, axes, bottom, left): the n axes in reading order, the second (last) row
    (for the x label) and the first panel of each row (for the y label).
    """
    nrows = 1 if n <= 3 else 2
    ncols = -(-n // nrows)
    fig = plt.figure(figsize=(panel_size[0] * ncols, panel_size[1] * nrows), **fig_kw)
    gs = fig.add_gridspec(nrows, 2 * ncols)
    axes = []
    for i in range(n):
        row, col = divmod(i, ncols)
        c0 = 2 * col + ncols - min(ncols, n - row * ncols)
        first = axes[0] if axes else None
        axes.append(fig.add_subplot(gs[row, c0:c0 + 2], sharex=first if sharex else None,
                                    sharey=first if sharey else None))
    left = axes[::ncols]
    bottom = axes[(nrows - 1) * ncols:]
    offset = n % ncols != 0
    for ax in axes:
        ax.tick_params(labelbottom=not sharex or offset or ax in bottom,
                       labelleft=not sharey or ax in left)
    return fig, axes, bottom, left


def legend_entries(axes, first=()):
    """(handles, labels) of every distinct label on the axes, those in first leading in that
    order: one legend for the figure even when no single panel has every entry."""
    found = {}
    for ax in axes:
        for h, name in zip(*ax.get_legend_handles_labels()):
            found.setdefault(name, h)
    names = [n for n in first if n in found] + [n for n in found if n not in first]
    return [found[n] for n in names], names
