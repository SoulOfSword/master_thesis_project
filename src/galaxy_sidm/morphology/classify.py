"""Disc vs spheroidal classification from MORDOR component masses."""

import numpy as np
import pandas as pd


def disc_fraction(df):
    """Fraction of rows with IsDisc == 1."""
    if len(df) == 0:
        return np.nan
    return float((df["IsDisc"] == 1).sum()) / len(df)


def component_fractions(df):
    """Return a DataFrame with per-galaxy component mass fractions."""
    m = df["Mstar"].replace(0, np.nan)
    return pd.DataFrame({
        "fthin":   df["Mthin"]   / m,
        "fthick":  df["Mthick"]  / m,
        "fpbulge": df["Mpbulge"] / m,
        "fbulge":  df["Mbulge"]  / m,
        "fhalo":   df["Mhalo"]   / m,
    }, index=df.index)


def disc_fraction_binned(log_m, is_disc, bins, n_min=5,
                         confidence_level=0.95):
    """Binned disc fraction with Wilson 95% CI errorbars.

    Used by mosaic plots where bins are shared across models to ensure
    visual comparability.

    Args:
        log_m: 1D array of log10(M_star).
        is_disc: 1D 0/1 array, same length as log_m.
        bins: bin edges (1D array).
        n_min: minimum per-bin galaxy count to plot a point.
        confidence_level: Wilson interval level.

    Returns:
        (centres, fracs, lo_err, hi_err) — all 1D, with lo_err and
        hi_err already clipped to non-negative.
    """
    from astropy.stats import binom_conf_interval
    log_m = np.asarray(log_m)
    is_disc = np.asarray(is_disc).astype(int)
    centres, fracs, lo, hi = [], [], [], []
    for i in range(len(bins) - 1):
        sel = (log_m >= bins[i]) & (log_m < bins[i + 1])
        n_total = int(sel.sum())
        if n_total < n_min:
            continue
        n_disc = int((is_disc[sel] == 1).sum())
        f = n_disc / n_total
        l, h = binom_conf_interval(
            n_disc, n_total,
            confidence_level=confidence_level, interval="wilson")
        centres.append(0.5 * (bins[i] + bins[i + 1]))
        fracs.append(f)
        lo.append(f - l)
        hi.append(h - f)
    centres = np.asarray(centres)
    fracs = np.asarray(fracs)
    yerr = np.clip(np.array([lo, hi]), 0, None) if len(centres) \
        else np.zeros((2, 0))
    return centres, fracs, yerr[0], yerr[1]
