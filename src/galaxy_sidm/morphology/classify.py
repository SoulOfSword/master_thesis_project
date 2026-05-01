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
