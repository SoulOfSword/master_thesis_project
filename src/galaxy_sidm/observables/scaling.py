"""Galaxy scaling relations.

Implements the key scaling relations for comparison between
semi-analytical models and simulations:
- Tully-Fisher relation: M_star vs V_rot
- Mass-size relation: M_star vs R_half
- Fall relation: M_star vs j_star (specific angular momentum)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ScalingRelationFit:
    """Container for scaling relation fit results."""

    slope: float
    intercept: float
    scatter: float  # intrinsic scatter in dex
    x_pivot: float  # pivot point for the fit


def fit_power_law(x, y, x_pivot = None):
    """Fit a power-law relation log(y) = a * log(x/x_pivot) + b.

    Parameters
    ----------
    x, y : array
        Data points.
    x_pivot : float, optional
        Pivot point. If None, uses median of x.

    Returns
    -------
    ScalingRelationFit
        Fit results.
    """
    log_x = np.log10(x)
    log_y = np.log10(y)

    if x_pivot is None:
        x_pivot = np.median(x)

    log_x_pivot = np.log10(x_pivot)

    # Simple linear regression
    slope, intercept = np.polyfit(log_x - log_x_pivot, log_y, 1)

    # Scatter around the relation
    residuals = log_y - (slope * (log_x - log_x_pivot) + intercept)
    scatter = np.std(residuals)

    return ScalingRelationFit(slope, intercept, scatter, x_pivot)


def tully_fisher(M_star, V_rot):
    """Fit the Tully-Fisher relation.

    The stellar Tully-Fisher relation: M_star ~ V_rot^alpha
    with alpha ~ 3-4.

    Parameters
    ----------
    M_star : array
        Stellar masses in Msun.
    V_rot : array
        Rotation velocities in km/s.

    Returns
    -------
    ScalingRelationFit
        Fit with slope, intercept, and scatter.
    """
    return fit_power_law(V_rot, M_star, x_pivot=100.0)  # pivot at 100 km/s


def mass_size_relation(M_star, R_half):
    """Fit the mass-size relation.

    R_half ~ M_star^alpha with alpha ~ 0.2-0.3 for late-types.

    Parameters
    ----------
    M_star : array
        Stellar masses in Msun.
    R_half : array
        Half-mass radii in kpc.

    Returns
    -------
    ScalingRelationFit
        Fit with slope, intercept, and scatter.
    """
    return fit_power_law(M_star, R_half, x_pivot=1e10)  # pivot at 10^10 Msun


def fall_relation(M_star, j_star):
    """Fit the Fall relation (angular momentum - mass).

    j_star ~ M_star^alpha with alpha ~ 2/3 expected from theory.

    Parameters
    ----------
    M_star : array
        Stellar masses in Msun.
    j_star : array
        Specific angular momentum in kpc km/s.

    Returns
    -------
    ScalingRelationFit
        Fit with slope, intercept, and scatter.
    """
    return fit_power_law(M_star, j_star, x_pivot=1e10)


def compare_relations(sim_fit, model_fit):
    """Compare scaling relations between simulation and model.

    Returns
    -------
    dict
        Comparison metrics (slope difference, normalization offset, scatter comparison).
    """
    return {
        "delta_slope": model_fit.slope - sim_fit.slope,
        "delta_intercept": model_fit.intercept - sim_fit.intercept,
        "scatter_ratio": model_fit.scatter / sim_fit.scatter,
    }
