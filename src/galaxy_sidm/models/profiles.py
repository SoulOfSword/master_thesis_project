"""Dark matter halo density profiles (JAX implementation)."""

import jax.numpy as jnp
from jax import jit
import numpy as np
from scipy.optimize import curve_fit


@jit
def nfw_density(r, rho_s, r_s):
    """NFW density profile.

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_s : float
        Scale density in Msun/kpc^3.
    r_s : float
        Scale radius in kpc.

    Returns
    -------
    array
        Density at each radius in Msun/kpc^3.
    """
    x = r / r_s
    return rho_s / (x * (1 + x) ** 2)


@jit
def nfw_mass(r, rho_s, r_s):
    """Enclosed mass for NFW profile.

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_s : float
        Scale density in Msun/kpc^3.
    r_s : float
        Scale radius in kpc.

    Returns
    -------
    array
        Enclosed mass at each radius in Msun.
    """
    x = r / r_s
    return 4 * jnp.pi * rho_s * r_s**3 * (jnp.log(1 + x) - x / (1 + x))


@jit
def nfw_circular_velocity(r, rho_s, r_s):
    """Circular velocity for NFW profile.

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_s : float
        Scale density in Msun/kpc^3.
    r_s : float
        Scale radius in kpc.

    Returns
    -------
    array
        Circular velocity in km/s.
    """
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    M_enc = nfw_mass(r, rho_s, r_s)
    return jnp.sqrt(G * M_enc / r)


@jit
def cored_nfw_density(r, rho_s, r_s, r_core):
    """Cored NFW profile.

    Multiplies the NFW profile by tanh(r/r_core) so the inner cusp is
    smoothly suppressed. Asymptotes to NFW for r >> r_core and to a
    constant central density for r << r_core.

    Parameters
    ----------
    r : array
        Radius (in whatever length unit r_s, r_core are; e.g. ckpc).
    rho_s : float
        Scale density (e.g. Msun/ckpc^3).
    r_s : float
        Scale radius.
    r_core : float
        Core radius. r_core -> 0 recovers pure NFW; larger values
        produce more pronounced cores.

    Returns
    -------
    array
        Density at each radius, same units as rho_s.
    """
    x = r / r_s
    rho_nfw = rho_s / (x * (1.0 + x) ** 2)
    return jnp.tanh(r / r_core) * rho_nfw


@jit
def einasto_density(r, rho_s, r_s, alpha=0.18):
    """Einasto density profile.

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_s : float
        Scale density in Msun/kpc^3.
    r_s : float
        Scale radius in kpc.
    alpha : float
        Shape parameter (typically 0.16-0.20 for CDM halos).

    Returns
    -------
    array
        Density at each radius in Msun/kpc^3.
    """
    x = r / r_s
    return rho_s * jnp.exp(-(2 / alpha) * (x**alpha - 1))


@jit
def isothermal_core_density(r, rho_0, r_core):
    """Pseudo-isothermal core profile (often used for SIDM cores).

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_0 : float
        Central density in Msun/kpc^3.
    r_core : float
        Core radius in kpc.

    Returns
    -------
    array
        Density at each radius in Msun/kpc^3.
    """
    return rho_0 / (1 + (r / r_core) ** 2)


@jit
def concentration_duffy08(M200, z=0.0):
    """Concentration-mass relation from Duffy et al. (2008).

    Parameters
    ----------
    M200 : float or array
        Halo mass in Msun.
    z : float
        Redshift.

    Returns
    -------
    float or array
        Concentration c200.
    """
    # Duffy+08 relaxed halos, 200c
    A, B, C = 5.71, -0.084, -0.47
    M_pivot = 2e12  # Msun/h (ignoring h for simplicity)
    return A * (M200 / M_pivot) ** B * (1 + z) ** C


@jit
def nfw_scale_density(M200, c200):
    """Compute NFW scale density from M200 and concentration.

    Parameters
    ----------
    M200 : float
        Virial mass in Msun.
    c200 : float
        Concentration.

    Returns
    -------
    float
        Scale density rho_s in Msun/kpc^3.
    """
    # R200 from M200 (assuming rho_crit ~ 127 Msun/kpc^3 at z=0)
    rho_crit = 127.0  # Msun/kpc^3
    R200 = (M200 / (4/3 * jnp.pi * 200 * rho_crit)) ** (1/3)
    r_s = R200 / c200

    # rho_s from mass normalization
    rho_s = M200 / (4 * jnp.pi * r_s**3 * (jnp.log(1 + c200) - c200 / (1 + c200)))
    return rho_s


def _nfw_log_density(log_r, log_rho_s, log_r_s):
    """NFW profile in log-log space for fitting.

    Args:
        log_r: log10(r / kpc).
        log_rho_s: log10(rho_s / (Msun/kpc^3)).
        log_r_s: log10(r_s / kpc).

    Returns:
        log10(rho) in Msun/kpc^3.
    """
    r = 10**log_r
    rho_s = 10**log_rho_s
    r_s = 10**log_r_s
    x = r / r_s
    rho = rho_s / (x * (1 + x)**2)
    return np.log10(rho)


def fit_nfw(r_mid, rho, r_fit_min=None, r_fit_max=None):
    """Fit an NFW profile to a measured density profile in log space.

    Args:
        r_mid: Bin centres in kpc, shape (n_bins,).
        rho: Measured density in Msun/kpc^3, shape (n_bins,).
        r_fit_min: Minimum radius for fit in kpc. If None, uses all bins.
        r_fit_max: Maximum radius for fit in kpc. If None, uses all bins.

    Returns:
        Dict with keys:
            rho_s: Best-fit scale density in Msun/kpc^3.
            r_s: Best-fit scale radius in kpc.
            rho_fit: Model density evaluated at r_mid, shape (n_bins,).
            chi2: Reduced chi-squared in log space (Despali+2026 Eq. A).
            success: Whether the fit converged.
    """
    r_mid = np.asarray(r_mid, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)

    # Select fitting range
    mask = rho > 0
    if r_fit_min is not None:
        mask &= r_mid >= r_fit_min
    if r_fit_max is not None:
        mask &= r_mid <= r_fit_max

    if mask.sum() < 3:
        return {"rho_s": np.nan, "r_s": np.nan, "rho_fit": np.full_like(rho, np.nan),
                "chi2": np.nan, "success": False}

    log_r = np.log10(r_mid[mask])
    log_rho = np.log10(rho[mask])

    # Initial guesses
    p0 = [6.5, 1.5]  # log10(rho_s), log10(r_s)

    try:
        popt, _ = curve_fit(_nfw_log_density, log_r, log_rho, p0=p0,
                            maxfev=10000)
        log_rho_s, log_r_s = popt
        rho_s = 10**log_rho_s
        r_s = 10**log_r_s

        # Model evaluated at all radii
        rho_fit = rho_s / ((r_mid / r_s) * (1 + r_mid / r_s)**2)

        # Chi-squared (Despali+2026 Appendix A)
        log_rho_model = _nfw_log_density(log_r, *popt)
        n_dof = 2  # two free parameters
        chi2 = np.sum((log_rho_model - log_rho)**2 / np.abs(log_rho)) / (len(log_r) - n_dof)

        return {"rho_s": rho_s, "r_s": r_s, "rho_fit": rho_fit,
                "chi2": chi2, "success": True}
    except (RuntimeError, ValueError):
        return {"rho_s": np.nan, "r_s": np.nan, "rho_fit": np.full_like(rho, np.nan),
                "chi2": np.nan, "success": False}


def _cored_nfw_log_density(log_r, log_rho_s, log_r_s, log_r_core):
    """Cored NFW profile in log-log space for fitting."""
    r = 10**log_r
    rho_s = 10**log_rho_s
    r_s = 10**log_r_s
    r_core = 10**log_r_core
    x = r / r_s
    rho_nfw = rho_s / (x * (1.0 + x)**2)
    rho = np.tanh(r / r_core) * rho_nfw
    return np.log10(rho)


def fit_cored_nfw(r_mid, rho, r_fit_min=None, r_fit_max=None,
                  p0_from_nfw=None):
    """Fit a 3-parameter cored NFW profile to a measured density profile.

    Args:
        r_mid: Bin centres, shape (n_bins,).
        rho: Measured density, shape (n_bins,).
        r_fit_min, r_fit_max: Fit range. If None, uses all bins.
        p0_from_nfw: Optional dict with 'rho_s' and 'r_s' from a prior
            NFW fit, used as initial guesses. r_core init defaults to
            10% of r_s.

    Returns:
        Dict with keys rho_s, r_s, r_core, rho_fit, chi2, success.
    """
    r_mid = np.asarray(r_mid, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)

    mask = rho > 0
    if r_fit_min is not None:
        mask &= r_mid >= r_fit_min
    if r_fit_max is not None:
        mask &= r_mid <= r_fit_max

    if mask.sum() < 4:
        return {"rho_s": np.nan, "r_s": np.nan, "r_core": np.nan,
                "rho_fit": np.full_like(rho, np.nan),
                "chi2": np.nan, "success": False}

    log_r = np.log10(r_mid[mask])
    log_rho = np.log10(rho[mask])

    if p0_from_nfw is not None and np.isfinite(p0_from_nfw.get("r_s", np.nan)):
        log_rs0 = np.log10(p0_from_nfw["r_s"])
        log_rhos0 = np.log10(p0_from_nfw["rho_s"]) if np.isfinite(
            p0_from_nfw.get("rho_s", np.nan)) else 6.5
        log_rc0 = log_rs0 - 1.0
    else:
        log_rhos0 = 6.5
        log_rs0 = 1.5
        log_rc0 = 0.0

    p0 = [log_rhos0, log_rs0, log_rc0]
    bounds_lo = [3.0,  -1.0, -3.0]
    bounds_hi = [10.0,  4.0,  4.0]

    try:
        popt, _ = curve_fit(
            _cored_nfw_log_density, log_r, log_rho, p0=p0,
            bounds=(bounds_lo, bounds_hi), maxfev=20000)
        log_rho_s, log_r_s, log_r_core = popt
        rho_s = 10**log_rho_s
        r_s = 10**log_r_s
        r_core = 10**log_r_core

        x = r_mid / r_s
        rho_nfw = rho_s / (x * (1.0 + x)**2)
        rho_fit = np.tanh(r_mid / r_core) * rho_nfw

        log_rho_model = _cored_nfw_log_density(log_r, *popt)
        n_dof = 3
        chi2 = np.sum((log_rho_model - log_rho)**2 / np.abs(log_rho)) / max(
            len(log_r) - n_dof, 1)

        return {"rho_s": rho_s, "r_s": r_s, "r_core": r_core,
                "rho_fit": rho_fit, "chi2": chi2, "success": True}
    except (RuntimeError, ValueError):
        return {"rho_s": np.nan, "r_s": np.nan, "r_core": np.nan,
                "rho_fit": np.full_like(rho, np.nan),
                "chi2": np.nan, "success": False}
