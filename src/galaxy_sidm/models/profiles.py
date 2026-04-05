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
    """Cored NFW profile (approximation for SIDM halos).

    Uses the Read et al. (2016) coreNFW parameterization where
    SIDM scattering creates a constant-density core.

    Parameters
    ----------
    r : array
        Radius in kpc.
    rho_s : float
        Scale density in Msun/kpc^3.
    r_s : float
        Scale radius in kpc.
    r_core : float
        Core radius in kpc.

    Returns
    -------
    array
        Density at each radius in Msun/kpc^3.
    """
    x = r / r_s
    f_core = jnp.tanh(r / r_core)
    return rho_s / ((r / r_core + f_core * x) * (1 + x) ** 2)


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
