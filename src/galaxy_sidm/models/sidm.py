"""SIDM cross-section models and physics (JAX implementation)."""

import jax.numpy as jnp
from jax import jit

from ..config import get_sidm_params


# Unit conversion constants
CM_TO_KPC = 3.24078e-22
G_TO_MSUN = 5.0279e-34
KMS_TO_KPC_PER_GYR = 1.0227  # 1 km/s ~ 1.02 kpc/Gyr


@jit
def cross_section_constant(v, sigma_m=1.0):
    """Constant (velocity-independent) SIDM cross-section.

    Parameters
    ----------
    v : array
        Relative velocity in km/s (unused, for API consistency).
    sigma_m : float
        Cross-section per unit mass in cm^2/g.

    Returns
    -------
    array
        Cross-section sigma/m at each velocity in cm^2/g.
    """
    return jnp.full_like(v, sigma_m, dtype=jnp.float64)


@jit
def cross_section_velocity_dependent(v, sigma_0=50.0, w=30.0):
    """Velocity-dependent cross-section (Yukawa-like).

    Common parameterization: sigma/m = sigma_0 / (1 + (v/w)^2)^2

    This gives large cross-sections in dwarf galaxies (low v)
    and small cross-sections in clusters (high v).

    Parameters
    ----------
    v : array
        Relative velocity in km/s.
    sigma_0 : float
        Cross-section at v=0 in cm^2/g.
    w : float
        Characteristic velocity scale in km/s.

    Returns
    -------
    array
        Cross-section sigma/m at each velocity in cm^2/g.
    """
    return sigma_0 / (1 + (v / w) ** 2) ** 2


@jit
def cross_section_resonant(v, sigma_0=10.0, v_res=50.0, gamma=10.0):
    """Resonant cross-section model.

    Features a peak at a characteristic velocity, motivated by
    particle physics models with resonances.

    sigma/m = sigma_0 * gamma^2 / ((v - v_res)^2 + gamma^2)

    Parameters
    ----------
    v : array
        Relative velocity in km/s.
    sigma_0 : float
        Peak cross-section in cm^2/g.
    v_res : float
        Resonance velocity in km/s.
    gamma : float
        Width of resonance in km/s.

    Returns
    -------
    array
        Cross-section sigma/m at each velocity in cm^2/g.
    """
    return sigma_0 * gamma**2 / ((v - v_res)**2 + gamma**2)


def get_cross_section_function(config=None):
    """Get the appropriate cross-section function based on config.

    Returns a JIT-compiled function sigma_m(v).

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.

    Returns
    -------
    callable
        Cross-section function sigma_m(v) -> array.
    """
    params = get_sidm_params(config)
    model = params.get("model", "constant")

    if model == "constant":
        sigma_m = params.get("sigma_m", 1.0)

        @jit
        def sigma_fn(v):
            return cross_section_constant(v, sigma_m)
        return sigma_fn

    elif model == "velocity_dependent":
        sigma_0 = params.get("sigma_0", 50.0)
        w = params.get("w", 30.0)

        @jit
        def sigma_fn(v):
            return cross_section_velocity_dependent(v, sigma_0, w)
        return sigma_fn

    else:
        raise ValueError(f"Unknown SIDM model: {model}")


@jit
def scattering_rate(rho, v_rms, sigma_m):
    """DM self-scattering rate per particle.

    Parameters
    ----------
    rho : float or array
        Local DM density in Msun/kpc^3.
    v_rms : float or array
        RMS velocity in km/s.
    sigma_m : float or array
        Cross-section per unit mass in cm^2/g.

    Returns
    -------
    float or array
        Scattering rate in Gyr^-1.
    """
    # Convert sigma_m from cm^2/g to kpc^2/Msun
    sigma_m_kpc2_per_Msun = sigma_m * (CM_TO_KPC**2) / G_TO_MSUN

    # Convert v_rms from km/s to kpc/Gyr
    v_rms_kpc_Gyr = v_rms * KMS_TO_KPC_PER_GYR

    # Rate = rho * sigma/m * v
    return rho * sigma_m_kpc2_per_Msun * v_rms_kpc_Gyr


@jit
def core_formation_timescale(rho_s, r_s, sigma_m, v_max):
    """Estimate timescale for SIDM core formation.

    Based on the condition that ~1 scattering per particle
    occurs within the scale radius.

    Parameters
    ----------
    rho_s : float
        NFW scale density in Msun/kpc^3.
    r_s : float
        NFW scale radius in kpc.
    sigma_m : float
        Cross-section in cm^2/g.
    v_max : float
        Maximum circular velocity in km/s.

    Returns
    -------
    float
        Core formation timescale in Gyr.
    """
    # Characteristic density at r ~ r_s is rho_s / 4
    rho_char = rho_s / 4

    # Scattering rate at characteristic radius
    rate = scattering_rate(rho_char, v_max, sigma_m)

    # Timescale is inverse of rate
    return 1.0 / rate


@jit
def effective_cross_section(v_rms, sigma_0, w):
    """Thermally-averaged effective cross-section.

    For velocity-dependent cross-sections, computes the
    effective value averaged over a Maxwell-Boltzmann distribution.

    Approximation: <sigma v> / <v> evaluated at v = sqrt(2) * v_rms

    Parameters
    ----------
    v_rms : float
        RMS velocity of the halo in km/s.
    sigma_0 : float
        Cross-section at v=0 in cm^2/g.
    w : float
        Characteristic velocity in km/s.

    Returns
    -------
    float
        Effective cross-section in cm^2/g.
    """
    # Characteristic collision velocity
    v_coll = jnp.sqrt(2) * v_rms
    return cross_section_velocity_dependent(v_coll, sigma_0, w)
