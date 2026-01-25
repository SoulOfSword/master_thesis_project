"""Physics models for SIDM and galaxy evolution (JAX-accelerated)."""

from .profiles import (
    nfw_density,
    nfw_mass,
    nfw_circular_velocity,
    cored_nfw_density,
    einasto_density,
    isothermal_core_density,
    concentration_duffy08,
    nfw_scale_density,
)
from .sidm import (
    cross_section_constant,
    cross_section_velocity_dependent,
    cross_section_resonant,
    get_cross_section_function,
    scattering_rate,
    core_formation_timescale,
    effective_cross_section,
)
from .sam import (
    SAMParameters,
    stellar_mass_halo_mass,
    virial_radius,
    galaxy_size_cdm,
    galaxy_size_sidm,
    rotation_velocity_nfw,
    rotation_velocity_cored,
    specific_angular_momentum_galaxy,
    predict_galaxy_properties,
)
