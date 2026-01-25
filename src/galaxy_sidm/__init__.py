"""
galaxy-sidm: Semi-analytical SIDM galaxy evolution models.

This package provides tools for:
- Loading and analyzing AIDA-TNG simulation data
- Computing SIDM physics (cross-sections, core formation)
- Semi-analytical galaxy evolution modeling
- Calculating scaling relations (Tully-Fisher, mass-size, Fall relation)
- MCMC inference for model parameters
"""

from . import config
from . import cosmology
from . import data
from . import models
from . import observables
from . import inference

__version__ = "0.1.0"
