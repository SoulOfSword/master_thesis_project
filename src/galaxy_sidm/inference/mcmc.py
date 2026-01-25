"""MCMC sampling with emcee for parameter inference."""

import os
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import jax
import emcee
from loky import get_reusable_executor


# Default path for MCMC chains
DEFAULT_CHAIN_PATH = Path(__file__).parent.parent.parent.parent / "data" / "mcmc_chains"


@dataclass
class MCMCResult:
    """Container for MCMC results."""

    samples: np.ndarray      # Shape: (n_samples, n_params)
    log_prob: np.ndarray     # Shape: (n_samples,)
    param_names: list[str]
    acceptance_fraction: float
    backend_path: Path | None = None

    @property
    def median(self):
        """Median of each parameter."""
        return np.median(self.samples, axis=0)

    @property
    def std(self):
        """Standard deviation of each parameter."""
        return np.std(self.samples, axis=0)

    def percentiles(self, q=[16, 50, 84]):
        """Percentiles for each parameter."""
        return np.percentile(self.samples, q, axis=0)

    def to_dict(self):
        """Convert to dictionary with parameter names as keys."""
        return {name: self.samples[:, i] for i, name in enumerate(self.param_names)}


def _init_jax_worker():
    """Initialize JAX with float64 precision in worker processes."""
    os.environ['JAX_ENABLE_X64'] = '1'
    jax.config.update("jax_enable_x64", True)


def run_mcmc(
    log_prob_fn,
    n_walkers,
    n_dim,
    n_steps,
    param_names=None,
    filename=None,
    fresh_start=True,
    initial_pos=None,
    max_workers=12,
    n_steps_is_total=False,
    n_burn=0,
):
    """Run MCMC sampling with emcee, with HDF5 backend for checkpointing.

    Parameters
    ----------
    log_prob_fn : callable
        Log-probability function: log_prob(params) -> float.
    n_walkers : int
        Number of walkers.
    n_dim : int
        Number of parameters.
    n_steps : int
        Number of steps to run.
    param_names : list of str, optional
        Names for each parameter (for MCMCResult).
    filename : str or Path, optional
        Path to HDF5 backend file. If None, uses default location.
    fresh_start : bool
        If True, reset and start fresh. If False, continue from previous run.
    initial_pos : array-like, optional
        Initial positions for walkers, shape (n_walkers, n_dim).
        Required if fresh_start=True.
    max_workers : int
        Number of parallel workers.
    n_steps_is_total : bool
        If True, n_steps is the total target (will compute remaining).
        If False, n_steps is additional steps to run.
    n_burn : int
        Number of burn-in steps to discard when returning MCMCResult.

    Returns
    -------
    MCMCResult
        MCMC samples and diagnostics.
    """
    # Set up backend path
    if filename is None:
        DEFAULT_CHAIN_PATH.mkdir(parents=True, exist_ok=True)
        filename = DEFAULT_CHAIN_PATH / "mcmc_backend.h5"
    else:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

    backend = emcee.backends.HDFBackend(str(filename))

    if fresh_start:
        backend.reset(n_walkers, n_dim)
        if initial_pos is None:
            raise ValueError("initial_pos required for fresh start")
        pos = initial_pos
        steps_to_run = n_steps
        print(f"Starting fresh MCMC with {n_walkers} walkers, {steps_to_run} steps")
    else:
        current_steps = backend.iteration
        if current_steps == 0:
            raise ValueError("No previous run found. Use fresh_start=True")
        pos = backend.get_last_sample()

        if n_steps_is_total:
            steps_to_run = n_steps - current_steps
            if steps_to_run <= 0:
                print(f"Chain already has {current_steps} steps, target {n_steps}. Returning existing samples.")
                # Return existing samples
                samples = backend.get_chain(discard=n_burn, flat=True)
                log_prob = backend.get_log_prob(discard=n_burn, flat=True)
                return MCMCResult(
                    samples=samples,
                    log_prob=log_prob,
                    param_names=param_names or [f"p{i}" for i in range(n_dim)],
                    acceptance_fraction=np.nan,
                    backend_path=filename,
                )
        else:
            steps_to_run = n_steps
        print(f"Resuming from step {current_steps}, running {steps_to_run} more steps")

    # Set up parallel executor with JAX initialization
    executor = get_reusable_executor(max_workers=max_workers, initializer=_init_jax_worker)

    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_prob_fn,
        pool=executor, backend=backend
    )

    # Run MCMC
    sampler.run_mcmc(pos, steps_to_run, progress=True)

    print(f"Done! Total steps: {backend.iteration}")

    # Extract samples
    samples = backend.get_chain(discard=n_burn, flat=True)
    log_prob_samples = backend.get_log_prob(discard=n_burn, flat=True)

    return MCMCResult(
        samples=samples,
        log_prob=log_prob_samples,
        param_names=param_names or [f"p{i}" for i in range(n_dim)],
        acceptance_fraction=np.mean(sampler.acceptance_fraction),
        backend_path=filename,
    )


def get_sampler_from_backend(filename):
    """Load an existing backend to inspect chains without running.

    Parameters
    ----------
    filename : str or Path
        Path to HDF5 backend file.

    Returns
    -------
    backend : emcee.backends.HDFBackend
        The loaded backend with chain data.
    """
    return emcee.backends.HDFBackend(str(filename), read_only=True)
