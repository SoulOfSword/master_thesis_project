"""Configuration loading and management."""

from pathlib import Path
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "default_params.yaml"


def load_config(path = None):
    """Load configuration from YAML file.

    Parameters
    ----------
    path : Path or str, optional
        Path to config file. If None, loads default_params.yaml.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH

    with open(path) as f:
        return yaml.safe_load(f)


def get_cosmology_params(config = None):
    """Extract cosmology parameters from config."""
    if config is None:
        config = load_config()
    return config.get("cosmology", {})


def get_sidm_params(config = None):
    """Extract SIDM parameters from config."""
    if config is None:
        config = load_config()
    return config.get("sidm", {})
