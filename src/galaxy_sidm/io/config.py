"""Load shared script configuration from config/scripts.yaml."""

from pathlib import Path

import yaml


_DEFAULT_CONFIG = (Path(__file__).resolve().parents[3]
                   / "config" / "scripts.yaml")


def load_config(path=None):
    """Read the shared YAML config into a dict.

    Args:
        path: optional explicit path. Defaults to config/scripts.yaml at
            the project root.

    Returns:
        dict mirroring the YAML structure.
    """
    p = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)
