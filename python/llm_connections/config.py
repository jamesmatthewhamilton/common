"""YAML config loader. Only reads the 'llm-providers:' key, ignores everything else."""

import os
import re

import yaml

YAML_KEY = "llm-providers"


def _expand_env_vars(value):
    """Recursively expand ${ENV_VAR} references in strings."""
    if isinstance(value, str):
        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return re.sub(r'\$\{(\w+)\}', replacer, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def load_providers(yaml_path: str) -> dict:
    """Load LLM providers from a YAML file.

    Only reads the 'llm-providers:' key. All other keys are ignored.
    Expands ${ENV_VAR} references in string values.

    Returns:
        dict mapping provider name -> provider config dict
    """
    yaml_path = os.path.expanduser(yaml_path)
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    providers = raw.get(YAML_KEY, {})
    if not providers:
        raise ValueError(f"No '{YAML_KEY}:' key found in {yaml_path}")

    return _expand_env_vars(providers)
