"""YAML config loader for the optional slurm_session orchestrator.

Reads ~/.llm-connections/config.yaml (the same file LLMConnection.load uses)
and parses only the 'slurm-sessions:' top-level key. Schema:

    slurm-sessions:
      clusters:
        <cluster_name>:
          ssh: { user, host, password? }
          paths: { log_dir, ollama_bin, ollama_models_dir }
          endpoint:
            file_pattern: "~/ollama-endpoint-${job_id}.txt"
          readiness: { path, timeout }
          bootstrap: { enabled }                              # optional, currently inert

      sessions:
        <session_name>:
          cluster: <cluster_name>
          model: <ollama model tag>
          sbatch: <path to sbatch template>
          sbatch_params: { partition, gres, cpus, mem, time, port, ... }

LLMConnection's llm-providers: key and our slurm-sessions: key coexist in
the same YAML file. Other top-level keys are ignored.
"""

import os
import re

import yaml

YAML_KEY = "slurm-sessions"


def _expand_env_vars(value):
    """Recursively expand ${ENV_VAR} references in strings.

    Variables in the form ${WORD_LIKE} that aren't set in os.environ are
    left untouched, so substitution patterns like ${job_id} survive into
    sbatch templates.
    """
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


def _validate_cluster(name: str, cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"Cluster '{name}' must be a mapping")
    ssh = cfg.get("ssh") or {}
    if not ssh.get("user") or not ssh.get("host"):
        raise ValueError(f"Cluster '{name}' missing ssh.user or ssh.host")
    endpoint = cfg.get("endpoint") or {}
    pattern = endpoint.get("file_pattern")
    if not pattern:
        raise ValueError(
            f"Cluster '{name}' missing endpoint.file_pattern "
            f'(e.g. "~/ollama-endpoint-${{job_id}}.txt")'
        )
    if "${job_id}" not in pattern:
        raise ValueError(
            f"Cluster '{name}' endpoint.file_pattern must contain literal "
            f"'${{job_id}}' so client and sbatch agree on the rendezvous file. "
            f"Got: {pattern!r}"
        )


def _validate_session(name: str, cfg: dict, clusters: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"Session '{name}' must be a mapping")
    cluster_name = cfg.get("cluster")
    if not cluster_name:
        raise ValueError(f"Session '{name}' missing 'cluster' key")
    if cluster_name not in clusters:
        available = ", ".join(clusters.keys()) or "(none)"
        raise ValueError(
            f"Session '{name}' references unknown cluster '{cluster_name}'. "
            f"Available: [{available}]"
        )
    if not cfg.get("model"):
        raise ValueError(f"Session '{name}' missing 'model' key")
    if not cfg.get("sbatch"):
        raise ValueError(f"Session '{name}' missing 'sbatch' key (path to template)")


def load_slurm_sessions(yaml_path: str) -> tuple[dict, dict]:
    """Load and validate slurm-sessions: from the LLM-connections config file.

    Returns:
        (clusters, sessions): both dicts keyed by name. Env vars expanded.

    Raises:
        FileNotFoundError: yaml_path does not exist
        ValueError: schema violations OR no 'slurm-sessions:' key
    """
    yaml_path = os.path.expanduser(yaml_path)
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    section = raw.get(YAML_KEY)
    if not section:
        raise ValueError(
            f"No '{YAML_KEY}:' key found in {yaml_path}. "
            f"Add it alongside the existing 'llm-providers:' key."
        )

    clusters = _expand_env_vars(section.get("clusters") or {})
    sessions = _expand_env_vars(section.get("sessions") or {})

    if not clusters:
        raise ValueError(f"No clusters defined in '{YAML_KEY}:' in {yaml_path}")
    if not sessions:
        raise ValueError(f"No sessions defined in '{YAML_KEY}:' in {yaml_path}")

    for cname, cfg in clusters.items():
        _validate_cluster(cname, cfg)
    for sname, cfg in sessions.items():
        _validate_session(sname, cfg, clusters)

    return clusters, sessions
