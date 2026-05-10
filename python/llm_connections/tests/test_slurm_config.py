"""Unit tests for llm_connections.slurm_config — schema validation under
the new 'slurm-sessions:' top-level key in ~/.llm-connections/config.yaml.
"""

import os
import textwrap

import pytest

from llm_connections.slurm_config import _expand_env_vars, load_slurm_sessions


VALID_YAML = textwrap.dedent("""\
    llm-providers:
      default:
        provider: ollama
        model: llama3.1:8b

    slurm-sessions:
      clusters:
        acme:
          ssh:
            user: alice
            host: login.acme.example
          paths:
            log_dir: ~/log
            ollama_bin: ~/bin/ollama
            ollama_models_dir: ~/models
          endpoint:
            file_pattern: "~/ollama-endpoint-${job_id}.txt"
          readiness:
            path: /api/tags
            timeout: 30

      sessions:
        acme-llama-70b:
          cluster: acme
          model: llama3.1:70b
          sbatch: ~/.llm-connections/sbatch/ollama-generic.sbatch
          sbatch_params:
            partition: gpu
            gres: gpu:1
            cpus: 8
            mem: 64G
            time: "08:00:00"
            port: 11070
""")


def _write(tmp_path, body):
    p = tmp_path / "config.yaml"
    p.write_text(body)
    return str(p)


def test_load_valid_config(tmp_path):
    path = _write(tmp_path, VALID_YAML)
    clusters, sessions = load_slurm_sessions(path)
    assert "acme" in clusters
    assert "acme-llama-70b" in sessions
    assert sessions["acme-llama-70b"]["cluster"] == "acme"


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_slurm_sessions(str(tmp_path / "nope.yaml"))


def test_missing_top_level_key(tmp_path):
    path = _write(tmp_path, "llm-providers:\n  foo: {provider: ollama, model: x}\n")
    with pytest.raises(ValueError, match="slurm-sessions"):
        load_slurm_sessions(path)


def test_endpoint_pattern_missing_job_id(tmp_path):
    body = VALID_YAML.replace(
        '"~/ollama-endpoint-${job_id}.txt"',
        '"~/ollama-endpoint.txt"',
    )
    path = _write(tmp_path, body)
    with pytest.raises(ValueError, match=r"\$\{job_id\}"):
        load_slurm_sessions(path)


def test_session_references_unknown_cluster(tmp_path):
    body = VALID_YAML.replace("cluster: acme", "cluster: nonexistent")
    path = _write(tmp_path, body)
    with pytest.raises(ValueError, match="unknown cluster"):
        load_slurm_sessions(path)


def test_cluster_missing_ssh_user(tmp_path):
    body = VALID_YAML.replace("user: alice", "")
    path = _write(tmp_path, body)
    with pytest.raises(ValueError, match="ssh.user or ssh.host"):
        load_slurm_sessions(path)


def test_env_var_expanded():
    os.environ["LC_TEST_VAR"] = "alice"
    try:
        result = _expand_env_vars({"ssh": {"user": "${LC_TEST_VAR}"}})
        assert result["ssh"]["user"] == "alice"
    finally:
        del os.environ["LC_TEST_VAR"]


def test_unknown_env_var_left_alone():
    assert _expand_env_vars("~/x-${job_id}.txt") == "~/x-${job_id}.txt"


def test_session_missing_model(tmp_path):
    body = VALID_YAML.replace("model: llama3.1:70b", "")
    path = _write(tmp_path, body)
    with pytest.raises(ValueError, match="missing 'model'"):
        load_slurm_sessions(path)


def test_coexists_with_llm_providers(tmp_path):
    """The same YAML file can hold both top-level keys; each loader reads its own."""
    path = _write(tmp_path, VALID_YAML)
    # slurm_config only cares about slurm-sessions:
    clusters, sessions = load_slurm_sessions(path)
    assert "acme-llama-70b" in sessions
    # Doesn't barf on the presence of llm-providers:
