"""End-to-end tests for SlurmSession with mocked subprocess + tunnel.

Mirrors the original tests in slurm_manipulator/tests/test_session.py, but
with imports adjusted for the new home in llm_connections.slurm_session.
"""

import textwrap
from unittest.mock import MagicMock, patch

import pytest

from llm_connections.slurm_session import SessionHandle, SlurmSession


VALID_YAML = textwrap.dedent("""\
    slurm-sessions:
      clusters:
        acme:
          ssh:
            user: alice
            host: login.acme.example
          paths:
            log_dir: ~/log
            ollama_bin: /opt/ollama
            ollama_models_dir: /scratch/models
          endpoint:
            file_pattern: "~/ollama-endpoint-${job_id}.txt"
          readiness:
            path: /api/tags
            timeout: 30

      sessions:
        acme-llama-70b:
          cluster: acme
          model: llama3.1:70b
          sbatch: BUNDLED
          sbatch_params:
            partition: gpu
            gres: gpu:1
            cpus: 8
            mem: 64G
            time: "08:00:00"
            port: 11070
""")


@pytest.fixture
def loaded(tmp_path):
    """Reset the registry, load a tmp config, point sbatch at slurm-manipulator's bundled template."""
    SlurmSession._clusters.clear()
    SlurmSession._sessions.clear()
    SlurmSession._active.clear()

    import slurm_manipulator
    import os
    bundled = os.path.join(
        os.path.dirname(slurm_manipulator.__file__),
        "sbatch", "ollama-generic.sbatch",
    )
    body = VALID_YAML.replace("BUNDLED", bundled)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(body)
    SlurmSession.load(str(cfg))
    yield
    SlurmSession._clusters.clear()
    SlurmSession._sessions.clear()
    SlurmSession._active.clear()


def _mk_run(scripted):
    """Build a fake subprocess.run that pops scripted responses by command match."""
    queue = list(scripted)

    def fake_run(cmd, *args, **kwargs):
        joined = " ".join(cmd) if isinstance(cmd, list) else cmd
        for i, (needle, rc, out, err) in enumerate(queue):
            if needle in joined:
                queue.pop(i)
                result = MagicMock()
                result.returncode = rc
                result.stdout = out
                result.stderr = err
                return result
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    return fake_run


def test_list_sessions_and_clusters(loaded):
    assert "acme-llama-70b" in SlurmSession.list_sessions()
    assert "acme" in SlurmSession.list_clusters()


def test_get_unknown_session_raises(loaded):
    with pytest.raises(KeyError, match="not found"):
        SlurmSession.get("nope")


def test_start_fresh_submit(loaded):
    """No existing job — submit, poll, read endpoint, tunnel, probe."""
    scripted = [
        ("squeue -u alice", 0, "", ""),
        ("mkdir -p", 0, "", ""),
        ("cat >", 0, "", ""),
        ("sed -i", 0, "", ""),
        ("sbatch", 0, "Submitted batch job 12345\n", ""),
        ("squeue -j 12345", 0, "RUNNING", ""),
        ("cat ~/ollama-endpoint-12345.txt", 0,
         "http://node-1.acme.example:11070\n", ""),
        ("scontrol show job 12345", 0,
         "Partition=gpu Gres=gpu:1 NodeList=node-1.acme.example", ""),
    ]

    with patch("slurm_manipulator.slurm.subprocess.run", side_effect=_mk_run(scripted)), \
         patch("llm_connections.slurm_session.open_tunnel", return_value=54321), \
         patch("llm_connections.slurm_session.probe_http_ok", return_value=True):
        h = SlurmSession.get("acme-llama-70b").start(reuse_existing=True)

    assert h.job_id == "12345"
    assert h.local_url == "http://localhost:54321"
    assert h.remote_host == "node-1.acme.example"
    assert h.remote_port == 11070


def test_start_reuses_running_job(loaded):
    """Existing RUNNING job — skip submit."""
    scripted = [
        ("squeue -u alice", 0, "RUNNING acme-llama-70b 99999\n", ""),
        ("sbatch", 1, "", "sbatch should not have been called"),
        ("cat ~/ollama-endpoint-99999.txt", 0, "http://node-9.acme.example:11070\n", ""),
        ("scontrol show job 99999", 0, "Partition=gpu", ""),
    ]
    with patch("slurm_manipulator.slurm.subprocess.run", side_effect=_mk_run(scripted)), \
         patch("llm_connections.slurm_session.open_tunnel", return_value=54321), \
         patch("llm_connections.slurm_session.probe_http_ok", return_value=True):
        h = SlurmSession.get("acme-llama-70b").start()

    assert h.job_id == "99999"


def test_handle_kill_calls_stop_with_cancel(loaded):
    h = SessionHandle(
        name="x", model="m", job_id="42", local_url="http://localhost:1",
        remote_host="r", remote_port=1, local_port=1, gpu_info="",
        cluster="acme", ssh_user="u", ssh_host="h", ssh_password="",
    )
    with patch.object(SessionHandle, "stop") as mock_stop:
        h.kill()
    mock_stop.assert_called_once_with(cancel_job_on_cluster=True)


def test_active_by_local_url_finds_handle(loaded):
    SlurmSession._active.clear()
    h1 = SessionHandle(
        name="a", model="m", job_id="1", local_url="http://localhost:1111",
        remote_host="r", remote_port=1, local_port=1111, gpu_info="",
        cluster="c", ssh_user="u", ssh_host="h",
    )
    h2 = SessionHandle(
        name="b", model="m", job_id="2", local_url="http://localhost:2222",
        remote_host="r", remote_port=2, local_port=2222, gpu_info="",
        cluster="c", ssh_user="u", ssh_host="h",
    )
    SlurmSession._active["a"] = h1
    SlurmSession._active["b"] = h2
    try:
        assert SlurmSession.active_by_local_url("http://localhost:2222") is h2
        assert SlurmSession.active_by_local_url("http://localhost:1111") is h1
        assert SlurmSession.active_by_local_url("http://nowhere") is None
    finally:
        SlurmSession._active.clear()


def test_peek_existing(loaded):
    scripted = [("squeue -u alice", 0, "RUNNING acme-llama-70b 7777\n", "")]
    with patch("slurm_manipulator.slurm.subprocess.run", side_effect=_mk_run(scripted)):
        result = SlurmSession.get("acme-llama-70b").peek_existing()
    assert result == ("RUNNING", "7777")
