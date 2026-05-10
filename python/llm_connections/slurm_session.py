"""SlurmSession — config-driven orchestrator for cluster-spawned LLM endpoints.

Composes:
  - slurm_manipulator (Slurm primitives: submit/find/poll/cancel + render)
  - llm_connections.ssh (tunnel + readiness probe)

Reads ~/.llm-connections/config.yaml's 'slurm-sessions:' top-level key.
Optional dependency: importing this module requires slurm-manipulator to be
on the Python path (it's a nested optional submodule of llm-connections).

Usage:
    from llm_connections import SlurmSession

    SlurmSession.load()                                 # ~/.llm-connections/config.yaml
    handle = SlurmSession.get("pace-llama-70b").start()
    print(handle.local_url)                              # http://localhost:54123
    handle.kill()                                        # tear down tunnel + scancel
"""

import logging
import os
import shutil
from dataclasses import dataclass, field

# slurm-manipulator is an optional dependency. Import lazily and fail with a
# clear message if it's not installed.
try:
    from slurm_manipulator import (
        cancel_job,
        endpoint_file_for_client,
        find_existing_job,
        parse_job_name_from_sbatch,
        parse_scontrol_gpu_info,
        poll_until_running,
        query_gpu_info,
        read_endpoint_file,
        render_sbatch,
        submit_sbatch,
    )
except ImportError as exc:
    raise ImportError(
        "llm_connections.slurm_session requires the optional 'slurm-manipulator' "
        "package. If you're using llm-connections as a submodule, init the "
        "nested submodule:\n"
        "    cd llm-connections && git submodule update --init slurm-manipulator\n"
        "Then add llm-connections/slurm-manipulator/python to PYTHONPATH."
    ) from exc

from .slurm_config import load_slurm_sessions
from .ssh import close_tunnel, open_tunnel, probe_http_ok

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_DIR = os.path.expanduser("~/.llm-connections")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_CONFIG_DIR, "config.yaml")
DEFAULT_SBATCH_DIR = os.path.join(DEFAULT_CONFIG_DIR, "sbatch")

# When the user doesn't supply their own sbatch, we copy the bundled one from
# the slurm-manipulator submodule on first run.
def _bundled_sbatch_path() -> str:
    """Locate slurm-manipulator's bundled ollama-generic.sbatch."""
    import slurm_manipulator as _sm
    return os.path.join(
        os.path.dirname(_sm.__file__), "sbatch", "ollama-generic.sbatch",
    )


@dataclass
class SessionHandle:
    """A running model server reachable via local SSH tunnel.

    Attributes:
        name, model, job_id, cluster — identifiers
        local_url, local_port — what the caller talks to
        remote_host, remote_port — where the tunnel forwards to
        gpu_info — short summary, e.g. "partition: gpu-h200, gres: gpu:1, ..."

    The handle owns the local tunnel; stop() tears it down. The Slurm job is
    NOT cancelled by stop() unless cancel_job=True is passed (or via kill()).
    """
    name: str
    model: str
    job_id: str
    local_url: str
    remote_host: str
    remote_port: int
    local_port: int
    gpu_info: str
    cluster: str
    ssh_user: str = field(repr=False, default="")
    ssh_host: str = field(repr=False, default="")
    ssh_password: str = field(repr=False, default="")
    _stopped: bool = field(repr=False, default=False)

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.ssh_host}"

    def stop(self, cancel_job_on_cluster: bool = False) -> None:
        """Close the local SSH tunnel. Optionally scancel the Slurm job."""
        if self._stopped:
            return
        close_tunnel(self.local_port)
        if cancel_job_on_cluster:
            cancel_job(self.ssh_target, self.job_id, self.ssh_password)
        self._stopped = True

    def kill(self) -> None:
        """Tear down the tunnel AND scancel the Slurm job.

        Synonym for stop(cancel_job_on_cluster=True). Use this when you want to
        fully terminate the AI behind this session, not just disconnect.
        """
        self.stop(cancel_job_on_cluster=True)


class SlurmSession:
    """Class-level registry of (cluster, session) configs.

    Sessions are not started at load() time. Call .start() on a fetched
    SlurmSession to submit/find the Slurm job, open the tunnel, and verify
    readiness.
    """

    _clusters: dict = {}
    _sessions: dict = {}
    _active: dict = {}        # name -> SessionHandle

    def __init__(self, name: str, session_cfg: dict, cluster_cfg: dict):
        self._name = name
        self._session_cfg = session_cfg
        self._cluster_cfg = cluster_cfg

    # ── classmethods (registry) ──────────────────────────────────────────

    @classmethod
    def load(cls, yaml_path: str | None = None) -> None:
        """Load and validate slurm-sessions: from a YAML file.

        Defaults to ~/.llm-connections/config.yaml. On first run, also copies
        the bundled ollama-generic.sbatch into ~/.llm-connections/sbatch/ so
        sessions referencing that path Just Work without the user touching
        the submodule.
        """
        if yaml_path is None:
            yaml_path = DEFAULT_CONFIG_PATH

        # First-run: stub a sbatch copy if missing.
        user_sbatch = os.path.join(DEFAULT_SBATCH_DIR, "ollama-generic.sbatch")
        if not os.path.isfile(user_sbatch):
            try:
                bundled = _bundled_sbatch_path()
                if os.path.isfile(bundled):
                    os.makedirs(DEFAULT_SBATCH_DIR, exist_ok=True)
                    shutil.copy(bundled, user_sbatch)
                    logger.info(f"Copied bundled sbatch template to {user_sbatch}")
            except Exception as exc:
                logger.warning(f"Could not copy bundled sbatch: {exc}")

        clusters, sessions = load_slurm_sessions(yaml_path)
        cls._clusters.update(clusters)
        cls._sessions.update(sessions)

    @classmethod
    def get(cls, name: str) -> "SlurmSession":
        if name not in cls._sessions:
            available = ", ".join(cls._sessions.keys()) or "(none)"
            raise KeyError(
                f"Slurm session '{name}' not found. Available: [{available}]. "
                f"Add it under 'slurm-sessions:' in {DEFAULT_CONFIG_PATH}."
            )
        sess_cfg = cls._sessions[name]
        cluster_cfg = cls._clusters[sess_cfg["cluster"]]
        return cls(name, sess_cfg, cluster_cfg)

    @classmethod
    def list_sessions(cls) -> list:
        return list(cls._sessions.keys())

    @classmethod
    def list_clusters(cls) -> list:
        return list(cls._clusters.keys())

    @classmethod
    def active(cls, name: str) -> SessionHandle | None:
        return cls._active.get(name)

    @classmethod
    def active_by_local_url(cls, url: str) -> SessionHandle | None:
        """Find the active session whose tunnel is bound to this local URL.

        Lets a caller go from `OllamaProvider.base_url` (e.g. http://localhost:54321)
        back to the SessionHandle that owns it, so it can be killed.
        """
        for handle in cls._active.values():
            if handle.local_url == url:
                return handle
        return None

    # ── instance methods (lifecycle) ─────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._session_cfg.get("model", "")

    def peek_existing(self) -> tuple[str, str] | None:
        """Lightweight check: is a job for this session already in squeue?

        Returns (state, job_id) or None. Does not render the sbatch, submit
        anything, open a tunnel, or block.
        """
        ssh_cfg = self._cluster_cfg.get("ssh") or {}
        ssh_user = ssh_cfg["user"]
        ssh_host = ssh_cfg["host"]
        ssh_password = ssh_cfg.get("password", "") or ""
        ssh_target = f"{ssh_user}@{ssh_host}"
        sbatch_params = self._session_cfg.get("sbatch_params") or {}
        job_name = sbatch_params.get("job_name") or self._name
        return find_existing_job(ssh_target, ssh_user, job_name, ssh_password)

    def start(self,
              wait_for_endpoint: int = 600,
              wait_for_ready: int = 30,
              reuse_existing: bool = True) -> SessionHandle:
        """Submit-or-find Slurm job, tunnel, verify, return SessionHandle."""
        ssh_cfg = self._cluster_cfg.get("ssh") or {}
        ssh_user = ssh_cfg["user"]
        ssh_host = ssh_cfg["host"]
        ssh_password = ssh_cfg.get("password", "") or ""
        ssh_target = f"{ssh_user}@{ssh_host}"

        endpoint_pattern = self._cluster_cfg["endpoint"]["file_pattern"]

        # 1. Render sbatch (job_name comes from sbatch directive in template).
        sbatch_template = self._session_cfg["sbatch"]
        rendered = render_sbatch(
            sbatch_template, self._cluster_cfg, self._name, self._session_cfg,
        )
        job_name = parse_job_name_from_sbatch(rendered) or self._name
        remote_basename = f"{job_name}.sbatch"

        # 2. Find existing job or submit a new one.
        job_id = None
        existing = None
        if reuse_existing:
            existing = find_existing_job(
                ssh_target, ssh_user, job_name, ssh_password,
            )
        if existing is not None:
            state, job_id = existing
            logger.info(f"  Found existing job '{job_name}' ({job_id}) in state {state}")
            if state == "PENDING":
                poll_until_running(ssh_target, job_id, ssh_password)
            elif state != "RUNNING":
                logger.info(f"  Existing job in state {state}; submitting a new one.")
                existing = None

        if existing is None:
            log_dir = (self._cluster_cfg.get("paths") or {}).get("log_dir", "~/log")
            job_id = submit_sbatch(
                ssh_target, rendered, remote_basename, log_dir, ssh_password,
            )
            poll_until_running(ssh_target, job_id, ssh_password)

        # 3. Read endpoint file for (host, port).
        endpoint_path = endpoint_file_for_client(endpoint_pattern, job_id)
        remote_host, remote_port = read_endpoint_file(
            ssh_target, endpoint_path, ssh_password,
            total_timeout=wait_for_endpoint,
        )

        # 4. Tunnel.
        local_port = open_tunnel(
            ssh_user, ssh_host, remote_host, remote_port,
            ssh_password=ssh_password,
        )
        local_url = f"http://localhost:{local_port}"

        # 5. Readiness probe.
        readiness = self._cluster_cfg.get("readiness") or {}
        probe_path = readiness.get("path", "/")
        probe_timeout = int(readiness.get("timeout", wait_for_ready))
        if not probe_http_ok(local_url + probe_path,
                             total_timeout=probe_timeout):
            close_tunnel(local_port)
            raise ConnectionError(
                f"SSH tunnel up at {local_url} but {probe_path} did not "
                f"respond within {probe_timeout}s. "
                f"Check job log on {ssh_target}."
            )

        # 6. Build handle.
        gpu_info = query_gpu_info(ssh_target, job_id, ssh_password)
        handle = SessionHandle(
            name=self._name,
            model=self.model,
            job_id=job_id,
            local_url=local_url,
            remote_host=remote_host,
            remote_port=remote_port,
            local_port=local_port,
            gpu_info=gpu_info,
            cluster=self._session_cfg["cluster"],
            ssh_user=ssh_user,
            ssh_host=ssh_host,
            ssh_password=ssh_password,
        )
        type(self)._active[self._name] = handle
        return handle

    def stop(self, cancel_job_on_cluster: bool = False) -> None:
        """Convenience: stop the active handle for this session, if any."""
        handle = type(self)._active.pop(self._name, None)
        if handle is not None:
            handle.stop(cancel_job_on_cluster=cancel_job_on_cluster)
