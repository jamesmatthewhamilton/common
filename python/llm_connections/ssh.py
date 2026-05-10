"""SSH primitives: tunnel + probe + reachability + connectivity.

Stdlib only — readiness probes use urllib.request, no httpx/requests.
Used by OllamaProvider (for ssh_tunnel:) and SlurmSession (for cluster
tunneling). Slurm-side _ssh_cmd / _scp_cmd builders live in the optional
slurm-manipulator submodule (slurm_manipulator.ssh_cmd).
"""

import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


# ── tunnel + port primitives ─────────────────────────────────────────────


def find_free_port() -> int:
    """Find a free local port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def open_tunnel(
    ssh_user: str,
    ssh_host: str,
    remote_host: str,
    remote_port: int,
    local_port: int = 0,
    ssh_password: str = None,
    verify_url: str = None,
    verify_timeout: int = 15,
) -> int:
    """Open an SSH tunnel: localhost:local_port -> remote_host:remote_port via ssh_host.

    Args:
        ssh_user: SSH username
        ssh_host: SSH login/jump host
        remote_host: Target host accessible from ssh_host
        remote_port: Target port on remote_host
        local_port: Local port to bind (0 = auto-assign free port)
        ssh_password: Optional password (uses sshpass if provided, prefers SSH keys)
        verify_url: Optional URL or path to GET after tunnel is up. Both
                    http://remote_host:remote_port/path and bare /path work
                    — the URL is rewritten to localhost:local_port internally.
        verify_timeout: Seconds to wait for verify_url to respond

    Returns:
        The local port the tunnel is bound to.

    Raises:
        ConnectionError: If SSH tunnel fails or verification times out.
    """
    if local_port == 0:
        local_port = find_free_port()

    tunnel_spec = f"localhost:{local_port}:{remote_host}:{remote_port}"
    logger.info(f"Opening SSH tunnel: {tunnel_spec} via {ssh_user}@{ssh_host}")

    cmd = [
        "ssh", "-N", "-f",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ExitOnForwardFailure=yes",
        "-L", tunnel_spec,
        f"{ssh_user}@{ssh_host}",
    ]

    if ssh_password:
        cmd = ["sshpass", "-p", ssh_password] + cmd

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise ConnectionError(
                f"SSH tunnel failed (exit {result.returncode}): {result.stderr.strip()}"
            )
    except FileNotFoundError as exc:
        if ssh_password:
            raise ConnectionError(
                "sshpass not found. Install with: brew install sshpass "
                "OR set up SSH keys to avoid passwords."
            ) from exc
        raise ConnectionError("ssh command not found.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ConnectionError("SSH tunnel timed out after 30 seconds.") from exc

    logger.info(f"SSH tunnel established on localhost:{local_port}")

    if verify_url:
        target_url = verify_url.replace(
            f"{remote_host}:{remote_port}",
            f"localhost:{local_port}",
        )
        if not target_url.startswith("http"):
            target_url = f"http://localhost:{local_port}{verify_url}"
        if not probe_http_ok(target_url, total_timeout=verify_timeout,
                             retry_interval=1):
            raise ConnectionError(
                f"SSH tunnel is up but could not reach {target_url} "
                f"after {verify_timeout}s"
            )
        logger.info(f"Tunnel verified via {target_url}")

    return local_port


def close_tunnel(local_port: int) -> None:
    """Best-effort: kill the ssh process bound to local_port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{local_port}"],
            capture_output=True, text=True,
        )
        for pid in result.stdout.strip().split("\n"):
            if pid:
                try:
                    os.kill(int(pid), 9)
                    logger.info(f"Closed tunnel pid {pid} on port {local_port}")
                except ProcessLookupError:
                    pass
    except FileNotFoundError:
        logger.warning("lsof not found; cannot close tunnel automatically.")
    except Exception as exc:
        logger.warning(f"Could not close tunnel on port {local_port}: {exc}")


# ── reachability + readiness probes ──────────────────────────────────────


def can_reach(host: str, port: int = 22, attempts: int = 3,
              per_attempt_timeout: float = 2.0) -> bool:
    """TCP-level reachability check. Returns True on first success.

    Defaults: 3 attempts at 2s each = 6s max wallclock if every attempt
    times out. Faster than connect_ssh()'s SSH-handshake test — useful as
    a pre-flight so callers error out promptly when the user isn't on VPN.
    """
    for _ in range(attempts):
        try:
            with socket.create_connection((host, port),
                                          timeout=per_attempt_timeout):
                return True
        except (socket.timeout, OSError):
            pass
    return False


def probe_http_ok(url: str, total_timeout: int = 30,
                  per_request_timeout: int = 5,
                  retry_interval: int = 5) -> bool:
    """Poll url until any HTTP response (even 4xx) comes back.

    Returns True on success, False if total_timeout elapses with only
    network-level errors. Used to confirm an SSH tunnel is forwarding to
    a live server — we don't care about the response body.
    """
    start = time.time()
    while time.time() - start < total_timeout:
        try:
            with urllib.request.urlopen(url, timeout=per_request_timeout):
                return True
        except urllib.error.HTTPError:
            return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            elapsed = time.time() - start
            logger.info(f"Waiting for {url} to respond... ({elapsed:.0f}s elapsed)")
        time.sleep(retry_interval)
    return False


# ── connectivity helper ──────────────────────────────────────────────────


def connect_ssh(ssh_user: str, ssh_host: str, ssh_password: str = "",
                retry_interval: int = 10) -> None:
    """Block until SSH to ssh_user@ssh_host succeeds.

    Useful for prompting the user to start their VPN. Probes with BatchMode
    first and prints a key-setup hint if no keys are configured.
    """
    ssh_target = f"{ssh_user}@{ssh_host}"

    if not ssh_password:
        try:
            key_check = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                 ssh_target, "echo ok"],
                capture_output=True, text=True, timeout=10,
            )
            has_keys = key_check.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            has_keys = False
        if not has_keys:
            logger.info(
                f"  Warning: SSH keys not found for {ssh_target}.\n"
                f"  You may be prompted for a password.\n"
                f"  To avoid this:\n"
                f"    ssh-keygen                # generate keys (if you haven't)\n"
                f"    ssh-copy-id {ssh_target}  # copy keys to server\n"
            )

    logger.info(f"Connecting to {ssh_target}...")
    start = time.time()
    while True:
        elapsed = time.time() - start
        try:
            cmd = ["ssh", "-o", "ConnectTimeout=10", ssh_target, "echo ok"]
            if ssh_password:
                if not shutil.which("sshpass"):
                    raise ConnectionError(
                        "Password auth requires 'sshpass' but it's not installed."
                    )
                cmd = ["sshpass", "-p", ssh_password] + cmd
            test = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if test.returncode == 0:
                logger.info(f"  SSH connection established ({elapsed:.0f}s)")
                return
            logger.info(
                f"  Waiting for SSH connection... ({elapsed:.0f}s elapsed) "
                f"Are you connected to the VPN?"
            )
        except subprocess.TimeoutExpired:
            logger.info(
                f"  Waiting for SSH connection... ({elapsed:.0f}s elapsed, "
                f"timed out) Are you connected to the VPN?"
            )
        time.sleep(retry_interval)
