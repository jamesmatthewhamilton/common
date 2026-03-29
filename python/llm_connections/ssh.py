"""SSH tunnel utility. Reusable for any service that needs port forwarding."""

import logging
import os
import socket
import subprocess
import time

logger = logging.getLogger(__name__)


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
        verify_url: Optional URL to GET after tunnel is up to verify connectivity
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
    except FileNotFoundError:
        if ssh_password:
            raise ConnectionError(
                "sshpass not found. Install with: brew install sshpass "
                "OR set up SSH keys to avoid passwords."
            )
        raise ConnectionError("ssh command not found.")
    except subprocess.TimeoutExpired:
        raise ConnectionError("SSH tunnel timed out after 30 seconds.")

    logger.info(f"SSH tunnel established on localhost:{local_port}")

    # Verify connectivity if URL provided
    if verify_url:
        import httpx
        target_url = verify_url.replace(
            f"{remote_host}:{remote_port}",
            f"localhost:{local_port}"
        )
        # If verify_url is just a path, build full URL
        if not target_url.startswith("http"):
            target_url = f"http://localhost:{local_port}{verify_url}"

        for _ in range(verify_timeout):
            try:
                resp = httpx.get(target_url, timeout=2)
                if resp.status_code < 500:
                    logger.info(f"Tunnel verified via {target_url}")
                    return local_port
            except Exception:
                pass
            time.sleep(1)

        raise ConnectionError(
            f"SSH tunnel is up but could not reach {target_url} "
            f"after {verify_timeout}s"
        )

    return local_port


def close_tunnel(local_port: int):
    """Kill the SSH tunnel process bound to a local port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{local_port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                os.kill(int(pid), 9)
                logger.info(f"Killed tunnel process {pid} on port {local_port}")
    except Exception as e:
        logger.warning(f"Could not close tunnel on port {local_port}: {e}")
