"""Shared LLM client library — config-driven, multi-provider.

Core (always available):
    LLMConnection, LLMResponse, LLMChunk

Optional (requires the slurm-manipulator nested submodule):
    SlurmSession, SessionHandle, connect_ssh

Importing SlurmSession when slurm-manipulator isn't installed raises a
clear ImportError pointing at the submodule init command.
"""

from .client import LLMConnection
from .response import LLMChunk, LLMResponse
from .ssh import connect_ssh

__all__ = [
    "LLMConnection",
    "LLMResponse",
    "LLMChunk",
    "connect_ssh",
    # SlurmSession / SessionHandle exposed via __getattr__ — lazy so the
    # optional slurm-manipulator dependency is only resolved on access.
]


def __getattr__(name: str):
    if name in ("SlurmSession", "SessionHandle"):
        from .slurm_session import SessionHandle, SlurmSession
        return {"SlurmSession": SlurmSession, "SessionHandle": SessionHandle}[name]
    raise AttributeError(f"module 'llm_connections' has no attribute {name!r}")
