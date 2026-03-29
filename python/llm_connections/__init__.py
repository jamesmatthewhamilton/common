"""Shared LLM client library — config-driven, multi-provider."""

from .client import LLMConnection
from .response import LLMResponse, LLMChunk

__all__ = ["LLMConnection", "LLMResponse", "LLMChunk"]
