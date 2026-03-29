"""Base provider interface."""

from abc import ABC, abstractmethod
from ..response import LLMResponse


class BaseProvider(ABC):
    """Abstract base for LLM providers."""

    # Options that providers can accept (passed via YAML or per-call overrides)
    KNOWN_OPTIONS = {
        "temperature", "num_ctx", "num_predict", "top_p", "top_k",
        "repeat_penalty", "max_tokens",
    }

    def __init__(self, config: dict):
        self.model = config.get("model", "")
        self.config = config
        # Extract default options from config
        self.default_options = {
            k: v for k, v in config.items()
            if k in self.KNOWN_OPTIONS
        }

    def _merge_options(self, overrides: dict) -> dict:
        """Merge default options with per-call overrides."""
        opts = dict(self.default_options)
        for k, v in overrides.items():
            if k in self.KNOWN_OPTIONS:
                opts[k] = v
        return opts

    @abstractmethod
    def chat(self, messages: list, tools: list = None,
             stream: bool = False, **overrides) -> LLMResponse:
        """Send a chat request to the LLM.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            tools: Optional list of tool definitions (provider-specific format)
            stream: If True, response is iterable yielding LLMChunk objects
            **overrides: Per-call option overrides (temperature, num_ctx, etc.)

        Returns:
            LLMResponse with text, tool_calls, token counts, etc.
            If stream=True, iterate the response for chunks.
        """
        ...
