"""LLMConnection — config-driven, multi-provider LLM client."""

import os

from .config import load_providers
from .llm_providers import get_provider_class
from .response import LLMResponse


class LLMConnection:
    """A configured LLM client backed by a specific provider.

    Usage:
        # Load providers from a YAML file (only reads "providers:" key)
        LLMConnection.load("config/default.yaml")

        # Get a named client
        client = LLMConnection.get("local-big")

        # Chat
        response = client.chat(messages)
        print(response.text)

        # Stream
        response = client.chat(messages, stream=True)
        for chunk in response:
            print(chunk.text, end="")
    """

    _registry: dict = {}

    def __init__(self, provider):
        self._provider = provider

    @classmethod
    def load(cls, yaml_path: str):
        """Load provider configs from a YAML file and populate the registry.

        Only reads the 'providers:' key. All other YAML keys are ignored.
        Can be called multiple times — new providers are added, existing ones updated.
        """
        yaml_path = os.path.expanduser(yaml_path)
        providers = load_providers(yaml_path)

        for name, config in providers.items():
            provider_type = config.get("provider")
            if not provider_type:
                raise ValueError(f"Provider '{name}' missing 'provider' key in config")

            provider_cls = get_provider_class(provider_type)
            provider_instance = provider_cls(config)
            cls._registry[name] = cls(provider_instance)

    @classmethod
    def get(cls, name: str) -> "LLMConnection":
        """Get a named client from the registry."""
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "(none loaded)"
            raise KeyError(
                f"Provider '{name}' not found. Available: {available}. "
                f"Call LLMConnection.load('config.yaml') first."
            )
        return cls._registry[name]

    @classmethod
    def list_providers(cls) -> list:
        """List all registered provider names."""
        return list(cls._registry.keys())

    def chat(self, messages: list, tools: list = None,
             stream: bool = False, **overrides) -> LLMResponse:
        """Send a chat request.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            tools: Optional tool definitions (format depends on provider)
            stream: If True, iterate the response for LLMChunk objects
            **overrides: Per-call overrides (temperature, num_ctx, etc.)

        Returns:
            LLMResponse with .text, .tool_calls, .prompt_tokens, etc.
        """
        return self._provider.chat(messages, tools=tools, stream=stream, **overrides)

    def complete(self, prompt: str, system: str = None, **overrides) -> LLMResponse:
        """Convenience: simple prompt + optional system message.

        Builds a messages list and calls chat().
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **overrides)

    @property
    def model(self) -> str:
        """The model name this client is configured to use."""
        return self._provider.model

    def __repr__(self):
        return f"LLMConnection(provider={self._provider.__class__.__name__}, model={self.model})"
