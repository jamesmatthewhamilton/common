"""LLMConnection — config-driven, multi-provider LLM client."""

import os

from .config import load_providers
from .llm_providers import get_provider_class
from .response import LLMResponse


class _FailedConnection:
    """Placeholder for a provider that failed to connect at load time."""
    def __init__(self, name, error):
        self.name = name
        self.error = error
        self.model = f"(failed: {name})"
    def chat(self, *args, **kwargs):
        raise ConnectionError(
            f"Provider '{self.name}' is not available: {self.error}"
        )
    def __repr__(self):
        return f"FailedConnection({self.name}: {self.error})"


DEFAULT_CONFIG_DIR = os.path.expanduser("~/.llm-connections")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_CONFIG_DIR, "config.yaml")

DEFAULT_CONFIG_TEMPLATE = """\
llm-providers:
  default:
    provider: ollama
    model: llama3.1:8b
    num_ctx: 8192
    temperature: 0.0
"""


class LLMConnection:
    """A configured LLM client backed by a specific provider.

    Usage:
        # Load from default global config (~/.llm-connections/config.yaml)
        LLMConnection.load()

        # Or load from a specific YAML file
        LLMConnection.load("config/default.yaml")

        # Both can be called — providers merge into one registry
        LLMConnection.load()
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
    def load(cls, yaml_path: str = None):
        """Load provider configs from a YAML file and populate the registry.

        Args:
            yaml_path: Path to YAML file. If None, reads from
                       ~/.llm-connections/config.yaml

        Only reads the 'llm-providers:' key. All other YAML keys are ignored.
        Can be called multiple times — new providers merge into the registry.
        """
        if yaml_path is None:
            yaml_path = DEFAULT_CONFIG_PATH
            if not os.path.isfile(yaml_path):
                os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
                with open(yaml_path, "w") as f:
                    f.write(DEFAULT_CONFIG_TEMPLATE)
                import logging
                logging.warning(
                    f"Created default LLM config at {yaml_path}. "
                    f"Edit it to add your providers."
                )
        yaml_path = os.path.expanduser(yaml_path)
        providers = load_providers(yaml_path)

        for name, config in providers.items():
            provider_type = config.get("provider")
            if not provider_type:
                raise ValueError(f"Provider '{name}' missing 'provider' key in config")

            provider_cls = get_provider_class(provider_type)
            try:
                provider_instance = provider_cls(config)
                cls._registry[name] = cls(provider_instance)
            except ConnectionError as e:
                import logging
                logging.warning(f"Provider '{name}' failed to connect: {e}")
                cls._registry[name] = _FailedConnection(name, str(e))

    @classmethod
    def get(cls, name: str) -> "LLMConnection":
        """Get a named client from the registry."""
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "(none)"
            raise KeyError(
                f"LLM provider '{name}' not found in config. "
                f"Available providers: [{available}]. "
                f"Add '{name}' to llm-providers in {DEFAULT_CONFIG_PATH}"
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
