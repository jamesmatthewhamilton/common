"""Provider registry. Lazily imports providers to avoid unnecessary dependencies."""

from .base import BaseProvider

PROVIDER_NAMES = {"ollama", "litellm"}


def get_provider_class(provider_type: str) -> type:
    """Get provider class by type name. Imports lazily."""
    if provider_type not in PROVIDER_NAMES:
        available = ", ".join(PROVIDER_NAMES)
        raise ValueError(f"Unknown provider '{provider_type}'. Available: {available}")

    if provider_type == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider
    elif provider_type == "litellm":
        from .litellm import LitellmProvider
        return LitellmProvider
