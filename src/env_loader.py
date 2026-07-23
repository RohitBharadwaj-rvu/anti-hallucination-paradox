"""
src/env_loader.py — Loads and validates .env; exposes provider config helpers.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from src.utils import get_logger

logger = get_logger("ahp.env_loader")

# Fixed provider -> env var mappings (locked; do not change)
_KEY_VARS = {
    "nim": "NVIDIA_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
}
_BASE_URL_VARS = {
    "nim": ("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    "fireworks": ("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
    "lmstudio": ("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
}
_PLACEHOLDER_PREFIXES = {
    "nim": "nvapi-REPLACE",
    "fireworks": "fw_REPLACE",
}


def load_env(env_path: str | None = None) -> None:
    """Load the .env file. Called once at startup."""
    if env_path is None:
        # Search for .env relative to project root (two levels up from this file)
        candidate = Path(__file__).parent.parent / ".env"
        env_path = str(candidate)
    loaded = load_dotenv(env_path, override=False)
    if loaded:
        logger.debug("Loaded .env from %s", env_path)
    else:
        logger.debug(".env not found at %s; relying on shell environment", env_path)


def get_api_key(provider: str) -> str:
    """
    Return the API key for a provider. For lmstudio, use LMSTUDIO_API_KEY if non-empty
    else literal "lm-studio". For cloud providers, fail fast if missing or placeholder.
    """
    if provider == "lmstudio":
        val = os.environ.get("LMSTUDIO_API_KEY", "").strip()
        return val if val else "lm-studio"

    var = _KEY_VARS.get(provider)
    if var is None:
        raise ValueError(f"Unknown provider: {provider!r}")
    val = os.environ.get(var, "").strip()
    placeholder = _PLACEHOLDER_PREFIXES.get(provider, "REPLACE")
    if not val or val.startswith(placeholder):
        raise EnvironmentError(
            f"Provider '{provider}' requires a real API key in env var {var!r}. "
            f"Current value looks like a placeholder: {val!r}. "
            "Fill the key in .env before running cloud target experiments."
        )
    return val


def get_base_url(provider: str) -> str:
    """Return the base URL for a provider from env or hardcoded default."""
    var, default = _BASE_URL_VARS.get(provider, (None, None))
    if var is None:
        raise ValueError(f"Unknown provider: {provider!r}")
    val = os.environ.get(var, "").strip()
    url = val if val else default
    # Ensure lmstudio URL ends at /v1 (not /v1/ — SDK appends /chat/completions)
    if provider == "lmstudio":
        url = url.rstrip("/")
        if not url.endswith("/v1"):
            logger.warning(
                "LMSTUDIO_BASE_URL %r does not end with /v1 — SDK will append /chat/completions. "
                "Ensure you haven't included a trailing slash or extra path.",
                url,
            )
    return url


def is_billed(provider: str) -> bool:
    """Return True if the provider incurs cloud billing charges."""
    return provider in ("nim", "fireworks")
