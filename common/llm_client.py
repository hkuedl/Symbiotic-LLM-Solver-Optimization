"""Create OpenAI-compatible clients using explicit arguments or environment variables."""

import os


def create_client(api_key=None, base_url=None, *, asynchronous=False):
    """Resolve credentials without storing or logging their values."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY before running an LLM method.")

    from openai import AsyncOpenAI, OpenAI

    options = {"api_key": key}
    endpoint = base_url or os.environ.get("OPENAI_BASE_URL")
    if endpoint:
        options["base_url"] = endpoint
    client_type = AsyncOpenAI if asynchronous else OpenAI
    return client_type(**options)
