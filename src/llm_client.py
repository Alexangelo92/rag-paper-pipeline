"""Provider-agnostic chat completion client.

Provider is chosen at call time via the LLM_PROVIDER env var ("anthropic" or
"openai"), so you can decide later which API to wire up without touching the
RAG logic. Only the SDK for the provider you actually use needs to be
installed.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4.1",
}


def complete(system: str, user: str, provider: str | None = None, model: str | None = None) -> str:
    """Send a system+user prompt to the configured LLM and return the text reply."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "")).lower()

    if provider == "anthropic":
        return _complete_anthropic(system, user, model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS["anthropic"])
    if provider == "openai":
        return _complete_openai(system, user, model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS["openai"])

    raise ValueError(
        "No LLM provider configured. Set LLM_PROVIDER=anthropic or LLM_PROVIDER=openai "
        "(and the matching *_API_KEY) in your environment or .env file."
    )


def _complete_anthropic(system: str, user: str, model: str) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _complete_openai(system: str, user: str, model: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content
