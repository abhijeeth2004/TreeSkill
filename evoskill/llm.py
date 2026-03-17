"""LLM Client — thin, multimodal-aware wrapper around ``openai.OpenAI``.

Handles serialization of ``Message`` objects (including image content parts)
into the format expected by the OpenAI Chat Completions API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import openai

from evoskill.config import GlobalConfig
from evoskill.schema import Message


class LLMClient:
    """Stateless wrapper around the OpenAI Python SDK.

    Parameters
    ----------
    config : GlobalConfig
        Framework configuration (API key, base URL, model names, etc.).
    """

    def __init__(self, config: GlobalConfig) -> None:
        self._config = config
        self._client = openai.OpenAI(
            api_key=config.llm.api_key.get_secret_value(),
            base_url=config.llm.base_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Message:
        """Send a chat completion request and return the response as a ``Message``.

        Parameters
        ----------
        messages :
            Conversation history.  Each ``Message`` may contain plain text
            *or* a list of ``ContentPart`` objects (text + image_url).
        model :
            Override the default model for this call (useful for the APO
            judge step).
        **kwargs :
            Extra parameters forwarded to ``chat.completions.create``
            (e.g. ``max_tokens``, ``top_p``).
        """
        model = model or self._config.llm.model

        api_messages = [msg.to_api_dict() for msg in messages]

        completion = self._client.chat.completions.create(
            model=model,
            messages=api_messages,  # type: ignore[arg-type]
            temperature=kwargs.pop("temperature", self._config.llm.temperature),
            **kwargs,
        )

        choice = completion.choices[0]
        return Message(
            role="assistant",
            content=choice.message.content or "",
        )
