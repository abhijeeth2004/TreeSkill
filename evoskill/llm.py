"""LLM Client — thin, multimodal-aware wrapper around ``openai.OpenAI``.

Handles serialization of ``Message`` objects (including image content parts)
into the format expected by the OpenAI Chat Completions API.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

import openai

from evoskill.builtin_tools import format_tool_result
from evoskill.config import GlobalConfig
from evoskill.schema import Message
from evoskill.tools import BaseTool

_MAX_TOOL_ITERATIONS = 24


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
        tools: Optional[Mapping[str, BaseTool]] = None,
        on_tool_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
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
        tool_defs = None
        if tools:
            tool_defs = [
                {"type": "function", "function": tool.to_schema()}
                for tool in tools.values()
            ]
        temperature = kwargs.pop("temperature", self._config.llm.temperature)

        for _ in range(_MAX_TOOL_ITERATIONS):
            request_kwargs = dict(kwargs)
            request_kwargs.update(
                {
                    "model": model,
                    "messages": api_messages,  # type: ignore[arg-type]
                    "temperature": temperature,
                }
            )
            if tool_defs:
                request_kwargs["tools"] = tool_defs
                request_kwargs["tool_choice"] = "auto"

            completion = self._client.chat.completions.create(
                **request_kwargs,
            )

            choice = completion.choices[0]
            assistant_message = choice.message
            tool_calls = getattr(assistant_message, "tool_calls", None) or []

            if not tool_calls:
                return Message(
                    role="assistant",
                    content=assistant_message.content or "",
                )

            api_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        call.model_dump() if hasattr(call, "model_dump") else {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )

            for call in tool_calls:
                tool_name = call.function.name
                if on_tool_event:
                    on_tool_event("start", {"name": tool_name, "arguments": call.function.arguments})
                try:
                    parsed_args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    result = {"error": f"Invalid tool arguments: {exc}"}
                else:
                    tool = tools.get(tool_name) if tools else None
                    if tool is None:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    else:
                        try:
                            result = tool.execute(**parsed_args)
                        except Exception as exc:  # pragma: no cover - runtime safeguard
                            result = {"error": str(exc)}

                if on_tool_event:
                    on_tool_event(
                        "finish",
                        {"name": tool_name, "result": format_tool_result(result)},
                    )

                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": format_tool_result(result),
                    }
                )

        raise RuntimeError(
            f"Tool-calling loop exceeded the maximum number of iterations ({_MAX_TOOL_ITERATIONS})"
        )
