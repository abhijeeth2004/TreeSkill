"""Skill Management — load, save, and compile Skills into chat messages.

Skills are stored as YAML (or JSON) files on disk.  At runtime they are
validated against the Pydantic ``Skill`` schema and compiled into the
message sequence expected by the LLM client.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import yaml

from evoskill.schema import Message, Skill


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load(path: Union[str, Path]) -> Skill:
    """Load a Skill from a YAML or JSON file.

    Missing fields are filled with defaults defined in ``schema.Skill``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    pydantic.ValidationError
        If the file contents cannot be coerced into a valid ``Skill``.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw) or {}
    else:
        data = json.loads(raw)

    return Skill.model_validate(data)


def save(skill: Skill, path: Union[str, Path]) -> None:
    """Dump *skill* to a YAML file at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = skill.model_dump(mode="json")
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Message Compilation
# ---------------------------------------------------------------------------

def compile_messages(skill: Skill, user_input: List[Message]) -> List[Message]:
    """Build the full message list sent to the LLM.

    Order::

        [SystemMessage(skill.system_prompt)]
        + skill.few_shot_messages
        + user_input
    """
    system_msg = Message(role="system", content=skill.system_prompt)
    return [system_msg] + list(skill.few_shot_messages) + list(user_input)
