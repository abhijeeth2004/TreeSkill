"""Global Configuration — loads from YAML config file, .env, and environment variables.

Priority (highest to lowest):
1. Environment variables (``EVO_`` prefix)
2. ``.env`` file
3. YAML config file (``--config``)
4. Pydantic defaults

Usage::

    python -m evo_framework.main --config config.yaml --skill my-skill
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """OpenAI-compatible LLM connection settings."""

    model_config = SettingsConfigDict(env_prefix="EVO_LLM_")

    api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="API base URL (change for Azure / local proxies)",
    )
    model: str = Field(default="gpt-4o", description="Default chat model")
    judge_model: str = Field(
        default="gpt-4o", description="Model used by the APO evaluator"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class StorageConfig(BaseSettings):
    """Paths for traces and skill files."""

    model_config = SettingsConfigDict(env_prefix="EVO_STORAGE_")

    trace_path: Path = Field(default=Path("./data/traces.jsonl"))
    skill_path: Path = Field(default=Path("./skills"))


class APOConfig(BaseSettings):
    """Automatic Prompt Optimization hyper-parameters."""

    model_config = SettingsConfigDict(env_prefix="EVO_APO_")

    max_steps: int = Field(default=3, ge=1)
    gradient_accumulation_steps: int = Field(default=5, ge=1)


class RewardConfig(BaseSettings):
    """Auto-judge / reward provider settings.

    When ``model`` is None, falls back to ``LLMConfig.judge_model``.
    """

    model_config = SettingsConfigDict(env_prefix="EVO_REWARD_")

    enabled: bool = Field(
        default=False,
        description="启用自动 Judge 评分。关闭时只能手动反馈。",
    )
    model: Optional[str] = Field(
        default=None,
        description="Judge 模型。为空则回落到 llm.judge_model。",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Judge API 地址。为空则回落到 llm.base_url。",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Judge API 密钥。为空则回落到 llm.api_key。",
    )
    auto_judge: bool = Field(
        default=False,
        description="每次生成后自动运行 Judge 评分。",
    )
    default_rubric: Optional[str] = Field(
        default=None,
        description="默认评分标准 prompt。Skill 级的 judge_rubric 会覆盖此值。",
    )


class GlobalConfig(BaseSettings):
    """Top-level configuration that aggregates all sub-configs.

    Reads a ``.env`` file from the project root when present.
    Can also be initialized from a YAML config file via ``from_yaml()``.
    """

    model_config = SettingsConfigDict(
        env_prefix="EVO_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    apo: APOConfig = Field(default_factory=APOConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    verbose: bool = Field(default=False, description="Enable debug logging")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GlobalConfig":
        """Create a ``GlobalConfig`` from a YAML file.

        The YAML structure mirrors the nested config layout::

            llm:
              api_key: sk-xxx
              model: gpt-4o
            storage:
              trace_path: ./data/traces.jsonl
            apo:
              max_steps: 3
            reward:
              enabled: true
              auto_judge: false

        Environment variables and ``.env`` still take priority over
        the YAML values (Pydantic-settings precedence).
        """
        path = Path(path)
        raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(**raw)
