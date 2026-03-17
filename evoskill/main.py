"""Evo-Framework — Entry Point.

Usage::

    python -m evo_framework.main --skill <name-or-path>
    python -m evo_framework.main --skill <name-or-path> --optimize
    python -m evo_framework.main --ckpt <checkpoint-path>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from evoskill import skill as skill_module
from evoskill.checkpoint import CheckpointManager
from evoskill.cli import ChatCLI
from evoskill.config import GlobalConfig
from evoskill.llm import LLMClient
from evoskill.optimizer import APOEngine
from evoskill.skill_tree import SkillTree
from evoskill.storage import TraceStorage

console = Console()


def _resolve_skill_path(name_or_path: str, config: GlobalConfig) -> Path:
    """Resolve a skill name (looked up inside ``config.storage.skill_path``)
    or an explicit file path to an absolute ``Path``."""
    p = Path(name_or_path)
    if p.is_file() or p.is_dir():
        return p.resolve()
    # Try inside the configured skill directory
    for ext in (".yaml", ".yml", ".json"):
        candidate = Path(config.storage.skill_path) / f"{name_or_path}{ext}"
        if candidate.is_file():
            return candidate.resolve()
    # Try as a directory
    candidate_dir = Path(config.storage.skill_path) / name_or_path
    if candidate_dir.is_dir():
        return candidate_dir.resolve()
    # Fall back — create a minimal default skill
    default = Path(config.storage.skill_path) / f"{name_or_path}.yaml"
    default.parent.mkdir(parents=True, exist_ok=True)
    from evoskill.schema import Skill

    default_skill = Skill(
        name=name_or_path,
        system_prompt="You are a helpful assistant.",
    )
    skill_module.save(default_skill, default)
    console.print(f"[dim]Created default skill → {default}[/dim]")
    return default.resolve()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="evo",
        description="Evo-Framework: Multimodal Self-Evolving Agent",
    )
    parser.add_argument(
        "--skill",
        default="default",
        help="Skill name (resolved in skills/) or path to a YAML/JSON file or skill-tree directory.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (see demo/example/config.yaml for template).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run APO optimization on stored traces instead of starting chat.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Restore from a checkpoint path (e.g. ckpt/writing-assistant_v1.2_20260306_140000).",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="./ckpt",
        help="Directory for storing checkpoints (default: ./ckpt).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    # --- Config ---
    if args.config:
        config = GlobalConfig.from_yaml(args.config)
        console.print(f"[dim]Config loaded from {args.config}[/dim]")
    else:
        config = GlobalConfig()
    if args.verbose:
        config = config.model_copy(update={"verbose": True})
        logging.basicConfig(level=logging.DEBUG, format="%(name)s  %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # --- Checkpoint restore ---
    if args.ckpt:
        ckpt_mgr = CheckpointManager(args.ckpt_dir)
        info = ckpt_mgr.load(args.ckpt)
        skill_path = info["skill_path"]
        if info["trace_path"]:
            # Update storage config to use the checkpoint's traces
            config = config.model_copy(
                update={"storage": config.storage.model_copy(
                    update={"trace_path": info["trace_path"]}
                )}
            )
        console.print(
            f"[green]✓[/green] Restored from checkpoint: [cyan]{args.ckpt}[/cyan]"
        )
    else:
        skill_path = _resolve_skill_path(args.skill, config)

    # --- Load skill (file or tree) ---
    skill_tree = None
    if skill_path.is_dir():
        skill_tree = SkillTree.load(skill_path)
        loaded_skill = skill_tree.root.skill
        console.print(
            f"[dim]Loaded skill tree from {skill_path} "
            f"({skill_tree.root.leaf_count()} leaves)[/dim]"
        )
    else:
        loaded_skill = skill_module.load(skill_path)

    # --- Optimize mode ---
    if args.optimize:
        llm = LLMClient(config)
        storage = TraceStorage(config.storage)
        engine = APOEngine(config, llm)
        traces = storage.get_feedback_samples()
        if not traces:
            console.print("[yellow]No feedback traces found — nothing to optimize.[/yellow]")
            sys.exit(0)
        console.print(
            f"[bold]Running APO[/bold] on {len(traces)} feedback traces …"
        )

        if skill_tree:
            engine.evolve_tree(skill_tree, traces)
            skill_tree.save()
            new_skill = skill_tree.root.skill
        else:
            new_skill = engine.optimize(loaded_skill, traces)
            skill_module.save(new_skill, skill_path)

        # Save checkpoint
        ckpt_mgr = CheckpointManager(args.ckpt_dir)
        ckpt_path = ckpt_mgr.save(
            skill_path if skill_tree else new_skill,
            trace_path=Path(config.storage.trace_path),
        )

        console.print(
            f"[green]✓ Skill updated:[/green] {new_skill.name} "
            f"({loaded_skill.version} → {new_skill.version})"
        )
        console.print(f"[dim]Checkpoint saved → {ckpt_path}[/dim]")
        sys.exit(0)

    # --- Chat mode ---
    chat = ChatCLI(
        config,
        loaded_skill,
        skill_path,
        skill_tree=skill_tree,
        ckpt_dir=args.ckpt_dir,
    )
    chat.run()


if __name__ == "__main__":
    main()
