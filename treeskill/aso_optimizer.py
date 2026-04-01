"""AS(skill)O optimizer.

This module keeps the first implementation intentionally small:
- optimize a frontier of agent programs, not a single prompt
- analyze failures with textual gradients
- propose skill-level actions
- generate program candidates and keep the best frontier on validation
"""

from __future__ import annotations

import json
import logging
import hashlib
import traceback
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from treeskill.aso_program import ASOProgram, ASOSkill
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.schema import Feedback, Message, Trace
from treeskill.tasks.sealqa import SealQAExample

logger = logging.getLogger(__name__)

_GLOBAL_FAILURE_ROUTE = "__global__"


def _strip_thinking_blocks(text: str) -> str:
    if not text:
        return ""
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _extract_json_payload(text: str, *, expect_array: bool) -> Optional[Any]:
    candidates: list[str] = []
    cleaned = _strip_thinking_blocks(text)
    if not cleaned:
        return None
    candidates.append(cleaned)

    fenced = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", cleaned, flags=re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    regex = r"\[[\s\S]*?\]" if expect_array else r"\{[\s\S]*?\}"
    candidates.extend(
        match.group(0).strip()
        for match in re.finditer(regex, cleaned)
    )

    for raw in candidates:
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if expect_array and isinstance(parsed, list):
            return parsed
        if not expect_array and isinstance(parsed, dict):
            return parsed
    return None


@dataclass
class ASOSkillAction:
    action: str
    rationale: str = ""
    skill_name: Optional[str] = None
    description: Optional[str] = None
    skill_prompt: Optional[str] = None
    target_skill: Optional[str] = None
    merge_skills: List[str] = field(default_factory=list)
    selection_policy: Optional[str] = None
    focus_route: Optional[str] = None


@dataclass
class ASOIterationResult:
    iteration: int
    best_score: float
    frontier_scores: List[float]
    accepted_program_id: str
    actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ASOResult:
    best_program: ASOProgram
    frontier: List[ASOProgram]
    baseline_score: float
    final_score: float
    history: List[ASOIterationResult] = field(default_factory=list)
    postprocess: List[Dict[str, Any]] = field(default_factory=list)


class ASOOptimizer:
    """Skill-level optimizer driven by textual gradients and validation frontier."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        frontier_size: int = 3,
        branch_factor: int = 2,
        max_iterations: int = 3,
        max_workers: int = 1,
        auto_merge: bool = False,
        auto_prune: bool = False,
        apo_fallback_enabled: bool = False,
        apo_fallback_skill_limit: int = 1,
        trajectory_mode: bool = False,
        trajectory_focus_top_k: int = 3,
        artifact_dir: Optional[Path] = None,
    ) -> None:
        self._llm = llm
        self.frontier_size = frontier_size
        self.branch_factor = branch_factor
        self.max_iterations = max_iterations
        self.max_workers = max(1, max_workers)
        self.auto_merge = auto_merge
        self.auto_prune = auto_prune
        self.apo_fallback_enabled = apo_fallback_enabled
        self.apo_fallback_skill_limit = max(1, apo_fallback_skill_limit)
        self.trajectory_mode = trajectory_mode
        self.trajectory_focus_top_k = max(1, trajectory_focus_top_k)
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self._apo_engine: Optional[APOEngine] = (
            APOEngine(llm._config, llm) if apo_fallback_enabled else None
        )

    def run(
        self,
        seed_program: ASOProgram,
        train_data: Sequence[SealQAExample],
        val_data: Sequence[SealQAExample],
        runner: Callable[[ASOProgram, SealQAExample], Any],
        scorer: Callable[[SealQAExample, str], float],
        *,
        start_iteration: int = 1,
        initial_frontier: Optional[Sequence[ASOProgram]] = None,
        initial_best_program: Optional[ASOProgram] = None,
        initial_history: Optional[Sequence[ASOIterationResult]] = None,
        initial_baseline_score: Optional[float] = None,
    ) -> ASOResult:
        """Run a frontier-based ASO loop."""
        if not initial_frontier:
            frontier = [seed_program]
            if seed_program.score is None:
                seed_program.score = self._evaluate(seed_program, val_data, runner, scorer)
            baseline_score = seed_program.score
            best_program = initial_best_program or seed_program
        else:
            frontier = list(initial_frontier)
            if initial_best_program is not None:
                best_program = initial_best_program
            else:
                best_program = max(
                    frontier,
                    key=lambda item: 0.0 if item.score is None else item.score,
                )
            baseline_score = (
                initial_baseline_score
                if initial_baseline_score is not None
                else best_program.score
            )
            if baseline_score is None:
                baseline_score = self._evaluate(best_program, val_data, runner, scorer)
        if baseline_score is not None:
            best_program.score = baseline_score

        history: List[ASOIterationResult] = list(initial_history or [])
        postprocess: List[Dict[str, Any]] = []
        start_iteration = max(1, int(start_iteration))

        if start_iteration > self.max_iterations:
            logger.info(
                "ASO start iteration %d is beyond max iterations %d; skipping evolution",
                start_iteration,
                self.max_iterations,
            )
            return ASOResult(
                best_program=best_program,
                frontier=[best_program],
                baseline_score=baseline_score,
                final_score=float(best_program.score or 0.0),
                history=history,
                postprocess=postprocess,
            )

        for iteration in range(start_iteration, self.max_iterations + 1):
            logger.info("ASO iteration %d/%d", iteration, self.max_iterations)
            candidates: List[ASOProgram] = list(frontier)
            iteration_actions: List[Dict[str, Any]] = []
            iteration_signatures: set[str] = set()

            for parent in frontier:
                traces = self._collect_failure_traces(parent, train_data, runner, scorer)
                if not traces:
                    logger.info("  Program %s has no train failures", parent.program_id)
                    continue

                focus_groups = self._group_traces_by_route(parent, traces)
                focus_order = sorted(
                    focus_groups.items(),
                    key=lambda item: len(item[1]),
                    reverse=True,
                )
                for focus_route, focus_traces in focus_order[: self.trajectory_focus_top_k]:
                    if self.trajectory_mode and focus_route != _GLOBAL_FAILURE_ROUTE and len(focus_traces) > 0:
                        route_note = f" (focus_route={focus_route}, failures={len(focus_traces)})"
                    else:
                        route_note = ""
                    gradient = self.compute_program_gradient(
                        parent,
                        focus_traces,
                        focus_route=focus_route,
                    )
                    logger.info(
                        "  Program %s focus%s gradient=%s",
                        parent.program_id,
                        route_note,
                        str(gradient[:120]).replace("\n", " "),
                    )

                    for _ in range(self.branch_factor):
                        actions = self.propose_actions(
                            parent,
                            gradient,
                            focus_traces,
                            focus_route=focus_route,
                        )
                        if not actions and self.apo_fallback_enabled:
                            fallback_action = self.propose_apo_action(
                                parent,
                                gradient,
                                focus_traces,
                                focus_route=focus_route,
                            )
                            if fallback_action is not None:
                                actions = [fallback_action]

                        if not actions:
                            continue

                        iteration_actions.extend([action.__dict__ for action in actions])
                        actions = self._dedupe_actions(actions)
                        if not actions:
                            continue
                        candidate_signature = self._candidate_signature(
                            parent,
                            actions,
                            focus_route,
                        )
                        if candidate_signature in iteration_signatures:
                            logger.debug(
                                "  Skip duplicate candidate for parent=%s focus=%s",
                                parent.program_id,
                                focus_route,
                            )
                            continue
                        iteration_signatures.add(candidate_signature)
                        candidate = self.apply_actions(parent, actions)
                        candidate.metadata.setdefault("focus_routes", [])
                        for action in actions:
                            if action.focus_route:
                                candidate.metadata["focus_routes"].append(action.focus_route)
                        candidates.append(candidate)

            scored_candidates: List[ASOProgram] = []
            for candidate in candidates:
                candidate.score = self._evaluate(candidate, val_data, runner, scorer)
                scored_candidates.append(candidate)

            scored_candidates.sort(key=lambda item: item.score or 0.0, reverse=True)
            frontier = scored_candidates[: self.frontier_size]
            if frontier and (frontier[0].score or 0.0) >= (best_program.score or 0.0):
                best_program = frontier[0]
                best_program.score = float(best_program.score or 0.0)

            history.append(
                ASOIterationResult(
                    iteration=iteration,
                    best_score=float(best_program.score or 0.0),
                    frontier_scores=[float(item.score or 0.0) for item in frontier],
                    accepted_program_id=best_program.program_id,
                    actions=iteration_actions,
                )
            )
            self._write_iteration_artifacts(iteration, frontier, best_program, iteration_actions)

        if self.auto_merge:
            best_program, merge_event = self._auto_merge(best_program, val_data, runner, scorer)
            if merge_event:
                postprocess.append(merge_event)

        if self.auto_prune:
            best_program, prune_event = self._auto_prune(best_program, val_data, runner, scorer)
            if prune_event:
                postprocess.append(prune_event)

        best_program.score = self._evaluate(best_program, val_data, runner, scorer)

        return ASOResult(
            best_program=best_program,
            frontier=frontier,
            baseline_score=baseline_score,
            final_score=float(best_program.score or 0.0),
            history=history,
            postprocess=postprocess,
        )

    def _write_iteration_artifacts(
        self,
        iteration: int,
        frontier: Sequence[ASOProgram],
        best_program: ASOProgram,
        iteration_actions: Sequence[Dict[str, Any]],
    ) -> None:
        if self.artifact_dir is None:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        iteration_dir = self.artifact_dir / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        frontier_payload = []
        for index, program in enumerate(frontier, start=1):
            program_dir = iteration_dir / f"frontier_{index}"
            program.save_to_dir(program_dir, clean=True)
            frontier_payload.append(program.to_dict())

        best_program.save_to_dir(iteration_dir / "best_program", clean=True)
        payload = {
            "iteration": iteration,
            "best_program_id": best_program.program_id,
            "best_score": best_program.score,
            "frontier": frontier_payload,
            "actions": list(iteration_actions),
        }
        (iteration_dir / "frontier.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def compute_program_gradient(
        self,
        program: ASOProgram,
        traces: List[Trace],
        *,
        focus_route: Optional[str] = None,
    ) -> str:
        if focus_route == _GLOBAL_FAILURE_ROUTE or focus_route is None:
            focus_route = None
        failure_block = "\n".join(
            (
                f"- Question: {trace.inputs[-1].content}\n"
                f"  Prediction: {trace.prediction.content}\n"
                f"  Critique: {trace.feedback.critique if trace.feedback else 'n/a'}"
            )
            for trace in traces[:8]
        )
        route_hint = (
            f"Focus this optimization on route `{focus_route}`."
            if focus_route
            else "No explicit route hint; treat failures holistically."
        )
        messages = [
            Message(
                role="system",
                content=(
                    "You are analyzing failures for a skill-evolution loop. "
                    "Do not just critique wording. Identify missing skills, weak skills, "
                    "and bad skill-selection behavior. Return a concise bullet list."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Root prompt:\n{program.root_prompt}\n\n"
                    f"Selection policy:\n{program.selection_policy}\n\n"
                    f"Current skills:\n{self._render_skill_inventory(program)}\n\n"
                    f"{route_hint}\n\n"
                    f"Failures:\n{failure_block}"
                ),
            ),
        ]
        response = self._llm.generate(messages, role="judge")
        return response.content if isinstance(response.content, str) else str(response.content)

    def propose_actions(
        self,
        program: ASOProgram,
        gradient: str,
        traces: List[Trace],
        *,
        focus_route: Optional[str] = None,
    ) -> List[ASOSkillAction]:
        focus_route = None if focus_route == _GLOBAL_FAILURE_ROUTE else focus_route
        focus_note = (
            f"\n\nRoute focus: `{focus_route}`. Prefer editing skills under this route."
            if focus_route
            else ""
        )
        messages = [
            Message(
                role="system",
                content=(
                    "You are a proposer for AS(skill)O. Based on the gradient and failures, "
                    "propose up to 2 structured skill actions.\n\n"
                    "Allowed action values:\n"
                    "- add_skill\n"
                    "- revise_skill\n"
                    "- drop_skill\n"
                    "- merge_skills\n"
                    "- adjust_selection_policy\n\n"
                    "Return ONLY valid JSON: "
                    '[{"action":"add_skill","skill_name":"...","description":"...","skill_prompt":"...","rationale":"..."}]\n'
                    "For revise_skill use target_skill + skill_prompt.\n"
                    "For merge_skills use merge_skills + skill_name + description + skill_prompt.\n"
                    "For adjust_selection_policy use selection_policy.\n"
                    "Prefer actionable skill proposals over generic prompt edits."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Current program skills:\n{self._render_skill_inventory(program)}\n\n"
                    f"Selection policy:\n{program.selection_policy}\n\n"
                    f"Gradient:\n{gradient}\n\n"
                    f"Representative failure:\n{traces[0].feedback.critique if traces and traces[0].feedback else 'n/a'}"
                    f"{focus_note}"
                ),
            ),
        ]
        response = self._llm.generate(messages, role="rewrite")
        raw = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _extract_json_payload(raw, expect_array=True)
        if parsed is None:
            logger.warning("Failed to parse ASO actions JSON: %s", raw[:240])
            return []
        if not isinstance(parsed, list):
            return []
        actions: List[ASOSkillAction] = []
        for item in parsed:
            if not isinstance(item, dict) or "action" not in item:
                continue
            action = str(item.get("action", "")).strip()
            if action not in {
                "add_skill",
                "revise_skill",
                "drop_skill",
                "merge_skills",
                "adjust_selection_policy",
            }:
                continue
            actions.append(
                ASOSkillAction(
                    action=action,
                    rationale=str(item.get("rationale", "")).strip(),
                    skill_name=item.get("skill_name"),
                    description=item.get("description"),
                    skill_prompt=item.get("skill_prompt"),
                    target_skill=item.get("target_skill"),
                    merge_skills=list(item.get("merge_skills", []) or []),
                    selection_policy=item.get("selection_policy"),
                    focus_route=focus_route,
                )
            )
        return self._dedupe_actions(actions)

    @staticmethod
    def _action_signature(action: ASOSkillAction) -> str:
        payload = {
            "action": action.action,
            "target_skill": action.target_skill or "",
            "skill_name": action.skill_name or "",
            "description": action.description or "",
            "selection_policy": action.selection_policy or "",
            "focus_route": action.focus_route or "",
            "merge_skills": sorted(action.merge_skills),
            "skill_prompt_hash": hashlib.sha1((action.skill_prompt or "").encode("utf-8")).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _dedupe_actions(actions: Iterable[ASOSkillAction]) -> List[ASOSkillAction]:
        seen: set[str] = set()
        deduped: List[ASOSkillAction] = []
        for action in actions:
            signature = ASOOptimizer._action_signature(action)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(action)
        return deduped

    @staticmethod
    def _candidate_signature(
        parent: ASOProgram,
        actions: Sequence[ASOSkillAction],
        focus_route: Optional[str],
    ) -> str:
        payload = {
            "parent_id": parent.program_id,
            "focus_route": focus_route or "",
            "action_signatures": [ASOOptimizer._action_signature(action) for action in actions],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def apply_actions(self, parent: ASOProgram, actions: List[ASOSkillAction]) -> ASOProgram:
        candidate = parent.bump_version()
        candidate.skills = [
            ASOSkill(
                name=skill.name,
                description=skill.description,
                prompt=skill.prompt,
                version=skill.version,
                tags=list(skill.tags),
                path=skill.path,
                parent_skill=skill.parent_skill,
            )
            for skill in parent.skills
        ]
        candidate.metadata.setdefault("applied_actions", [])
        candidate.metadata["trajectory_mode"] = self.trajectory_mode

        for action in actions:
            candidate.metadata["applied_actions"].append(action.__dict__)
            if action.focus_route:
                candidate.metadata["applied_actions"][-1]["focus_route"] = action.focus_route
            if action.action == "add_skill" and action.skill_name and action.skill_prompt:
                if any(skill.name == action.skill_name for skill in candidate.skills):
                    continue
                candidate.skills.append(
                    ASOSkill(
                        name=action.skill_name,
                        description=action.description or "",
                        prompt=action.skill_prompt,
                        path=(action.focus_route or ""),
                        tags=["generated"],
                    )
                )
            elif action.action == "revise_skill" and action.target_skill and action.skill_prompt:
                for skill in candidate.skills:
                    if skill.name == action.target_skill:
                        skill.prompt = action.skill_prompt
                        if action.description:
                            skill.description = action.description
                        skill.version = _increment_version(skill.version)
                        if action.focus_route and not skill.path:
                            skill.path = action.focus_route
                        break
            elif action.action == "drop_skill" and action.target_skill:
                candidate.skills = [
                    skill for skill in candidate.skills if skill.name != action.target_skill
                ]
            elif action.action == "merge_skills" and action.merge_skills and action.skill_name and action.skill_prompt:
                candidate.skills = [
                    skill for skill in candidate.skills if skill.name not in set(action.merge_skills)
                ]
                candidate.skills.append(
                    ASOSkill(
                        name=action.skill_name,
                        description=action.description or "",
                        prompt=action.skill_prompt,
                        path=(action.focus_route or ""),
                        tags=["merged"],
                    )
                )
            elif action.action == "adjust_selection_policy" and action.selection_policy:
                candidate.selection_policy = action.selection_policy

        return candidate

    def _auto_prune(
        self,
        program: ASOProgram,
        val_data: Sequence[SealQAExample],
        runner: Callable[[ASOProgram, SealQAExample], Any],
        scorer: Callable[[SealQAExample, str], float],
    ) -> tuple[ASOProgram, Optional[Dict[str, Any]]]:
        current = program
        base_score = float(program.score if program.score is not None else self._evaluate(program, val_data, runner, scorer))
        pruned: List[str] = []
        non_root_skills = [skill for skill in current.skills if "root" not in skill.tags]
        for skill in non_root_skills:
            candidate = current.bump_version()
            candidate.skills = [
                ASOSkill(
                    name=item.name,
                    description=item.description,
                    prompt=item.prompt,
                    version=item.version,
                    tags=list(item.tags),
                )
                for item in current.skills
                if item.name != skill.name
            ]
            candidate.selection_policy = current.selection_policy
            candidate.metadata = dict(current.metadata)
            score = self._evaluate(candidate, val_data, runner, scorer)
            if score >= base_score:
                current = candidate
                current.score = score
                pruned.append(skill.name)
                base_score = score
        if not pruned:
            return current, None
        current.metadata["auto_pruned_skills"] = pruned
        return current, {
            "type": "auto_prune",
            "accepted": True,
            "score": base_score,
            "pruned_skills": pruned,
            "skills": [skill.name for skill in current.skills],
        }

    def _auto_merge(
        self,
        program: ASOProgram,
        val_data: Sequence[SealQAExample],
        runner: Callable[[ASOProgram, SealQAExample], Any],
        scorer: Callable[[SealQAExample, str], float],
    ) -> tuple[ASOProgram, Optional[Dict[str, Any]]]:
        merge_action = self._propose_merge_action(program)
        if merge_action is None:
            return program, None
        candidate = self.apply_actions(program, [merge_action])
        base_score = float(program.score if program.score is not None else self._evaluate(program, val_data, runner, scorer))
        merge_score = self._evaluate(candidate, val_data, runner, scorer)
        if merge_score < base_score:
            return program, {
                "type": "auto_merge",
                "accepted": False,
                "base_score": base_score,
                "candidate_score": merge_score,
                "merged_from": merge_action.merge_skills,
                "merged_skill": merge_action.skill_name,
            }
        candidate.score = merge_score
        candidate.metadata["auto_merged_from"] = list(merge_action.merge_skills)
        candidate.metadata["auto_merged_skill"] = merge_action.skill_name
        return candidate, {
            "type": "auto_merge",
            "accepted": True,
            "score": merge_score,
            "merged_from": merge_action.merge_skills,
            "merged_skill": merge_action.skill_name,
            "skills": [skill.name for skill in candidate.skills],
        }

    def propose_apo_action(
        self,
        program: ASOProgram,
        gradient: str,
        traces: List[Trace],
        focus_route: Optional[str] = None,
    ) -> Optional[ASOSkillAction]:
        if self._apo_engine is None or not program.skills:
            return None
        if not traces:
            return None

        target = self._select_skill_for_apo(program, gradient, traces, focus_route=focus_route)
        if target is None:
            return None

        skill_payload = target.to_skill()
        skill_payload.target = (
            f"Improve this skill for the current program based on failure traces. "
            f"Context: {program.selection_policy}"
        )

        optimized = self._apo_engine.optimize(
            skill_payload,
            traces[-min(len(traces), self.apo_fallback_skill_limit) :],
        )
        if optimized.system_prompt == target.prompt:
            return None
        return ASOSkillAction(
            action="revise_skill",
            rationale=(
                f"Fallback APO rewrite from ASO textual gradient, targeting skill "
                f"'{target.name}'."
            ),
            target_skill=target.name,
            description=target.description,
            skill_prompt=optimized.system_prompt,
            skill_name=target.name,
            focus_route=focus_route,
        )

    def _select_skill_for_apo(
        self,
        program: ASOProgram,
        gradient: str,
        traces: List[Trace],
        focus_route: Optional[str] = None,
    ) -> Optional[ASOSkill]:
        if not program.skills:
            return None
        focus_route = None if focus_route == _GLOBAL_FAILURE_ROUTE else focus_route

        candidates = program.skills
        if focus_route:
            focus_candidates = [
                skill
                for skill in program.skills
                if self._skill_route_matches(skill, focus_route)
            ]
            candidates = focus_candidates or program.skills

        if traces:
            critique_text = " ".join(
                str(trace.feedback.critique) for trace in traces if trace.feedback and trace.feedback.critique
            ).lower()
            sample_topic = str(traces[0].inputs[-1].content).lower() if traces and traces[0].inputs else ""
            fallback_text = f"{gradient} {critique_text} {sample_topic}"
        else:
            fallback_text = gradient

        tokens = {
            token.strip(",:;\"'`()[]{}")
            for token in re.findall(r"[A-Za-z0-9_./-]+", fallback_text.lower())
        }
        if not candidates:
            return None
        if not tokens:
            return candidates[0]

        best_skill = candidates[0]
        best_score = -1.0
        for skill in candidates:
            content = f"{skill.name} {skill.description} {skill.prompt}".lower()
            match_score = sum(1 for token in tokens if token and token in content)
            if match_score > best_score:
                best_score = match_score
                best_skill = skill
        return best_skill

    @staticmethod
    def _normalized_route(route: Optional[str]) -> str:
        if not route:
            return ""
        return route.strip().strip(".")

    @staticmethod
    def _skill_route_matches(skill: ASOSkill, focus_route: str) -> bool:
        normalized = ASOOptimizer._normalized_route(focus_route)
        if not normalized:
            return True
        route = (skill.path or "").strip().strip(".")
        if not route:
            return False
        return route == normalized or route.startswith(f"{normalized}.") or normalized.startswith(f"{route}.")


    def _propose_merge_action(self, program: ASOProgram) -> Optional[ASOSkillAction]:
        candidates = self._rank_merge_pairs(program)
        if not candidates:
            return None
        left, right = candidates[0]
        left_skill = next(skill for skill in program.skills if skill.name == left)
        right_skill = next(skill for skill in program.skills if skill.name == right)
        messages = [
            Message(
                role="system",
                content=(
                    "You are merging overlapping agent skills for AS(skill)O. "
                    "Return ONLY valid JSON with keys: skill_name, description, skill_prompt, rationale. "
                    "If the pair should not be merged, return [] instead."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Skill A\nname: {left_skill.name}\n"
                    f"description: {left_skill.description}\n"
                    f"prompt:\n{left_skill.prompt}\n\n"
                    f"Skill B\nname: {right_skill.name}\n"
                    f"description: {right_skill.description}\n"
                    f"prompt:\n{right_skill.prompt}\n\n"
                    "Create one merged skill that preserves the useful behavior of both and removes redundancy."
                ),
            ),
        ]
        response = self._llm.generate(messages, role="rewrite")
        raw = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _extract_json_payload(raw, expect_array=False)
        if raw == "[]":
            return None
        if parsed is None:
            logger.warning("Failed to parse ASO merge JSON: %s", raw[:240])
            return None
        if not isinstance(parsed, dict):
            return None
        skill_name = str(parsed.get("skill_name", "")).strip()
        skill_prompt = str(parsed.get("skill_prompt", "")).strip()
        if not skill_name or not skill_prompt:
            return None
        return ASOSkillAction(
            action="merge_skills",
            rationale=str(parsed.get("rationale", "")).strip(),
            skill_name=skill_name,
            description=str(parsed.get("description", "")).strip(),
            skill_prompt=skill_prompt,
            merge_skills=[left, right],
        )

    @staticmethod
    def _rank_merge_pairs(program: ASOProgram) -> List[tuple[str, str]]:
        skills = [skill for skill in program.skills if "root" not in skill.tags]
        ranked: List[tuple[float, tuple[str, str]]] = []
        for idx, left in enumerate(skills):
            for right in skills[idx + 1:]:
                left_tags = set(left.tags)
                right_tags = set(right.tags)
                tag_overlap = len(left_tags & right_tags)
                left_words = set((left.description + " " + left.prompt).lower().split())
                right_words = set((right.description + " " + right.prompt).lower().split())
                text_overlap = len(left_words & right_words)
                score = tag_overlap * 10 + text_overlap
                if score <= 0:
                    continue
                ranked.append((float(score), (left.name, right.name)))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [pair for _score, pair in ranked]

    def _collect_failure_traces(
        self,
        program: ASOProgram,
        train_data: Sequence[SealQAExample],
        runner: Callable[[ASOProgram, SealQAExample], Any],
        scorer: Callable[[SealQAExample, str], float],
    ) -> List[Trace]:
        traces: List[Trace] = []
        if self.max_workers == 1:
            for sample in train_data:
                trace = self._score_sample(program, sample, runner, scorer)
                if trace is not None:
                    traces.append(trace)
            return traces

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._score_sample, program, sample, runner, scorer)
                for sample in train_data
            ]
            for future in as_completed(futures):
                try:
                    trace = future.result()
                    if trace is not None:
                        traces.append(trace)
                except Exception:
                    logger.warning("Failure trace worker error:\n%s", traceback.format_exc())
        return traces

    def _group_traces_by_route(
        self,
        program: ASOProgram,
        traces: Sequence[Trace],
    ) -> Dict[str, List[Trace]]:
        if not self.trajectory_mode:
            return {_GLOBAL_FAILURE_ROUTE: list(traces)}

        groups: Dict[str, List[Trace]] = {}
        for trace in traces:
            route = self._infer_trace_route(program, trace)
            groups.setdefault(route, []).append(trace)
        return groups

    def _infer_trace_route(self, program: ASOProgram, trace: Trace) -> str:
        if trace.node_path:
            return trace.node_path
        metadata = trace.metadata if hasattr(trace, "metadata") else {}
        if isinstance(metadata, dict):
            route = metadata.get("route") or metadata.get("focus_route") or metadata.get("path")
            if isinstance(route, str) and route.strip():
                return route.strip()
            selected_skill = metadata.get("selected_skill") or metadata.get("skill_name")
            if isinstance(selected_skill, str) and selected_skill.strip():
                route_skill = self._infer_focus_skill(program, selected_skill)
                if route_skill:
                    return self._route_key(route_skill)

        user_text = str(trace.inputs[-1].content) if trace.inputs else ""
        selected_skill = self._infer_failure_focus_skill(program, trace, user_text)
        if selected_skill is not None:
            return self._route_key(selected_skill)
        return _GLOBAL_FAILURE_ROUTE

    def _infer_failure_focus_skill(
        self,
        program: ASOProgram,
        trace: Trace,
        user_text: str,
    ) -> Optional[ASOSkill]:
        candidate_text = []
        candidate_text.append(user_text or "")
        if trace.feedback and trace.feedback.critique:
            candidate_text.append(str(trace.feedback.critique))
        if isinstance(trace.metadata, dict):
            candidate_text.append(str(trace.metadata.get("topic", "")))
            candidate_text.append(str(trace.metadata.get("question", "")))
        fallback_text = " ".join(part for part in candidate_text if part).strip()
        return self._infer_focus_skill(program, fallback_text)

    def _infer_focus_skill(self, program: ASOProgram, text: str) -> Optional[ASOSkill]:
        if not program.skills:
            return None
        candidate_text = str(text or "").strip().lower()
        tokens = {
            token.strip(",:;\"'`()[]{}")
            for token in re.findall(r"[A-Za-z0-9_./-]+", candidate_text.lower())
        }
        if not tokens:
            return program.skills[0]

        best_skill = program.skills[0]
        best_score = -1.0
        for skill in program.skills:
            content = f"{skill.name} {skill.description} {skill.prompt} {skill.path}".lower()
            match_score = sum(1 for token in tokens if token and token in content)
            if match_score > best_score:
                best_score = match_score
                best_skill = skill
        return best_skill

    @staticmethod
    def _route_key(skill: ASOSkill) -> str:
        return ASOOptimizer._normalized_route(skill.path) or ASOOptimizer._normalize_name(skill.name)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", name).strip("._-").lower()

    def _collect_sample_metadata(
        self,
        sample: SealQAExample,
        prediction: str,
        route_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if sample.topic:
            meta["topic"] = sample.topic
        if sample.metadata:
            meta.update(sample.metadata)
        if sample.question:
            meta["question"] = sample.question
        if prediction:
            meta["prediction"] = prediction[:512]
        if route_metadata:
            for key, value in route_metadata.items():
                if isinstance(key, str) and key.strip() and value is not None:
                    meta[key.strip()] = value
        meta.setdefault("source", "aso")
        return meta

    def _evaluate(
        self,
        program: ASOProgram,
        data: Sequence[SealQAExample],
        runner: Callable[[ASOProgram, SealQAExample], Any],
        scorer: Callable[[SealQAExample, str], float],
    ) -> float:
        if not data:
            return 0.0

        if self.max_workers == 1:
            total = 0.0
            for sample in data:
                total += self._score_value(program, sample, runner, scorer)
            return total / len(data)

        total = 0.0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._score_value, program, sample, runner, scorer)
                for sample in data
            ]
            for future in as_completed(futures):
                try:
                    total += future.result()
                except Exception:
                    logger.warning("Scoring worker error:\n%s", traceback.format_exc())
                    total += 0.0
        return total / len(data)

    @staticmethod
    def _score_value(
        program: ASOProgram,
        sample: SealQAExample,
        runner: Callable[[ASOProgram, SealQAExample], Any],
        scorer: Callable[[SealQAExample, str], float],
    ) -> float:
        try:
            prediction, _route_metadata = ASOOptimizer._run_program(program, sample, runner)
            return scorer(sample, prediction)
        except Exception:
            logger.warning("Sample scoring failed:\n%s", traceback.format_exc())
            return 0.0

    def _score_sample(
        self,
        program: ASOProgram,
        sample: SealQAExample,
        runner: Callable[[ASOProgram, SealQAExample], Any],
        scorer: Callable[[SealQAExample, str], float],
    ) -> Optional[Trace]:
        try:
            prediction, route_metadata = self._run_program(program, sample, runner)
            score = scorer(sample, prediction)
        except Exception:
            logger.warning("Failure trace scoring failed:\n%s", traceback.format_exc())
            prediction = ""
            score = 0.0
            route_metadata: Dict[str, Any] = {}
        # Keep trace-level metadata lightweight; used by route inference in trajectory mode.
        route_metadata = self._collect_sample_metadata(sample, prediction, route_metadata)
        if score >= 1.0:
            return None
        return Trace(
            session_id=program.program_id,
            inputs=[Message(role="user", content=sample.question)],
            prediction=Message(role="assistant", content=prediction),
            feedback=Feedback(
                score=score,
                critique=(
                    f"Incorrect or incomplete answer for topic={sample.topic}. "
                    f"Expected: {sample.answer}. Got: {prediction}"
                ),
                correction=sample.answer,
            ),
            metadata=route_metadata,
        )

    @staticmethod
    def _run_program(
        program: ASOProgram,
        sample: SealQAExample,
        runner: Callable[[ASOProgram, SealQAExample], Any],
    ) -> Tuple[str, Dict[str, Any]]:
        result = runner(program, sample)
        return ASOOptimizer._normalize_runner_output(result)

    @staticmethod
    def _normalize_runner_output(result: Any) -> Tuple[str, Dict[str, Any]]:
        """Normalize heterogeneous runner outputs to `(prediction, metadata)`."""
        if result is None:
            return "", {}

        if isinstance(result, (tuple, list)):
            if not result:
                return "", {}
            prediction, metadata = ASOOptimizer._normalize_runner_output(result[0])
            if len(result) > 1 and isinstance(result[1], dict):
                if metadata:
                    merged = dict(metadata)
                    merged.update(result[1])
                    metadata = merged
                else:
                    metadata = result[1]
            return prediction, metadata

        if isinstance(result, dict):
            prediction = result.get("result", "")
            if prediction in (None, ""):
                prediction = result.get("output", "")
            if prediction in (None, ""):
                prediction = result.get("answer", "")
            if prediction is None:
                prediction = ""

            metadata: Dict[str, Any] = {}
            nested_metadata = result.get("metadata")
            if isinstance(nested_metadata, dict):
                metadata.update(nested_metadata)

            for key, value in result.items():
                if key in {"result", "output", "answer", "metadata"}:
                    continue
                if isinstance(key, str):
                    metadata[key] = value

            return str(prediction).strip(), metadata

        return str(result).strip() if result is not None else "", {}

    @staticmethod
    def _render_skill_inventory(program: ASOProgram) -> str:
        if not program.skills:
            return "(none)"
        return "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in program.skills
        )


def _increment_version(version: str) -> str:
    raw = version[1:] if version.startswith("v") else version
    parts = raw.split(".")
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].isdigit():
            parts[idx] = str(int(parts[idx]) + 1)
            return "v" + ".".join(parts)
    return f"{version}.1"
