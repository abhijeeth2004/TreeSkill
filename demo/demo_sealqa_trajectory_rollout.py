#!/usr/bin/env python3
"""Single-trajectory rollout demo for SealQA ASO.

This utility runs one trajectory at a time: for each sample, evaluate current
program, optionally run one ASO cycle on the failure trace, then evaluate again.
It is designed to answer:
 - 为什么只优化一条轨迹时会出现精度下降
 - 在何种条件下是单样本过拟合（测试集跌幅）
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()

from treeskill.aso_optimizer import ASOOptimizer
from treeskill.aso_program import ASOProgram, ASOSkill
from treeskill.config import APOConfig, GlobalConfig, LLMConfig
from treeskill.llm import LLMClient
from treeskill.schema import Message
from treeskill.tasks.sealqa import SealQAExample

import importlib.util

_TOOLKIT_PATH = Path(__file__).resolve().parent / "tools" / "search_toolkit.py"
_TOOLKIT_SPEC = importlib.util.spec_from_file_location("demo_search_toolkit", _TOOLKIT_PATH)
if _TOOLKIT_SPEC is None or _TOOLKIT_SPEC.loader is None:
    raise RuntimeError(f"Cannot load search toolkit: {_TOOLKIT_PATH}")
_TOOLKIT_MODULE = importlib.util.module_from_spec(_TOOLKIT_SPEC)
_TOOLKIT_SPEC.loader.exec_module(_TOOLKIT_MODULE)
build_search_web_script = _TOOLKIT_MODULE.build_search_web_script
build_fetch_script = _TOOLKIT_MODULE.build_fetch_script


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


BASELINE_ROOT_PROMPT = (
    "You are a search-augmented QA assistant working in a coding-agent environment. "
    "Answer the user's question as accurately as possible. Use tools when needed. "
    "Return only the final answer."
)


@dataclass
class RolloutStep:
    step: int
    sample_id: str
    sample_question: str
    before_sample_score: float
    before_overall_score: float
    optimized_sample_score: float
    optimized_overall_score: float
    accepted: bool
    fallback_reason: str | None = None
    candidate_score: float | None = None
    applied_actions: int = 0
    candidate_program_id: str | None = None
    previous_program_id: str | None = None


def _safe_to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, set):
        return [str(item).strip() for item in sorted(value) if str(item).strip()]
    if not isinstance(value, str):
        return [str(value).strip()] if str(value).strip() else []
    text = value.strip()
    if not text:
        return []
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in re.split(r"[;,]", text) if item.strip()]


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return {str(k).strip(): _clean_value(v) for k, v in value.items() if str(k).strip()}
    return str(value)


def _extract_cache_records(samples: list[SealQAExample]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for idx, sample in enumerate(samples, start=1):
        metadata = sample.metadata or {}
        urls = _safe_to_list(metadata.get("urls"))
        keywords = _safe_to_list(metadata.get("keywords"))
        golds = _safe_to_list(metadata.get("golds") or metadata.get("answers") or metadata.get("answer"))
        texts = [sample.question, sample.answer]
        texts.extend(_safe_to_list(metadata.get("snippet")))
        texts.extend(_safe_to_list(metadata.get("search_results")))
        if golds:
            texts.extend(golds)
        snippet = "; ".join(part for part in texts if part).strip()
        records.append(
            {
                "id": f"sample_{idx:04d}",
                "topic": sample.topic or "unknown",
                "question": sample.question or "",
                "keywords": ", ".join(sorted(set(keywords))),
                "title": sample.question or "",
                "url": urls[0] if urls else "",
                "snippet": snippet[:1200],
                "text": "\n".join(texts).strip(),
            }
        )
    return records


def _serialize_step(step: RolloutStep) -> dict[str, Any]:
    return step.__dict__.copy()


def _deserialize_step(payload: dict[str, Any]) -> RolloutStep:
    return RolloutStep(
        step=int(payload["step"]),
        sample_id=str(payload["sample_id"]),
        sample_question=payload.get("sample_question", "")[:120],
        before_sample_score=float(payload["before_sample_score"]),
        before_overall_score=float(payload["before_overall_score"]),
        optimized_sample_score=float(payload["optimized_sample_score"]),
        optimized_overall_score=float(payload["optimized_overall_score"]),
        accepted=bool(payload["accepted"]),
        fallback_reason=payload.get("fallback_reason"),
        candidate_score=(None if payload.get("candidate_score") is None else float(payload["candidate_score"])),
        applied_actions=int(payload.get("applied_actions", 0)),
        candidate_program_id=payload.get("candidate_program_id"),
        previous_program_id=payload.get("previous_program_id"),
    )


def _program_to_dict(program: ASOProgram) -> dict[str, Any]:
    return program.to_dict()


def _program_from_dict(payload: dict[str, Any]) -> ASOProgram:
    skills = [
        ASOSkill(
            name=skill_payload.get("name", ""),
            description=skill_payload.get("description", ""),
            prompt=skill_payload.get("prompt", ""),
            version=skill_payload.get("version", "v1.0"),
            tags=list(skill_payload.get("tags", [])),
            path=str(skill_payload.get("path", "")),
            parent_skill=str(skill_payload.get("parent_skill", "")),
        )
        for skill_payload in payload.get("skills", [])
    ]
    return ASOProgram(
        root_prompt=payload.get("root_prompt", BASELINE_ROOT_PROMPT),
        skills=skills,
        selection_policy=payload.get("selection_policy", ""),
        version=payload.get("version", "v1.0"),
        score=payload.get("score"),
        parent_id=payload.get("parent_id"),
        program_id=str(payload.get("program_id") or uuid.uuid4()),
        metadata=dict(payload.get("metadata", {})),
    )


def _load_resume_state(output_dir: Path) -> tuple[int, ASOProgram | None, list[RolloutStep]]:
    state_path = output_dir / "run_state.json"
    if not state_path.exists():
        return 0, None, []

    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return 0, None, []

    next_step = int(raw.get("next_step", 0) or 0)
    history_payload = raw.get("history", [])
    history: list[RolloutStep] = []
    for item in history_payload:
        if not isinstance(item, dict):
            continue
        try:
            history.append(_deserialize_step(item))
        except Exception:
            continue

    current_program_payload = raw.get("current_program")
    current_program: ASOProgram | None = None
    if isinstance(current_program_payload, dict):
        try:
            current_program = _program_from_dict(current_program_payload)
        except Exception:
            current_program = None

    return next_step, current_program, history


def _save_resume_state(output_dir: Path, next_step: int, program: ASOProgram, history: list[RolloutStep]) -> None:
    state_path = output_dir / "run_state.json"
    payload = {
        "next_step": next_step,
        "current_program": _program_to_dict(program),
        "history": [_serialize_step(item) for item in history],
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _stable_workspace_id(sample: SealQAExample, label: str) -> str:
    topic = (sample.topic or "no_topic").strip().replace("/", "_")
    question = (sample.question or "").strip().replace("\n", " ")
    digest = hashlib.sha1(f"{label}|{topic}|{question}".encode("utf-8")).hexdigest()[:16]
    return digest


def _prepare_search_workspace(workspace: Path, cache_records: list[dict[str, str]]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "search_web.py").write_text(build_search_web_script(), encoding="utf-8")
    (workspace / "search_web.py").chmod(0o755)
    (workspace / "fetch_url.py").write_text(build_fetch_script(), encoding="utf-8")
    (workspace / "fetch_url.py").chmod(0o755)
    (workspace / "search_cache.json").write_text(json.dumps(cache_records, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_metadata_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return {str(key).strip(): _normalize_metadata_value(val) for key, val in value.items() if str(key).strip()}
    return str(value)


def _extract_kode_payload(raw: str) -> tuple[str, Dict[str, Any]]:
    if not raw:
        return "", {}

    def _is_metadata_key(key: str) -> bool:
        return bool(key) and key.strip() not in {"result", "output", "answer", "metadata"}

    def _from_parsed(payload: Any) -> tuple[str, Dict[str, Any]]:
        if payload is None:
            return "", {}
        if isinstance(payload, (list, tuple)):
            if not payload:
                return "", {}
            return _from_parsed(payload[0])
        if isinstance(payload, dict):
            if not payload:
                return "", {}
            prediction = payload.get("result", "")
            if prediction in (None, ""):
                prediction = payload.get("output", "")
            if prediction in (None, ""):
                prediction = payload.get("answer", "")
            if prediction is None:
                prediction = ""
            metadata: Dict[str, Any] = {}
            nested_metadata = payload.get("metadata", {})
            if isinstance(nested_metadata, dict):
                metadata.update(
                    {str(k).strip(): _normalize_metadata_value(v) for k, v in nested_metadata.items() if str(k).strip()}
                )
            for key, value in payload.items():
                if _is_metadata_key(str(key)):
                    metadata[str(key).strip()] = _normalize_metadata_value(value)
            return str(prediction).strip(), metadata
        return str(payload).strip(), {}

    parsed: Any | None = None
    raw_text = raw.strip()
    try:
        parsed = json.loads(raw_text)
    except Exception:
        pass

    if parsed is None:
        match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```", raw_text, flags=re.IGNORECASE)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except Exception:
                parsed = None

    if parsed is None:
        for marker in ("{", "["):
            index = raw_text.find(marker)
            if index == -1:
                continue
            candidate = raw_text[index:]
            for suffix in ("}", "]"):
                end = candidate.rfind(suffix)
                if end <= 0:
                    continue
                try:
                    parsed = json.loads(candidate[: end + 1])
                    if parsed:
                        break
                except Exception:
                    pass
            if parsed is not None:
                break

    if parsed is None:
        return raw_text, {}
    return _from_parsed(parsed)


def _load_samples(path: Path, limit: int | None = None) -> list[SealQAExample]:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    samples: list[SealQAExample] = []
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must be an array of sample objects.")
        for item in payload:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            topic = str(item.get("topic", "unknown")).strip()
            metadata = {
                k: str(v).strip()
                for k, v in item.items()
                if k not in {"question", "answer", "topic"} and v is not None
            }
            samples.append(SealQAExample(question=question, answer=answer, topic=topic, metadata=metadata))
    elif suffix == ".jsonl":
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            topic = str(item.get("topic", "unknown")).strip()
            metadata = {
                k: str(v).strip()
                for k, v in item.items()
                if k not in {"question", "answer", "topic"} and v is not None
            }
            samples.append(SealQAExample(question=question, answer=answer, topic=topic, metadata=metadata))
    else:
        from treeskill.tasks.sealqa import SealQATaskAdapter

        adapter = SealQATaskAdapter(path)
        samples = adapter.load()

    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError(f"No usable samples loaded from {path}")
    return samples


def _build_baseline_program() -> ASOProgram:
    return ASOProgram(
        root_prompt=BASELINE_ROOT_PROMPT,
        skills=[
            ASOSkill(
                name="answer-format",
                description="Keep final answers concise and avoid extra explanation.",
                prompt=(
                    "When you have enough evidence, produce a short final answer. "
                    "Do not include chain-of-thought. Prefer direct factual wording."
                ),
                tags=["baseline"],
            ),
            ASOSkill(
                name="search_web_lookup",
                description="Use local cache first, then external fallback if available.",
                prompt=(
                    "For factual questions, run:\n"
                    "python search_web.py --query \"<full user question>\" --top-k 3\n"
                    "Then run `python fetch_url.py --url \"<best url>\"` and answer from fetched evidence."
                ),
                tags=["retrieval", "baseline"],
            ),
            ASOSkill(
                name="verified_fact_lookup",
                description="Do not answer beyond evidence from tool output.",
                prompt=(
                    "Before finalizing, require evidence from search/fetch output. "
                    "If no sufficient evidence exists, answer with uncertainty instead of fabrication."
                ),
                tags=["verification", "baseline"],
            ),
        ],
        selection_policy=(
            "Use `search_web_lookup` for factual questions first. "
            "If cache fails and fallback is enabled, use it once and then answer from evidence."
        ),
    )


def _judge_answer(
    llm: LLMClient,
    sample: SealQAExample,
    prediction: str,
    timeout_seconds: int = 30,
) -> float:
    prompt = (
        f"Question: {sample.question}\n"
        f"Ground truth answer: {sample.answer}\n"
        f"Predicted answer: {prediction}\n\n"
        "Judge only the correctness of the answer. Return ONLY 1 or 0."
    )

    def _sync_call() -> float:
        response = llm.generate(
            [
                Message(role="system", content="You are a strict answer judge. Return only 1 or 0."),
                Message(role="user", content=prompt),
            ],
            role="judge",
            max_tokens=32,
            temperature=0.0,
        )
        raw = response.content.strip()
        match = re.search(r"\b([01])\b", raw)
        return 1.0 if match and match.group(1) == "1" else 0.0

    with ThreadPoolExecutor(max_workers=1) as judge_executor:
        future = judge_executor.submit(_sync_call)
        try:
            return future.result(timeout=timeout_seconds)
        except TimeoutError:
            logger.warning(
                "judge timeout: sample=%s timeout=%ss",
                sample.question[:60],
                timeout_seconds,
            )
            return 0.0
        except Exception as exc:
            logger.warning("judge error: %s", exc)
            return 0.0


def _run_kode(
    program: ASOProgram,
    sample: SealQAExample,
    args: argparse.Namespace,
    cache_records: list[dict[str, str]],
    workspace_root: Path,
    *,
    label: str,
) -> tuple[str, Dict[str, Any]]:
    workspace = workspace_root / f"{label}_{_stable_workspace_id(sample, label)}"
    if workspace.exists():
        for item in workspace.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    _prepare_search_workspace(workspace, cache_records)
    (workspace / "AGENTS.md").write_text(program.render_agents_markdown(), encoding="utf-8")

    env = os.environ.copy()
    if args.actor_api_key:
        env["TREE_LLM_API_KEY"] = args.actor_api_key
    if args.actor_base_url:
        env["TREE_LLM_BASE_URL"] = args.actor_base_url
    if args.actor_protocol:
        env["TREE_LLM_PROTOCOL"] = args.actor_protocol
    if args.actor_model:
        env["TREE_LLM_MODEL"] = args.actor_model

    command = [
        "kode",
        "-p",
        sample.question,
        "--model",
        args.actor_model,
        "--cwd",
        str(workspace),
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
    ]

    with tempfile.NamedTemporaryFile("w+", suffix=".stdout", encoding="utf-8", delete=True) as stdout_file, tempfile.NamedTemporaryFile(
        "w+", suffix=".stderr", encoding="utf-8", delete=True
    ) as stderr_file:
        try:
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                start_new_session=True,
            )
            proc.wait(timeout=args.kode_timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(proc.pid) if proc.pid else None
            except Exception:
                pgid = None
            if pgid:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(pgid, sig)
                    except Exception:
                        pass
            else:
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
            logger.warning("Kode timeout sample=%s", sample.question[:60])
            return "", {}

        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()

    if proc.returncode != 0 and not (stdout and stdout.strip()):
        logger.warning("Kode exit=%s stderr=%s", proc.returncode, (stderr or "").strip()[:200])
        return "", {}
    if not stdout.strip():
        return "", {}

    try:
        prediction, metadata = _extract_kode_payload(stdout)
    except Exception:
        logger.warning("Could not parse Kode output: %s", stdout[:200])
        return stdout.strip(), {}
    return prediction, metadata


def _run_sample(
    program: ASOProgram,
    sample: SealQAExample,
    llm: LLMClient,
    args: argparse.Namespace,
    cache_records: list[dict[str, str]],
    workspace_root: Path,
    idx: int,
    split: str,
    *,
    judge_timeout_seconds: int = 30,
) -> Tuple[float, Dict[str, object]]:
    t0 = time.time()
    prediction, route_metadata = _run_kode(
        program,
        sample,
        args=args,
        cache_records=cache_records,
        workspace_root=workspace_root,
        label=f"{split}_{idx:03d}",
    )
    kode_seconds = time.time() - t0
    score = _judge_answer(llm, sample, prediction, timeout_seconds=judge_timeout_seconds)
    return score, {
        "sample_id": idx,
        "question": sample.question,
        "answer": sample.answer,
        "topic": sample.topic,
        "prediction": prediction,
        "score": score,
        "route_metadata": route_metadata,
        "kode_seconds": round(kode_seconds, 2),
    }


def _evaluate_program(
    program: ASOProgram,
    samples: Sequence[SealQAExample],
    llm: LLMClient,
    args: argparse.Namespace,
    cache_records: list[dict[str, str]],
    workspace_root: Path,
    *,
    label: str,
) -> Tuple[float, list[dict[str, object]], int]:
    if not samples:
        return 0.0, [], 0
    indexed_rows: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.eval_workers) as executor:
        futures = [
            executor.submit(
                _run_sample,
                program,
                sample,
                llm,
                args,
                cache_records,
                workspace_root / label,
                idx + 1,
                label,
                judge_timeout_seconds=args.judge_timeout_seconds,
            )
            for idx, sample in enumerate(samples)
        ]
        for future in as_completed(futures):
            score, row = future.result()
            # ThreadPoolExecutor returns futures out of order; keep deterministic
            # sample ordering by assigning explicit ids inside the loop and sorting later.
            # Find the sample idx embedded in row (safe fallback to next index by insertion).
            idx = len(indexed_rows) + 1
            if "sample_id" in row:
                try:
                    parsed_idx = (
                        int(row["sample_id"])
                        if isinstance(row["sample_id"], int)
                        else int(re.search(r"\d+", str(row["sample_id"])).group(0))
                    )
                    idx = parsed_idx
                except Exception:
                    pass
            indexed_rows[idx] = row
            row["score"] = score
        ordered_rows = [indexed_rows[i] for i in sorted(indexed_rows)]

    correct = sum(float(row["score"]) for row in ordered_rows)
    accuracy = correct / len(ordered_rows) if ordered_rows else 0.0
    hits = int(sum(1 for row in ordered_rows if float(row["score"]) >= 1.0))
    return accuracy, ordered_rows, hits


def _run_single_trajectory(
    step: int,
    sample: SealQAExample,
    current_program: ASOProgram,
    llm: LLMClient,
    args: argparse.Namespace,
    cache_records: list[dict[str, str]],
    rollout_samples: Sequence[SealQAExample],
) -> Tuple[RolloutStep, ASOProgram]:
    workspace_root = args.output_dir / "workspaces"
    before_sample_score = _run_sample(
        current_program,
        sample,
        llm,
        args=args,
        cache_records=cache_records,
        workspace_root=workspace_root,
        idx=step,
        split="before",
        judge_timeout_seconds=args.judge_timeout_seconds,
    )[0]

    before_overall, before_rows, before_hits = _evaluate_program(
        current_program,
        rollout_samples,
        llm,
        args=args,
        cache_records=cache_records,
        workspace_root=workspace_root,
        label=f"overall_before_step_{step}",
    )

    if before_sample_score >= 1.0:
        return RolloutStep(
            step=step,
            sample_id=str(step),
            sample_question=sample.question[:120],
            before_sample_score=before_sample_score,
            before_overall_score=before_overall,
            optimized_sample_score=before_sample_score,
            optimized_overall_score=before_overall,
            accepted=False,
            fallback_reason="sample_already_correct",
            candidate_score=None,
            applied_actions=0,
            candidate_program_id=None,
            previous_program_id=current_program.program_id,
        ), current_program

    optimizer = ASOOptimizer(
        llm,
        frontier_size=args.frontier_size,
        branch_factor=args.branch_factor,
        max_iterations=max(1, args.trajectory_iterations),
        max_workers=args.aso_workers,
        trajectory_mode=args.trajectory_mode,
        trajectory_focus_top_k=max(1, args.trajectory_focus_top_k),
        auto_merge=False,
        auto_prune=False,
    )

    def runner(program: ASOProgram, item: SealQAExample):
        pred, _ = _run_kode(
            program,
            item,
            args=args,
            cache_records=cache_records,
            workspace_root=workspace_root / f"iter_{step:03d}",
            label=f"opt_train",
        )
        return pred

    result = optimizer.run(
        current_program,
        [sample],
        [sample],
        runner=runner,
        scorer=lambda s, p: _judge_answer(llm, s, p, timeout_seconds=args.judge_timeout_seconds),
        start_iteration=1,
    )
    candidate = result.best_program
    candidate_score = result.final_score
    latest_history = result.history[-1] if result.history else None

    holdout_set = (
        list(rollout_samples[step:]) if args.validate_on_remaining else list(rollout_samples)
    )
    if not holdout_set:
        holdout_set = list(rollout_samples)
    before_holdout_acc, before_holdout_rows, before_holdout_hits = _evaluate_program(
        current_program,
        holdout_set,
        llm,
        args=args,
        cache_records=cache_records,
        workspace_root=workspace_root,
        label=f"candidate_before_step_{step}",
    )
    after_candidate_acc, holdout_rows, holdout_hits = _evaluate_program(
        candidate,
        holdout_set,
        llm,
        args=args,
        cache_records=cache_records,
        workspace_root=workspace_root,
        label=f"candidate_after_step_{step}",
    )

    after_sample_score = _run_sample(
        candidate,
        sample,
        llm,
        args=args,
        cache_records=cache_records,
        workspace_root=workspace_root,
        idx=step,
        split="after",
        judge_timeout_seconds=args.judge_timeout_seconds,
    )[0]

    accepted = False
    fallback = None
    candidate_program_id = candidate.program_id
    next_program = current_program
    if holdout_set:
        # Strict mode: prevent obvious overfitting to the single trajectory.
        after_metric = after_candidate_acc
        before_metric = before_holdout_acc
        if after_metric + args.regression_tolerance >= before_metric:
            accepted = True
            after_overall = after_candidate_acc
            next_program = candidate
        else:
            fallback = (
                f"rollout_regression_guard: before_holdout={before_metric:.3f}, "
                f"candidate_holdout={after_metric:.3f}"
            )
            after_overall = before_overall
            after_sample_score = before_sample_score
    else:
        accepted = True
        after_overall = after_candidate_acc
        next_program = candidate

    return RolloutStep(
        step=step,
        sample_id=str(step),
        sample_question=sample.question[:120],
        before_sample_score=before_sample_score,
        before_overall_score=before_overall,
        optimized_sample_score=after_sample_score if accepted else before_sample_score,
        optimized_overall_score=after_overall,
        accepted=accepted,
        fallback_reason=fallback,
        candidate_score=candidate_score,
        applied_actions=(
            len(latest_history.actions) if latest_history is not None and hasattr(latest_history, "actions") else 0
        ),
        candidate_program_id=candidate_program_id,
        previous_program_id=current_program.program_id,
    ), next_program


def _resolve_protocol(model: str, base_url: str, explicit: str) -> str:
    if explicit:
        return explicit
    lowered = (base_url or "").lower()
    if "anthropic" in lowered or "minimax" in (model or "").lower():
        return "anthropic"
    return "openai"


def _build_llm(args: argparse.Namespace) -> LLMClient:
    return LLMClient(
        GlobalConfig(
            llm=LLMConfig(
                api_key=args.judge_api_key,
                base_url=args.judge_base_url,
                model=args.actor_model or "gpt-4o",
                protocol=args.actor_protocol,
                temperature=0.0,
                judge_api_key=args.judge_api_key,
                judge_base_url=args.judge_base_url,
                judge_model=args.judge_model,
                judge_protocol=args.judge_protocol,
                judge_temperature=0.0,
                judge_extra_body={"thinking": {"type": "disabled"}} if "anthropic" in args.judge_protocol else None,
                rewrite_api_key=args.rewrite_api_key or args.judge_api_key,
                rewrite_base_url=args.rewrite_base_url or args.judge_base_url,
                rewrite_model=args.judge_model,
                rewrite_protocol=args.judge_protocol,
                rewrite_temperature=0.0,
                rewrite_extra_body={"thinking": {"type": "disabled"}} if "anthropic" in args.judge_protocol else None,
            ),
            apo=APOConfig(beam_width=2, branch_factor=args.branch_factor, beam_rounds=1),
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-trajectory ASO rollouts on SealQA samples."
    )
    parser.add_argument("--dataset", default=os.getenv("SEALQA_PATH", "/Users/mzm/Downloads/sealqa/seal-0.parquet"))
    parser.add_argument("--rollout-size", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--aso-workers", type=int, default=8)
    parser.add_argument("--actor-model", default=os.getenv("KODE_ACTOR_MODEL", "MiniMax-M2.7"))
    parser.add_argument("--actor-base-url", default=os.getenv("KODE_ACTOR_BASE_URL", "https://api.minimaxi.com/anthropic"))
    parser.add_argument("--actor-protocol", default=os.getenv("KODE_ACTOR_PROTOCOL", "anthropic"))
    parser.add_argument("--actor-api-key", default=os.getenv("KODE_ACTOR_API_KEY", os.getenv("MINIMAX_API_KEY", "")))
    parser.add_argument("--judge-model", default=os.getenv("TREE_LLM_JUDGE_MODEL", "MiniMax-M2.7"))
    parser.add_argument("--judge-base-url", default=os.getenv("TREE_LLM_JUDGE_BASE_URL", "https://api.minimaxi.com/anthropic"))
    parser.add_argument("--judge-protocol", default=os.getenv("TREE_LLM_JUDGE_PROTOCOL", ""))
    parser.add_argument("--judge-api-key", default=os.getenv("TREE_LLM_JUDGE_API_KEY", os.getenv("MINIMAX_API_KEY", "")))
    parser.add_argument("--rewrite-api-key", default=os.getenv("TREE_LLM_REWRITE_API_KEY", ""))
    parser.add_argument("--rewrite-base-url", default=os.getenv("TREE_LLM_REWRITE_BASE_URL", ""))
    parser.add_argument("--rewrite-protocol", default=os.getenv("TREE_LLM_REWRITE_PROTOCOL", ""))
    parser.add_argument("--kode-timeout-seconds", type=int, default=120)
    parser.add_argument("--judge-timeout-seconds", type=int, default=45)
    parser.add_argument("--frontier-size", type=int, default=2)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--trajectory-iterations", type=int, default=1, help="ASO iterations per trajectory")
    parser.add_argument("--trajectory-mode", action="store_true", default=True)
    parser.add_argument("--trajectory-focus-top-k", type=int, default=3)
    parser.add_argument("--validate-on-remaining", action="store_true", default=True)
    parser.add_argument("--regression-tolerance", type=float, default=0.0)
    parser.add_argument("--force-restart", action="store_true", help="Ignore existing run_state and start from step 1")
    parser.add_argument("--output-dir", default="demo/outputs/sealqa-trajectory-rollout")
    return parser.parse_args()


def _make_diagnostics(step_logs: list[RolloutStep]) -> dict[str, Any]:
    if not step_logs:
        return {"drop_events": [], "net_delta": 0.0}
    drops = []
    for item in step_logs:
        if item.optimized_overall_score < item.before_overall_score - 1e-9:
            drops.append(
                {
                    "step": item.step,
                    "before": item.before_overall_score,
                    "after": item.optimized_overall_score,
                    "delta": item.optimized_overall_score - item.before_overall_score,
                    "accepted": item.accepted,
                    "reason": item.fallback_reason,
                }
            )
    return {
        "drop_events": drops,
        "net_delta": step_logs[-1].optimized_overall_score - step_logs[0].before_overall_score,
        "accept_rate": sum(1 for item in step_logs if item.accepted) / len(step_logs),
    }


def main() -> None:
    args = _parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize protocol settings before creating the LLM client
    args.actor_protocol = args.actor_protocol or _resolve_protocol(args.actor_model, args.actor_base_url, args.actor_protocol)
    args.judge_protocol = _resolve_protocol(args.judge_model, args.judge_base_url, args.judge_protocol)
    if not args.rewrite_api_key:
        args.rewrite_api_key = args.judge_api_key
    args.rewrite_protocol = args.rewrite_protocol or args.judge_protocol
    args.rewrite_base_url = args.rewrite_base_url or args.judge_base_url

    samples = _load_samples(Path(args.dataset), limit=max(1, args.rollout_size))
    cache_records = _extract_cache_records(samples)
    logger.info("Loaded %d samples", len(samples))

    llm = _build_llm(args)
    program = _build_baseline_program()
    history: list[RolloutStep] = []
    current = program
    start_step = 1
    if args.force_restart:
        run_state_path = args.output_dir / "run_state.json"
        if run_state_path.exists():
            logger.info("force_restart enabled, removing prior state: %s", run_state_path)
            run_state_path.unlink()
    else:
        resume_step, resumed_program, resumed_history = _load_resume_state(args.output_dir)
        if resume_step > 0:
            if resumed_program is not None:
                current = resumed_program
                logger.info("Resuming ASO rollout from step %d using persisted program state", resume_step)
            else:
                logger.warning(
                    "Found run_state at %s but current_program missing; resuming samples from step %d with baseline program",
                    args.output_dir / "run_state.json",
                    resume_step,
                )
            history = resumed_history
            start_step = resume_step

    if start_step <= 0:
        start_step = 1
    if start_step > len(samples):
        logger.info("No remaining steps to run. Start step %d exceeds dataset size %d", start_step, len(samples))
        diagnostics = _make_diagnostics(history)
        summary = {
            "dataset": str(Path(args.dataset).resolve()),
            "steps": len(history),
            "output_dir": str(args.output_dir.resolve()),
            "rollout_samples": [sample.question for sample in samples],
            "steps_log": [step.__dict__ for step in history],
            "diagnostics": diagnostics,
            "config": {
                "actor_model": args.actor_model,
                "judge_model": args.judge_model,
                "rollout_size": args.rollout_size,
                "frontier_size": args.frontier_size,
                "branch_factor": args.branch_factor,
                "trajectory_iterations": args.trajectory_iterations,
                "trajectory_mode": args.trajectory_mode,
                "trajectory_focus_top_k": args.trajectory_focus_top_k,
                "eval_workers": args.eval_workers,
                "aso_workers": args.aso_workers,
                "regression_tolerance": args.regression_tolerance,
                "validate_on_remaining": args.validate_on_remaining,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (args.output_dir / "trajectory_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Done. Final baseline->final drift = %.3f", diagnostics["net_delta"])
        logger.info("Summary written to %s", args.output_dir / "trajectory_summary.json")
        state_path = args.output_dir / "run_state.json"
        if state_path.exists():
            state_path.unlink()
        return

    for step, sample in enumerate(samples[start_step - 1 :], start=start_step):
        logger.info("=== trajectory step %d/%d ===", step, len(samples))
        step_out, next_program = _run_single_trajectory(
            step,
            sample,
            current,
            llm,
            args,
            cache_records=cache_records,
            rollout_samples=samples,
        )
        history.append(step_out)
        current = next_program

        progress_path = args.output_dir / "trajectory_progress.jsonl"
        with progress_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(step_out.__dict__, ensure_ascii=False) + "\n")
        _save_resume_state(args.output_dir, step + 1, current, history)

    state_path = args.output_dir / "run_state.json"
    if state_path.exists():
        state_path.unlink()

    diagnostics = _make_diagnostics(history)
    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "steps": len(history),
        "output_dir": str(args.output_dir.resolve()),
        "rollout_samples": [sample.question for sample in samples],
        "steps_log": [step.__dict__ for step in history],
        "diagnostics": diagnostics,
        "config": {
            "actor_model": args.actor_model,
            "judge_model": args.judge_model,
            "rollout_size": args.rollout_size,
            "frontier_size": args.frontier_size,
            "branch_factor": args.branch_factor,
            "trajectory_iterations": args.trajectory_iterations,
            "trajectory_mode": args.trajectory_mode,
            "trajectory_focus_top_k": args.trajectory_focus_top_k,
            "eval_workers": args.eval_workers,
            "aso_workers": args.aso_workers,
            "regression_tolerance": args.regression_tolerance,
            "validate_on_remaining": args.validate_on_remaining,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    (args.output_dir / "trajectory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Done. Final baseline->final drift = %.3f", diagnostics["net_delta"])
    logger.info("Summary written to %s", args.output_dir / "trajectory_summary.json")


if __name__ == "__main__":
    main()
