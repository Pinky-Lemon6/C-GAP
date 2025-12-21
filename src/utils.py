"""Utilities for C-GAP.

Includes:
- JSONL load/save helpers
- Intermediate result dumping for debugging each pipeline phase
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import json
from datetime import datetime


JsonValue = Any
JsonDict = Dict[str, JsonValue]


def load_jsonl(path: str | Path) -> List[JsonDict]:
    """Load a JSONL file into a list of dicts."""

    p = Path(path)
    items: List[JsonDict] = []
    if not p.exists():
        raise FileNotFoundError(str(p))

    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {p}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object per line (dict) in {p}, got {type(obj)}")
            items.append(obj)
    return items


def save_jsonl(path: str | Path, records: Iterable[JsonDict]) -> None:
    """Save an iterable of dict records to JSONL."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError(f"Each record must be a dict, got {type(rec)}")
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def save_intermediate_result(
    data: Any,
    phase_name: str,
    session_id: str,
    root_dir: str | Path = "data/intermediate",
) -> Path:
    """Save debug/intermediate data to data/intermediate/{session_id}/{phase_name}_{timestamp}.json.

    Returns the written file path.
    """

    if not phase_name:
        raise ValueError("phase_name must be non-empty")
    if not session_id:
        raise ValueError("session_id must be non-empty")


    out_dir = Path(root_dir) / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{phase_name}_{timestamp}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def normalize_role(raw_role: Any) -> str:
    """Normalize role strings from benchmarks into a stable form.

    Examples:
    - "Orchestrator (thought)" -> "orchestrator"
    - "Orchestrator (-> WebSurfer)" -> "orchestrator"
    - "human" -> "user"
    """

    role = "" if raw_role is None else str(raw_role)
    role = role.strip()
    if not role:
        return ""

    # Strip annotation suffixes like "(thought)" or "(-> WebSurfer)".
    if "(" in role:
        role = role.split("(", 1)[0].strip()

    role_lc = role.lower()
    if role_lc in {"human", "user"}:
        return "user"
    if role_lc in {"assistant", "model"}:
        return role_lc

    return role_lc


def load_benchmark_input(
    path: str | Path,
    dataset_type: str = "hand_crafted",
) -> Tuple[str, str, str, List[JsonDict]]:
    """Load input data for different benchmark formats.

    Args:
        path: input file path
        dataset_type: "hand_crafted" | "algorithm" (algorithm not implemented yet)

    Returns:
        dataset_source, session_id, error_info, raw_steps

    Notes:
        - For hand_crafted Who&When format, this parses {"history": [...]} into steps.
        - For other JSON/JSONL formats (already supported by main.py previously), this
          falls back to reading step objects directly.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    dataset_type = (dataset_type or "").strip().lower()
    if dataset_type not in {"hand_crafted", "algorithm"}:
        raise ValueError("dataset_type must be 'hand_crafted' or 'algorithm'")

    if dataset_type == "algorithm":
        # Who&When Algorithm-Generated format: JSON object with 'history' list.
        # Each history item may contain: {content, role, name}
        if p.suffix.lower() == ".jsonl":
            steps = load_jsonl(p)
            return "algorithm", p.stem, "", steps

        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("history"), list):
            raise ValueError("algorithm input must be a JSON object with a 'history' list")

        history = data.get("history") or []
        question = str(data.get("question", ""))
        ground_truth = str(data.get("ground_truth", ""))
        mistake_agent = str(data.get("mistake_agent", ""))
        mistake_step = str(data.get("mistake_step", ""))
        mistake_reason = str(data.get("mistake_reason", ""))

        error_parts: List[str] = []
        if question:
            error_parts.append(f"Question: {question}")
        if ground_truth:
            error_parts.append(f"GroundTruth: {ground_truth}")
        if mistake_reason:
            error_parts.append(f"MistakeReason: {mistake_reason}")
        if mistake_agent or mistake_step:
            error_parts.append(f"MistakeAgent: {mistake_agent} | MistakeStep: {mistake_step}")
        error_info = "\n".join([x for x in error_parts if x]).strip()

        session_id = str(data.get("question_ID") or p.stem)
        dataset_source = "algorithm"

        raw_steps: List[JsonDict] = []
        for i, item in enumerate(history, start=0):
            if not isinstance(item, dict):
                continue
            base_role = normalize_role(item.get("role"))
            name = item.get("name")
            name_str = str(name).strip() if name is not None else ""
            content = str(item.get("content", ""))

            # Keep the benchmark's base role for stack/action logic (user/assistant/system).
            # Preserve the agent name inside raw_content to help Phase I parsing.
            if name_str:
                raw_content = f"SPEAKER_NAME: {name_str}\n{content}"
            else:
                raw_content = content

            raw_steps.append({"step_id": i, "role": base_role, "raw_content": raw_content})

        return dataset_source, session_id, error_info, raw_steps

    # --- hand_crafted ---
    # Supports both:
    # 1) JSONL of step objects
    # 2) JSON object with steps
    # 3) Who&When Hand-Crafted JSON object with 'history'

    if p.suffix.lower() == ".jsonl":
        steps = load_jsonl(p)
        return "hand_crafted", p.stem, "", steps

    data = json.loads(p.read_text(encoding="utf-8"))

    # Who&When Hand-Crafted format: { history: [{role, content}, ...], question, ground_truth, ... }
    if isinstance(data, dict) and isinstance(data.get("history"), list):
        history = data.get("history") or []
        question = str(data.get("question", ""))
        ground_truth = str(data.get("ground_truth", ""))
        mistake_agent = str(data.get("mistake_agent", ""))
        mistake_step = str(data.get("mistake_step", ""))
        mistake_reason = str(data.get("mistake_reason", ""))

        error_parts: List[str] = []
        if question:
            error_parts.append(f"Question: {question}")
        if ground_truth:
            error_parts.append(f"GroundTruth: {ground_truth}")
        if mistake_reason:
            error_parts.append(f"MistakeReason: {mistake_reason}")
        if mistake_agent or mistake_step:
            error_parts.append(f"MistakeAgent: {mistake_agent} | MistakeStep: {mistake_step}")

        error_info = "\n".join([p for p in error_parts if p]).strip()

        session_id = str(data.get("question_ID") or p.stem)
        dataset_source = "hand_crafted"

        raw_steps: List[JsonDict] = []
        for i, item in enumerate(history, start=0):
            if not isinstance(item, dict):
                continue
            raw_steps.append(
                {
                    "step_id": i,
                    "role": normalize_role(item.get("role")),
                    "raw_content": str(item.get("content", "")),
                }
            )

        return dataset_source, session_id, error_info, raw_steps

    # Fallback formats: list of step dicts or dict with steps
    if isinstance(data, list):
        return "hand_crafted", p.stem, "", data

    if isinstance(data, dict):
        dataset_source = str(data.get("dataset_source", "hand_crafted"))
        session_id = str(data.get("session_id", p.stem))
        error_info = str(data.get("error_info", ""))
        steps = data.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("Input JSON must contain 'steps' as a list")
        return dataset_source, session_id, error_info, steps

    raise ValueError("Unsupported input JSON format")
