"""Phase II: Builder

Builds a causal dependency graph among steps using:
- Sliding Window (last K steps)
- Instruction Stack (most recent instruction as a strong candidate)
- LLM verification for each candidate edge

This module defines Agent B (Builder) system prompt as a constant.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import json

import networkx as nx

from src.llm_client import LLMClient
from src.models import StandardLogItem


BUILDER_SYSTEM_PROMPT_AGENT_B = """You are Agent B (Builder) in a Multi-Agent Failure Attribution System (C-GAP).

Task:
- Decide whether a target step causally depends on a source step.

Definition:
- "Causally depends" means the target step would likely be different or impossible without the source step.
- Consider direct dependencies (e.g., using outputs, decisions, tool results) and strong contextual dependencies.
- Ignore weak topical similarity.

Output requirements (STRICT):
- You MUST output a single JSON object and nothing else.
- The JSON object MUST contain exactly these keys:
  - depends: boolean
  - confidence: number between 0 and 1
  - reason: string (brief)
- Do NOT wrap JSON in markdown.
- Do NOT include additional keys.

Guidelines:
- Prefer depends=false when uncertain.
- confidence should reflect your certainty.
"""


INSTRUCTION_ROLES = {"user", "orchestrator", "planner", "manager", "system", "admin"}

ACTION_ROLES = {
    "websurfer",
    "coder",
    "codeinterpreter",
    "toolexecutor",
    "solver",
    "executor",
    "assistant",
    "model",
}


class GraphBuilder:
    """Builds a causal graph for a session using LLM verification."""

    def __init__(
        self,
        llm: LLMClient,
        model_name: str,
        window_k: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.model_name = model_name
        self.window_k = window_k
        self.temperature = temperature
        self.instruction_stack: List[int] = []

    def process_session(self, steps: List[StandardLogItem]) -> nx.DiGraph:
        """Build a causal graph for the given ordered steps."""

        g = nx.DiGraph()

        # Add nodes first (stable regardless of edges)
        for s in steps:
            g.add_node(
                s.step_id,
                session_id=s.session_id,
                role=s.role,
                raw_content=s.raw_content,
                parsed_iaot=(s.parsed_iaot.model_dump() if hasattr(s.parsed_iaot, "model_dump") else dict(s.parsed_iaot)),
            )

        self.instruction_stack = []

        for idx, target in enumerate(steps):
            t_type = _step_type(target)

            if t_type == "instruction":
                self.instruction_stack.append(target.step_id)
                continue

            if t_type != "action":
                # For non-action steps, we skip dependency checks by default.
                continue

            candidate_parents = self._candidate_parents(steps=steps, target_index=idx)

            for source_id in candidate_parents:
                if source_id == target.step_id:
                    continue
                if g.has_edge(source_id, target.step_id):
                    continue

                if self._verify_dependency(source_id=source_id, target_step=target, steps=steps):
                    g.add_edge(source_id, target.step_id, relation="causal")

        return g

    def _candidate_parents(self, steps: List[StandardLogItem], target_index: int) -> List[int]:
        """Collect candidate parent step_ids using stack + sliding window."""

        candidates: List[int] = []
        seen: Set[int] = set()

        # Stack candidate: most recent instruction
        if self.instruction_stack:
            top = self.instruction_stack[-1]
            if top not in seen:
                candidates.append(top)
                seen.add(top)

        # Window candidates: last K steps
        start = max(0, target_index - self.window_k)
        for i in range(start, target_index):
            sid = steps[i].step_id
            if sid not in seen:
                candidates.append(sid)
                seen.add(sid)

        return candidates

    def _verify_dependency(self, source_id: int, target_step: StandardLogItem, steps: List[StandardLogItem]) -> bool:
        """Ask the LLM if target causally depends on source."""

        source_step = _find_step_by_id(steps, source_id)
        if source_step is None:
            return False

        # Prompt required by spec (included verbatim).
        question = f"Does Step {target_step.step_id} causally depend on Step {source_id}?"

        user_content = (
            f"{question}\n\n"
            "You are given two steps from the same session. Decide causal dependency.\n\n"
            f"SOURCE_STEP_ID: {source_step.step_id}\n"
            f"SOURCE_ROLE: {source_step.role}\n"
            f"SOURCE_RAW: {source_step.raw_content}\n\n"
            f"TARGET_STEP_ID: {target_step.step_id}\n"
            f"TARGET_ROLE: {target_step.role}\n"
            f"TARGET_RAW: {target_step.raw_content}\n"
        )

        messages = [
            {"role": "system", "content": BUILDER_SYSTEM_PROMPT_AGENT_B},
            {"role": "user", "content": user_content},
        ]

        text = self.llm.one_step_chat(
            messages=messages,
            model_name=self.model_name,
            json_mode=True,
            temperature=self.temperature,
        )

        verdict = _safe_parse_verdict(text)
        return bool(verdict.get("depends"))


def _safe_parse_verdict(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"depends": False, "confidence": 0.0, "reason": "empty"}

    try:
        obj: Any = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            raise

    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object verdict, got {type(obj)}")

    depends = bool(obj.get("depends"))
    confidence_raw = obj.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    reason = obj.get("reason")
    reason = reason if isinstance(reason, str) else str(reason) if reason is not None else ""

    return {"depends": depends, "confidence": confidence, "reason": reason}


def _step_type(step: StandardLogItem) -> str:
    """Infer step type from `role` or parsed IAOT fields.

    Returns: "instruction" | "action" | "other"
    """

    role = (step.role or "").strip().lower()

    # Step 1: explicit allow-list for instruction providers.
    if role in INSTRUCTION_ROLES:
        return "instruction"

    # Step 2: explicit allow-list for executors.
    if role in ACTION_ROLES:
        return "action"

    # Step 3: fuzzy match for executor-like roles.
    if "agent" in role:
        return "action"

    # Step 4 (Fallback): infer from parsed_iaot if role naming is inconsistent.
    iaot = step.parsed_iaot
    instruction = getattr(iaot, "instruction", None)
    action = getattr(iaot, "action", None)

    if instruction and not action:
        return "instruction"
    if action:
        return "action"

    return "other"


def _find_step_by_id(steps: List[StandardLogItem], step_id: int) -> Optional[StandardLogItem]:
    for s in steps:
        if s.step_id == step_id:
            return s
    return None
