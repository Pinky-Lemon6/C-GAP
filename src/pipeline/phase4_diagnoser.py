"""Phase IV: Diagnoser (Judge)

Given pruned "Golden Context" steps and the causal graph, ask the LLM to:
- Identify root cause step id
- Identify responsible agent
- Provide reasoning

Crucial requirement:
- For each kept step, if it has an incoming edge from another kept step,
  append a tag like: (Ref: Caused by Step X)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

import networkx as nx

from src.llm_client import LLMClient
from src.models import StandardLogItem


DIAGNOSER_SYSTEM_PROMPT_AGENT_D = """You are Agent D (Diagnoser/Judge) in a Multi-Agent Failure Attribution System (C-GAP).

Task:
- Given ERROR_INFO, a set of pruned log steps (Golden Context), and causal reference tags,
  identify the most plausible root-cause step.

Output requirements (STRICT):
- You MUST output a single JSON object and nothing else.
- The JSON object MUST contain exactly these keys:
  - root_cause_step_id: integer
  - responsible_agent: string
  - reasoning: string
- Do NOT wrap JSON in markdown.
- Do NOT include additional keys.

Guidelines:
- Prefer a single root-cause step.
- Use the (Ref: Caused by Step X) tags as causal hints.
- If uncertain, choose the step that best explains the error.
"""


class RootCauseDiagnoser:
    def __init__(self, llm: LLMClient, model_name: str, temperature: float = 0.0) -> None:
        self.llm = llm
        self.model_name = model_name
        self.temperature = temperature

    def diagnose(
        self,
        keep_steps: List[StandardLogItem],
        graph: nx.DiGraph,
        error_info: str,
    ) -> Dict[str, Any]:
        """Run diagnosis over the pruned steps."""

        golden_context = self._build_golden_context(keep_steps=keep_steps, graph=graph)

        user_content = (
            "You will diagnose the root cause of a failure.\n\n"
            f"ERROR_INFO:\n{error_info}\n\n"
            "GOLDEN_CONTEXT (chronological; may include causal tags):\n"
            f"{golden_context}\n"
        )

        messages = [
            {"role": "system", "content": DIAGNOSER_SYSTEM_PROMPT_AGENT_D},
            {"role": "user", "content": user_content},
        ]

        text = self.llm.one_step_chat(
            messages=messages,
            model_name=self.model_name,
            json_mode=True,
            temperature=self.temperature,
        )

        return _safe_parse_diagnosis(text)

    def _build_golden_context(self, keep_steps: List[StandardLogItem], graph: nx.DiGraph) -> str:
        kept_ids = {s.step_id for s in keep_steps}
        ordered = sorted(keep_steps, key=lambda s: s.step_id)

        parts: List[str] = []
        for step in ordered:
            incoming_refs = [
                int(src)
                for src in graph.predecessors(step.step_id)
                if int(src) in kept_ids
            ]

            tag = ""
            if incoming_refs:
                incoming_refs_sorted = sorted(set(incoming_refs))
                # Crucial tag format requirement
                if len(incoming_refs_sorted) == 1:
                    tag = f" (Ref: Caused by Step {incoming_refs_sorted[0]})"
                else:
                    refs = ", ".join(str(x) for x in incoming_refs_sorted)
                    tag = f" (Ref: Caused by Step {refs})"

            parts.append(
                f"Step {step.step_id} | Role={step.role}:\n{step.raw_content}{tag}\n"
            )

        return "\n".join(parts).strip()


def _safe_parse_diagnosis(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"root_cause_step_id": -1, "responsible_agent": "", "reasoning": "empty"}

    try:
        obj: Any = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            return {"root_cause_step_id": -1, "responsible_agent": "", "reasoning": "invalid_json"}

    if not isinstance(obj, dict):
        return {"root_cause_step_id": -1, "responsible_agent": "", "reasoning": "invalid_type"}

    rid = obj.get("root_cause_step_id", -1)
    try:
        root_id = int(rid)
    except Exception:
        root_id = -1

    agent = obj.get("responsible_agent")
    agent = agent if isinstance(agent, str) else str(agent) if agent is not None else ""

    reasoning = obj.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, str) else str(reasoning) if reasoning is not None else ""

    return {"root_cause_step_id": root_id, "responsible_agent": agent, "reasoning": reasoning}
