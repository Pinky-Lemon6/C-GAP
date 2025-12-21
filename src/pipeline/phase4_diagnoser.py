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


DIAGNOSER_SYSTEM_PROMPT_AGENT_D = """You are Agent D (Diagnoser) in a Multi-Agent Failure Attribution System.
Your task is to identify the **ROOT CAUSE** step (The Origin), not just the PROXIMATE CAUSE (The Symptom).

**INPUTS:**
- QUESTION: User's goal.
- GROUND_TRUTH: The correct answer (Use as the "Gold Standard" for logic verification).
- ERROR_INFO: The final failure state (Crash/Wrong Answer).
- GOLDEN_CONTEXT: Structured log steps.

**DIAGNOSIS STRATEGY (CRITICAL):**

1. **Beware the "Last Step Bias"**:
   - If the log ends in a Crash/Traceback (e.g., Step 85), that is usually just the **Symptom**.
   - **DO NOT** blame the crash step unless it was a distinct, isolated coding error.
   - **LOOK UPSTREAM**: The crash likely happened because an EARLIER step provided bad data (e.g., empty list, wrong URL, null variable).

2. **Gap Analysis (Use Ground Truth)**:
   - Compare the System's trace with the GROUND_TRUTH.
   - Find the **FIRST** step where the system drifted away from the path to the Ground Truth.
   - Example: If GT is "CSI: Cyber", find the first search/filter step that **failed to retrieve** or **discarded** "CSI: Cyber". THAT step is the Root Cause.

3. **Distinguish Errors**:
   - **Execution Error**: Tool failed (Network down). -> Blame that step.
   - **Logic Error**: Agent searched "Ted Danson movies" instead of "TV shows". -> Blame the Search step.
   - **Extraction Error**: Information was on the page (Observation), but Agent ignored it. -> Blame the Thought/Action step.

**OUTPUT FORMAT (JSON only):**
{
  "root_cause_step_id": integer,
  "responsible_agent": "string (Role)",
  "reasoning": "string (Explain: 1. What expected info was missed? 2. Why is this earlier step the true cause instead of the final crash?)"
}

**Constraint:**
If the final step is a generic error (timeout, loop, content filter), you MUST find a step **at least 2 steps prior** that triggered this unstable state.
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
        question: str,
        ground_truth: str,
        error_info: str,
    ) -> Dict[str, Any]:
        """Run diagnosis over the pruned steps.
        
        Args:
            keep_steps: List of pruned steps to analyze.
            graph: The causal dependency graph.
            question: The original user query/goal.
            ground_truth: The expected correct answer.
            error_info: Heuristic error signal from execution.
        """

        golden_context = self._build_golden_context(keep_steps=keep_steps, graph=graph)
        
        print("Golden Context for Diagnosis:\n", golden_context)

        user_content = (
            "You will diagnose the root cause of a multi-agent system failure.\n\n"
            "==================================================\n"
            f"QUESTION:\n{question}\n\n"
            "==================================================\n"
            f"GROUND_TRUTH (Expected Correct Answer):\n{ground_truth}\n\n"
            "==================================================\n"
            f"ERROR_INFO (Observed Failure):\n{error_info}\n\n"
            "==================================================\n"
            "GOLDEN_CONTEXT (Structured I-A-O-T format; may include causal tags):\n"
            f"{golden_context}\n"
            "==================================================\n"
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
        """Build golden context string using structured I-A-O-T content."""
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
                    tag = f"\n  (Ref: Caused by Step {incoming_refs_sorted[0]})"
                else:
                    refs = ", ".join(str(x) for x in incoming_refs_sorted)
                    tag = f"\n  (Ref: Caused by Step {refs})"

            # Use structured I-A-O-T content from parsed_iaot
            iaot = step.parsed_iaot
            if hasattr(iaot, "model_dump"):
                iaot_dict = iaot.model_dump()
            elif hasattr(iaot, "dict"):
                iaot_dict = iaot.dict()
            else:
                iaot_dict = dict(iaot) if iaot else {}

            instruction = iaot_dict.get("instruction") or "(none)"
            action = iaot_dict.get("action") or "(none)"
            observation = iaot_dict.get("observation") or "(none)"
            thought = iaot_dict.get("thought") or "(none)"

            step_content = (
                f"Step {step.step_id} | Role={step.role}:\n"
                f"  [Instruction]: {instruction}\n"
                f"  [Action]: {action}\n"
                f"  [Observation]: {observation}\n"
                f"  [Thought]: {thought}{tag}\n"
            )
            parts.append(step_content)

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
