"""Phase III: Pruner

Selects top-K nodes based on:
- PageRank scores on the causal graph
- Semantic relevance to the error information (LLM-scored)

Returns a list of step_ids to keep.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

import networkx as nx

from src.llm_client import LLMClient
from src.models import StandardLogItem


PRUNER_SYSTEM_PROMPT_AGENT_C = """You are Agent C (Pruner) in a Multi-Agent Failure Attribution System (C-GAP).

Task:
- Score how relevant a single log step is to the given error information.

Output requirements (STRICT):
- You MUST output a single JSON object and nothing else.
- The JSON object MUST contain exactly these keys:
  - score: number between 0 and 1
  - reason: string (brief)
- Do NOT wrap JSON in markdown.
- Do NOT include additional keys.

Scoring rubric:
- 1.0: the step directly causes or explains the error.
- 0.7: the step is a strong prerequisite or sets up the failure condition.
- 0.4: weakly related context.
- 0.0: unrelated.

If uncertain, prefer a lower score.
"""


class GraphPruner:
    def __init__(
        self,
        llm: LLMClient,
        model_name: str,
        alpha: float = 0.5,
        beta: float = 0.5,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.model_name = model_name
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def calculate_pagerank(self, graph: nx.DiGraph) -> Dict[int, float]:
        """Calculate PageRank scores for nodes."""

        if graph.number_of_nodes() == 0:
            return {}

        pr = nx.pagerank(graph)
        # networkx returns keys as node IDs (we use int step_id)
        return {int(k): float(v) for k, v in pr.items()}

    def calculate_semantic_score(
        self,
        step: StandardLogItem,
        error_info: str,
        llm_client: Optional[LLMClient] = None,
    ) -> float:
        """Ask LLM to score step relevance to error_info (0.0-1.0)."""

        client = llm_client or self.llm

        user_content = (
            "Score the relevance of the STEP to the ERROR.\n\n"
            f"ERROR_INFO:\n{error_info}\n\n"
            f"STEP_ID: {step.step_id}\n"
            f"STEP_ROLE: {step.role}\n"
            f"STEP_RAW: {step.raw_content}\n"
        )

        messages = [
            {"role": "system", "content": PRUNER_SYSTEM_PROMPT_AGENT_C},
            {"role": "user", "content": user_content},
        ]

        text = client.one_step_chat(
            messages=messages,
            model_name=self.model_name,
            json_mode=True,
            temperature=self.temperature,
        )

        score = _safe_parse_score(text)
        return score

    def prune_graph(
        self,
        graph: nx.DiGraph,
        steps: List[StandardLogItem],
        error_info: str,
        top_k: int = 10,
    ) -> List[int]:
        """Combine PageRank and semantic scores to pick top-K step ids."""

        if top_k <= 0:
            return []

        pagerank = self.calculate_pagerank(graph)

        scores: Dict[int, float] = {}
        for step in steps:
            pr = pagerank.get(step.step_id, 0.0)
            sem = self.calculate_semantic_score(step=step, error_info=error_info)
            scores[step.step_id] = (self.alpha * pr) + (self.beta * sem)

            # Store debug labels on the step (optional but useful downstream)
            try:
                step.pruning_labels["pagerank"] = pr
                step.pruning_labels["semantic"] = sem
                step.pruning_labels["combined"] = scores[step.step_id]
            except Exception:
                pass

        keep = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        keep_step_ids = [sid for sid, _ in keep]
        return keep_step_ids


def _safe_parse_score(text: str) -> float:
    if not text or not text.strip():
        return 0.0

    try:
        obj: Any = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            return 0.0

    if not isinstance(obj, dict):
        return 0.0

    raw = obj.get("score", 0.0)
    try:
        score = float(raw)
    except Exception:
        score = 0.0

    if score != score:  # NaN
        score = 0.0

    return max(0.0, min(1.0, score))
