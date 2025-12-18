from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from .config import GraphConfig, ParseConfig, PruneConfig
from .schema import IAOTContent, NodeStore, StandardLogItem
from .utils import jaccard_similarity, normalize_whitespace, truncate_middle


@dataclass
class ParseResult:
    node_store: NodeStore


class AgentAParser:
    """Phase I: Parsing & Compression.

    Current implementation is a deterministic baseline:
    - Splits raw log into steps via simple heuristics.
    - Extracts I/A/O/T with tag-like patterns if present.
    - Compresses long fields into a single-sentence summary.

    Later you can replace `parse()` with a LoRA/SFT model.
    """

    def __init__(self, config: ParseConfig) -> None:
        self.config = config

    def parse(self, raw_text: str, role: str = "Unknown") -> ParseResult:
        raw_text = raw_text.replace("\r\n", "\n")
        chunks = [c.strip() for c in raw_text.split("\n\n") if c.strip()]
        items: List[StandardLogItem] = []

        next_id = 1
        for chunk in chunks:
            content = self._extract_iaot(chunk)
            content = self._compress(content)
            items.append(StandardLogItem(id=next_id, role=role, content=content))
            next_id += 1

        return ParseResult(node_store=NodeStore(items=items))

    def _extract_iaot(self, chunk: str) -> IAOTContent:
        text = normalize_whitespace(chunk)

        def find(tag: str) -> Optional[str]:
            tag_lower = tag.lower()
            for prefix in (f"<{tag}>", f"{tag}:", f"{tag_lower}:"):
                if prefix.lower() in text.lower():
                    # naive split; later replaced by robust parser/LLM
                    idx = text.lower().find(prefix.lower())
                    value = text[idx + len(prefix) :].strip()
                    return value or None
            return None

        instruction = find("Instruction")
        action = find("Action")
        observation = find("Observation")
        thought = find("Thought")

        if not any([instruction, action, observation, thought]):
            # fallback: treat as observation
            observation = text

        return IAOTContent(
            instruction=instruction,
            action=action,
            observation=observation,
            thought=thought,
        )

    def _compress(self, content: IAOTContent) -> IAOTContent:
        # Replace too-long action/observation with summary (baseline: truncated middle)
        action = content.action
        obs = content.observation

        summary_parts: List[str] = []
        if action and len(action) > self.config.max_field_chars:
            summary_parts.append(f"Action: {truncate_middle(action, self.config.summary_chars)}")
            content.action = truncate_middle(action, self.config.summary_chars)
        if obs and len(obs) > self.config.max_field_chars:
            summary_parts.append(f"Obs: {truncate_middle(obs, self.config.summary_chars)}")
            content.observation = truncate_middle(obs, self.config.summary_chars)

        if summary_parts:
            content.summary = " | ".join(summary_parts)
        return content


@dataclass
class GraphResult:
    graph: nx.DiGraph


class AgentBCausalAnalyst:
    """Phase II: Sparse causal graph construction.

    Current implementation is a rule+similarity baseline:
    - Candidate recall: local window + instruction stack + (optional) similarity fallback.
    - Scoring: heuristic confidence derived from overlap & presence of instruction/action.

    Later you can replace `score_edge()` with your SFT causal classifier.
    """

    def __init__(self, config: GraphConfig) -> None:
        self.config = config

    def build_graph(self, store: NodeStore) -> GraphResult:
        g = nx.DiGraph()
        for item in store.items:
            g.add_node(item.id)

        instruction_stack: List[int] = []
        by_id = store.by_id()

        for current in store.items:
            if current.content.instruction and self.config.use_instruction_stack:
                instruction_stack.append(current.id)

            candidates: List[int] = []
            # 1) local window
            start = max(1, current.id - self.config.window_k)
            for cid in range(start, current.id):
                if cid in by_id:
                    candidates.append(cid)

            # 2) long-range from stack
            if self.config.use_instruction_stack and current.content.action and instruction_stack:
                top = instruction_stack[-1]
                if top != current.id and top not in candidates:
                    candidates.append(top)

            # 3) hybrid retrieval fallback (top-1 by similarity)
            if (
                self.config.use_hybrid_retrieval
                and current.content.action
                and (not instruction_stack)
                and store.items
            ):
                best_id, best_sim = self._retrieve_best_instruction(store, current.id)
                if best_id is not None and best_id not in candidates and best_sim > 0:
                    candidates.append(best_id)

            # score edges
            for cand_id in candidates:
                cand = by_id[cand_id]
                conf = self.score_edge(cand, current)
                if conf >= self.config.min_confidence:
                    g.add_edge(cand_id, current.id, weight=conf)
                    current.causal_parents[cand_id] = conf

        return GraphResult(graph=g)

    def _retrieve_best_instruction(self, store: NodeStore, current_id: int) -> Tuple[Optional[int], float]:
        cur = next((x for x in store.items if x.id == current_id), None)
        if cur is None:
            return None, 0.0
        query = cur.to_node_text()
        best_id: Optional[int] = None
        best_sim = 0.0
        for it in store.items:
            if it.id >= current_id:
                continue
            if not it.content.instruction:
                continue
            sim = jaccard_similarity(it.to_node_text(), query)
            if sim > best_sim:
                best_sim = sim
                best_id = it.id
        return best_id, best_sim

    def score_edge(self, candidate: StandardLogItem, current: StandardLogItem) -> float:
        # baseline heuristic
        cand_text = candidate.to_node_text()
        cur_text = current.to_node_text()
        sim = jaccard_similarity(cand_text, cur_text)

        bonus = 0.0
        if candidate.content.instruction and current.content.action:
            bonus += 0.15
        if candidate.role == "Orchestrator":
            bonus += 0.05

        conf = min(1.0, sim + bonus)
        return conf


@dataclass
class PruneResult:
    keep_list: List[int]
    scores: Dict[int, float]


class AgentCPruner:
    """Phase III: Hybrid pruning.

    - Structural: PageRank on back-traced subgraph.
    - Semantic: similarity(NodeSummary, ErrorInfo) baseline.

    Later you can replace semantic scoring with your SFT regressor.
    """

    def __init__(self, config: PruneConfig) -> None:
        self.config = config

    def prune(self, graph: nx.DiGraph, store: NodeStore, error_step_id: int, error_info: str) -> PruneResult:
        if error_step_id not in graph:
            # if unknown, fallback to last step
            error_step_id = store.items[-1].id if store.items else 0

        # 1) back-tracing
        reachable = self._reverse_reachable(graph, error_step_id)
        sub = graph.subgraph(reachable).copy()

        # 2) structural: pagerank
        pr = nx.pagerank(sub, weight="weight") if sub.number_of_nodes() else {}

        # 3) semantic: similarity to error
        by_id = store.by_id()
        sem: Dict[int, float] = {}
        for nid in sub.nodes:
            it = by_id.get(nid)
            if it is None:
                continue
            text = it.content.summary or it.to_node_text()
            sem[nid] = jaccard_similarity(text, error_info)

        # 4) combine
        scores: Dict[int, float] = {}
        for nid in sub.nodes:
            scores[nid] = self.config.alpha * pr.get(nid, 0.0) + self.config.beta * sem.get(nid, 0.0)

        keep = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[: self.config.top_k]
        return PruneResult(keep_list=keep, scores=scores)

    def _reverse_reachable(self, graph: nx.DiGraph, target: int) -> set[int]:
        visited = {target}
        stack = [target]
        while stack:
            v = stack.pop()
            for u in graph.predecessors(v):
                if u not in visited:
                    visited.add(u)
                    stack.append(u)
        return visited


@dataclass
class Diagnosis:
    root_cause_step_id: int
    responsible_agent: str
    reason: str


class AgentDRootCauseSolver:
    """Phase IV: Diagnosis.

    Current implementation is a baseline (no external LLM calls):
    - Chooses the highest-score kept node as root cause.
    - Produces a short rationale.

    Later you can swap in a real LLM and keep the same input format.
    """

    def diagnose(
        self,
        store: NodeStore,
        graph: nx.DiGraph,
        keep_list: List[int],
        scores: Dict[int, float],
        error_info: str,
    ) -> Diagnosis:
        if not keep_list:
            return Diagnosis(root_cause_step_id=0, responsible_agent="Unknown", reason="Empty keep_list")

        best = max(keep_list, key=lambda k: scores.get(k, 0.0))
        by_id = store.by_id()
        it = by_id.get(best)
        role = it.role if it else "Unknown"

        reason = (
            f"Selected by highest hybrid score among Top-K. "
            f"Error='{truncate_middle(error_info, 200)}'. "
            f"Step={best}, role={role}."
        )
        return Diagnosis(root_cause_step_id=best, responsible_agent=role, reason=reason)
