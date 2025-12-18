from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParseConfig:
    """Phase I config."""

    max_field_chars: int = 1200
    summary_chars: int = 220


@dataclass(frozen=True)
class GraphConfig:
    """Phase II config."""

    window_k: int = 5
    min_confidence: float = 0.55
    use_instruction_stack: bool = True
    use_hybrid_retrieval: bool = True


@dataclass(frozen=True)
class PruneConfig:
    """Phase III config."""

    alpha: float = 0.5
    beta: float = 0.5
    top_k: int = 15


@dataclass(frozen=True)
class PipelineConfig:
    parse: ParseConfig = ParseConfig()
    graph: GraphConfig = GraphConfig()
    prune: PruneConfig = PruneConfig()
