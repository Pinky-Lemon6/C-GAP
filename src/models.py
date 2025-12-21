"""Core data models for C-GAP.

This module defines:
- StandardLogItem: normalized representation of a single agent log step
- DependencyGraph: lightweight wrapper around a NetworkX directed graph
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "pydantic is required for src/models.py. Install with: pip install pydantic"
    ) from exc


class IAOT(BaseModel):
    """Instruction-Action-Observation-Thought (IAOT) container."""

    instruction: Optional[str] = None
    action: Optional[str] = None
    observation: Optional[str] = None
    thought: Optional[str] = None


class StandardLogItem(BaseModel):
    """A standardized representation of one log step for downstream pipeline phases."""

    dataset_source: str
    session_id: str
    step_id: int
    role: str
    raw_content: str

    # Pydantic will accept either a dict with the required keys or an IAOT instance.
    parsed_iaot: IAOT = Field(default_factory=IAOT)

    # Used in graph construction (relations, parent/child, causal hints, etc.)
    topology_labels: Dict[str, Any] = Field(default_factory=dict)

    # Used in pruning (scores, keep/drop decisions, confidence, etc.)
    pruning_labels: Dict[str, Any] = Field(default_factory=dict)


class DependencyGraph:
    """Directed dependency graph wrapper using NetworkX internally."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or "dependency_graph"
        try:
            import networkx as nx
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "networkx is required for DependencyGraph. Install with: pip install networkx"
            ) from exc

        self._nx = nx
        self._g = nx.DiGraph(name=self.name)

    @property
    def graph(self):
        """Access the underlying NetworkX DiGraph (advanced use)."""

        return self._g

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self._g.add_node(node_id, **attrs)

    def add_edge(self, src: str, dst: str, **attrs: Any) -> None:
        self._g.add_edge(src, dst, **attrs)

    def nodes(self) -> List[str]:
        return list(self._g.nodes)

    def edges(self) -> List[Tuple[str, str]]:
        return list(self._g.edges)

    def to_edge_list(self) -> List[Dict[str, Any]]:
        """Serialize edges as a list of dicts suitable for JSON."""

        out: List[Dict[str, Any]] = []
        for u, v, data in self._g.edges(data=True):
            out.append({"src": u, "dst": v, **(data or {})})
        return out

    def to_node_link_data(self) -> Dict[str, Any]:
        """Serialize graph in NetworkX node-link format (JSON-friendly)."""

        from networkx.readwrite import json_graph

        return json_graph.node_link_data(self._g)

    @classmethod
    def from_node_link_data(cls, data: Dict[str, Any], name: str | None = None) -> "DependencyGraph":
        from networkx.readwrite import json_graph

        obj = cls(name=name)
        obj._g = json_graph.node_link_graph(data, directed=True)
        return obj
