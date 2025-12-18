from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx

from .agents import (
    AgentAParser,
    AgentBCausalAnalyst,
    AgentCPruner,
    AgentDRootCauseSolver,
    Diagnosis,
    GraphResult,
    ParseResult,
    PruneResult,
)
from .config import PipelineConfig
from .schema import NodeStore


@dataclass
class PipelineArtifacts:
    store: NodeStore
    graph: nx.DiGraph
    prune: PruneResult
    diagnosis: Diagnosis


class CGAPPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.agent_a = AgentAParser(self.config.parse)
        self.agent_b = AgentBCausalAnalyst(self.config.graph)
        self.agent_c = AgentCPruner(self.config.prune)
        self.agent_d = AgentDRootCauseSolver()

    def run(
        self,
        raw_text: str,
        error_info: str,
        error_step_id: Optional[int] = None,
        role: str = "Unknown",
    ) -> PipelineArtifacts:
        parse: ParseResult = self.agent_a.parse(raw_text=raw_text, role=role)
        graph_res: GraphResult = self.agent_b.build_graph(parse.node_store)

        if error_step_id is None:
            error_step_id = parse.node_store.items[-1].id if parse.node_store.items else 0

        prune_res = self.agent_c.prune(
            graph=graph_res.graph,
            store=parse.node_store,
            error_step_id=error_step_id,
            error_info=error_info,
        )

        diagnosis = self.agent_d.diagnose(
            store=parse.node_store,
            graph=graph_res.graph,
            keep_list=prune_res.keep_list,
            scores=prune_res.scores,
            error_info=error_info,
        )

        return PipelineArtifacts(
            store=parse.node_store,
            graph=graph_res.graph,
            prune=prune_res,
            diagnosis=diagnosis,
        )
