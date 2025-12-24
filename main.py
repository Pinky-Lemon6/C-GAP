"""C-GAP main execution script.

Runs the 4-phase pipeline with Atomic Node architecture:
1) Phase I  - Parse raw logs into AtomicNodes (INTENT/EXEC/INFO/COMM)
2) Phase II - Build causal graph (Intra-Step + Inter-Step linking)
3) Phase III- Prune graph nodes (PageRank + semantic relevance)
4) Phase IV - Diagnose root cause from pruned golden context

Assumptions:
- You provide a sample input file under data/raw.
- This script supports either JSON or JSONL.

Recommended input formats:
A) JSON object:
{
  "dataset_source": "sample",
  "session_id": "session_X",
  "error_info": "...",
  "steps": [
    {"step_id": 1, "role": "Instruction", "raw_content": "..."},
    {"step_id": 2, "role": "Action", "raw_content": "..."}
  ]
}

B) JSON array (steps only):
[
  {"step_id": 1, "role": "Instruction", "raw_content": "..."},
  ...
]

C) JSONL (one step per line):
{"step_id": 1, "role": "Instruction", "raw_content": "..."}
...

Environment:
- Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) in .env or env vars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse
import json

import networkx as nx

from src.llm_client import LLMClient
from src.models import StandardLogItem, AtomicNode
from src.pipeline.phase1_parser import LogParser
from src.pipeline.phase2_builder import CausalGraphBuilder
from src.pipeline.phase3_pruner import GraphPruner
from src.pipeline.phase4_diagnoser import RootCauseDiagnoser
from src.utils import load_benchmark_input, normalize_role, save_intermediate_result


def _graph_to_adjacency_edges(g: nx.DiGraph) -> List[Dict[str, Any]]:
    """Convert graph edges to serializable format."""
    edges: List[Dict[str, Any]] = []
    for u, v, data in g.edges(data=True):
        edges.append({"src": str(u), "dst": str(v), **(data or {})})
    return edges


def _graph_to_node_list(g: nx.DiGraph) -> List[Dict[str, Any]]:
    """Convert graph nodes to serializable format."""
    nodes: List[Dict[str, Any]] = []
    for node_id, data in g.nodes(data=True):
        nodes.append({"node_id": str(node_id), **(data or {})})
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the C-GAP 4-phase pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/sample_session.json",
        help="Path to sample input JSON/JSONL under data/raw",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model name for all phases (adjust for your provider)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="hand_crafted",
        choices=["hand_crafted", "algorithm"],
        help="Benchmark dataset type: hand_crafted (Who&When-style) or algorithm (Who&When-style)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-K steps to keep in pruning",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=15,
        help="Window size for inter-step candidate selection",
    )
    parser.add_argument(
        "--phase1-batch-size",
        type=int,
        default=4,
        help="Phase I parallel batch size (smaller is safer for local/small models)",
    )
    parser.add_argument(
        "--phase1-max-workers",
        type=int,
        default=2,
        help="Phase I thread workers for concurrent LLM calls",
    )
    parser.add_argument(
        "--phase1-max-chars",
        type=int,
        default=6000,
        help="Max chars per raw log passed into Phase I (avoid small-model context overflow)",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        default=True,
        help="Use embedding-based candidate selection in Phase II",
    )
    args = parser.parse_args()

    # Load input data
    dataset_source, session_id, question, ground_truth, error_info, raw_steps = load_benchmark_input(
        args.input,
        dataset_type=args.dataset_type,
    )
    print("\n-----------Question:--------------\n", question)
    print("\n-----------Ground Truth:--------------\n", ground_truth)
    print("\n-----------Error Info:--------------\n", error_info)
    if not error_info:
        error_info = "(no error_info provided in input)"

    llm = LLMClient()

    # =========================================================================
    # Phase I: Atomic Node Extraction
    # =========================================================================
    print("\n========== Phase I: Atomic Node Extraction ==========")
    
    phase1 = LogParser(llm=llm, model_name=args.model, use_atomic=True)
    
    # Prepare items for parallel processing
    phase1_items: List[Dict[str, Any]] = []
    for rec in raw_steps:
        phase1_items.append({
            "raw_log": str(rec.get("raw_content", "")),
            "step_id": int(rec.get("step_id")),
            "role": normalize_role(rec.get("role", "")),
        })
    
    # Parse to atomic nodes (parallel)
    atomic_nodes_per_step = phase1.parse_to_atomic_nodes_parallel(
        items=phase1_items,
        batch_size=args.phase1_batch_size,
        max_workers=args.phase1_max_workers,
        max_input_chars=args.phase1_max_chars,
    )
    
    # Build StandardLogItem list with atomic_nodes populated
    structured_steps: List[StandardLogItem] = []
    for item, node_dicts in zip(phase1_items, atomic_nodes_per_step):
        # Convert node dicts to AtomicNode objects
        atomic_nodes = [
            AtomicNode(
                node_id=nd["node_id"],
                step_id=nd["step_id"],
                role=nd["role"],
                type=nd["type"],
                content=nd["content"],
                original_text=nd.get("original_text"),
            )
            for nd in node_dicts
        ]
        
        structured_steps.append(
            StandardLogItem(
                dataset_source=dataset_source,
                session_id=session_id,
                step_id=item["step_id"],
                role=item["role"],
                raw_content=item["raw_log"],
                atomic_nodes=atomic_nodes,
            )
        )
    
    # Count total atomic nodes
    total_atomic_nodes = sum(len(s.atomic_nodes) for s in structured_steps)
    print(f"Phase I complete: {len(structured_steps)} steps -> {total_atomic_nodes} atomic nodes")
    
    # Save Phase I results
    phase1_output = []
    for step in structured_steps:
        step_data = step.model_dump()
        # Ensure atomic_nodes are serializable
        step_data["atomic_nodes"] = [
            node.model_dump() if hasattr(node, "model_dump") else dict(node)
            for node in step.atomic_nodes
        ]
        phase1_output.append(step_data)
    
    save_intermediate_result(
        data=phase1_output,
        phase_name="phase1_atomic",
        session_id=session_id,
    )
    
    # =========================================================================
    # Phase II: Causal Graph Building
    # =========================================================================
    print("\n========== Phase II: Causal Graph Building ==========")
    
    phase2 = CausalGraphBuilder(
        llm=llm,
        model_name=args.model,
        window_size=args.window_size,
        rule_candidates=3,
        semantic_candidates=2,
        confidence_threshold=0.5,
        use_embeddings=args.use_embeddings,
        max_workers=4,
        batch_size=8,
        enable_early_stop=True,
    )
    
    graph = phase2.build(structured_steps)
    stats = phase2.get_stats()
    
    print(f"Phase II complete:")
    print(f"  - Total nodes: {stats.get('total_nodes', 0)}")
    print(f"  - Intra-step edges: {stats.get('intra_step_edges', 0)}")
    print(f"  - Inter-step edges: {stats.get('inter_step_edges', 0)}")
    print(f"  - LLM calls: {stats.get('llm_calls', 0)}")
    print(f"  - Wall time: {stats.get('wall_time_seconds', 0):.2f}s")
    
    # Save Phase II results (graph)
    phase2_output = {
        "nodes": _graph_to_node_list(graph),
        "edges": _graph_to_adjacency_edges(graph),
        "stats": stats,
    }
    
    save_intermediate_result(
        data=phase2_output,
        phase_name="phase2_graph",
        session_id=session_id,
    )

    # =========================================================================
    # Phase III: Graph Pruning (Optional - commented for now)
    # =========================================================================
    # print("\n========== Phase III: Graph Pruning ==========")
    # phase3 = GraphPruner(llm=llm, model_name=args.model)
    # keep_ids = phase3.prune_graph(
    #     graph=graph,
    #     steps=structured_steps,
    #     error_info=error_info,
    #     top_k=args.top_k,
    # )
    # 
    # save_intermediate_result(
    #     data={"keep_step_ids": keep_ids},
    #     phase_name="phase3_pruned",
    #     session_id=session_id,
    # )

    # =========================================================================
    # Phase IV: Root Cause Diagnosis (Optional - commented for now)
    # =========================================================================
    # print("\n========== Phase IV: Root Cause Diagnosis ==========")
    # kept_steps = [s for s in structured_steps if s.step_id in set(keep_ids)]
    # phase4 = RootCauseDiagnoser(llm=llm, model_name=args.model)
    # diagnosis = phase4.diagnose(
    #     keep_steps=kept_steps,
    #     graph=graph,
    #     question=question,
    #     ground_truth=ground_truth,
    #     error_info=error_info,
    # )
    # 
    # print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    # 
    # save_intermediate_result(
    #     data=diagnosis,
    #     phase_name="final_result",
    #     session_id=session_id,
    # )

    print("\n========== Pipeline Complete ==========")


if __name__ == "__main__":
    main()
