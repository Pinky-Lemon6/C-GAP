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
from src.models import StandardLogItem, AtomicNode, TaskContext
from src.pipeline.causal_types import NodeType
from src.pipeline.phase1_parser import LogParser
from src.pipeline.phase2_builder import CausalGraphBuilder
from src.pipeline.phase3_pruner import CausalGraphSlicer
from src.pipeline.phase4_diagnoser import RootCauseDiagnoser, DiagnosisResult
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


def _find_target_node(graph: nx.DiGraph, structured_steps: List[StandardLogItem]) -> str:
    """
    Find the target node for slicing
    
    Strategy:
    1. The last step (Max Step ID).
    2. Within that Step, prioritize the type with the highest information entropy (INFO > EXEC > COMM > INTENT).
    3. If the type is the same, choose the last one in the sequence (Max Index).
    """
    
    
    # Step 1: Identify the last step ID
    if not graph.nodes():
        raise ValueError("Cannot find target node: Graph is empty")
    
    candidates = []
    
    TYPE_WEIGHTS = {
        "INFO": 40,
        "EXEC": 30,
        "COMM": 20,
        "INTENT": 10,
        "UNKNOWN": 0
    }
    
    for idx, (node_id, data) in enumerate(graph.nodes(data=True)):
        step_id = data.get("step_id", -1)
        
        # Normalize type (handle enum or string)
        raw_type = data.get("type", "UNKNOWN")
        node_type = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
        node_type = node_type.upper()
        
        weight = TYPE_WEIGHTS.get(node_type, 0)
        
        candidates.append({
            "id": node_id,
            "step": step_id,
            "idx": idx,      # Original sequence position
            "weight": weight
        })
        
    max_step_id = max(c["step"] for c in candidates)
    last_step_nodes = [c for c in candidates if c["step"] == max_step_id]
    
    last_step_nodes.sort(key=lambda x: (x["weight"], x["idx"]), reverse=True)
    
    target = last_step_nodes[0]
    
    return target["id"]


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
        default=16,
        help="Phase I parallel batch size (smaller is safer for local/small models)",
    )
    parser.add_argument(
        "--phase1-max-workers",
        type=int,
        default=16,
        help="Phase I thread workers for concurrent LLM calls",
    )
    parser.add_argument(
        "--phase1-max-chars",
        type=int,
        default=100000,
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

    taskContext = TaskContext(
        question=question,
        ground_truth=ground_truth,
        error_info=error_info,
    )
    llm = LLMClient()

    # =========================================================================
    # Phase I: Atomic Node Extraction
    # =========================================================================
    print("\n========== Phase I: Atomic Node Extraction ==========")
    
    phase1 = LogParser(llm=llm, model_name=args.model)
    
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
        max_workers=16,
        batch_size=16,
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
    # Phase III: Causal Graph Slicing
    # =========================================================================
    print("\n========== Phase III: Causal Graph Slicing ==========")
    
    phase3 = CausalGraphSlicer(
        max_depth=30,
        max_cost=50.0,
        enable_loop_compression=False,
    )
    
    # Find target node (typically the last ERROR node or last node in graph)
    target_node_id = _find_target_node(graph, structured_steps)
    print(f"Target node for slicing: {target_node_id}")
    
    # Slice the graph
    sliced_nodes = phase3.slice(graph=graph, target_node_id=target_node_id)
    slice_stats = phase3.get_stats()
    
    print(f"Phase III complete:")
    print(f"  - Backward reachable: {slice_stats.get('backward_reachable', 0)}")
    print(f"  - After filtering: {slice_stats.get('after_filtering', 0)}")
    print(f"  - After compression: {slice_stats.get('after_compression', 0)}")
    print(f"  - Final sliced nodes: {len(sliced_nodes)}")
    
    # Save Phase III results
    phase3_output = {
        "sliced_node_ids": [n.node_id for n in sliced_nodes],
        "sliced_nodes": [
            n.model_dump() if hasattr(n, "model_dump") else dict(n)
            for n in sliced_nodes
        ],
        "target_node_id": target_node_id,
        "stats": slice_stats,
    }
    
    save_intermediate_result(
        data=phase3_output,
        phase_name="phase3_sliced",
        session_id=session_id,
    )

    # =========================================================================
    # Phase IV: Root Cause Diagnosis
    # =========================================================================
    print("\n========== Phase IV: Root Cause Diagnosis ==========")
    
    phase4 = RootCauseDiagnoser(llm_client=llm, model_name="deepseek-reasoner")
    
    # Generate golden context for debugging (optional print)
    golden_context = phase4.get_golden_context_only(sliced_nodes, graph)
    print("Golden Context Preview")
    print(golden_context)
    
    # Run diagnosis
    diagnosis_result = phase4.diagnose(
        trace_nodes=sliced_nodes,
        full_graph=graph,
        task_context=taskContext,
    )
    
    print(f"\nDiagnosis Result:")
    print(f"  - Root Cause Step: {diagnosis_result.root_cause_step_id}")
    print(f"  - Culprit: {diagnosis_result.root_cause_culprit}")
    print(f"  - Confidence: {diagnosis_result.confidence_score:.2f}")
    print(f"  - Reasoning: {diagnosis_result.reasoning[:200]}...")
    
    # Save Phase IV results
    phase4_output = {
        "root_cause_step_id": diagnosis_result.root_cause_step_id,
        "root_cause_culprit": diagnosis_result.root_cause_culprit,
        "reasoning": diagnosis_result.reasoning,
        "confidence_score": diagnosis_result.confidence_score,
        "golden_context": diagnosis_result.golden_context,
        "metadata": {
            "trace_length": diagnosis_result.trace_length,
            "skipped_steps": diagnosis_result.skipped_steps,
            "weak_links_count": diagnosis_result.weak_links_count,
            "implicit_refs_count": diagnosis_result.implicit_refs_count,
        },
    }
    
    save_intermediate_result(
        data=phase4_output,
        phase_name="phase4_diagnosis",
        session_id=session_id,
    )

    print("\n========== Pipeline Complete ==========")


if __name__ == "__main__":
    main()
