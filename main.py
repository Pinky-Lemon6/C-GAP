"""C-GAP main execution script.

Runs the 4-phase pipeline:
1) Phase I  - Parse raw logs into I-A-O-T
2) Phase II - Build causal graph (sliding window + instruction stack)
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
from src.models import StandardLogItem
from src.pipeline.phase1_parser import LogParser
from src.pipeline.phase2_builder import GraphBuilder
from src.pipeline.phase3_pruner import GraphPruner
from src.pipeline.phase4_diagnoser import RootCauseDiagnoser
from src.utils import load_benchmark_input, normalize_role, save_intermediate_result


def _graph_to_adjacency_edges(g: nx.DiGraph) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    for u, v, data in g.edges(data=True):
        edges.append({"src": int(u), "dst": int(v), **(data or {})})
    return edges


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
        default="gpt-5",
        help="Model name for all phases (adjust for your provider)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="hand_crafted",
        choices=["hand_crafted", "algorithm"],
        help="Benchmark dataset type: hand_crafted (Who&When-style) or algorithm (TODO)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-K steps to keep in pruning",
    )
    parser.add_argument(
        "--window-k",
        type=int,
        default=5,
        help="Sliding window size for graph builder",
    )
    args = parser.parse_args()

    dataset_source, session_id, question, ground_truth, error_info, raw_steps = load_benchmark_input(
        args.input,
        dataset_type=args.dataset_type,
    )
    print("\n-----------Question:--------------\n", question)
    print("\n-----------Ground Truth:--------------\n", ground_truth)
    print("\n-----------Error Info:--------------\n", error_info)
    if not error_info:
        # Minimal fallback: user can embed error text here or in input JSON.
        error_info = "(no error_info provided in input)"

    llm = LLMClient()  # loads from .env / environment variables

    # Phase I
    phase1 = LogParser(llm=llm, model_name=args.model)
    structured_steps: List[StandardLogItem] = []
    for rec in raw_steps:
        step_id = int(rec.get("step_id"))
        role = normalize_role(rec.get("role", ""))
        raw_content = str(rec.get("raw_content", ""))

        iaot = phase1.parse_log_segment(raw_content)

        structured_steps.append(
            StandardLogItem(
                dataset_source=dataset_source,
                session_id=session_id,
                step_id=step_id,
                role=role,
                raw_content=raw_content,
                parsed_iaot=iaot,
                topology_labels={},
                pruning_labels={},
            )
        )

    save_intermediate_result(
        data=[s.model_dump() for s in structured_steps],
        phase_name="phase1",
        session_id=session_id,
    )

    # Phase II
    phase2 = GraphBuilder(llm=llm, model_name=args.model, window_k=args.window_k)
    graph = phase2.process_session(structured_steps)

    save_intermediate_result(
        data=_graph_to_adjacency_edges(graph),
        phase_name="phase2_graph",
        session_id=session_id,
    )

    # Phase III
    phase3 = GraphPruner(llm=llm, model_name=args.model)
    keep_ids = phase3.prune_graph(graph=graph, steps=structured_steps, error_info=error_info, top_k=args.top_k)

    save_intermediate_result(
        data={"keep_step_ids": keep_ids},
        phase_name="phase3_pruned",
        session_id=session_id,
    )

    # Phase IV
    kept_steps = [s for s in structured_steps if s.step_id in set(keep_ids)]
    phase4 = RootCauseDiagnoser(llm=llm, model_name=args.model)
    diagnosis = phase4.diagnose(
        keep_steps=kept_steps,
        graph=graph,
        question=question,
        ground_truth=ground_truth,
        error_info=error_info,
    )

    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))

    save_intermediate_result(
        data=diagnosis,
        phase_name="final_result",
        session_id=session_id,
    )


if __name__ == "__main__":
    main()
