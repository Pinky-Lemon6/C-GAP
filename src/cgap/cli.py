from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import CGAPPipeline
from .utils import read_text, write_json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cgap", description="C-GAP research pipeline (baseline implementation).")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run Parse->Graph->Prune->Diagnose on a raw log file")
    run.add_argument("--log", type=Path, required=True, help="Path to raw log text file")
    run.add_argument("--error", type=str, required=True, help="Final error info (string)")
    run.add_argument("--error-step", type=int, default=None, help="Step id where error happened (optional)")
    run.add_argument("--role", type=str, default="Unknown", help="Default role for parsed steps")
    run.add_argument("--out", type=Path, default=Path("data/processed/run_artifacts.json"), help="Output JSON")

    demo = sub.add_parser("demo", help="Run a minimal built-in demo")
    demo.add_argument("--out", type=Path, default=Path("data/processed/demo_artifacts.json"), help="Output JSON")

    return p


def cmd_run(args: argparse.Namespace) -> int:
    raw = read_text(args.log)
    pipeline = CGAPPipeline()
    artifacts = pipeline.run(raw_text=raw, error_info=args.error, error_step_id=args.error_step, role=args.role)

    out = {
        "diagnosis": artifacts.diagnosis.__dict__,
        "keep_list": artifacts.prune.keep_list,
        "scores": artifacts.prune.scores,
        "nodes": [it.model_dump() for it in artifacts.store.items],
        "edges": [
            {"src": u, "dst": v, "weight": d.get("weight", None)}
            for u, v, d in artifacts.graph.edges(data=True)
        ],
    }
    write_json(args.out, out)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    raw = """
<Instruction> Find Ted Danson's series.

<Action> Open IMDb.
<Observation> Page loaded, search bar visible.

<Action> Search for Ted Danson.
<Observation> Results show 'Ted Danson' profile.

<Action> Click Credits.
<Observation> Error: element not found: #credits
""".strip()

    pipeline = CGAPPipeline()
    artifacts = pipeline.run(raw_text=raw, error_info="element not found: #credits", error_step_id=4, role="WebSurfer")

    out = {
        "diagnosis": artifacts.diagnosis.__dict__,
        "keep_list": artifacts.prune.keep_list,
        "scores": artifacts.prune.scores,
        "nodes": [it.model_dump() for it in artifacts.store.items],
        "edges": [
            {"src": u, "dst": v, "weight": d.get("weight", None)}
            for u, v, d in artifacts.graph.edges(data=True)
        ],
    }
    write_json(args.out, out)
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        raise SystemExit(cmd_run(args))
    if args.cmd == "demo":
        raise SystemExit(cmd_demo(args))

    raise SystemExit(2)
