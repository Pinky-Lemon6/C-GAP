"""
Phase IV: Root Cause Diagnoser
==============================
Formats the sliced causal graph into "Golden Context" and calls LLM for diagnosis. 

Two responsibilities:
1. Generate structured prompt from pruned trace nodes (Golden Context)
2. Invoke LLM and parse structured diagnosis output

Key features:
- Task Context: Include original question, expected answer, and error info
- Linearization: Sort nodes by step_id
- Gap Summaries: Insert summaries for skipped steps
- Edge Annotation: Show implicit context and weak links explicitly
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from src.models import AtomicNode, TaskContext
from src.pipeline.causal_types import NodeType

if TYPE_CHECKING:
    from src.llm_client import LLMClient





@dataclass
class DiagnosisResult: 
    """Structured output from root cause diagnosis."""
    
    root_cause_step_id: str
    root_cause_culprit: str
    reasoning: str
    confidence_score: float
    raw_response: str = ""
    golden_context: str = ""
    
    # Additional metadata
    trace_length: int = 0
    skipped_steps: int = 0
    weak_links_count: int = 0
    implicit_refs_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_cause_step_id": self.root_cause_step_id,
            "root_cause_culprit": self. root_cause_culprit,
            "reasoning": self. reasoning,
            "confidence_score": self.confidence_score,
            "metadata": {
                "trace_length": self.trace_length,
                "skipped_steps": self. skipped_steps,
                "weak_links_count": self.weak_links_count,
                "implicit_refs_count": self.implicit_refs_count,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) analyzing multi-agent system execution traces. 

Your task:  Identify the ROOT CAUSE of the failure - the earliest step where something went wrong that led to the incorrect result.

## Trace Annotations

Step markers: 
- [ERROR]:  Step that encountered an error or produced incorrect output
- [COMPRESSED]: Multiple similar steps were compressed into one

Node types:
- INTENT: Agent's thought, plan, or decision
- EXEC: Tool call, code execution, API call
- INFO:  Observation, result, error, feedback
- COMM:  Message to/from other agents or users

Dependency annotations:
- [Implicit Context]: Data dependencies from other steps, with causal relationship types: 
  - INSTRUCTION: This step executes based on instructions from source step
  - DATA:  This step uses data/results produced by source step
  - STATE: This step depends on state established by source step
- [WEAK LINK]: Only temporal sequence exists, no verified causal relationship (be skeptical of these)

## Analysis Guidelines

1. Compare the EXPECTED ANSWER with what the system produced to understand the error
2. The root cause is often NOT the final error step, but an EARLIER step where a wrong decision was made
3. Look for: incorrect assumptions, wrong data interpretation, missed validations, premature conclusions
4. Pay attention to INTENT nodes - they show the agent's reasoning and decisions
5. Follow [Implicit Context] annotations to trace the causal chain
6. Be skeptical of [WEAK LINK] connections - they may not represent true causality

## Common Root Cause Patterns

- Agent misinterpreted user request (early INTENT step)
- Agent accepted incorrect/incomplete data without verification
- Agent made wrong decision based on observations
- Agent failed to validate results before concluding

## Output Format

Output MUST be valid JSON: 
{
  "root_cause_step_id": <integer step number>,
  "root_cause_culprit": "Agent or component name",
  "reasoning": "Detailed explanation of what went wrong and why",
  "confidence_score": 0.0 to 1.0
}"""


USER_PROMPT_TEMPLATE = """## Task Information

**Original Question:**
{question}

**Expected Answer:**
{ground_truth}

**Error Description:**
{error_info}

---

{golden_context}

---

## Your Analysis

Based on the execution trace above, identify the root cause of the incorrect result. 

1. **root_cause_step_id**: The step number where the fundamental error occurred (integer)
2. **root_cause_culprit**: The agent/component responsible for the error
3. **reasoning**:  Step-by-step explanation - what should have happened vs what actually happened
4. **confidence_score**:  Your confidence in this diagnosis (0.0-1.0)

Respond with ONLY the JSON object, no additional text."""


# ─────────────────────────────────────────────────────────────────────────────
# Main Diagnoser Class
# ─────────────────────────────────────────────────────────────────────────────
class RootCauseDiagnoser:
    """
    Phase IV:  Formats sliced graph into Golden Context and invokes LLM for diagnosis.
    """
    
    INDENT = "        "  # 8 spaces
    
    def __init__(
        self,
        llm_client:  "LLMClient",
        model_name: str = "gpt-4",
        max_context_tokens: int = 65536,
        verbose: bool = False,
    ):
        """
        Initialize the diagnoser. 
        
        Args:
            llm_client: LLM client instance for API calls
            model_name: Model to use for diagnosis
            max_context_tokens: Maximum tokens for context
            verbose: Whether to print debug information
        """
        self.llm = llm_client
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.verbose = verbose
        
        self._stats:  Dict[str, int] = {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Node Extraction from Graph
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_node_from_graph(
        self,
        graph: nx.DiGraph,
        node_id: str
    ) -> Optional[AtomicNode]:
        """Extract AtomicNode from graph node data."""
        if node_id not in graph: 
            return None
        
        data = graph. nodes[node_id]
        
        if "node" in data and isinstance(data["node"], AtomicNode):
            return data["node"]
        
        return AtomicNode(
            node_id=node_id,
            step_id=data. get("step_id", 0),
            role=data.get("role", "unknown"),
            type=data.get("type", "INFO"),
            content=data.get("content", ""),
            original_text=data.get("original_text"),
        )
    
    def _get_all_nodes_from_graph(
        self,
        graph: nx.DiGraph
    ) -> Dict[str, AtomicNode]:
        """Extract all nodes from graph as AtomicNode objects."""
        all_nodes:  Dict[str, AtomicNode] = {}
        
        for node_id in graph.nodes():
            node = self._extract_node_from_graph(graph, node_id)
            if node: 
                all_nodes[node_id] = node
        
        return all_nodes
    
    # ─────────────────────────────────────────────────────────────────────────
    # Golden Context Generation
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_golden_context(
        self,
        trace_nodes: List[AtomicNode],
        full_graph: nx.DiGraph,
    ) -> str:
        """
        Format sliced nodes into a structured prompt string.
        
        Format per step:
        [Step X] (role)
                [TYPE]:  content
                [TYPE]: content
                [Implicit Context]:  Step A (INSTRUCTION), Step B (DATA)
                [WEAK LINK]: Step C
        """
        if not trace_nodes: 
            return "[Empty trace - no nodes to analyze]"
        
        # Reset stats
        self._stats = {
            "trace_length": len(trace_nodes),
            "skipped_steps": 0,
            "weak_links_count":  0,
            "implicit_refs_count": 0,
        }
        
        # Sort by step_id, then node_id
        sorted_nodes = sorted(trace_nodes, key=lambda n: (n.step_id, n. node_id))
        
        # Build lookup structures
        trace_node_ids = {n.node_id for n in sorted_nodes}
        trace_step_ids = {n.step_id for n in sorted_nodes}
        
        # Get all nodes from full graph for gap analysis
        all_graph_nodes = self._get_all_nodes_from_graph(full_graph)
        
        # Group trace nodes by step
        nodes_by_step:  Dict[int, List[AtomicNode]] = defaultdict(list)
        for n in sorted_nodes: 
            nodes_by_step[n. step_id].append(n)
        
        lines:  List[str] = []
        lines.append("=" * 60)
        lines.append("EXECUTION TRACE")
        lines.append("=" * 60)
        lines.append("")
        
        unique_steps = sorted(nodes_by_step. keys())
        prev_step_id:  Optional[int] = None
        
        for step_id in unique_steps: 
            step_nodes = nodes_by_step[step_id]
            
            # Gap Summary for skipped steps
            if prev_step_id is not None and step_id > prev_step_id + 1:
                gap_summary = self._summarize_gap(
                    start_step=prev_step_id + 1,
                    end_step=step_id - 1,
                    all_nodes=all_graph_nodes,
                    included_step_ids=trace_step_ids,
                )
                if gap_summary: 
                    lines.append(gap_summary)
                    lines.append("")
            
            # Format the entire step block
            step_block = self._format_step_block(
                step_id=step_id,
                step_nodes=step_nodes,
                full_graph=full_graph,
                all_nodes=all_graph_nodes,
                trace_node_ids=trace_node_ids,
            )
            lines.append(step_block)
            lines.append("")
            
            prev_step_id = step_id
        
        lines.append("=" * 60)
        lines.append("END OF TRACE")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _format_step_block(
        self,
        step_id:  int,
        step_nodes: List[AtomicNode],
        full_graph: nx.DiGraph,
        all_nodes: Dict[str, AtomicNode],
        trace_node_ids: Set[str],
    ) -> str:
        """
        Format a single step with all its nodes.
        
        Output format:
        [Step X] (role)
                [TYPE]: content
                [TYPE]: content
                [Implicit Context]:  Step A (INSTRUCTION), Step B (DATA)
                [WEAK LINK]: Step C
        """
        if not step_nodes: 
            return ""
        
        # Get role from first node
        role = step_nodes[0].role
        if role == "unknown":
            role = "System"
        
        # Check if any node in this step is an error
        has_error = any(self._is_error_node(n) for n in step_nodes)
        has_compressed = any(n.type == "COMPRESSED" for n in step_nodes)
        
        # Build step header
        prefix = ""
        if has_error:
            prefix = "[ERROR] "
        elif has_compressed:
            prefix = "[COMPRESSED] "
        
        lines:  List[str] = []
        lines. append(f"{prefix}[Step {step_id}] ({role})")
        
        # Format each node's content
        for node in step_nodes: 
            node_line = self._format_node_content(node)
            lines.append(node_line)
        
        # Get edge annotations (Implicit Context and WEAK LINK)
        implicit_context, weak_links = self._get_edge_annotations(
            step_nodes=step_nodes,
            full_graph=full_graph,
            all_nodes=all_nodes,
        )
        
        if implicit_context: 
            lines.append(f"{self.INDENT}[Implicit Context]: {implicit_context}")
        
        if weak_links:
            lines.append(f"{self.INDENT}[WEAK LINK]: {weak_links}")
        
        return "\n". join(lines)
    
    def _format_node_content(self, node:  AtomicNode) -> str:
        """Format a single node's content with proper indentation."""
        if isinstance(node.type, NodeType):
            type_str = node.type. value
        else:
            type_str = str(node.type)
        
        content = node.content or ""
        # if len(content) > self.max_content_length:
        #     content = content[:self.max_content_length - 3] + "..."
        content = " ". join(content.split())
        
        return f"{self. INDENT}[{type_str}]: {content}"
    
    def _is_error_node(self, node: AtomicNode) -> bool:
        """Check if a node represents an error."""
        type_str = node. type. value if isinstance(node.type, NodeType) else str(node.type)
        if type_str. upper() == "ERROR":
            return True
        
        error_keywords = [
            "error", "exception", "failed", "failure", "traceback",
            "fatal", "crash", "refused", "denied", "timeout",
        ]
        content_lower = (node.content or "").lower()
        return any(kw in content_lower for kw in error_keywords)
    
    def _summarize_gap(
        self,
        start_step: int,
        end_step: int,
        all_nodes:  Dict[str, AtomicNode],
        included_step_ids: Set[int],
    ) -> str:
        """Generate a summary for skipped steps."""
        skipped_nodes = [
            n for n in all_nodes.values()
            if start_step <= n. step_id <= end_step
            and n.step_id not in included_step_ids
        ]
        
        gap_size = end_step - start_step + 1
        self._stats["skipped_steps"] += gap_size
        
        if not skipped_nodes:
            if gap_size == 1:
                return f"{self.INDENT}...  [Skipped Step {start_step}] ..."
            return f"{self. INDENT}... [Skipped Steps {start_step}-{end_step}] ..."
        
        type_counts:  Counter = Counter()
        for n in skipped_nodes:
            type_str = n.type. value if isinstance(n.type, NodeType) else str(n.type)
            type_counts[type_str] += 1
        
        type_summary = ", ".join(f"{count} {t}" for t, count in type_counts. most_common())
        
        if gap_size == 1:
            return f"{self. INDENT}... [Skipped Step {start_step}:  {type_summary}] ..."
        return f"{self. INDENT}... [Skipped Steps {start_step}-{end_step}: {type_summary}] ..."
    
    def _get_edge_annotations(
        self,
        step_nodes: List[AtomicNode],
        full_graph: nx.DiGraph,
        all_nodes: Dict[str, AtomicNode],
    ) -> Tuple[str, str]:
        """
        Get edge annotations for all nodes in a step.
        
        Returns:
            (implicit_context_str, weak_links_str)
            
        implicit_context_str format: "Step A (INSTRUCTION), Step B (DATA)"
        weak_links_str format: "Step C, Step D"
        """
        current_step_id = step_nodes[0].step_id if step_nodes else -1
        
        # Collect implicit context:  step_id -> causal_type
        implicit_refs:  Dict[int, str] = {}
        # Collect weak links:  set of step_ids
        weak_link_steps: Set[int] = set()
        
        for node in step_nodes: 
            if node.node_id not in full_graph: 
                continue
            
            for src_id in full_graph.predecessors(node.node_id):
                if not full_graph.has_edge(src_id, node.node_id):
                    continue
                
                edge_data = full_graph. edges[src_id, node.node_id]
                src_node = all_nodes. get(src_id)
                
                if not src_node or src_node.step_id == current_step_id:
                    continue
                
                src_step = src_node.step_id
                
                # Check if this is a weak link (fallback edge)
                is_fallback = edge_data.get("is_fallback", False)
                
                if is_fallback:
                    weak_link_steps. add(src_step)
                    self._stats["weak_links_count"] += 1
                else:
                    # Get causal type for implicit context
                    causal_type = edge_data.get("causal_type", "")
                    
                    # Normalize causal type
                    if not causal_type or causal_type == "SEQUENTIAL":
                        layer = edge_data. get("layer", "")
                        if layer == "PRIMARY":
                            causal_type = "INSTRUCTION"
                        elif layer == "SECONDARY":
                            causal_type = "DATA"
                        else:
                            causal_type = "UNKNOWN"
                    
                    # Only keep one causal type per source step (prefer more specific)
                    if src_step not in implicit_refs:
                        implicit_refs[src_step] = causal_type
                        self._stats["implicit_refs_count"] += 1
                    else:
                        # Prefer INSTRUCTION > DATA > STATE > UNKNOWN
                        priority = {"INSTRUCTION": 0, "DATA": 1, "STATE":  2, "UNKNOWN": 3}
                        existing_priority = priority. get(implicit_refs[src_step], 99)
                        new_priority = priority. get(causal_type, 99)
                        if new_priority < existing_priority: 
                            implicit_refs[src_step] = causal_type
        
        # Format implicit context string
        implicit_context_str = ""
        if implicit_refs: 
            sorted_refs = sorted(implicit_refs.items(), key=lambda x: x[0])
            parts = [f"Step {step} ({ctype})" for step, ctype in sorted_refs]
            implicit_context_str = ", ".join(parts)
        
        # Format weak links string
        weak_links_str = ""
        if weak_link_steps:
            # Remove any steps already in implicit_refs (prefer causal over weak)
            weak_only = weak_link_steps - set(implicit_refs. keys())
            if weak_only: 
                sorted_weak = sorted(weak_only)
                weak_links_str = ", ".join(f"Step {s}" for s in sorted_weak)
        
        return implicit_context_str, weak_links_str
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Diagnosis Method
    # ─────────────────────────────────────────────────────────────────────────
    def diagnose(
        self,
        trace_nodes: List[AtomicNode],
        full_graph: nx.DiGraph,
        task_context: TaskContext,
    ) -> DiagnosisResult:
        """
        Perform root cause diagnosis on the pruned trace.
        
        Args:
            trace_nodes:  Pruned nodes from Phase III
            full_graph: Complete causal graph for edge analysis
            task_context: Task information including question, ground_truth, and error_info
            
        Returns: 
            DiagnosisResult with root cause analysis
        """
        # Step 1: Generate Golden Context (trace only)
        golden_context = self._generate_golden_context(trace_nodes, full_graph)
        
        # Step 2: Build user prompt with task context
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=task_context. question or "[No question provided]",
            ground_truth=task_context.ground_truth or "[No expected answer provided]",
            error_info=task_context.error_info or "The system produced an incorrect result.",
            golden_context=golden_context,
        )
        
        messages = [
            {"role": "system", "content":  SYSTEM_PROMPT},
            {"role":  "user", "content": user_prompt},
        ]
        
        total_chars = len(SYSTEM_PROMPT) + len(user_prompt)
        if self.verbose:
            print(f"[Phase IV] Prompt chars: {total_chars}, ~{total_chars // 4} tokens")
        
        # Step 3: Call LLM
        raw_response = self._call_llm(messages)
        
        # Step 4: Parse response
        diagnosis = self._parse_response(raw_response)
        
        
        return DiagnosisResult(
            root_cause_step_id=str(diagnosis.get("root_cause_step_id", "unknown")),
            root_cause_culprit=diagnosis.get("root_cause_culprit", "unknown"),
            reasoning=diagnosis.get("reasoning", "Failed to parse LLM response"),
            confidence_score=float(diagnosis.get("confidence_score", 0.0)),
            raw_response=raw_response,
            golden_context=golden_context,
            trace_length=self._stats. get("trace_length", 0),
            skipped_steps=self._stats. get("skipped_steps", 0),
            weak_links_count=self._stats.get("weak_links_count", 0),
            implicit_refs_count=self._stats. get("implicit_refs_count", 0),
        )
    
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM with retry logic."""
        raw_response = ""
        
        try:
            raw_response = self. llm.one_step_chat(
                messages=messages,
                model_name=self.model_name,
                json_mode=True,
                temperature=0.1,
            )
            
            if self.verbose:
                print(f"[Phase IV] Response length: {len(raw_response)} chars")
            
            if not raw_response. strip():
                if self.verbose:
                    print("[Phase IV] Empty response, retrying without json_mode...")
                raw_response = self.llm.one_step_chat(
                    messages=messages,
                    model_name=self.model_name,
                    json_mode=False,
                    temperature=0.1,
                )
                
        except Exception as e: 
            if self. verbose:
                print(f"[Phase IV] LLM call failed: {type(e).__name__}: {e}")
            raw_response = ""
        
        return raw_response
    
    def _parse_response(self, response: str) -> Dict[str, Any]: 
        """Parse LLM response into structured dict."""
        if not response or not response.strip():
            return {
                "root_cause_step_id": "empty_response",
                "root_cause_culprit": "unknown",
                "reasoning": "LLM returned empty response",
                "confidence_score": 0.0,
            }
        
        response = response.strip()
        
        # Handle markdown code blocks
        if "```json" in response: 
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match: 
                response = match.group(1)
        elif "```" in response:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match: 
                response = match.group(1)
        
        # Try direct JSON parse
        try:
            result = json.loads(response)
            return self._normalize_result(result)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in response
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            json_str = response[start: end]
            result = json.loads(json_str)
            return self._normalize_result(result)
        except (ValueError, json.JSONDecodeError) as e:
            if self.verbose:
                print(f"[Phase IV] JSON parse error: {e}")
        
        return {
            "root_cause_step_id": "parse_error",
            "root_cause_culprit": "unknown",
            "reasoning": f"Failed to parse:  {response[: 200]}",
            "confidence_score": 0.0,
        }
    
    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize the parsed result."""
        if "root_cause_step_id" in result:
            step_id = result["root_cause_step_id"]
            
            if isinstance(step_id, (int, float)):
                result["root_cause_step_id"] = int(step_id)
            elif isinstance(step_id, str):
                match = re.search(r'(\d+)', step_id)
                if match:
                    result["root_cause_step_id"] = int(match. group(1))
        
        if "confidence_score" in result:
            try:
                result["confidence_score"] = float(result["confidence_score"])
            except (ValueError, TypeError):
                result["confidence_score"] = 0.0
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    def get_golden_context_only(
        self,
        trace_nodes: List[AtomicNode],
        full_graph: nx.DiGraph,
    ) -> str:
        """Generate golden context without calling LLM."""
        trace_context = self._generate_golden_context(trace_nodes, full_graph)
        return trace_context
    
    def get_last_stats(self) -> Dict[str, int]: 
        """Get statistics from the last diagnosis."""
        return self._stats. copy()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Function
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_root_cause(
    trace_nodes: List[AtomicNode],
    full_graph: nx.DiGraph,
    llm_client: "LLMClient",
    task_context: TaskContext,
    model_name:  str = "gpt-4",
) -> DiagnosisResult: 
    """Convenience function for one-shot diagnosis."""
    diagnoser = RootCauseDiagnoser(llm_client, model_name)
    return diagnoser. diagnose(trace_nodes, full_graph, task_context)


# ─────────────────────────────────────────────────────────────────────────────
# Backward Compatibility
# ─────────────────────────────────────────────────────────────────────────────
ContextGenerator = RootCauseDiagnoser