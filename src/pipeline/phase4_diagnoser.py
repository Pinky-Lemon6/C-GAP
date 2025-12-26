"""
Phase IV: Root Cause Diagnoser
==============================
Formats the sliced causal graph into "Golden Context" and calls LLM for diagnosis.

Two responsibilities:
1. Generate structured prompt from pruned trace nodes (Golden Context)
2. Invoke LLM and parse structured diagnosis output

Key features:
- Linearization: Sort nodes by step_id
- Gap Summaries: Insert summaries for skipped steps
- Edge Annotation: Mark weak links and implicit context references
- Anomaly Highlighting: Prefix errors with visual markers
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import networkx as nx

from src.models import AtomicNode
from src.pipeline.causal_types import NodeType

if TYPE_CHECKING:
    from src.llm_client import LLMClient


# ─────────────────────────────────────────────────────────────────────────────
# Output Data Model
# ─────────────────────────────────────────────────────────────────────────────
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
            "root_cause_culprit": self.root_cause_culprit,
            "reasoning": self.reasoning,
            "confidence_score": self.confidence_score,
            "metadata": {
                "trace_length": self.trace_length,
                "skipped_steps": self.skipped_steps,
                "weak_links_count": self.weak_links_count,
                "implicit_refs_count": self.implicit_refs_count,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE). Analyze the provided Causal Trace to identify the Root Cause of the failure.

Key annotations in the trace:
- [ ERROR]: Indicates a step that encountered an error
- [ WEAK LINK]: Temporal sequence only, no strong causal evidence
- [ Implicit Context]: Data dependency or reference to another step

Analysis guidelines:
1. Follow the causal chain backwards from the error
2. Pay attention to [Implicit Context] for data dependencies
3. Be skeptical of [WEAK LINK] connections - they may not be causal
4. The root cause is often NOT the error step itself, but an earlier step

Output MUST be valid JSON with this exact structure:
{
  "root_cause_step_id": "step_X",
  "root_cause_culprit": "Agent or component name",
  "reasoning": "Detailed explanation of why this is the root cause",
  "confidence_score": 0.0 to 1.0
}"""

USER_PROMPT_TEMPLATE = """Here is the execution trace:

{golden_context}

Analyze and identify:
1. root_cause_step_id: The step that is the actual root cause (may differ from error step)
2. root_cause_culprit: The agent/component responsible
3. reasoning: Step-by-step explanation of your diagnosis
4. confidence_score: Your confidence in this diagnosis (0.0-1.0)

Respond with ONLY the JSON object, no additional text."""


# ─────────────────────────────────────────────────────────────────────────────
# Main Diagnoser Class
# ─────────────────────────────────────────────────────────────────────────────
class RootCauseDiagnoser:
    """
    Phase IV: Formats sliced graph into Golden Context and invokes LLM for diagnosis.
    
    Usage:
        diagnoser = RootCauseDiagnoser(llm_client, model_name="gpt-4")
        result = diagnoser.diagnose(trace_nodes, full_graph)
    """
    
    def __init__(
        self,
        llm_client: "LLMClient",
        model_name: str = "gpt-4",
        max_context_tokens: int = 8000,
    ):
        """
        Initialize the diagnoser.
        
        Args:
            llm_client: LLM client instance for API calls
            model_name: Model to use for diagnosis
            max_context_tokens: Maximum tokens for context (truncation threshold)
        """
        self.llm = llm_client
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        
        # Statistics for the last diagnosis
        self._stats: Dict[str, int] = {}
    
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
        
        Steps:
        1. Linearization: Sort nodes by step_id
        2. Gap Summaries: Insert summaries for skipped steps
        3. Edge Annotation: Add weak link and implicit context markers
        4. Anomaly Highlighting: Prefix errors with markers
        
        Args:
            trace_nodes: Pruned nodes from Phase III
            full_graph: Complete causal graph for edge lookup
            
        Returns:
            Formatted golden context string
        """
        if not trace_nodes:
            return "[Empty trace - no nodes to analyze]"
        
        # Reset stats
        self._stats = {
            "trace_length": len(trace_nodes),
            "skipped_steps": 0,
            "weak_links_count": 0,
            "implicit_refs_count": 0,
        }
        
        # Step 1: Linearization - sort by step_id
        sorted_nodes = sorted(trace_nodes, key=lambda n: n.step_id)
        
        # Build node_id to node mapping for edge lookup
        trace_node_ids = {n.node_id for n in sorted_nodes}
        
        # Collect all nodes from full_graph for gap analysis
        all_graph_nodes: Dict[str, AtomicNode] = {}
        for node_id in full_graph.nodes():
            node_data = full_graph.nodes[node_id]
            if "node" in node_data:
                all_graph_nodes[node_id] = node_data["node"]
        
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("CAUSAL EXECUTION TRACE")
        lines.append("=" * 60)
        lines.append("")
        
        prev_step_id: Optional[int] = None
        
        for node in sorted_nodes:
            # Step 2: Gap Summaries
            if prev_step_id is not None and node.step_id > prev_step_id + 1:
                gap_summary = self._summarize_gap(
                    prev_step_id + 1, 
                    node.step_id - 1, 
                    all_graph_nodes,
                    trace_node_ids
                )
                if gap_summary:
                    lines.append(gap_summary)
                    lines.append("")
            
            # Step 4: Anomaly Highlighting
            prefix = ""
            if self._is_error_node(node):
                prefix = "[ ERROR] "
            
            # Format the node
            node_line = self._format_node(node, prefix)
            lines.append(node_line)
            
            # Step 3: Edge Annotation
            edge_annotations = self._get_edge_annotations(node, full_graph, all_graph_nodes)
            for annotation in edge_annotations:
                lines.append(annotation)
            
            lines.append("")  # Blank line between nodes
            prev_step_id = node.step_id
        
        lines.append("=" * 60)
        lines.append(f"END OF TRACE ({len(sorted_nodes)} nodes)")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _format_node(self, node: AtomicNode, prefix: str = "") -> str:
        """Format a single node for display."""
        type_str = node.type.value if isinstance(node.type, NodeType) else str(node.type)
        role_str = f" ({node.role})" if node.role else ""
        
        # Truncate content if too long
        content = node.content
        if len(content) > 200:
            content = content[:197] + "..."
        
        return f"{prefix}[Step {node.step_id}] [{type_str}]{role_str}: {content}"
    
    def _is_error_node(self, node: AtomicNode) -> bool:
        """Check if a node represents an error."""
        # Check type
        if isinstance(node.type, NodeType) and node.type == NodeType.ERROR:
            return True
        if str(node.type).upper() == "ERROR":
            return True
        
        # Check content for error indicators
        error_keywords = ["error", "exception", "failed", "failure", "traceback", "fatal"]
        content_lower = node.content.lower()
        return any(kw in content_lower for kw in error_keywords)
    
    def _summarize_gap(
        self,
        start_step: int,
        end_step: int,
        all_nodes: Dict[str, AtomicNode],
        included_ids: set,
    ) -> str:
        """Generate a summary for skipped steps."""
        # Find nodes in the gap that were not included
        skipped_nodes = [
            n for n in all_nodes.values()
            if start_step <= n.step_id <= end_step and n.node_id not in included_ids
        ]
        
        if not skipped_nodes:
            return ""
        
        # Count by type
        type_counts: Counter = Counter()
        for n in skipped_nodes:
            type_str = n.type.value if isinstance(n.type, NodeType) else str(n.type)
            type_counts[type_str] += 1
        
        # Format summary
        step_range = f"step {start_step}" if start_step == end_step else f"steps {start_step}-{end_step}"
        type_summary = ", ".join(f"{count} {t}" for t, count in type_counts.most_common())
        
        self._stats["skipped_steps"] += end_step - start_step + 1
        
        return f"    ... [Skipped {step_range}: {type_summary}] ..."
    
    def _get_edge_annotations(
        self,
        node: AtomicNode,
        full_graph: nx.DiGraph,
        all_nodes: Dict[str, AtomicNode],
    ) -> List[str]:
        """Get edge annotations for incoming edges to this node."""
        annotations: List[str] = []
        
        if node.node_id not in full_graph:
            return annotations
        
        # Check incoming edges
        for src_id in full_graph.predecessors(node.node_id):
            edge_data = full_graph.edges[src_id, node.node_id]
            
            # Check for fallback (weak) link
            if edge_data.get("is_fallback", False):
                annotations.append("    <- [ WEAK LINK]: Temporal sequence only, uncertain causality.")
                self._stats["weak_links_count"] += 1
            
            # Check for secondary layer (implicit context)
            elif edge_data.get("layer") == "SECONDARY":
                src_node = all_nodes.get(src_id)
                if src_node:
                    content_preview = src_node.content[:50]
                    if len(src_node.content) > 50:
                        content_preview += "..."
                    annotations.append(
                        f'    <- [ Implicit Context]: Refers to Step {src_node.step_id}: "{content_preview}"'
                    )
                else:
                    annotations.append(
                        f"    <- [ Implicit Context]: Reference to {src_id}"
                    )
                self._stats["implicit_refs_count"] += 1
        
        return annotations
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Diagnosis Method
    # ─────────────────────────────────────────────────────────────────────────
    def diagnose(
        self,
        trace_nodes: List[AtomicNode],
        full_graph: nx.DiGraph,
        additional_context: Optional[str] = None,
    ) -> DiagnosisResult:
        """
        Perform root cause diagnosis on the pruned trace.
        
        Args:
            trace_nodes: Pruned nodes from Phase III (CausalGraphSlicer)
            full_graph: Complete causal graph for edge analysis
            additional_context: Optional extra context to append
            
        Returns:
            DiagnosisResult with root cause analysis
        """
        # Step 1: Generate Golden Context
        golden_context = self._generate_golden_context(trace_nodes, full_graph)
        
        if additional_context:
            golden_context += f"\n\nAdditional Context:\n{additional_context}"
        
        # Step 2: Construct LLM payload
        user_prompt = USER_PROMPT_TEMPLATE.format(golden_context=golden_context)
        
        # Step 3: Call LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        # Debug: print prompt length
        total_chars = len(SYSTEM_PROMPT) + len(user_prompt)
        print(f"[Phase IV] Prompt total chars: {total_chars}, estimated tokens: ~{total_chars // 4}")
        
        raw_response = ""
        try:
            raw_response = self.llm.one_step_chat(
                messages=messages,
                model_name=self.model_name,
                json_mode=True,
                temperature=0.1,  # Low temperature for consistent analysis
            )
            print(f"[Phase IV] LLM response length: {len(raw_response)} chars")
            
            if not raw_response.strip():
                print("[Phase IV] WARNING: LLM returned empty response with json_mode=True, retrying without json_mode...")
                # Retry without json_mode - some models don't support it
                raw_response = self.llm.one_step_chat(
                    messages=messages,
                    model_name=self.model_name,
                    json_mode=False,
                    temperature=0.1,
                )
                print(f"[Phase IV] Retry response length: {len(raw_response)} chars")
                
            if not raw_response.strip():
                print("[Phase IV] WARNING: Still empty, trying with simplified prompt...")
                # Try with a simpler prompt
                simple_messages = [
                    {"role": "user", "content": f"Analyze this execution trace and identify the root cause of failure. Output JSON with root_cause_step_id, root_cause_culprit, reasoning, confidence_score.\n\n{golden_context[:8000]}"}
                ]
                raw_response = self.llm.one_step_chat(
                    messages=simple_messages,
                    model_name=self.model_name,
                    json_mode=False,
                    temperature=0.1,
                )
                print(f"[Phase IV] Simple prompt response length: {len(raw_response)} chars")
                
        except Exception as e:
            print(f"[Phase IV] LLM call failed with exception: {type(e).__name__}: {e}")
            raw_response = ""
        
        # Debug: print first 500 chars of response
        if raw_response:
            print(f"[Phase IV] Response preview: {raw_response[:500]}...")
        
        # Step 4: Parse JSON output
        diagnosis = self._parse_response(raw_response)
        
        # Populate result
        result = DiagnosisResult(
            root_cause_step_id=diagnosis.get("root_cause_step_id", "unknown"),
            root_cause_culprit=diagnosis.get("root_cause_culprit", "unknown"),
            reasoning=diagnosis.get("reasoning", "Failed to parse LLM response"),
            confidence_score=float(diagnosis.get("confidence_score", 0.0)),
            raw_response=raw_response,
            golden_context=golden_context,
            trace_length=self._stats.get("trace_length", 0),
            skipped_steps=self._stats.get("skipped_steps", 0),
            weak_links_count=self._stats.get("weak_links_count", 0),
            implicit_refs_count=self._stats.get("implicit_refs_count", 0),
        )
        
        return result
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured dict."""
        if not response or not response.strip():
            return {
                "root_cause_step_id": "empty_response",
                "root_cause_culprit": "unknown",
                "reasoning": "LLM returned empty response",
                "confidence_score": 0.0,
            }
        
        # Try to extract JSON from response
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
            # Normalize step_id format
            if "root_cause_step_id" in result:
                step_id = result["root_cause_step_id"]
                # Handle various formats like "Step 1", "step_1", 1, etc.
                if isinstance(step_id, str):
                    # Extract number if present
                    match = re.search(r'(\d+)', str(step_id))
                    if match:
                        result["root_cause_step_id"] = f"step_{match.group(1)}"
                elif isinstance(step_id, int):
                    result["root_cause_step_id"] = f"step_{step_id}"
            return result
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in response
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            json_str = response[start:end]
            result = json.loads(json_str)
            # Normalize step_id
            if "root_cause_step_id" in result:
                step_id = result["root_cause_step_id"]
                if isinstance(step_id, str):
                    match = re.search(r'(\d+)', str(step_id))
                    if match:
                        result["root_cause_step_id"] = f"step_{match.group(1)}"
                elif isinstance(step_id, int):
                    result["root_cause_step_id"] = f"step_{step_id}"
            return result
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[Phase IV] JSON parse error: {e}")
            print(f"[Phase IV] Attempted to parse: {response[:500]}...")
        
        # Return default structure if parsing fails
        return {
            "root_cause_step_id": "parse_error",
            "root_cause_culprit": "unknown",
            "reasoning": f"Failed to parse LLM response: {response[:200]}",
            "confidence_score": 0.0,
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    def get_golden_context_only(
        self,
        trace_nodes: List[AtomicNode],
        full_graph: nx.DiGraph,
    ) -> str:
        """
        Generate golden context without calling LLM.
        Useful for debugging or manual analysis.
        """
        return self._generate_golden_context(trace_nodes, full_graph)
    
    def get_last_stats(self) -> Dict[str, int]:
        """Get statistics from the last diagnosis."""
        return self._stats.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Function
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_root_cause(
    trace_nodes: List[AtomicNode],
    full_graph: nx.DiGraph,
    llm_client: "LLMClient",
    model_name: str = "gpt-4",
) -> DiagnosisResult:
    """
    Convenience function for one-shot diagnosis.
    
    Args:
        trace_nodes: Pruned nodes from Phase III
        full_graph: Complete causal graph
        llm_client: LLM client instance
        model_name: Model to use
        
    Returns:
        DiagnosisResult with root cause analysis
    """
    diagnoser = RootCauseDiagnoser(llm_client, model_name)
    return diagnoser.diagnose(trace_nodes, full_graph)


# ─────────────────────────────────────────────────────────────────────────────
# Backward Compatibility Alias
# ─────────────────────────────────────────────────────────────────────────────
ContextGenerator = RootCauseDiagnoser  # Alias for old name
