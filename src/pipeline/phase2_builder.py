"""Phase II: Atomic Node-based Causal Graph Builder

Builds a causal graph from AtomicNodes with two distinct stages:
1. Intra-Step Linking: Hard rules within each step (sequential flow)
2. Inter-Step Linking: LLM-verified causality across steps

Core Features:
- Graph nodes are AtomicNodes (identified by node_id)
- Type-aware candidate filtering (VALID_CAUSAL_SOURCES)
- Dual-track candidate selection (Rule + Semantic)
- Pairwise LLM verification with counterfactual reasoning
"""

from __future__ import annotations

import json
import re
import time
from typing import List, Optional, Dict, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx

from src.llm_client import LLMClient
from src.models import StandardLogItem, AtomicNode
from src.pipeline.causal_types import NodeType, CausalType, VALID_CAUSAL_SOURCES
from src.pipeline.candidate_selector import (
    AtomicNodeCandidateSelector,
    SimpleAtomicNodeSelector,
)


# =============================================================================
# System Prompts
# =============================================================================

INTER_STEP_VERIFICATION_SYSTEM_PROMPT = """You are a causal relationship judge analyzing multi-agent system logs.

## Task
Determine if the Target event causally depends on the Source event.

## Event Types
- INTENT: Agent's thought, plan, decision
- EXEC: Tool call, code execution, action
- INFO: Observation, result, error, feedback
- COMM: Message to/from other agents or users

## Causal Types (Choose ONE)

**INSTRUCTION**: Source requests/commands, Target responds/executes
  - "Search for weather" → "Calling search_api('weather')"
  
**DATA**: Target uses information produced by Source
  - "API returned temperature=25°C" → "The temperature is 25 degrees"
  
**STATE**: Target depends on state/condition from Source
  - "Login successful" → "Accessing dashboard"
  
**NONE**: No causal relationship - Target happens regardless of Source

## Reasoning Method
Apply COUNTERFACTUAL test:
1. If Source did NOT happen, would Target still occur the same way?
2. If NO → Causality exists (choose type)
3. If YES → NONE

## Output Format
Brief analysis, then JSON:
Analysis: <1-2 sentences>
JSON: {"type": "<INSTRUCTION|DATA|STATE|NONE>", "confidence": <0.0-1.0>, "reason": "<brief>"}"""


INTER_STEP_VERIFICATION_USER_TEMPLATE = """## Source Event ({source_type})
{source_content}

## Target Event ({target_type})
{target_content}

Does Target causally depend on Source?"""


# =============================================================================
# Result Types
# =============================================================================

class CausalResult:
    """Result of causal inference."""
    
    def __init__(
        self,
        causal_type: CausalType,
        confidence: float,
        reason: str,
    ):
        self.causal_type = causal_type
        self.confidence = confidence
        self.reason = reason


# =============================================================================
# Causal Graph Builder
# =============================================================================

class CausalGraphBuilder:
    """
    Causal Graph Builder for AtomicNodes.
    
    Two-Stage Process:
    1. Intra-Step: Link nodes within each step (sequential flow)
    2. Inter-Step: LLM-verified links across steps
    """
    
    def __init__(
        self,
        llm: LLMClient,
        model_name: str,
        # Candidate selection
        window_size: int = 15,
        rule_candidates: int = 3,
        semantic_candidates: int = 2,
        use_embeddings: bool = True,
        # Verification thresholds
        confidence_threshold: float = 0.5,
        # Performance
        max_workers: int = 4,
        batch_size: int = 8,
        enable_early_stop: bool = True,
        early_stop_confidence: float = 0.75,
        # Content processing
        max_content_length: int = 500,
        temperature: float = 0.1,
    ):
        """
        Initialize the Atomic Causal Graph Builder.
        
        Args:
            llm: LLM client for verification
            model_name: Model to use for LLM calls
            window_size: Number of previous steps to consider as candidates
            rule_candidates: Number of candidates from rule track
            semantic_candidates: Number of candidates from semantic track
            use_embeddings: Whether to use embedding-based selection
            confidence_threshold: Minimum confidence to accept a causal link
            max_workers: Parallel workers for LLM calls
            batch_size: Batch size for parallel processing
            enable_early_stop: Stop searching after finding high-confidence parent
            early_stop_confidence: Confidence threshold for early stopping
            max_content_length: Max chars for node content in prompts
            temperature: LLM temperature
        """
        self.llm = llm
        self.model_name = model_name
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_early_stop = enable_early_stop
        self.early_stop_confidence = early_stop_confidence
        self.max_content_length = max_content_length
        self.temperature = temperature
        
        # Initialize candidate selector
        if use_embeddings:
            self.candidate_selector = AtomicNodeCandidateSelector(
                rule_candidates=rule_candidates,
                semantic_candidates=semantic_candidates,
                use_embeddings=True,
                filter_invalid_types=True,
            )
        else:
            self.candidate_selector = SimpleAtomicNodeSelector(
                max_candidates=rule_candidates + semantic_candidates,
                filter_invalid_types=True,
            )
        
        # Statistics
        self._stats = {
            "total_nodes": 0,
            "intra_step_edges": 0,
            "inter_step_edges": 0,
            "llm_calls": 0,
            "skipped_by_type": 0,
            "skipped_by_early_stop": 0,
            "wall_time_seconds": 0.0,
        }
    
    def build(self, steps: List[StandardLogItem]) -> nx.DiGraph:
        """
        Build causal graph from steps containing AtomicNodes.
        
        Args:
            steps: List of StandardLogItem, each with atomic_nodes populated
            
        Returns:
            NetworkX DiGraph with AtomicNodes as nodes
        """
        start_time = time.time()
        self._reset_stats()
        
        g = nx.DiGraph()
        
        # Collect all atomic nodes
        all_nodes: List[AtomicNode] = []
        for step in steps:
            all_nodes.extend(step.atomic_nodes)
        
        self._stats["total_nodes"] = len(all_nodes)
        
        if not all_nodes:
            return g
        
        # Add all nodes to graph
        self._add_all_nodes(g, all_nodes)
        
        # Precompute embeddings if using semantic selection
        if hasattr(self.candidate_selector, 'precompute_embeddings'):
            self.candidate_selector.precompute_embeddings(all_nodes)
        
        # Stage 1: Intra-Step Linking (hard rules)
        for step in steps:
            self._link_intra_step(g, step)
        
        # Stage 2: Inter-Step Linking (LLM verification)
        self._link_inter_step(g, steps)
        
        self._stats["wall_time_seconds"] = time.time() - start_time
        
        return g
    
    def get_stats(self) -> Dict[str, Any]:
        """Get build statistics."""
        return self._stats.copy()
    
    def _reset_stats(self):
        """Reset statistics."""
        self._stats = {
            "total_nodes": 0,
            "intra_step_edges": 0,
            "inter_step_edges": 0,
            "llm_calls": 0,
            "skipped_by_type": 0,
            "skipped_by_early_stop": 0,
            "wall_time_seconds": 0.0,
        }
    
    def _add_all_nodes(self, g: nx.DiGraph, nodes: List[AtomicNode]):
        """Add all AtomicNodes to the graph."""
        for node in nodes:
            g.add_node(
                node.node_id,
                step_id=node.step_id,
                role=node.role,
                type=node.type,
                content=node.content,
                original_text=node.original_text,
            )
    
    # =========================================================================
    # Stage 1: Intra-Step Linking
    # =========================================================================
    
    def _link_intra_step(self, g: nx.DiGraph, step: StandardLogItem):
        """
        Link nodes within a single step using hard rules.
        
        Logic: Sequential flow within step
        - node[0] -> node[1] -> node[2] -> ...
        
        This captures the natural ordering:
        INFO (from prev) -> INTENT -> EXEC -> INFO -> COMM
        """
        nodes = step.atomic_nodes
        
        if len(nodes) < 2:
            return
        
        for i in range(len(nodes) - 1):
            source = nodes[i]
            target = nodes[i + 1]
            
            g.add_edge(
                source.node_id,
                target.node_id,
                edge_type="INTRA_STEP",
                causal_type="SEQUENTIAL",
                confidence=1.0,
                reason="Sequential flow within step",
            )
            self._stats["intra_step_edges"] += 1
    
    # =========================================================================
    # Stage 2: Inter-Step Linking
    # =========================================================================
    
    def _link_inter_step(self, g: nx.DiGraph, steps: List[StandardLogItem]):
        """
        Link nodes across steps using LLM verification.
        
        For each target node in current step:
        1. Filter candidates by valid causal types
        2. Select top-K candidates using dual-track selection
        3. Verify causality with LLM
        4. Add edges for confirmed relationships
        """
        # Build step index for efficient lookup
        step_nodes: Dict[int, List[AtomicNode]] = {}
        for step in steps:
            step_nodes[step.step_id] = step.atomic_nodes
        
        # Collect all previous nodes for each step
        prev_nodes_cache: Dict[int, List[AtomicNode]] = {}
        accumulated: List[AtomicNode] = []
        
        for step in steps:
            prev_nodes_cache[step.step_id] = accumulated.copy()
            accumulated.extend(step.atomic_nodes)
        
        # Collect verification tasks
        all_tasks: List[Tuple[AtomicNode, AtomicNode]] = []
        
        for step in steps:
            prev_nodes = prev_nodes_cache.get(step.step_id, [])
            if not prev_nodes:
                continue
            
            # Apply window size limit
            if len(prev_nodes) > self.window_size * 3:
                prev_nodes = prev_nodes[-(self.window_size * 3):]
            
            for target_node in step.atomic_nodes:
                # Get valid candidates (type-filtered + selected)
                candidates = self._get_inter_step_candidates(target_node, prev_nodes)
                
                for source_node in candidates:
                    all_tasks.append((source_node, target_node))
        
        # Execute verification in parallel
        if all_tasks:
            self._execute_inter_step_verification(g, all_tasks)
    
    def _get_inter_step_candidates(
        self,
        target: AtomicNode,
        prev_nodes: List[AtomicNode],
    ) -> List[AtomicNode]:
        """
        Get candidate source nodes for a target.
        
        Steps:
        1. Filter by valid causal types (VALID_CAUSAL_SOURCES)
        2. Apply candidate selection (rule + semantic)
        """
        # Step 1: Type filtering
        try:
            target_type = NodeType(target.type.upper())
            valid_source_types = VALID_CAUSAL_SOURCES.get(target_type, set())
        except (ValueError, AttributeError):
            valid_source_types = set(NodeType)
        
        type_filtered = []
        for node in prev_nodes:
            try:
                source_type = NodeType(node.type.upper())
                if source_type in valid_source_types:
                    type_filtered.append(node)
                else:
                    self._stats["skipped_by_type"] += 1
            except (ValueError, AttributeError):
                # Unknown type, include it
                type_filtered.append(node)
        
        if not type_filtered:
            return []
        
        # Step 2: Candidate selection
        selected = self.candidate_selector.select(target, type_filtered)
        
        return selected
    
    def _execute_inter_step_verification(
        self,
        g: nx.DiGraph,
        tasks: List[Tuple[AtomicNode, AtomicNode]],
    ):
        """
        Execute LLM verification for inter-step edges in parallel.
        
        Groups tasks by target for early stopping.
        """

        # Track which targets already have high-confidence parents
        satisfied_targets: Set[str] = set()
        
        def verify_single(task: Tuple[AtomicNode, AtomicNode]) -> Tuple[AtomicNode, AtomicNode, CausalResult]:
            source, target = task
            try:
                result = self._verify_causality(source, target)
                return (source, target, result)
            except Exception as e:
                print(f"Warning: Verification failed for {source.node_id} -> {target.node_id}: {e}")
                return (source, target, CausalResult(CausalType.NONE, 0.0))
        
        
        # Process in batches
        all_task_list = list(tasks)
        
        for batch_start in range(0, len(all_task_list), self.batch_size):
            batch_end = min(len(all_task_list), batch_start + self.batch_size)
            raw_batch = all_task_list[batch_start:batch_end]
            
            batch = raw_batch
            # Filter out tasks for already-satisfied targets
            if self.enable_early_stop:
                batch = [(s, t) for s, t in raw_batch if not (self.enable_early_stop and t.node_id in satisfied_targets)]
            
                if not batch:
                    self._stats["skipped_by_early_stop"] += len(raw_batch)
                    continue
            
            # Execute batch in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(verify_single, task) for task in batch]
                
                for future in as_completed(futures):
                    try:
                        source, target, result = future.result()
                        
                        # Skip if target already satisfied
                        if self.enable_early_stop and target.node_id in satisfied_targets:
                            self._stats["skipped_by_early_stop"] += 1
                            continue
                        
                        # Add edge if causality confirmed
                        if (result.causal_type != CausalType.NONE and 
                            result.confidence >= self.confidence_threshold):
                            
                            self._add_inter_step_edge(g, source, target, result)
                            
                            # Mark target as satisfied if high confidence
                            if (self.enable_early_stop and 
                                result.confidence >= self.early_stop_confidence):
                                satisfied_targets.add(target.node_id)
                    
                    except Exception as e:
                        # Log error but continue
                        print(f"Warning: Verification failed: {e}")
    
    def _verify_causality(
        self,
        source: AtomicNode,
        target: AtomicNode,
    ) -> CausalResult:
        """
        Verify causal relationship between source and target using LLM.
        """
        self._stats["llm_calls"] += 1
        
        source_content = self._truncate_content(source.content)
        target_content = self._truncate_content(target.content)
        
        user_message = INTER_STEP_VERIFICATION_USER_TEMPLATE.format(
            source_type=source.type,
            source_content=source_content,
            target_type=target.type,
            target_content=target_content,
        )
        
        messages = [
            {"role": "system", "content": INTER_STEP_VERIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        try:
            response = self.llm.one_step_chat(
                messages=messages,
                model_name=self.model_name,
                temperature=self.temperature,
                json_mode=False,
            )
            return self._parse_verification_response(response)
        except Exception as e:
            return CausalResult(
                causal_type=CausalType.NONE,
                confidence=0.0,
                reason=f"LLM error: {str(e)[:100]}",
            )
    
    def _parse_verification_response(self, response: str) -> CausalResult:
        """Parse LLM response to extract causal judgment."""
        if not response:
            return CausalResult(CausalType.NONE, 0.0, "Empty response")
        
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]+\}', response)
        
        if json_match:
            try:
                obj = json.loads(json_match.group())
                
                type_str = obj.get("type", "NONE").upper()
                confidence = float(obj.get("confidence", 0.0))
                reason = obj.get("reason", "")
                
                # Map type string to CausalType
                try:
                    causal_type = CausalType(type_str)
                except ValueError:
                    causal_type = CausalType.NONE
                
                return CausalResult(
                    causal_type=causal_type,
                    confidence=min(1.0, max(0.0, confidence)),
                    reason=reason,
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        
        return CausalResult(CausalType.NONE, 0.0, "Parse error")
    
    def _add_inter_step_edge(
        self,
        g: nx.DiGraph,
        source: AtomicNode,
        target: AtomicNode,
        result: CausalResult,
    ):
        """Add an inter-step causal edge to the graph."""
        g.add_edge(
            source.node_id,
            target.node_id,
            edge_type="INTER_STEP",
            causal_type=result.causal_type.value,
            confidence=result.confidence,
            reason=result.reason,
        )
        self._stats["inter_step_edges"] += 1
    
    def _truncate_content(
        self,
        content: Optional[str],
        max_length: Optional[int] = None,
    ) -> str:
        """Truncate content to max length."""
        if not content:
            return "[Empty]"
        
        max_len = max_length or self.max_content_length
        
        if len(content) <= max_len:
            return content
        
        # Keep head and tail
        head = max_len // 2
        tail = max_len // 2
        return content[:head] + "..." + content[-tail:]



