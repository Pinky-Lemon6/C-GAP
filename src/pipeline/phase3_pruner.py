"""Phase III: Causal Graph Slicer

Deterministic, algorithm-based graph slicing for failure attribution.
NO LLM calls - pure graph algorithms for reproducibility and speed.

Key Features:
1. Backward Reachability: Priority-queue based traversal from error node
2. Semantic Filtering: Keep relevant nodes, drop trivial pass-throughs
3. Loop Compression: Collapse repetitive execution patterns
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from src.models import AtomicNode


# =============================================================================
# Edge Cost Constants
# =============================================================================

EDGE_COSTS = {
    "PRIMARY": 0.1,      # Backbone/control flow - highest priority
    "SECONDARY": 1.0,    # Associative/data flow - medium priority
    "FALLBACK": 5.0,     # Temporal fallback - low priority
}

# Default cost for unknown edge types
DEFAULT_EDGE_COST = 1.0


# =============================================================================
# Destructive Keywords (never drop EXEC nodes containing these)
# =============================================================================

DESTRUCTIVE_KEYWORDS = {
    "delete", "remove", "kill", "format", "rm", "drop", "truncate",
    "destroy", "wipe", "purge", "erase", "clear", "reset", "uninstall",
}

# Error-indicating keywords for INFO nodes
ERROR_KEYWORDS = {
    "error", "fail", "exception", "traceback", "crash", "fatal",
    "abort", "panic", "timeout", "refused", "denied", "invalid",
}


# =============================================================================
# Slice Result Container
# =============================================================================

@dataclass
class SliceResult:
    """Result of graph slicing operation."""
    
    nodes: List[AtomicNode]
    """Sliced nodes sorted by step_id."""
    
    node_ids: Set[str]
    """Set of included node IDs for quick lookup."""
    
    subgraph: nx.DiGraph
    """The sliced subgraph."""
    
    stats: Dict[str, Any] = field(default_factory=dict)
    """Slicing statistics."""


# =============================================================================
# Causal Graph Slicer
# =============================================================================

class CausalGraphSlicer:
    """
    Deterministic causal graph slicer for failure attribution.
    
    Uses weighted backward traversal to find relevant causal ancestors
    of a target (error) node, then applies semantic filters to remove
    irrelevant nodes.
    """
    
    def __init__(
        self,
        max_depth: int = 30,
        max_cost: float = 50.0,
        high_degree_threshold: int = 6,
        enable_loop_compression: bool = True,
        loop_repeat_threshold: int = 3,
    ):
        """
        Initialize the slicer.
        
        Args:
            max_depth: Maximum traversal depth (hops from target)
            max_cost: Maximum cumulative edge cost
            high_degree_threshold: Nodes with out-degree > this are kept
            enable_loop_compression: Whether to compress repetitive patterns
            loop_repeat_threshold: Compress if pattern repeats > this many times
        """
        self.max_depth = max_depth
        self.max_cost = max_cost
        self.high_degree_threshold = high_degree_threshold
        self.enable_loop_compression = enable_loop_compression
        self.loop_repeat_threshold = loop_repeat_threshold
        
        self._stats: Dict[str, Any] = {}
    
    def slice(
        self,
        graph: nx.DiGraph,
        target_node_id: str,
        root_node_id: Optional[str] = None,
    ) -> List[AtomicNode]:
        """
        Slice the graph to extract relevant causal context.
        
        Args:
            graph: The full causal graph
            target_node_id: ID of the target (error) node
            root_node_id: ID of the root node (optional, auto-detected if None)
            
        Returns:
            List of AtomicNode sorted by step_id
        """
        self._reset_stats()
        
        if graph.number_of_nodes() == 0:
            return []
        
        if target_node_id not in graph:
            raise ValueError(f"Target node '{target_node_id}' not in graph")
        
        # Auto-detect root node if not provided
        if root_node_id is None:
            root_node_id = self._find_root_node(graph)
        
        # Step 1: Backward reachability from target
        reachable_ids = self._backward_slice(
            graph=graph,
            target_node_id=target_node_id,
        )
        self._stats["backward_reachable"] = len(reachable_ids)
        
        # Always include target and root
        reachable_ids.add(target_node_id)
        if root_node_id and root_node_id in graph:
            reachable_ids.add(root_node_id)
        
        # Step 2: Extract nodes from graph
        nodes = self._extract_nodes(graph, reachable_ids)
        self._stats["extracted_nodes"] = len(nodes)
        
        # Step 3: Build subgraph for filtering decisions
        subgraph = graph.subgraph(reachable_ids).copy()
        
        # Step 4: Apply semantic filters
        filtered_nodes = self._apply_filters(
            nodes=nodes,
            subgraph=subgraph,
            target_node_id=target_node_id,
            root_node_id=root_node_id,
        )
        filtered_nodes = self._collapse_chains(filtered_nodes, subgraph)
        self._stats["after_filtering"] = len(filtered_nodes)
        
        # Step 5: Loop compression (optional)
        if self.enable_loop_compression:
            filtered_nodes = self._compress_loops(filtered_nodes)
            self._stats["after_compression"] = len(filtered_nodes)
        
        # Sort by step_id
        filtered_nodes.sort(key=lambda n: (n.step_id, n.node_id))
        
        return filtered_nodes
    
    def slice_full(
        self,
        graph: nx.DiGraph,
        target_node_id: str,
        root_node_id: Optional[str] = None,
    ) -> SliceResult:
        """
        Slice and return full result with subgraph and stats.
        """
        nodes = self.slice(graph, target_node_id, root_node_id)
        node_ids = {n.node_id for n in nodes}
        subgraph = graph.subgraph(node_ids).copy()
        
        return SliceResult(
            nodes=nodes,
            node_ids=node_ids,
            subgraph=subgraph,
            stats=self._stats.copy(),
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get slicing statistics."""
        return self._stats.copy()
    
    def _reset_stats(self):
        """Reset statistics."""
        self._stats = {
            "backward_reachable": 0,
            "extracted_nodes": 0,
            "after_filtering": 0,
            "after_compression": 0,
            "dropped_isolated_exec": 0,
            "dropped_trivial_info": 0,
            "compressed_loops": 0,
        }
    
    # =========================================================================
    # Step 1: Backward Reachability (Weighted BFS with Priority Queue)
    # =========================================================================
    
    def _backward_slice(self, graph: nx.DiGraph, target_node_id: str) -> Set[str]:
        """Weighted BFS with Exponential Decay."""
        reachable: Set[str] = set()
        # queue: (cost, depth, node_id)
        pq: List[Tuple[float, int, str]] = [(0.0, 0, target_node_id)]
        best_cost: Dict[str, float] = {target_node_id: 0.0}
        
        while pq:
            cost, depth, node_id = heapq.heappop(pq)
            
            if cost > self.max_cost: continue 
            if depth > self.max_depth: continue
            
            if cost > best_cost.get(node_id, float('inf')): continue
            
            reachable.add(node_id)
            
            for pred_id in graph.predecessors(node_id):
                edge_data = graph.get_edge_data(pred_id, node_id, default={})
                
                base_cost = self._get_edge_cost(edge_data)
                edge_cost = base_cost
                
                if edge_data.get("layer") == "PRIMARY":
                    edge_cost = base_cost * (1.1 ** depth)
                
                new_cost = cost + edge_cost
                new_depth = depth + 1
                
                if new_cost < best_cost.get(pred_id, float('inf')):
                    best_cost[pred_id] = new_cost
                    heapq.heappush(pq, (new_cost, new_depth, pred_id))
                    
        return reachable
    
    def _get_edge_cost(self, edge_data: Dict[str, Any]) -> float:
        """
        Get traversal cost for an edge.
        
        Cost is determined by:
        1. Edge layer (PRIMARY < SECONDARY)
        2. Fallback status (fallback edges are costly)
        """
        # Check if fallback
        is_fallback = edge_data.get("is_fallback", False)
        if is_fallback:
            return EDGE_COSTS["FALLBACK"]
        
        # Check layer
        layer = edge_data.get("layer", "SECONDARY")
        if layer == "PRIMARY":
            return EDGE_COSTS["PRIMARY"]
        
        # Check causal type for hints
        causal_type = edge_data.get("causal_type", "")
        if causal_type in ("INSTRUCTION", "DATA"):
            return EDGE_COSTS["PRIMARY"]
        elif causal_type in ("SEQUENTIAL",):
            return 0.5  # Intra-step sequential is cheap
        
        return EDGE_COSTS.get(layer, DEFAULT_EDGE_COST)
    
    # =========================================================================
    # Step 2: Node Extraction
    # =========================================================================
    
    def _extract_nodes(
        self,
        graph: nx.DiGraph,
        node_ids: Set[str],
    ) -> List[AtomicNode]:
        """
        Extract AtomicNode objects from graph node data.
        """
        nodes = []
        
        for node_id in node_ids:
            if node_id not in graph:
                continue
            
            data = graph.nodes[node_id]
            
            # Construct AtomicNode from graph data
            node = AtomicNode(
                node_id=node_id,
                step_id=data.get("step_id", 0),
                role=data.get("role", "unknown"),
                type=data.get("type", "INFO"),
                content=data.get("content", ""),
                original_text=data.get("original_text"),
            )
            nodes.append(node)
        
        return nodes
    
    def _find_root_node(self, graph: nx.DiGraph) -> Optional[str]:
        """
        Find the root node (node with in-degree 0 and smallest step_id).
        """
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        
        if not roots:
            # No clear root, use node with smallest step_id
            nodes_with_step = [
                (n, graph.nodes[n].get("step_id", float('inf')))
                for n in graph.nodes()
            ]
            if nodes_with_step:
                nodes_with_step.sort(key=lambda x: x[1])
                return nodes_with_step[0][0]
            return None
        
        # Return root with smallest step_id
        roots_with_step = [
            (n, graph.nodes[n].get("step_id", float('inf')))
            for n in roots
        ]
        roots_with_step.sort(key=lambda x: x[1])
        return roots_with_step[0][0]
    
    # =========================================================================
    # Step 3: Semantic Filtering
    # =========================================================================
    
    def _apply_filters(
        self,
        nodes: List[AtomicNode],
        subgraph: nx.DiGraph,
        target_node_id: str,
        root_node_id: Optional[str],
    ) -> List[AtomicNode]:
        """
        Apply semantic filters to keep only relevant nodes.
        
        Rules:
        1. Always keep: INTENT, COMM, ERROR-containing nodes, target, root
        2. Always keep: High out-degree nodes (hubs)
        3. Drop isolated EXEC: in-degree=1, out-degree=1, no destructive keywords
        4. Drop trivial INFO: leaf nodes (out-degree=0) that aren't target
        """
        filtered = []
        # kept_nodes = set()
        # Precompute degrees in subgraph
        in_degrees = dict(subgraph.in_degree())
        out_degrees = dict(subgraph.out_degree())
        
        for node in nodes:
            # kept_nodes.add(node.node_id)
            keep, reason = self._should_keep_node(
                node=node,
                subgraph=subgraph,
                target_node_id=target_node_id,
                root_node_id=root_node_id,
                in_degree=in_degrees.get(node.node_id, 0),
                out_degree=out_degrees.get(node.node_id, 0),
            )
            
            if keep:
                filtered.append(node)
            else:
                # Track dropped nodes
                if node.type == "EXEC":
                    self._stats["dropped_isolated_exec"] += 1
                elif node.type == "INFO":
                    self._stats["dropped_trivial_info"] += 1
        
        # additional_nodes = []
        
        # for node in filtered:
        #     if node.type == "INTENT":
                
        #         for successor in full_graph.successors(node.node_id):
        #             child_data = full_graph.nodes[successor]
                    
                    
        #             if child_data.get("type") == "EXEC":
                        
        #                 if successor not in kept_nodes:
                            
        #                     child_node = self._extract_single_node(full_graph, successor)
        #                     additional_nodes.append(child_node)
        #                     kept_nodes.add(successor) 
                            
        #                     for grand_child in full_graph.successors(successor):
        #                         gc_data = full_graph.nodes[grand_child]
        #                         if gc_data.get("type") == "INFO" and grand_child not in kept_nodes:
        #                             gc_node = self._extract_single_node(full_graph, grand_child)
        #                             additional_nodes.append(gc_node)
        #                             kept_nodes.add(grand_child)
        
        return filtered
    
    def _should_keep_node(
        self,
        node: AtomicNode,
        subgraph: nx.DiGraph,
        target_node_id: str,
        root_node_id: Optional[str],
        in_degree: int,
        out_degree: int,
    ) -> Tuple[bool, str]:
        """
        Decide whether to keep a node.
        
        Returns:
            (keep, reason) tuple
        """
        # Base Anchor Rules
        if node.node_id == target_node_id:
            return True, "target_node"
        if root_node_id and node.node_id == root_node_id:
            return True, "root_node"
        
        
        # Keep nodes with error indicators
        content_lower = (node.content or "").lower()
        if self._contains_error_keywords(content_lower):
            return True, "error_content"
        
        # Keep high out-degree nodes (hubs in causal chain)
        secondary_in = 0
        secondary_out = 0
        if node.node_id in subgraph:
             for _, _, d in subgraph.in_edges(node.node_id, data=True):
                 if d.get("layer") == "SECONDARY": secondary_in += 1
             for _, _, d in subgraph.out_edges(node.node_id, data=True):
                 if d.get("layer") == "SECONDARY": secondary_out += 1

        if (in_degree + out_degree > self.high_degree_threshold) and (secondary_in + secondary_out > 0):
             return True, "structural_hub"
        
        if node.type == "INTENT":
            return True, "intent"
        
        secondary_out_degree = 0
        if node.node_id in subgraph:
            for succ in subgraph.successors(node.node_id):
                edge_data = subgraph.get_edge_data(node.node_id, succ)
                if edge_data.get("layer") == "SECONDARY":
                    secondary_out_degree += 1
                    
        if node.type == "INFO":
            # Secondary Out = 0 -> likely dead-end info
            if secondary_out_degree == 0:
                return False, "dead_end_info"
            return True, "useful_info"
        
        if node.type == "EXEC":
            if secondary_out_degree == 0:
                return False, "ineffectual_exec"
            return True, "useful_exec"
        
        if node.type == "COMM":
            if secondary_out_degree == 0:
                return False, "routing_comm"
            return True, "useful_comm"
            
        # Default: keep
        return True, "default_keep"
    
    def _contains_error_keywords(self, text: str) -> bool:
        """Check if text contains error-indicating keywords."""
        return any(kw in text for kw in ERROR_KEYWORDS)
    
    def _contains_destructive_keywords(self, text: str) -> bool:
        """Check if text contains destructive operation keywords."""
        return any(kw in text for kw in DESTRUCTIVE_KEYWORDS)
    
    # =========================================================================
    # Step 4: Loop Compression
    # =========================================================================
    
    def _compress_loops(
        self,
        nodes: List[AtomicNode],
    ) -> List[AtomicNode]:
        """
        Compress repetitive execution patterns.
        
        Detects patterns like INTENT->EXEC->INFO repeating and collapses them.
        """
        if len(nodes) < self.loop_repeat_threshold * 2:
            return nodes
        
        # Sort by step_id first
        sorted_nodes = sorted(nodes, key=lambda n: (n.step_id, n.node_id))
        
        # Build fingerprint sequence
        fingerprints = [self._node_fingerprint(n) for n in sorted_nodes]
        
        # Find repeated patterns (length 2-5)
        compressed = []
        i = 0
        
        while i < len(sorted_nodes):
            # Try to find a repeating pattern starting here
            pattern_found = False
            
            for pattern_len in range(2, min(6, (len(sorted_nodes) - i) // 2 + 1)):
                pattern = fingerprints[i:i + pattern_len]
                repeat_count = 1
                
                # Count consecutive repeats
                j = i + pattern_len
                while j + pattern_len <= len(fingerprints):
                    if fingerprints[j:j + pattern_len] == pattern:
                        repeat_count += 1
                        j += pattern_len
                    else:
                        break
                
                # Compress if repeats > threshold
                if repeat_count > self.loop_repeat_threshold:
                    pattern_found = True
                    total_steps = repeat_count * pattern_len
                    
                    # Keep first iteration
                    compressed.extend(sorted_nodes[i:i + pattern_len])
                    
                    # Add synthetic compressed node
                    compressed_node = AtomicNode(
                        node_id=f"compressed_{sorted_nodes[i].step_id}",
                        step_id=sorted_nodes[i].step_id,
                        role="system",
                        type="COMPRESSED",
                        content=f"[Loop compressed: {repeat_count - 2} iterations of {pattern_len}-node pattern omitted]",
                    )
                    compressed.append(compressed_node)
                    self._stats["compressed_loops"] += 1
                    
                    # Keep last iteration
                    last_start = i + (repeat_count - 1) * pattern_len
                    compressed.extend(sorted_nodes[last_start:last_start + pattern_len])
                    
                    # Move index past all repetitions
                    i = i + repeat_count * pattern_len
                    break
            
            if not pattern_found:
                compressed.append(sorted_nodes[i])
                i += 1
        
        return compressed
    
    def _node_fingerprint(self, node: AtomicNode) -> str:
        """
        Create a fingerprint for pattern detection.
        
        Uses type and role as the fingerprint.
        """
        return f"{node.type}:{node.role}"
    
    def _collapse_chains(self, nodes: List[AtomicNode], subgraph: nx.DiGraph) -> List[AtomicNode]:
        """
        Collapse chains of similar nodes (INTENT/INFO) into single nodes.
        """
        if len(nodes) < 3: return nodes
        
        nodes.sort(key=lambda n: (n.step_id, n.node_id))
        
        final_nodes = []
        chain = []
        
        for node in nodes:
            if not chain:
                chain.append(node)
                continue
                
            last = chain[-1]
            if (node.type == last.type and 
                node.type in ("INTENT", "INFO") and 
                (node.step_id - last.step_id <= 1)): 
                chain.append(node)
            else:
                final_nodes.extend(self._process_chain(chain, subgraph))
                chain = [node]
                
        if chain:
            final_nodes.extend(self._process_chain(chain, subgraph))
            
        return final_nodes

    def _process_chain(self, chain: List[AtomicNode], subgraph: nx.DiGraph) -> List[AtomicNode]:
        """
        Process a chain of similar nodes to decide which to keep.
        """
        if len(chain) <= 2: return chain
        
        kept = [chain[0]] 
        
        for mid_node in chain[1:-1]:
            has_secondary = False
            if mid_node.node_id in subgraph:
                 for u, v, d in subgraph.in_edges(mid_node.node_id, data=True):
                    if d.get("layer") == "SECONDARY": 
                        has_secondary = True; break
            
            if has_secondary:
                kept.append(mid_node)
        
        kept.append(chain[-1]) 
        return kept


# =============================================================================
# Legacy Compatibility: GraphPruner wrapper
# =============================================================================

class GraphPruner:
    """
    Legacy compatibility wrapper.
    
    Wraps CausalGraphSlicer for backward compatibility with old API.
    """
    
    def __init__(
        self,
        max_depth: int = 30,
        max_cost: float = 50.0,
        **kwargs,
    ):
        self.slicer = CausalGraphSlicer(
            max_depth=max_depth,
            max_cost=max_cost,
        )
    
    def prune_graph(
        self,
        graph: nx.DiGraph,
        target_node_id: str,
        **kwargs,
    ) -> List[AtomicNode]:
        """
        Prune graph to relevant nodes.
        
        Returns:
            List of AtomicNode objects
        """
        return self.slicer.slice(graph, target_node_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        return self.slicer.get_stats()
