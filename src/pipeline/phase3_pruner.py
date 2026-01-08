"""Phase III:  Causal Graph Slicer

Deterministic, algorithm-based graph slicing for failure attribution. 
NO LLM calls - pure graph algorithms for reproducibility and speed. 

Key Features:
1. Backward Reachability:  Priority-queue based traversal from error node
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
    "PRIMARY":  0.1,      # Backbone/control flow - highest priority
    "SECONDARY": 1.0,    # Associative/data flow - medium priority
    "FALLBACK": 5.0,     # Temporal fallback - low priority
}

# Default cost for unknown edge types
DEFAULT_EDGE_COST = 1.0


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
    
    subgraph: nx. DiGraph
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
    
    v2 Improvements:
    - Ensure full reachability via relaxed cost bounds
    - Aggressive filtering AFTER full traversal
    - Enhanced compression for INFO chains
    """
    
    def __init__(
        self,
        max_depth: int = 50,           # Increased from 30
        max_cost: float = 100.0,       # Increased from 50. 0
        high_degree_threshold:  int = 6,
        enable_loop_compression: bool = True,
        loop_repeat_threshold: int = 3,
    ):
        """
        Initialize the slicer.
        
        Args: 
            max_depth:  Maximum traversal depth (hops from target)
            max_cost: Maximum cumulative edge cost
            high_degree_threshold:  Nodes with out-degree > this are kept
            enable_loop_compression: Whether to compress repetitive patterns
            loop_repeat_threshold: Compress if pattern repeats > this many times
        """
        self.max_depth = max_depth
        self.max_cost = max_cost
        self. high_degree_threshold = high_degree_threshold
        self. enable_loop_compression = enable_loop_compression
        self. loop_repeat_threshold = loop_repeat_threshold
        
        self._stats:  Dict[str, Any] = {}
    
    def slice(
        self,
        graph: nx. DiGraph,
        target_node_id: str,
        root_node_id:  Optional[str] = None,
    ) -> List[AtomicNode]: 
        """
        Slice the graph to extract relevant causal context.
        
        Args:
            graph: The full causal graph
            target_node_id:  ID of the target (error) node
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
        
        # Step 1: Backward reachability - get ALL reachable nodes first
        reachable_ids = self._backward_slice(
            graph=graph,
            target_node_id=target_node_id,
        )
        self._stats["backward_reachable"] = len(reachable_ids)
        
        # Always include target and root
        reachable_ids. add(target_node_id)
        if root_node_id and root_node_id in graph:
            reachable_ids.add(root_node_id)
        
        # Step 2: Extract nodes from graph
        nodes = self._extract_nodes(graph, reachable_ids)
        self._stats["extracted_nodes"] = len(nodes)
        
        # Step 3: Build subgraph for filtering decisions
        subgraph = graph.subgraph(reachable_ids).copy()
        
        # Step 4: Apply AGGRESSIVE semantic filters (this is where we compress)
        filtered_nodes = self._apply_filters(
            nodes=nodes,
            subgraph=subgraph,
            target_node_id=target_node_id,
            root_node_id=root_node_id,
        )
        self._stats["after_filtering"] = len(filtered_nodes)
        
        # Step 5: Chain collapse for additional compression
        filtered_nodes = self._collapse_chains(filtered_nodes, subgraph)
        self._stats["after_chain_collapse"] = len(filtered_nodes)
        
        # Step 6: Loop compression (optional)
        if self.enable_loop_compression:
            filtered_nodes = self._compress_loops(filtered_nodes)
            self._stats["after_compression"] = len(filtered_nodes)
        
        # Sort by step_id
        filtered_nodes.sort(key=lambda n: (n.step_id, n.node_id))
        
        return filtered_nodes
    
    def slice_full(
        self,
        graph: nx. DiGraph,
        target_node_id: str,
        root_node_id: Optional[str] = None,
    ) -> SliceResult:
        """
        Slice and return full result with subgraph and stats.
        """
        nodes = self.slice(graph, target_node_id, root_node_id)
        node_ids = {n.node_id for n in nodes}
        subgraph = graph. subgraph(node_ids).copy()
        
        return SliceResult(
            nodes=nodes,
            node_ids=node_ids,
            subgraph=subgraph,
            stats=self._stats. copy(),
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
            "after_chain_collapse":  0,
            "after_compression":  0,
            "dropped_isolated_exec": 0,
            "dropped_trivial_info": 0,
            "compressed_loops": 0,
        }
    
    # =========================================================================
    # Step 1: Backward Reachability (Full traversal, no early cutoff)
    # =========================================================================
    
    def _backward_slice(self, graph: nx. DiGraph, target_node_id: str) -> Set[str]:
        """
        Full backward reachability with relaxed bounds.
        
        Key change: Use simple BFS first to ensure we don't miss nodes,
        then use weighted traversal for ranking.
        """
        # Phase A: Simple BFS to get ALL reachable ancestors
        all_ancestors = nx.ancestors(graph, target_node_id)
        all_ancestors. add(target_node_id)
        
        self._stats["total_ancestors"] = len(all_ancestors)
        
        # If graph is small enough, just return all ancestors
        if len(all_ancestors) <= 100:
            return all_ancestors
        
        # Phase B: For large graphs, use weighted selection
        reachable:  Set[str] = set()
        pq:  List[Tuple[float, int, str]] = [(0.0, 0, target_node_id)]
        best_cost:  Dict[str, float] = {target_node_id: 0.0}
        
        while pq: 
            cost, depth, node_id = heapq.heappop(pq)
            
            if cost > self.max_cost:
                continue
            if depth > self.max_depth:
                continue
            if cost > best_cost. get(node_id, float('inf')):
                continue
            
            reachable.add(node_id)
            
            for pred_id in graph. predecessors(node_id):
                edge_data = graph. get_edge_data(pred_id, node_id, default={})
                
                base_cost = self._get_edge_cost(edge_data)
                edge_cost = base_cost
                
                # Mild decay for PRIMARY edges
                if edge_data.get("layer") == "PRIMARY":
                    edge_cost = base_cost * (1.02 ** depth)  # Very mild decay
                
                new_cost = cost + edge_cost
                new_depth = depth + 1
                
                if new_cost < best_cost. get(pred_id, float('inf')):
                    best_cost[pred_id] = new_cost
                    heapq. heappush(pq, (new_cost, new_depth, pred_id))
        
        return reachable
    
    def _get_edge_cost(self, edge_data: Dict[str, Any]) -> float:
        """
        Get traversal cost for an edge.
        """
        is_fallback = edge_data.get("is_fallback", False)
        if is_fallback: 
            return EDGE_COSTS["FALLBACK"]
        
        layer = edge_data. get("layer", "SECONDARY")
        if layer == "PRIMARY": 
            return EDGE_COSTS["PRIMARY"]
        
        causal_type = edge_data.get("causal_type", "")
        if causal_type in ("INSTRUCTION", "DATA"):
            return EDGE_COSTS["PRIMARY"]
        elif causal_type in ("SEQUENTIAL",):
            return 0.5
        
        return EDGE_COSTS. get(layer, DEFAULT_EDGE_COST)
    
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
            
            data = graph. nodes[node_id]
            
            node = AtomicNode(
                node_id=node_id,
                step_id=data. get("step_id", 0),
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
        roots = [n for n in graph. nodes() if graph.in_degree(n) == 0]
        
        if not roots: 
            nodes_with_step = [
                (n, graph.nodes[n]. get("step_id", float('inf')))
                for n in graph.nodes()
            ]
            if nodes_with_step:
                nodes_with_step. sort(key=lambda x: x[1])
                return nodes_with_step[0][0]
            return None
        
        roots_with_step = [
            (n, graph.nodes[n]. get("step_id", float('inf')))
            for n in roots
        ]
        roots_with_step.sort(key=lambda x:  x[1])
        return roots_with_step[0][0]
    
    # =========================================================================
    # Step 3: Semantic Filtering (AGGRESSIVE for compression)
    # =========================================================================
    
    def _apply_filters(
        self,
        nodes:  List[AtomicNode],
        subgraph: nx. DiGraph,
        target_node_id: str,
        root_node_id: Optional[str],
    ) -> List[AtomicNode]:
        """
        Apply semantic filters - now MORE aggressive since we have full reachability. 
        
        Strategy: 
        1.  Protect anchor nodes (target, root, error-containing, INTENT)
        2. Protect structural hubs
        3. Aggressively filter INFO nodes that are "pass-through"
        4. Filter EXEC nodes without meaningful output
        """
        filtered = []
        
        in_degrees = dict(subgraph. in_degree())
        out_degrees = dict(subgraph.out_degree())
        
        for node in nodes: 
            keep, reason = self._should_keep_node(
                node=node,
                subgraph=subgraph,
                target_node_id=target_node_id,
                root_node_id=root_node_id,
                in_degree=in_degrees. get(node.node_id, 0),
                out_degree=out_degrees.get(node. node_id, 0),
            )
            
            if keep:
                filtered. append(node)
            else:
                if node.type == "EXEC":
                    self._stats["dropped_isolated_exec"] += 1
                elif node.type == "INFO":
                    self._stats["dropped_trivial_info"] += 1
        
        return filtered
    
    def _should_keep_node(
        self,
        node: AtomicNode,
        subgraph:  nx.DiGraph,
        target_node_id: str,
        root_node_id: Optional[str],
        in_degree: int,
        out_degree: int,
    ) -> Tuple[bool, str]: 
        """
        Decide whether to keep a node - BALANCED for recall + compression.
        """
        # === Anchor Rules (never drop) ===
        if node.node_id == target_node_id:
            return True, "target_node"
        if root_node_id and node.node_id == root_node_id:
            return True, "root_node"
        
        content_lower = (node.content or "").lower()
        
        # Keep nodes with error indicators
        if self._contains_error_keywords(content_lower):
            return True, "error_content"
        
        # === INTENT:  Always keep (decisions are important) ===
        if node.type == "INTENT": 
            return True, "intent"
        
        # === Compute meaningful Inter-Step edge degrees ===
        # Only count INTER_STEP edges (PRIMARY or SECONDARY layer)
        # Exclude INTRA_STEP SEQUENTIAL edges which every node has
        meaningful_in = 0
        meaningful_out = 0
        if node.node_id in subgraph: 
            for pred in subgraph.predecessors(node.node_id):
                edge_data = subgraph.get_edge_data(pred, node.node_id)
                if edge_data:
                    edge_type = edge_data.get("edge_type", "")
                    layer = edge_data.get("layer", "")
                    # Inter-Step edges with PRIMARY or SECONDARY layer are meaningful
                    if edge_type == "INTER_STEP" and layer in ("PRIMARY", "SECONDARY"):
                        meaningful_in += 1
            for succ in subgraph.successors(node.node_id):
                edge_data = subgraph.get_edge_data(node.node_id, succ)
                if edge_data:
                    edge_type = edge_data.get("edge_type", "")
                    layer = edge_data.get("layer", "")
                    if edge_type == "INTER_STEP" and layer in ("PRIMARY", "SECONDARY"):
                        meaningful_out += 1
        
        # === Structural Hub Protection ===
        # Keep nodes that are important junctions (high total degree + meaningful edges)
        if (in_degree + out_degree > self.high_degree_threshold) and (meaningful_in + meaningful_out > 0):
            return True, "structural_hub"
        
        # === INFO Filtering (most aggressive) ===
        if node.type == "INFO": 
            # Drop if no meaningful outgoing edges (doesn't causally influence anything)
            if meaningful_out == 0:
                return False, "dead_end_info"
            return True, "useful_info"
        
        # === EXEC Filtering ===
        if node.type == "EXEC":
            # Drop if no meaningful outgoing edges (execution had no tracked effect)
            if meaningful_out == 0:
                return False, "ineffectual_exec"
            return True, "useful_exec"
        
        # === COMM Filtering ===
        if node.type == "COMM": 
            # Drop if no meaningful outgoing edges (message didn't influence anything)
            if meaningful_out == 0:
                return False, "routing_comm"
            return True, "useful_comm"
        
        return True, "default_keep"
    
    def _contains_error_keywords(self, text: str) -> bool:
        """Check if text contains error-indicating keywords."""
        return any(kw in text for kw in ERROR_KEYWORDS)
    
    # =========================================================================
    # Step 4: Chain Collapse (Additional Compression)
    # =========================================================================
    
    def _collapse_chains(
        self, 
        nodes: List[AtomicNode], 
        subgraph:  nx.DiGraph
    ) -> List[AtomicNode]: 
        """
        Collapse chains of similar consecutive nodes.
        
        Focus on INFO chains which are most compressible.
        """
        if len(nodes) < 3:
            return nodes
        
        nodes. sort(key=lambda n: (n. step_id, n.node_id))
        
        final_nodes = []
        chain:  List[AtomicNode] = []
        
        for node in nodes:
            if not chain:
                chain.append(node)
                continue
            
            last = chain[-1]
            
            # Chain continuation conditions
            same_type = node.type == last. type
            close_step = (node.step_id - last.step_id) <= 1
            collapsible_type = node.type in ("INFO",)  # Only INFO chains
            
            if same_type and close_step and collapsible_type: 
                chain.append(node)
            else: 
                final_nodes. extend(self._process_chain(chain, subgraph))
                chain = [node]
        
        if chain:
            final_nodes.extend(self._process_chain(chain, subgraph))
        
        return final_nodes
    
    def _process_chain(
        self, 
        chain: List[AtomicNode], 
        subgraph: nx.DiGraph
    ) -> List[AtomicNode]:
        """
        Process a chain:  keep first, last, and important middle nodes.
        """
        if len(chain) <= 2:
            return chain
        
        kept = [chain[0]]  # First
        
        for mid_node in chain[1:-1]:
            # Keep if has SECONDARY edges
            has_secondary = False
            if mid_node.node_id in subgraph:
                for u, v, d in subgraph. in_edges(mid_node.node_id, data=True):
                    if d. get("layer") == "SECONDARY": 
                        has_secondary = True
                        break
                if not has_secondary: 
                    for u, v, d in subgraph. out_edges(mid_node.node_id, data=True):
                        if d.get("layer") == "SECONDARY": 
                            has_secondary = True
                            break
            
            # Keep if contains error keywords
            has_error = self._contains_error_keywords((mid_node.content or "").lower())
            
            if has_secondary or has_error:
                kept.append(mid_node)
        
        kept.append(chain[-1])  # Last
        
        return kept
    
    # =========================================================================
    # Step 5: Loop Compression
    # =========================================================================
    
    def _compress_loops(
        self,
        nodes: List[AtomicNode],
    ) -> List[AtomicNode]: 
        """
        Compress repetitive execution patterns. 
        """
        if len(nodes) < self.loop_repeat_threshold * 2:
            return nodes
        
        sorted_nodes = sorted(nodes, key=lambda n: (n. step_id, n.node_id))
        fingerprints = [self._node_fingerprint(n) for n in sorted_nodes]
        
        compressed = []
        i = 0
        
        while i < len(sorted_nodes):
            pattern_found = False
            
            for pattern_len in range(2, min(6, (len(sorted_nodes) - i) // 2 + 1)):
                pattern = fingerprints[i: i + pattern_len]
                repeat_count = 1
                
                j = i + pattern_len
                while j + pattern_len <= len(fingerprints):
                    if fingerprints[j: j + pattern_len] == pattern: 
                        repeat_count += 1
                        j += pattern_len
                    else: 
                        break
                
                if repeat_count > self. loop_repeat_threshold:
                    pattern_found = True
                    
                    compressed.extend(sorted_nodes[i: i + pattern_len])
                    
                    compressed_node = AtomicNode(
                        node_id=f"compressed_{sorted_nodes[i]. step_id}",
                        step_id=sorted_nodes[i].step_id,
                        role="system",
                        type="COMPRESSED",
                        content=f"[Loop compressed: {repeat_count - 2} iterations of {pattern_len}-node pattern omitted]",
                    )
                    compressed.append(compressed_node)
                    self._stats["compressed_loops"] += 1
                    
                    last_start = i + (repeat_count - 1) * pattern_len
                    compressed. extend(sorted_nodes[last_start:last_start + pattern_len])
                    
                    i = i + repeat_count * pattern_len
                    break
            
            if not pattern_found: 
                compressed.append(sorted_nodes[i])
                i += 1
        
        return compressed
    
    def _node_fingerprint(self, node:  AtomicNode) -> str:
        """Create a fingerprint for pattern detection."""
        return f"{node.type}:{node.role}"


# =============================================================================
# Legacy Compatibility:  GraphPruner wrapper
# =============================================================================

class GraphPruner:
    """
    Legacy compatibility wrapper. 
    """
    
    def __init__(
        self,
        max_depth: int = 50,
        max_cost:  float = 100.0,
        **kwargs,
    ):
        self.slicer = CausalGraphSlicer(
            max_depth=max_depth,
            max_cost=max_cost,
        )
    
    def prune_graph(
        self,
        graph: nx.DiGraph,
        target_node_id:  str,
        **kwargs,
    ) -> List[AtomicNode]:
        """Prune graph to relevant nodes."""
        return self.slicer.slice(graph, target_node_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        return self.slicer. get_stats()