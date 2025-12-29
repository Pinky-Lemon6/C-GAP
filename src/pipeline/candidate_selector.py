"""Candidate Selector for Atomic Node-based Causal Graph Building.

This module provides candidate selection strategies for finding potential
causal parents of a target AtomicNode.

Key Features:
- Dual-track selection (Rule + Semantic)
- Type-aware scoring with causal pattern bonuses
- Embedding-based semantic similarity
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict

import numpy as np

from src.models import AtomicNode
from src.pipeline.causal_types import NodeType, VALID_CAUSAL_SOURCES


# =============================================================================
# Stop Words for Lexical Processing
# =============================================================================

STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'been', 'being', 'having', 'doing',
}


# =============================================================================
# Type Bonus Patterns
# =============================================================================

# Common causal patterns that deserve score boosts
# (source_type, target_type) -> bonus score
TYPE_BONUS_PATTERNS: Dict[Tuple[str, str], float] = {
    # Observation -> Thought: INFO informs next decision
    (NodeType.INFO.value, NodeType.INTENT.value): 0.15,
    
    # Decision -> Action: INTENT leads to EXEC
    (NodeType.INTENT.value, NodeType.EXEC.value): 0.20,
    
    # Action -> Result: EXEC produces INFO
    (NodeType.EXEC.value, NodeType.INFO.value): 0.20,
    
    # Message -> Decision: COMM triggers INTENT
    (NodeType.COMM.value, NodeType.INTENT.value): 0.15,
    
    # Result -> Communication: INFO leads to COMM
    (NodeType.INFO.value, NodeType.COMM.value): 0.10,
    
    # Thought chain: INTENT -> INTENT
    (NodeType.INTENT.value, NodeType.INTENT.value): 0.05,
}


# =============================================================================
# Evidence Signals
# =============================================================================

@dataclass
class NodeEvidenceSignals:
    """Evidence signals between two AtomicNodes."""
    
    lexical_overlap: float = 0.0
    shared_words: List[str] = field(default_factory=list)
    temporal_distance: int = 0
    same_role: bool = False
    type_bonus: float = 0.0
    is_valid_causal_pair: bool = True
    
    def rule_score(self) -> float:
        """
        Calculate rule-based score for candidate ranking.
        
        Components:
        - Temporal distance (closer = better): 40%
        - Lexical overlap: 25%
        - Type bonus (causal pattern): 25%
        - Same role: 10%
        """
        score = 0.0
        
        # Temporal distance (closer nodes more likely causally related)
        # Max distance considered: 20 steps
        score += max(0, 1 - self.temporal_distance / 20) * 0.40
        
        # Lexical overlap (shared content suggests relationship)
        score += min(self.lexical_overlap, 1.0) * 0.25
        
        # Type bonus (known causal patterns)
        score += self.type_bonus * 0.25
        
        # Same role bonus (agent's own actions often chain)
        if self.same_role:
            score += 0.10
        
        return score


# =============================================================================
# Node Evidence Extractor
# =============================================================================

class NodeEvidenceExtractor:
    """Extract evidence signals between AtomicNodes."""
    
    def extract(self, source: AtomicNode, target: AtomicNode) -> NodeEvidenceSignals:
        """
        Extract evidence signals between source and target nodes.
        
        Args:
            source: Potential cause node
            target: Effect node
            
        Returns:
            NodeEvidenceSignals with computed features
        """
        source_content = source.content or ""
        target_content = target.content or ""
        
        # Lexical overlap
        lexical_overlap, shared_words = self._compute_lexical_overlap(
            source_content, target_content
        )
        
        # Temporal distance (using step_id as proxy)
        temporal_distance = abs(target.step_id - source.step_id)
        
        # Role comparison
        same_role = (source.role or "").lower() == (target.role or "").lower()
        
        # Type bonus
        type_bonus = self._compute_type_bonus(source.type, target.type)
        
        # Valid causal pair check
        is_valid = self._is_valid_causal_pair(source.type, target.type)
        
        return NodeEvidenceSignals(
            lexical_overlap=lexical_overlap,
            shared_words=shared_words,
            temporal_distance=temporal_distance,
            same_role=same_role,
            type_bonus=type_bonus,
            is_valid_causal_pair=is_valid,
        )
    
    def _compute_lexical_overlap(
        self, 
        source: str, 
        target: str
    ) -> Tuple[float, List[str]]:
        """Compute lexical overlap ratio between two texts."""
        source_words = self._tokenize(source)
        target_words = self._tokenize(target)
        
        if not source_words or not target_words:
            return 0.0, []
        
        shared = source_words & target_words
        
        # Overlap relative to target (what portion of target is explained by source)
        overlap_ratio = len(shared) / len(target_words)
        
        return overlap_ratio, list(shared)[:10]
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text and filter stopwords."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return words - STOPWORDS
    
    def _compute_type_bonus(
        self, 
        source_type: str, 
        target_type: str
    ) -> float:
        """
        Compute type bonus based on known causal patterns.
        
        Args:
            source_type: NodeType value of source
            target_type: NodeType value of target
            
        Returns:
            Bonus score (0.0 - 0.2)
        """
        # Normalize type values
        src_type = source_type.upper() if isinstance(source_type, str) else source_type
        tgt_type = target_type.upper() if isinstance(target_type, str) else target_type
        
        return TYPE_BONUS_PATTERNS.get((src_type, tgt_type), 0.0)
    
    def _is_valid_causal_pair(
        self, 
        source_type: str, 
        target_type: str
    ) -> bool:
        """
        Check if source->target is a valid causal relationship per type constraints.
        
        Args:
            source_type: NodeType value of source
            target_type: NodeType value of target
            
        Returns:
            True if valid according to VALID_CAUSAL_SOURCES
        """
        try:
            src_node_type = NodeType(source_type.upper())
            tgt_node_type = NodeType(target_type.upper())
            valid_sources = VALID_CAUSAL_SOURCES.get(tgt_node_type, set())
            return src_node_type in valid_sources
        except (ValueError, AttributeError):
            # If types can't be parsed, allow the pair (let LLM decide)
            return True


# =============================================================================
# Dual-Track Candidate Selector for AtomicNodes
# =============================================================================

class AtomicNodeCandidateSelector:
    """
    Dual-track candidate selector for AtomicNodes.
    
    Design:
    - Track A (Rule): Captures explicit causality via lexical overlap + type patterns
    - Track B (Semantic): Captures implicit causality via embedding similarity
    
    Solves:
    - Pure rule selection misses "no overlap but causal" pairs
    - Semantic similarity identifies implicit relations (e.g., "search weather" → "25°C")
    """
    
    def __init__(
        self,
        rule_candidates: int = 3,
        semantic_candidates: int = 2,
        use_embeddings: bool = True,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        filter_invalid_types: bool = True,
    ):
        """
        Initialize the selector.
        
        Args:
            rule_candidates: Number of candidates from rule track
            semantic_candidates: Number of candidates from semantic track
            use_embeddings: Whether to use embedding-based selection
            embedding_model: SentenceTransformer model name
            filter_invalid_types: Whether to filter out invalid causal type pairs
        """
        self.rule_k = rule_candidates
        self.semantic_k = semantic_candidates
        self.use_embeddings = use_embeddings
        self.filter_invalid_types = filter_invalid_types
        
        self.evidence_extractor = NodeEvidenceExtractor()
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._embedder = None
        
        if use_embeddings:
            self._init_embedder(embedding_model)
    
    def _init_embedder(self, model_name: str):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(model_name)
        except ImportError:
            print("Warning: sentence-transformers not installed. "
                  "Falling back to rule-only selection.")
            self.use_embeddings = False
    
    def select(
        self, 
        target: AtomicNode, 
        candidates: List[AtomicNode]
    ) -> List[AtomicNode]:
        """
        Select candidate parent nodes for a target using dual-track approach.
        
        Args:
            target: The target node to find parents for
            candidates: List of potential parent nodes
            
        Returns:
            De-duplicated list of selected candidates
        """
        if not candidates:
            return []
        
        # Optionally filter by valid causal types
        if self.filter_invalid_types:
            candidates = self._filter_valid_candidates(target, candidates)
            if not candidates:
                return []
        
        selected_ids: Set[str] = set()
        selected: List[AtomicNode] = []
        
        # === Track A: Rule-based scoring ===
        rule_scored = self._rule_score_all(target, candidates)
        for candidate, score in rule_scored[:self.rule_k]:
            if candidate.node_id not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate.node_id)
        
        # === Track B: Semantic similarity ===
        if self.use_embeddings and self._embedder is not None:
            semantic_scored = self._semantic_score_all(target, candidates)
            for candidate, score in semantic_scored:
                if candidate.node_id not in selected_ids:
                    selected.append(candidate)
                    selected_ids.add(candidate.node_id)
                    if len(selected) >= self.rule_k + self.semantic_k:
                        break
        
        # Backfill from rule track if semantic track insufficient
        if len(selected) < self.rule_k + self.semantic_k:
            for candidate, score in rule_scored[self.rule_k:]:
                if candidate.node_id not in selected_ids:
                    selected.append(candidate)
                    selected_ids.add(candidate.node_id)
                    if len(selected) >= self.rule_k + self.semantic_k:
                        break
        
        return selected
    
    def _filter_valid_candidates(
        self, 
        target: AtomicNode, 
        candidates: List[AtomicNode]
    ) -> List[AtomicNode]:
        """Filter candidates to only valid causal type pairs."""
        valid = []
        for cand in candidates:
            if self.evidence_extractor._is_valid_causal_pair(cand.type, target.type):
                valid.append(cand)
        return valid
    
    def _rule_score_all(
        self, 
        target: AtomicNode, 
        candidates: List[AtomicNode]
    ) -> List[Tuple[AtomicNode, float]]:
        """Score all candidates using rule-based features."""
        scored = []
        
        for source in candidates:
            evidence = self.evidence_extractor.extract(source, target)
            score = evidence.rule_score()
            scored.append((source, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _semantic_score_all(
        self, 
        target: AtomicNode, 
        candidates: List[AtomicNode]
    ) -> List[Tuple[AtomicNode, float]]:
        """Score all candidates using semantic similarity."""
        if not self._embedder:
            return []
        
        target_emb = self._get_embedding(target)
        
        scored = []
        for source in candidates:
            source_emb = self._get_embedding(source)
            similarity = self._cosine_similarity(source_emb, target_emb)
            
            # === Correction 1: Type Bonus ===
            bonus = 0.0
            s_type, t_type = source.type, target.type
            
            # INFO -> INTENT/COMM/EXEC
            if s_type == "INFO" and t_type in ["INTENT", "COMM", "EXEC"]:
                bonus = 0.15
            # INTENT -> EXEC 
            elif s_type == "INTENT" and t_type == "EXEC":
                bonus = 0.20
            # EXEC -> INFO 
            elif s_type == "EXEC" and t_type == "INFO":
                bonus = 0.10
            
            # === Correction 2: Gentle Temporal Decay ===
            dist = abs(target.step_id - source.step_id)
            time_penalty = 0.0005 * dist  
            
            final_score = similarity + bonus - time_penalty
            scored.append((source, final_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _get_embedding(self, node: AtomicNode) -> np.ndarray:
        """Get embedding for a node (with caching)."""
        if node.node_id not in self._embedding_cache:
            # Embed the content field (summarized, de-noised)
            text = (node.content or "")[:1024]
            self._embedding_cache[node.node_id] = self._embedder.encode(text)
        return self._embedding_cache[node.node_id]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def precompute_embeddings(self, nodes: List[AtomicNode]):
        """Pre-compute embeddings for all nodes."""
        if not self._embedder:
            return
        for node in nodes:
            _ = self._get_embedding(node)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()


# =============================================================================
# Simple Selector (Fallback)
# =============================================================================

class SimpleAtomicNodeSelector:
    """
    Simple rule-based selector for AtomicNodes (no embeddings).
    
    Use as fallback when sentence-transformers is not available.
    """
    
    def __init__(
        self, 
        max_candidates: int = 5,
        filter_invalid_types: bool = True,
    ):
        self.max_candidates = max_candidates
        self.filter_invalid_types = filter_invalid_types
        self.evidence_extractor = NodeEvidenceExtractor()
    
    def select(
        self, 
        target: AtomicNode, 
        candidates: List[AtomicNode]
    ) -> List[AtomicNode]:
        """Select candidates using rule-based scoring only."""
        if not candidates:
            return []
        
        # Filter by type if enabled
        if self.filter_invalid_types:
            candidates = [
                c for c in candidates
                if self.evidence_extractor._is_valid_causal_pair(c.type, target.type)
            ]
        
        if not candidates:
            return []
        
        scored = []
        for source in candidates:
            evidence = self.evidence_extractor.extract(source, target)
            score = evidence.rule_score()
            scored.append((source, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:self.max_candidates]]
