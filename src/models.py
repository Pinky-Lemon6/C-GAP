"""Core data models for C-GAP.

This module defines:
- StandardLogItem: normalized representation of a single agent log step
- DependencyGraph: lightweight wrapper around a NetworkX directed graph
"""

from __future__ import annotations
from enum import Enum
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple,NamedTuple, Set

try:
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "pydantic is required for src/models.py. Install with: pip install pydantic"
    ) from exc

# Import NodeType for AtomicNode (lazy import to avoid circular dependency)
try:
    from src.pipeline.causal_types import NodeType
except ImportError:
    # Fallback: define NodeType locally if causal_types not available
    class NodeType(str, Enum):
        INTENT = "INTENT"
        EXEC = "EXEC"
        INFO = "INFO"
        COMM = "COMM"


class AtomicNode(BaseModel):
    """
    Atomic Node for fine-grained causal graph.
    
    Each StandardLogItem can be decomposed into multiple AtomicNodes,
    enabling more precise causal relationship tracking.
    """
    
    node_id: str
    """Unique ID (e.g., 'step_1_0', 'step_1_1')"""
    
    step_id: int
    """Reference to original step"""
    
    role: str
    """Role of the agent"""
    
    type: NodeType
    """The atomic type (INTENT, EXEC, INFO, COMM)"""
    
    content: str
    """Summarized content (de-noised)"""
    
    original_text: Optional[str] = None
    """Optional reference to raw text segment"""
    
    class Config:
        use_enum_values = True


class StandardLogItem(BaseModel):
    """A standardized representation of one log step for downstream pipeline phases."""

    dataset_source: str
    session_id: str
    step_id: int
    role: str
    raw_content: str

    # Used in graph construction (relations, parent/child, causal hints, etc.)
    topology_labels: Dict[str, Any] = Field(default_factory=dict)

    # Used in pruning (scores, keep/drop decisions, confidence, etc.)
    pruning_labels: Dict[str, Any] = Field(default_factory=dict)
    
    # Atomic nodes decomposed from this log step (for node-based causal graph)
    atomic_nodes: List["AtomicNode"] = Field(default_factory=list)


class DependencyGraph:
    """Directed dependency graph wrapper using NetworkX internally."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or "dependency_graph"
        try:
            import networkx as nx
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "networkx is required for DependencyGraph. Install with: pip install networkx"
            ) from exc

        self._nx = nx
        self._g = nx.DiGraph(name=self.name)

    @property
    def graph(self):
        """Access the underlying NetworkX DiGraph (advanced use)."""

        return self._g

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self._g.add_node(node_id, **attrs)

    def add_edge(self, src: str, dst: str, **attrs: Any) -> None:
        self._g.add_edge(src, dst, **attrs)

    def nodes(self) -> List[str]:
        return list(self._g.nodes)

    def edges(self) -> List[Tuple[str, str]]:
        return list(self._g.edges)

    def to_edge_list(self) -> List[Dict[str, Any]]:
        """Serialize edges as a list of dicts suitable for JSON."""

        out: List[Dict[str, Any]] = []
        for u, v, data in self._g.edges(data=True):
            out.append({"src": u, "dst": v, **(data or {})})
        return out

    def to_node_link_data(self) -> Dict[str, Any]:
        """Serialize graph in NetworkX node-link format (JSON-friendly)."""

        from networkx.readwrite import json_graph

        return json_graph.node_link_data(self._g)

    @classmethod
    def from_node_link_data(cls, data: Dict[str, Any], name: str | None = None) -> "DependencyGraph":
        from networkx.readwrite import json_graph

        obj = cls(name=name)
        obj._g = json_graph.node_link_graph(data, directed=True)
        return obj



class CausalType(Enum):
    """
    因果关系类型
    
    设计原则：
    1. 类别边界清晰，标注一致性高
    2. 覆盖故障归因所需的核心因果类型
    3. 对 SFT 小模型友好
    """
    
    INSTRUCTION = "INSTRUCTION"
    """
    指令/控制流因果
    
    定义：Source 发起请求/命令/任务，Target 响应/执行
    示例：
    - "Please search for X" → "I searched for X"
    - "What is the weather?" → "The temperature is 25°C"
    """
    
    DATA = "DATA"
    """
    数据/信息流因果
    
    定义：Target 使用了 Source 产生的信息/数据/结论
    示例：
    - "Search results show:  [list]" → "Based on results, clicking option A"
    - "I found that X=5" → "Since X=5, we should..."
    """
    
    STATE = "STATE"
    """
    状态/条件因果
    
    定义：Target 的执行依赖于 Source 建立的状态或条件
    示例：
    - "Login successful" → "Accessing user dashboard"
    - "Error:  connection failed" → "Retrying with different approach"
    """
    
    NONE = "NONE"
    """
    无因果关系
    
    定义：Target 的发生与 Source 无关
    判断标准：即使 Source 没有发生，Target 仍会以相同方式发生
    """


class CausalResult(NamedTuple):
    """因果推理结果"""
    causal_type: CausalType
    confidence: float
    reason: str


class CausalEdge(NamedTuple):
    """因果边"""
    source_id: int
    target_id: int
    causal_type: CausalType
    confidence: float
    reason: str
    edge_type: str = "primary"  # primary, skip, repair, fallback
    
    
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
    'am', 'been', 'being', 'having', 'doing', 'would', 'should', 'could',
}


@dataclass
class EvidenceSignals:
    """证据信号集合"""
    lexical_overlap: float = 0.0
    entity_overlap: float = 0.0
    shared_entities: List[str] = field(default_factory=list)
    shared_words: List[str] = field(default_factory=list)
    temporal_distance: int = 0
    same_sender: bool = False
    
    def rule_score(self) -> float:
        """计算规则评分（用于候选排序）"""
        score = 0.0
        
        # 时序距离（越近越好）
        score += max(0, 1 - self.temporal_distance / 15) * 0.35
        
        # 实体重叠
        score += self. entity_overlap * 0.35
        
        # 词汇重叠
        score += self.lexical_overlap * 0.20
        
        # 发送者相同
        if self.same_sender:
            score += 0.10
        
        return score


class EvidenceExtractor:
    """证据提取器"""
    
    def extract(
        self, 
        source:  StandardLogItem, 
        target: StandardLogItem
    ) -> EvidenceSignals: 
        """提取两个消息之间的证据信号"""
        source_content = source.raw_content or ""
        target_content = target.raw_content or ""
        
        # 词汇重叠
        lexical_overlap, shared_words = self._compute_lexical_overlap(
            source_content, target_content
        )
        
        # 实体重叠
        entity_overlap, shared_entities = self._compute_entity_overlap(
            source_content, target_content
        )
        
        # 时序距离
        temporal_distance = target.step_id - source.step_id
        
        # 发送者
        same_sender = (source.role or "").lower() == (target.role or "").lower()
        
        return EvidenceSignals(
            lexical_overlap=lexical_overlap,
            entity_overlap=entity_overlap,
            shared_entities=shared_entities,
            shared_words=shared_words,
            temporal_distance=temporal_distance,
            same_sender=same_sender,
        )
    
    def _compute_lexical_overlap(
        self, 
        source:  str, 
        target: str
    ) -> Tuple[float, List[str]]:
        """计算词汇重叠度"""
        source_words = self._tokenize(source)
        target_words = self._tokenize(target)
        
        if not source_words or not target_words: 
            return 0.0, []
        
        shared = source_words & target_words
        
        # 相对于目标的重叠比例
        overlap_ratio = len(shared) / len(target_words) if target_words else 0.0
        
        return overlap_ratio, list(shared)[:10]
    
    def _compute_entity_overlap(
        self, 
        source:  str, 
        target: str
    ) -> Tuple[float, List[str]]:
        """计算实体重叠度"""
        source_entities = self._extract_entities(source)
        target_entities = self._extract_entities(target)
        
        if not source_entities or not target_entities:
            return 0.0, []
        
        shared = source_entities & target_entities
        union = source_entities | target_entities
        
        overlap_ratio = len(shared) / len(union) if union else 0.0
        
        return overlap_ratio, list(shared)[:10]
    
    def _tokenize(self, text: str) -> Set[str]:
        """分词并过滤停用词"""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return words - STOPWORDS
    
    def _extract_entities(self, text: str) -> Set[str]: 
        """提取命名实体/关键对象"""
        entities = set()
        
        # URL
        entities.update(re. findall(r'https?://[^\s<>"]+', text))
        
        # 文件路径
        entities. update(re.findall(r'[\w./\\-]+\.\w{2,4}\b', text))
        
        # 引号内容
        entities.update(re.findall(r'"([^"]{3,50})"', text))
        entities.update(re. findall(r"'([^']{3,50})'", text))
        
        # 数字标识符（4位以上）
        entities. update(re.findall(r'\b\d{4,}\b', text))
        
        # 驼峰命名或下划线命名（代码标识符）
        entities.update(re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', text))
        entities.update(re. findall(r'\b\w+_\w+\b', text))
        
        return entities

class DualTrackCandidateSelector: 
    """
    双轨候选选择器

    设计原理：
    - Track A（规则轨道）：捕获显式因果（词汇/实体重叠）
    - Track B（语义轨道）：捕获隐式因果（语义相似）

    解决问题：
    - 纯规则筛选会漏掉"无重叠但有因果"的消息对
    - 语义相似度能识别"查天气"与"25度"的隐式关联
    """

    def __init__(
        self,
        rule_candidates: int = 3,
        semantic_candidates: int = 2,
        use_embeddings: bool = True,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.rule_k = rule_candidates
        self.semantic_k = semantic_candidates
        self.use_embeddings = use_embeddings
        
        self.evidence_extractor = EvidenceExtractor()
        self._embedding_cache: dict = {}
        self._embedder = None
        
        if use_embeddings: 
            self._init_embedder(embedding_model)

    def _init_embedder(self, model_name: str):
        """初始化嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(model_name)
        except ImportError: 
            print("Warning: sentence-transformers not installed.  Falling back to rule-only selection.")
            self.use_embeddings = False

    def select(
        self, 
        target: StandardLogItem, 
        candidates: List[StandardLogItem]
    ) -> List[StandardLogItem]:
        """
        双轨选择候选
        
        返回：去重后的候选列表（rule_k + semantic_k 个）
        """
        if not candidates:
            return []
        
        selected_ids = set()
        selected = []
        
        # === Track A：规则排序 ===
        rule_scored = self._rule_score_all(target, candidates)
        for candidate, score in rule_scored[: self.rule_k]:
            if candidate.step_id not in selected_ids: 
                selected. append(candidate)
                selected_ids. add(candidate.step_id)
        
        # === Track B：语义相似度 ===
        if self.use_embeddings and self._embedder is not None:
            semantic_scored = self._semantic_score_all(target, candidates)
            for candidate, score in semantic_scored: 
                if candidate. step_id not in selected_ids: 
                    selected.append(candidate)
                    selected_ids. add(candidate.step_id)
                    if len(selected) >= self.rule_k + self. semantic_k: 
                        break
        
        # 如果语义轨道不可用，从规则轨道补充
        if len(selected) < self.rule_k + self.semantic_k:
            for candidate, score in rule_scored[self.rule_k: ]:
                if candidate.step_id not in selected_ids: 
                    selected. append(candidate)
                    selected_ids.add(candidate. step_id)
                    if len(selected) >= self.rule_k + self.semantic_k:
                        break
        
        return selected

    def _rule_score_all(
        self, 
        target: StandardLogItem, 
        candidates: List[StandardLogItem]
    ) -> List[Tuple[StandardLogItem, float]]: 
        """规则评分所有候选"""
        scored = []
        
        for source in candidates:
            evidence = self.evidence_extractor.extract(source, target)
            score = evidence.rule_score()
            scored.append((source, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _semantic_score_all(
        self, 
        target: StandardLogItem, 
        candidates:  List[StandardLogItem]
    ) -> List[Tuple[StandardLogItem, float]]:
        """语义相似度评分所有候选"""
        if not self._embedder:
            return []
        
        target_emb = self._get_embedding(target)
        
        scored = []
        for source in candidates:
            source_emb = self._get_embedding(source)
            similarity = self._cosine_similarity(source_emb, target_emb)
            dist = target.step_id - source.step_id
            decay = 1.0 / (1.0 + 0.1 * dist)  # 简单的反比例衰减
            final_score = similarity * decay
            scored.append((source, final_score))
        
        scored.sort(key=lambda x:  x[1], reverse=True)
        return scored

    def _get_embedding(self, msg: StandardLogItem) -> np.ndarray:
        """获取嵌入（带缓存）"""
        if msg.step_id not in self._embedding_cache:
            text = (msg.raw_content or "")[: 1024]  # 截断以提升速度
            self._embedding_cache[msg.step_id] = self._embedder.encode(text)
        return self._embedding_cache[msg.step_id]

    def _cosine_similarity(self, a: np.ndarray, b: np. ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np. linalg.norm(a)
        norm_b = np.linalg. norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def precompute_embeddings(self, messages: List[StandardLogItem]):
        """预计算所有消息的嵌入"""
        if not self._embedder:
            return
        
        for msg in messages: 
            _ = self._get_embedding(msg)

    def clear_cache(self):
        """清空嵌入缓存"""
        self._embedding_cache. clear()


class SimpleSelector:
    """
    简单候选选择器（无嵌入时的后备方案）
    """

    def __init__(self, max_candidates: int = 5):
        self.max_candidates = max_candidates
        self.evidence_extractor = EvidenceExtractor()

    def select(
        self, 
        target: StandardLogItem, 
        candidates: List[StandardLogItem]
    ) -> List[StandardLogItem]: 
        """基于规则评分选择"""
        if not candidates:
            return []
        
        scored = []
        for source in candidates:
            evidence = self.evidence_extractor.extract(source, target)
            score = evidence. rule_score()
            scored.append((source, score))
        
        scored. sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[: self.max_candidates]]
    
    
@dataclass
class TaskContext:
    """Context about the task being diagnosed."""
    
    question: str
    """The original user question/request."""
    
    ground_truth: str
    """The expected correct answer."""
    
    error_info:  str = ""
    """Additional error information or description."""