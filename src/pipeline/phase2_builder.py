"""Phase II: LLM 驱动因果图构建器

基于第一性原理的通用因果图构建方案

核心特性：
1. Pairwise LLM 推理（稳定可靠）
2. 双轨候选选择（规则 + 语义，解决漏斗风险）
3. 无显式证据输入（强迫语义理解，防止捷径学习）
4. 3+1 因果分类（INSTRUCTION, DATA, STATE, NONE）
5. CoT 引导（先分析后判断）
6. 多层鲁棒性（主链 + 跳跃边 + 宽松修复）
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Dict, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import networkx as nx

from src.llm_client import LLMClient
from src.models import StandardLogItem, CausalType, CausalResult, CausalEdge, DualTrackCandidateSelector, SimpleSelector


# 主系统提示词（带 CoT 引导）
CAUSAL_REASONING_SYSTEM_PROMPT = """You are a causal relationship judge in a multi-agent system analyzer. 

## Task
Determine if Target message causally depends on Source message. 

## Causal Types (Choose ONE)

**INSTRUCTION**:  Source gives a request/command/task, Target responds/executes
  Examples:
  - "Please search for X" → "I searched for X"
  - "What is the weather?" → "The temperature is 25°C"
  - "Click the link" → "I clicked the link"

**DATA**: Target uses information/results produced by Source
  Examples:
  - "Search results show: [list]" → "Based on results, let's click option A"
  - "I found that X=5" → "Since X=5, we should..."
  - "The page contains: ..." → "I extracted the following data:  ..."

**STATE**: Target depends on state/condition established by Source
  Examples:
  - "Login successful" → "Accessing user dashboard"
  - "Error: connection failed" → "Retrying with different approach"
  - "Page loaded" → "Scrolling down to find content"

**NONE**: No causal relationship - Target would happen the same way without Source

## Reasoning Method (IMPORTANT)
Apply COUNTERFACTUAL test:
1. Imagine Source did NOT happen
2. Would Target still occur exactly the same way? 
3. If NO → Causality exists (choose INSTRUCTION/DATA/STATE based on the relationship)
4. If YES → NONE

## Critical Warnings
⚠️ Do NOT rely on word overlap.  "Weather query" and "Temperature report" have no shared words but ARE causally linked. 
⚠️ Do NOT assume no causality just because topics seem different.  Focus on logical dependency.
⚠️ Use SEMANTIC understanding to identify implicit connections.

## Output Format
First, provide a brief analysis (1-2 sentences), then output JSON. 

Analysis: <identify the key dependency or explain why there is none>
JSON: {"type": "<INSTRUCTION|DATA|STATE|NONE>", "confidence": <0. 0-1.0>, "reason": "<counterfactual explanation>"}"""


# 用户消息模板
CAUSAL_REASONING_USER_TEMPLATE = """## Source Message (ID: {source_id})
{source_content}

## Target Message (ID: {target_id})
{target_content}

Does Target causally depend on Source?  Apply counterfactual reasoning."""


# 快速验证提示词（用于跳跃边和修复）
QUICK_VERIFICATION_SYSTEM_PROMPT = """You are a causal relationship detector.  Be concise and precise. 

Determine if Target causally depends on Source using counterfactual reasoning: 
- If Source hadn't happened, would Target still occur the same way?
- If NO → There is causality
- If YES → No causality

Types:  INSTRUCTION (request→response), DATA (info usage), STATE (condition dependency), NONE

Output JSON only:  {"type": "<TYPE>", "confidence":  <0.0-1.0>, "reason": "<brief>"}"""


QUICK_VERIFICATION_USER_TEMPLATE = """Source: {source_content}

Target: {target_content}

Causal relationship? """


class CausalGraphBuilder:
    """
    LLM 驱动的通用因果图构建器
    
    设计原则：
    1. 风格无关：不依赖角色标签，基于第一性原理
    2. 信噪比优先：每条边都承载真实因果信息
    3. 鲁棒性保证：三层保护机制确保图的连通性
    """
    
    def __init__(
        self,
        llm:  LLMClient,
        model_name: str,
        window_size: int = 10,
        rule_candidates: int = 3,
        semantic_candidates: int = 2,
        confidence_threshold: float = 0.6,
        skip_interval: int = 5,
        repair_threshold: float = 0.3,
        temperature: float = 0.1,
        max_content_length: int = 400,
        use_embeddings: bool = True,
        # === 新增：性能优化参数 ===
        max_workers: int = 8,              # 并行线程数
        enable_early_stop: bool = True,    # 找到高置信父节点后早停
        early_stop_confidence: float = 0.8, # 早停置信度阈值
        rule_score_threshold: float = 0.15, # 规则分低于此值跳过 LLM
        batch_size: int = 10,              # 并行批次大小
    ):
        """
        初始化因果图构建器
        
        Args:
            llm: LLM 客户端
            model_name:  使用的模型名称
            window_size: 候选窗口大小
            rule_candidates:  规则轨道的候选数量
            semantic_candidates: 语义轨道的候选数量
            confidence_threshold: 主链置信度阈值
            skip_interval: 跳跃边间隔
            repair_threshold: 修复阶段的置信度阈值（较宽松）
            temperature: LLM 生成温度
            max_content_length: 消息内容最大长度
            use_embeddings:  是否使用嵌入模型
        """
        self.llm = llm
        self.model_name = model_name
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.skip_interval = skip_interval
        self.repair_threshold = repair_threshold
        self.temperature = temperature
        self.max_content_length = max_content_length
        
        # 性能优化参数
        self.max_workers = max_workers
        self.enable_early_stop = enable_early_stop
        self.early_stop_confidence = early_stop_confidence
        self.rule_score_threshold = rule_score_threshold
        self.batch_size = batch_size
        
        # 初始化候选选择器
        if use_embeddings: 
            self.candidate_selector = DualTrackCandidateSelector(
                rule_candidates=rule_candidates,
                semantic_candidates=semantic_candidates,
                use_embeddings=True,
            )
        else:
            self.candidate_selector = SimpleSelector(
                max_candidates=rule_candidates + semantic_candidates
            )
        
        # 统计信息
        self._stats = {
            "total_llm_calls": 0,
            "primary_edges": 0,
            "skip_edges": 0,
            "repair_edges": 0,
            "fallback_edges": 0,
            "skipped_by_rule": 0,      # 被规则过滤跳过的
            "skipped_by_early_stop": 0, # 被早停跳过的
            "parallel_batches": 0,
            "wall_time_seconds": 0.0,
        }
    
    def build(self, steps: List[StandardLogItem]) -> nx.DiGraph:
        """
        构建因果图
        
        Args:
            steps:  标准化的日志步骤列表
            
        Returns:
            因果图（NetworkX DiGraph）
        """
        start_time = time.time()
        g = nx.DiGraph()
        
        # 重置统计
        self._reset_stats()
        
        # 添加所有节点
        self._add_nodes(g, steps)
        
        # 预计算嵌入（如果使用）
        if hasattr(self.candidate_selector, 'precompute_embeddings'):
            self.candidate_selector.precompute_embeddings(steps)
        
        # Phase 1: 局部因果推理（主链）—— 并行优化版
        self._infer_local_causality_parallel(g, steps)
        
        # Phase 2: 添加跳跃边（鲁棒性）—— 简化版
        self._add_skip_edges_lite(g, steps)
        
        # Phase 3: 断裂修复（兜底）—— 简化版
        self._repair_broken_chains_lite(g, steps)
        
        self._stats["wall_time_seconds"] = time.time() - start_time
        
        return g
    
    def get_stats(self) -> Dict[str, int]:
        """获取构建统计信息"""
        return self._stats. copy()
    
    def _reset_stats(self):
        """重置统计信息"""
        self._stats = {
            "total_llm_calls": 0,
            "primary_edges": 0,
            "skip_edges": 0,
            "repair_edges": 0,
            "fallback_edges": 0,
            "skipped_by_rule": 0,
            "skipped_by_early_stop": 0,
            "parallel_batches": 0,
            "wall_time_seconds": 0.0,
        }
    
    def _add_nodes(self, g: nx.DiGraph, steps: List[StandardLogItem]):
        """添加所有节点到图"""
        for step in steps:
            g.add_node(
                step.step_id,
                session_id=step.session_id,
                role=step.role,
                raw_content=step.raw_content,
                parsed_iaot=(
                    step.parsed_iaot.model_dump() 
                    if hasattr(step.parsed_iaot, "model_dump") 
                    else dict(step.parsed_iaot)
                ),
            )
    
    # =========================================================================
    # 优化后的 Phase 1：并行 + 早停 + 规则预过滤
    # =========================================================================
    
    def _infer_local_causality_parallel(self, g: nx.DiGraph, steps: List[StandardLogItem]):
        """
        Phase 1: 并行化局部因果推理
        
        优化策略：
        1. 收集所有待验证的 (source, target) 对
        2. 按规则分过滤低分候选
        3. 批量并行调用 LLM
        4. 应用早停机制
        """
        # Step 1: 收集所有待验证对及其规则分数
        verification_tasks: List[Tuple[int, StandardLogItem, StandardLogItem, float]] = []
        # (target_idx, source, target, rule_score)
        
        for i, target in enumerate(steps):
            if i == 0:
                continue
            
            window_start = max(0, i - self.window_size)
            candidates = steps[window_start:i]
            
            # 获取候选及其规则分数
            if hasattr(self.candidate_selector, '_rule_score_all'):
                scored_candidates = self.candidate_selector._rule_score_all(target, candidates)
            else:
                # 简单选择器没有分数，给默认分
                scored_candidates = [(c, 0.5) for c in candidates[-5:]]
            
            # 双轨选择
            selected = self.candidate_selector.select(target, candidates)
            
            # 为选中的候选附加规则分数
            score_map = {c.step_id: s for c, s in scored_candidates}
            for source in selected:
                rule_score = score_map.get(source.step_id, 0.5)
                verification_tasks.append((i, source, target, rule_score))
        
        # Step 2: 按 target 分组，准备批量处理
        tasks_by_target: Dict[int, List[Tuple[StandardLogItem, StandardLogItem, float]]] = {}
        for target_idx, source, target, rule_score in verification_tasks:
            if target_idx not in tasks_by_target:
                tasks_by_target[target_idx] = []
            tasks_by_target[target_idx].append((source, target, rule_score))
        
        # Step 3: 批量并行处理
        all_results: Dict[int, List[Tuple[StandardLogItem, CausalResult]]] = {}
        
        # 收集所有需要 LLM 验证的任务（过滤低分）
        llm_tasks: List[Tuple[int, StandardLogItem, StandardLogItem]] = []
        for target_idx, task_list in tasks_by_target.items():
            for source, target, rule_score in task_list:
                if rule_score < self.rule_score_threshold:
                    # 规则分太低，跳过 LLM，直接标记为 NONE
                    self._stats["skipped_by_rule"] += 1
                    if target_idx not in all_results:
                        all_results[target_idx] = []
                    all_results[target_idx].append((source, CausalResult(
                        causal_type=CausalType.NONE,
                        confidence=0.0,
                        reason="Skipped: low rule score"
                    )))
                else:
                    llm_tasks.append((target_idx, source, target))
        
        # 并行执行 LLM 调用
        if llm_tasks:
            self._execute_parallel_inference(llm_tasks, all_results)
        
        # Step 4: 应用结果到图（带早停）
        for i, target in enumerate(steps):
            if i == 0:
                continue
            
            results = all_results.get(i, [])
            found_parent = False
            high_conf_found = False
            
            # 按置信度排序，优先处理高置信结果
            results_sorted = sorted(results, key=lambda x: x[1].confidence, reverse=True)
            
            for source, result in results_sorted:
                # 早停检查
                if high_conf_found and self.enable_early_stop:
                    self._stats["skipped_by_early_stop"] += 1
                    continue
                
                if (result.causal_type != CausalType.NONE and 
                    result.confidence >= self.confidence_threshold):
                    self._add_edge(g, source.step_id, target.step_id, result, edge_type="primary")
                    self._stats["primary_edges"] += 1
                    found_parent = True
                    
                    # 检查是否触发早停
                    if result.confidence >= self.early_stop_confidence:
                        high_conf_found = True
            
            # 保底
            if not found_parent:
                self._add_fallback_edge(g, steps[i-1], target)
    
    def _execute_parallel_inference(
        self,
        tasks: List[Tuple[int, StandardLogItem, StandardLogItem]],
        results: Dict[int, List[Tuple[StandardLogItem, CausalResult]]]
    ):
        """并行执行 LLM 推理"""
        
        def worker(task: Tuple[int, StandardLogItem, StandardLogItem]) -> Tuple[int, StandardLogItem, CausalResult]:
            target_idx, source, target = task
            result = self._pairwise_causal_inference(source, target)
            return (target_idx, source, result)
        
        # 分批处理，避免一次性提交太多
        for batch_start in range(0, len(tasks), self.batch_size):
            batch = tasks[batch_start:batch_start + self.batch_size]
            self._stats["parallel_batches"] += 1
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(worker, task): task for task in batch}
                
                for future in as_completed(futures):
                    try:
                        target_idx, source, result = future.result()
                        if target_idx not in results:
                            results[target_idx] = []
                        results[target_idx].append((source, result))
                    except Exception as e:
                        # 出错时返回 NONE
                        task = futures[future]
                        target_idx, source, target = task
                        if target_idx not in results:
                            results[target_idx] = []
                        results[target_idx].append((source, CausalResult(
                            causal_type=CausalType.NONE,
                            confidence=0.0,
                            reason=f"Error: {str(e)[:50]}"
                        )))
    
    # =========================================================================
    # 简化版 Phase 2：跳跃边（减少验证频率）
    # =========================================================================
    
    def _add_skip_edges_lite(self, g: nx.DiGraph, steps: List[StandardLogItem]):
        """
        Phase 2 简化版：只在关键位置添加跳跃边
        
        优化：增大间隔，减少 LLM 调用
        """
        skip_interval = self.skip_interval * 2  # 翻倍间隔
        
        tasks = []
        for i, target in enumerate(steps):
            skip_source_idx = i - 2 * skip_interval
            
            if skip_source_idx >= 0 and i % skip_interval == 0:
                source = steps[skip_source_idx]
                if not g.has_edge(source.step_id, target.step_id):
                    tasks.append((source, target))
        
        # 并行执行
        if tasks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._quick_causal_check, src, tgt): (src, tgt)
                    for src, tgt in tasks
                }
                
                for future in as_completed(futures):
                    source, target = futures[future]
                    try:
                        result = future.result()
                        if (result.causal_type != CausalType.NONE and 
                            result.confidence > 0.4):
                            self._add_edge(g, source.step_id, target.step_id, result, edge_type="skip")
                            self._stats["skip_edges"] += 1
                    except Exception:
                        pass
    
    # =========================================================================
    # 简化版 Phase 3：断裂修复（减少候选数）
    # =========================================================================
    
    def _repair_broken_chains_lite(self, g: nx.DiGraph, steps: List[StandardLogItem]):
        """
        Phase 3 简化版：快速修复孤岛
        
        优化：只验证最近的3个候选，而非10个
        """
        orphans = [
            step for step in steps[1:]
            if g.in_degree(step.step_id) == 0
        ]
        
        repair_tasks = []
        for orphan in orphans:
            orphan_idx = orphan.step_id
            # 只看最近的5步
            candidates = steps[max(0, orphan_idx - 5):orphan_idx]
            if candidates:
                # 只验证最近的3个
                for source in list(reversed(candidates))[:3]:
                    repair_tasks.append((orphan, source))
        
        # 并行执行
        if repair_tasks:
            orphan_results: Dict[int, List[Tuple[StandardLogItem, CausalResult]]] = {}
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._quick_causal_check, src, orphan): (orphan, src)
                    for orphan, src in repair_tasks
                }
                
                for future in as_completed(futures):
                    orphan, source = futures[future]
                    try:
                        result = future.result()
                        if orphan.step_id not in orphan_results:
                            orphan_results[orphan.step_id] = []
                        orphan_results[orphan.step_id].append((source, result))
                    except Exception:
                        pass
            
            # 为每个孤岛选择最佳父节点
            for orphan in orphans:
                results = orphan_results.get(orphan.step_id, [])
                best_parent = None
                best_result = None
                best_score = 0
                
                for source, result in results:
                    if result.causal_type != CausalType.NONE and result.confidence > best_score:
                        best_score = result.confidence
                        best_parent = source
                        best_result = result
                
                if best_parent and best_score >= self.repair_threshold:
                    self._add_edge(g, best_parent.step_id, orphan.step_id, best_result, edge_type="repair")
                    self._stats["repair_edges"] += 1
                else:
                    # 强制连接到前一步
                    prev_idx = orphan.step_id - 1
                    if prev_idx >= 0:
                        self._add_fallback_edge(g, steps[prev_idx], orphan)
    
    # =========================================================================
    # 保留原有方法（供向后兼容或单独调用）
    # =========================================================================
    
    def _infer_local_causality(self, g: nx.DiGraph, steps: List[StandardLogItem]):
        """
        Phase 1: 局部因果推理
        
        对每个节点，在窗口内选择候选并进行 Pairwise 推理
        """
        for i, target in enumerate(steps):
            if i == 0:
                continue  # 第一个节点没有父节点
            
            # 获取候选窗口
            window_start = max(0, i - self.window_size)
            candidates = steps[window_start: i]
            
            # 双轨候选选择
            selected = self.candidate_selector.select(target, candidates)
            
            # Pairwise 推理
            found_parent = False
            for source in selected:
                result = self._pairwise_causal_inference(source, target)
                
                if (result.causal_type != CausalType.NONE and 
                    result.confidence >= self.confidence_threshold):
                    self._add_edge(
                        g, source.step_id, target.step_id,
                        result, edge_type="primary"
                    )
                    self._stats["primary_edges"] += 1
                    found_parent = True
            
            # 保底：如果没有找到任何父节点，连接到前一个
            if not found_parent:
                self._add_fallback_edge(g, steps[i-1], target)
    
    def _add_skip_edges(self, g: nx.DiGraph, steps: List[StandardLogItem]):
        """
        Phase 2: 添加跳跃边
        
        每隔 skip_interval 步，验证与 2*skip_interval 之前节点的因果关系
        防止连续断裂导致链条崩溃
        """
        for i, target in enumerate(steps):
            skip_source_idx = i - 2 * self.skip_interval
            
            if skip_source_idx >= 0 and i % self.skip_interval == 0:
                source = steps[skip_source_idx]
                
                # 如果已有边，跳过
                if g.has_edge(source.step_id, target.step_id):
                    continue
                
                # 快速验证
                result = self._quick_causal_check(source, target)
                
                if (result.causal_type != CausalType.NONE and 
                    result.confidence > 0.4):  # 跳跃边使用较低阈值
                    self._add_edge(
                        g, source.step_id, target.step_id,
                        result, edge_type="skip"
                    )
                    self._stats["skip_edges"] += 1
    
    def _repair_broken_chains(self, g: nx.DiGraph, steps: List[StandardLogItem]):
        """
        Phase 3: 修复断裂的因果链
        
        检测孤岛节点（入度为0的非根节点），进行更广泛的搜索修复
        """
        # 根节点（第一个节点）
        root_id = steps[0].step_id
        
        # 检测孤岛
        orphans = [
            step for step in steps[1:]
            if g.in_degree(step.step_id) == 0
        ]
        
        for orphan in orphans:
            self._repair_single_orphan(g, orphan, steps)
    
    def _repair_single_orphan(
        self, 
        g: nx.DiGraph, 
        orphan: StandardLogItem, 
        steps: List[StandardLogItem]
    ):
        """修复单个孤岛节点"""
        orphan_idx = orphan.step_id
        
        # 扩大搜索范围（20步）
        search_range = min(orphan_idx, 20)
        candidates = steps[orphan_idx - search_range:orphan_idx]
        
        if not candidates:
            return
        
        # 使用语义排序（如果可用）
        if hasattr(self.candidate_selector, '_semantic_score_all'):
            scored = self.candidate_selector._semantic_score_all(orphan, candidates)
            to_verify = [c for c, _ in scored[:10]]
        else:
            # 后备：按时序从近到远
            to_verify = list(reversed(candidates))[:10]
        
        # 逐个验证，找最佳父节点
        best_parent = None
        best_result = None
        best_score = 0
        
        for source in to_verify:
            result = self._pairwise_causal_inference(source, orphan)
            
            if result.causal_type != CausalType.NONE:
                score = result.confidence
                if score > best_score:
                    best_score = score
                    best_parent = source
                    best_result = result
        
        # 添加修复边（使用较低阈值）
        if best_parent and best_score >= self.repair_threshold:
            self._add_edge(
                g, best_parent.step_id, orphan.step_id,
                best_result, edge_type="repair"
            )
            self._stats["repair_edges"] += 1
        else:
            # 最后手段：强制连接到前一步
            prev_idx = orphan_idx - 1
            if prev_idx >= 0:
                self._add_fallback_edge(g, steps[prev_idx], orphan)
    
    def _pairwise_causal_inference(
        self, 
        source: StandardLogItem, 
        target: StandardLogItem
    ) -> CausalResult:
        """
        Pairwise 因果推理
        
        使用完整的 CoT 引导 Prompt
        """
        # 构建用户消息
        source_content = self._truncate_content(source.raw_content)
        target_content = self._truncate_content(target.raw_content)
        
        user_message = CAUSAL_REASONING_USER_TEMPLATE.format(
            source_id=source.step_id,
            source_content=source_content,
            target_id=target.step_id,
            target_content=target_content,
        )
        
        # LLM 调用
        response = self._call_llm(
            system_prompt=CAUSAL_REASONING_SYSTEM_PROMPT,
            user_message=user_message,
        )
        
        return self._parse_causal_response(response)
    
    def _quick_causal_check(
        self, 
        source: StandardLogItem, 
        target: StandardLogItem
    ) -> CausalResult:
        """
        快速因果检查
        
        使用简化的 Prompt，用于跳跃边和修复
        """
        source_content = self._truncate_content(source.raw_content, max_length=600)
        target_content = self._truncate_content(target.raw_content, max_length=600)
        
        user_message = QUICK_VERIFICATION_USER_TEMPLATE.format(
            source_content=source_content,
            target_content=target_content,
        )
        
        response = self._call_llm(
            system_prompt=QUICK_VERIFICATION_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=100,
        )
        
        return self._parse_causal_response(response)
    
    def _call_llm(
        self, 
        system_prompt: str, 
        user_message: str,
        max_tokens: int = 150,
    ) -> str:
        """调用 LLM"""
        self._stats["total_llm_calls"] += 1
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        response = self.llm.one_step_chat(
            messages=messages,
            model_name=self.model_name,
            temperature=self.temperature,
            json_mode=False,  # 我们需要解析 Analysis + JSON
        )
        
        return response
    
    def _parse_causal_response(self, response: str) -> CausalResult:
        """
        解析 LLM 响应
        
        响应格式：
        Analysis: <分析内容>
        JSON: {"type": "...", "confidence": ..., "reason": "..."}
        """
        if not response:
            return CausalResult(
                causal_type=CausalType.NONE,
                confidence=0.0,
                reason="Empty response"
            )
        
        # 尝试提取 JSON 部分
        json_match = re.search(r'\{[^{}]+\}', response)
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                
                # 解析类型
                type_str = data.get("type", "NONE").upper()
                if type_str in CausalType.__members__:
                    causal_type = CausalType[type_str]
                else:
                    causal_type = CausalType.NONE
                
                # 解析置信度
                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                
                # 解析理由
                reason = data.get("reason", "")
                
                return CausalResult(
                    causal_type=causal_type,
                    confidence=confidence,
                    reason=reason
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        # 解析失败
        return CausalResult(
            causal_type=CausalType.NONE,
            confidence=0.0,
            reason="Parse error"
        )
    
    def _add_edge(
        self, 
        g: nx.DiGraph, 
        source_id: int, 
        target_id: int,
        result: CausalResult,
        edge_type: str = "primary"
    ):
        """添加因果边到图"""
        g.add_edge(
            source_id,
            target_id,
            causal_type=result.causal_type.value,
            confidence=result.confidence,
            reason=result.reason,
            edge_type=edge_type,
        )
    
    def _add_fallback_edge(
        self, 
        g: nx.DiGraph, 
        source: StandardLogItem, 
        target: StandardLogItem
    ):
        """添加保底边（时序相邻）"""
        g.add_edge(
            source.step_id,
            target.step_id,
            causal_type="temporal",
            confidence=0.2,
            reason="Fallback: temporal adjacency",
            edge_type="fallback",
        )
        self._stats["fallback_edges"] += 1
    
    def _truncate_content(
        self, 
        content: Optional[str], 
        max_length: Optional[int] = None
    ) -> str:
        """截断内容"""
        if not content:
            return ""
        
        max_len = max_length or self.max_content_length
        
        if len(content) <= max_len: return content
        head = max_len // 2
        tail = max_len // 2
        return content[:head] + "...[TRUNCATED]..." + content[-tail:]

