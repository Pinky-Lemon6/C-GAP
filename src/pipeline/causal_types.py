"""Causal types and constraints for C-GAP pipeline.

This module defines:
- NodeType: Atomic event types for node-based causal graph
- CausalType: Causal relationship types between nodes
- VALID_CAUSAL_SOURCES: Type constraints for valid causal relationships
"""

from __future__ import annotations

from enum import Enum
from typing import Set, Dict


class NodeType(str, Enum):
    """
    Atomic Node Types for fine-grained causal graph.
    
    Design Principles:
    1. Cover all semantic categories in multi-agent logs
    2. Enable type-based causal filtering
    3. Support granular failure attribution
    """
    
    INTENT = "INTENT"
    """
    Internal thought, plan, goal, decision.
    
    Examples:
    - "I need to search for weather information"
    - "Let me analyze the search results"
    - "I will click the first link"
    - "My plan is to extract the temperature"
    """
    
    EXEC = "EXEC"
    """
    Tool usage, code execution, API calls.
    
    Examples:
    - "Calling search_api('weather')"
    - "Executing click(element_id=5)"
    - "Running Python code: print(result)"
    - "Sending HTTP request to endpoint"
    """
    
    INFO = "INFO"
    """
    Observations, results, errors, feedback.
    
    Examples:
    - "Search returned 10 results: [...]"
    - "Error: Element not found"
    - "Page loaded successfully"
    - "API response: {'temperature': 25}"
    """
    
    COMM = "COMM"
    """
    Messages sent to other agents.
    
    Examples:
    - "User: What is the weather?"
    - "Sending message to Agent B: Please verify"
    - "Response to user: The temperature is 25°C"
    - "Forwarding data to next agent"
    """
    
    COMPRESSED = "COMPRESSED"
    """
    Compressed or aggregated node representing multiple events.
    """


class CausalType(str, Enum):
    """
    Causal relationship types between nodes.
    
    Design Principles:
    1. Clear category boundaries for consistent labeling
    2. Cover core causal types needed for failure attribution
    3. SFT-friendly for small model training
    """
    
    INSTRUCTION = "INSTRUCTION"
    """
    Instruction/Control flow causality.
    
    Definition: Source issues request/command/task, Target responds/executes.
    Examples:
    - "Please search for X" → "I searched for X"
    - "What is the weather?" → "The temperature is 25°C"
    """
    
    DATA = "DATA"
    """
    Data/Information flow causality.
    
    Definition: Target uses information/data/conclusions produced by Source.
    Examples:
    - "Search results show: [list]" → "Based on results, clicking option A"
    - "I found that X=5" → "Since X=5, we should..."
    """
    
    STATE = "STATE"
    """
    State/Condition causality.
    
    Definition: Target execution depends on state/condition established by Source.
    Examples:
    - "Login successful" → "Accessing user dashboard"
    - "Error: connection failed" → "Retrying with different approach"
    """
    
    NONE = "NONE"
    """
    No causal relationship.
    
    Definition: Target would happen the same way even if Source didn't occur.
    Criterion: Apply counterfactual test - if Source hadn't happened, 
               would Target still occur exactly the same way?
    """


# =============================================================================
# Type Constraints: Valid causal source types for each target type
# =============================================================================

VALID_CAUSAL_SOURCES: Dict[NodeType, Set[NodeType]] = {
    # INTENT can be caused by:
    # - INFO: Observations trigger new thoughts/plans
    # - COMM: Messages from others influence decisions
    # - INTENT: One thought leads to another
    NodeType.INTENT: {NodeType.INFO, NodeType.COMM, NodeType.INTENT},
    
    # EXEC can only be caused by:
    # - INTENT: Decisions/plans lead to actions
    NodeType.EXEC: {NodeType.INTENT},
    
    # COMM can be caused by:
    # - INFO: Observations lead to communication
    # - INTENT: Decisions to communicate
    # - EXEC: Execution results to report
    NodeType.COMM: {NodeType.INFO, NodeType.INTENT, NodeType.EXEC},
    
    # INFO can be caused by:
    # - EXEC: Execution produces results/errors
    # - COMM: Messages bring new information
    NodeType.INFO: {NodeType.EXEC, NodeType.COMM},
}


def is_valid_causal_pair(source_type: NodeType, target_type: NodeType) -> bool:
    """
    Check if a causal relationship from source_type to target_type is valid.
    
    Args:
        source_type: NodeType of the potential cause
        target_type: NodeType of the potential effect
        
    Returns:
        True if the relationship is valid according to type constraints
    """
    valid_sources = VALID_CAUSAL_SOURCES.get(target_type, set())
    return source_type in valid_sources


def get_valid_source_types(target_type: NodeType) -> Set[NodeType]:
    """
    Get the set of valid source types for a given target type.
    
    Args:
        target_type: The NodeType of the target node
        
    Returns:
        Set of NodeTypes that can validly cause the target type
    """
    return VALID_CAUSAL_SOURCES.get(target_type, set())
