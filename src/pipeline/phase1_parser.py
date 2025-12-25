"""Phase I: Atomic Extraction Parser

Extracts atomic nodes from raw log text using LLM.

Each raw log step is decomposed into a list of AtomicNode objects:
- INTENT: Internal thought, plan, goal, decision
- EXEC: Tool usage, code execution, API calls
- INFO: Observations, results, errors, feedback
- COMM: Messages sent to other agents

This module defines Agent A (Parser) with an atomization-focused system prompt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor

from src.llm_client import LLMClient


# =============================================================================
# System Prompt for Atomic Extraction
# =============================================================================

PARSER_SYSTEM_PROMPT_ATOMIC = """You are an Atomic Event Extractor in a Multi-Agent Log Analysis System.

## MISSION
Convert a raw log segment into a **list of Atomic Nodes**. Each node represents ONE distinct semantic event.

## ATOMIC NODE TYPES

**INTENT** - Internal thought, plan, goal, or decision
- Agent reasoning: "I need to search for weather data"
- Planning: "Let me try a different approach"
- Decision: "I will click the first result"

**EXEC** - Tool usage, code execution, API calls
- Tool calls: "Calling search_api('weather')"
- Code execution: "Running: print(result)"
- Actions: "Clicking button #submit"

**INFO** - Observations, results, errors, feedback
- Results: "Search returned 10 items"
- Errors: "ValueError: invalid input"
- System feedback: "Page loaded successfully"

**COMM** - Messages to/from other agents or users
- User messages: "User asked: What is the weather?"
- Agent responses: "Replying: The temperature is 25°C"
- Inter-agent: "Forwarding request to Agent B"

## EXTRACTION PRINCIPLES

1. **Split Intents from Facts**
   - Separate what the agent THINKS from what it DOES or SEES
   - "I will search" (INTENT) + "Calling search_api" (EXEC) = 2 nodes

2. **Merge Similar Consecutive Types**
   - Multiple related thoughts → ONE INTENT node
   - Multiple related observations → ONE INFO node

3. **Drop Fluff**
   - Remove boilerplate, repeated info, formatting artifacts
   - Skip empty or meaningless content

4. **Summarize Content**
   - Be concise but preserve debugging signals
   - Keep error messages, entity names, key values
   - Truncate long outputs (e.g., "List of 50 items, first 3: ...")

5. **Preserve Error Details** (CRITICAL)
   - Keep exception types: "ValueError", "TimeoutError"
   - Keep error messages and codes
   - Keep last few traceback frames if present

## OUTPUT FORMAT

Return a JSON array of objects. Each object has:
- "type": One of "INTENT", "EXEC", "INFO", "COMM"
- "content": Summarized content string (1-3 sentences)

```json
[
  {"type": "INTENT", "content": "Planning to search for weather information"},
  {"type": "EXEC", "content": "Calling weather_api(city='Beijing')"},
  {"type": "INFO", "content": "API returned: temperature=25°C, humidity=60%"}
]
```

## RULES

- Output ONLY the JSON array, no explanation
- Minimum 1 node, maximum ~5 nodes per log segment
- If the log is empty or meaningless, return: [{"type": "INFO", "content": "Empty or unparseable log"}]
- Order nodes chronologically as they appear in the log(Cause before Effect)
"""


class LogParser:
    """LLM-backed log parser for atomic node extraction."""

    def __init__(
        self, 
        llm: LLMClient, 
        model_name: str, 
        temperature: float = 0.0,
    ) -> None:
        """
        Initialize the log parser.
        
        Args:
            llm: LLM client instance
            model_name: Model to use for parsing
            temperature: Sampling temperature
        """
        self.llm = llm
        self.model_name = model_name
        self.temperature = temperature

    # =========================================================================
    # Main Atomic Extraction Method
    # =========================================================================

    def parse_to_atomic_nodes(
        self, 
        raw_log: str, 
        step_id: int = 0,
        role: str = "unknown",
        *,
        max_input_chars: int = 6000,
    ) -> List[Dict[str, Any]]:
        """
        Parse a raw log segment into a list of atomic node dicts.
        
        Args:
            raw_log: Raw log text to parse
            step_id: Step ID for generating node IDs
            role: Role of the agent (for node metadata)
            max_input_chars: Maximum input characters (truncate if exceeded)
            
        Returns:
            List of dicts with keys: node_id, step_id, role, type, content, original_text
        """
        if raw_log is None or not str(raw_log).strip():
            return [self._create_fallback_node(step_id, role, "Empty log")]

        # Truncate if needed
        safe_log = _truncate_raw_log(str(raw_log), max_chars=max_input_chars)

        messages = [
            {"role": "system", "content": PARSER_SYSTEM_PROMPT_ATOMIC},
            {
                "role": "user",
                "content": (
                    "Extract atomic nodes from the following log:\n\n"
                    "```\n"
                    f"{safe_log}\n"
                    "```"
                ),
            },
        ]

        try:
            text = self.llm.one_step_chat(
                messages=messages,
                model_name=self.model_name,
                json_mode=True,
                temperature=self.temperature,
            )
            raw_nodes = _safe_parse_json_list(text)
            
            # Convert to full node format with IDs
            nodes = []
            for idx, node in enumerate(raw_nodes):
                node_type = node.get("type", "INFO")
                content = node.get("content", "")
                
                # Validate type
                if node_type not in {"INTENT", "EXEC", "INFO", "COMM"}:
                    node_type = "INFO"
                
                nodes.append({
                    "node_id": f"step_{step_id}_{idx}",
                    "step_id": step_id,
                    "role": role,
                    "type": node_type,
                    "content": content,
                    "original_text": safe_log if idx == 0 else None,
                })
            
            # Ensure at least one node
            if not nodes:
                return [self._create_fallback_node(step_id, role, "No nodes extracted")]
            
            return nodes
            
        except Exception as exc:
            msg = str(exc)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            return [self._create_fallback_node(step_id, role, f"Parse error: {msg}")]

    def _create_fallback_node(
        self, 
        step_id: int, 
        role: str, 
        reason: str
    ) -> Dict[str, Any]:
        """Create a fallback INFO node when parsing fails."""
        return {
            "node_id": f"step_{step_id}_0",
            "step_id": step_id,
            "role": role,
            "type": "INFO",
            "content": f"[Fallback] {reason}",
            "original_text": None,
        }

    # =========================================================================
    # Parallel Processing
    # =========================================================================

    def parse_to_atomic_nodes_parallel(
        self,
        items: List[Dict[str, Any]],
        *,
        batch_size: int = 8,
        max_workers: Optional[int] = None,
        max_input_chars: int = 6000,
    ) -> List[List[Dict[str, Any]]]:
        """
        Parse multiple log segments in parallel.
        
        Args:
            items: List of dicts with keys: raw_log, step_id, role
            batch_size: Number of items per batch
            max_workers: Max parallel workers
            max_input_chars: Max input chars per item
            
        Returns:
            List of atomic node lists, one per input item (order preserved)
        """
        if not items:
            return []

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        n = len(items)

        if max_workers is None:
            cpu = os.cpu_count() or 4
            max_workers = min(4, cpu, n)

        # Pre-allocate results to preserve order
        results: List[List[Dict[str, Any]]] = [[] for _ in range(n)]

        def _worker(item: Dict[str, Any]) -> List[Dict[str, Any]]:
            return self.parse_to_atomic_nodes(
                raw_log=item.get("raw_log", ""),
                step_id=item.get("step_id", 0),
                role=item.get("role", "unknown"),
                max_input_chars=max_input_chars,
            )

        # Process in batches
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch = items[start:end]

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for offset, parsed in enumerate(ex.map(_worker, batch)):
                    results[start + offset] = parsed

        return results



# =============================================================================
# Helper Functions
# =============================================================================

def _truncate_raw_log(raw_log: str, *, max_chars: int = 6000) -> str:
    """
    Truncate raw log text to reduce prompt/context pressure.
    
    Strategy:
    - Keep head and tail (tail often contains errors/tracebacks)
    - Insert a clear marker to indicate truncation
    """
    s = "" if raw_log is None else str(raw_log)
    s = s.strip("\ufeff")  # tolerate BOM
    
    if max_chars is None or max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s

    # Bias to keep more tail if it looks like an exception/traceback
    s_lc = s.lower()
    tail_weighted = any(k in s_lc for k in ("traceback", "exception", "error", "stack trace"))

    if max_chars < 200:
        return s[:max_chars]

    if tail_weighted:
        head = max(100, int(max_chars * 0.35))
        tail = max_chars - head
    else:
        head = max(150, int(max_chars * 0.6))
        tail = max_chars - head

    head_part = s[:head].rstrip()
    tail_part = s[-tail:].lstrip()
    return (
        head_part
        + "\n...<TRUNCATED>...\n"
        + tail_part
    )


def _safe_parse_json_list(text: str) -> List[Dict[str, Any]]:
    """
    Parse a JSON list from LLM response with error recovery.
    
    Handles common LLM JSON errors:
    - Markdown code blocks
    - Trailing commas
    - Unclosed brackets
    - Extra text before/after JSON
    """
    if not text or not text.strip():
        return []

    # Remove markdown code blocks if present
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Try direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [item for item in obj if isinstance(item, dict)]
        elif isinstance(obj, dict):
            return [obj]
        return []
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from text
    start = text.find("[")
    end = text.rfind("]")
    
    if start >= 0 and end > start:
        json_str = text[start:end + 1]
        
        # Fix common issues
        # Remove trailing commas before ] or }
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        try:
            obj = json.loads(json_str)
            if isinstance(obj, list):
                return [item for item in obj if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass
    
    # Try to extract individual JSON objects
    objects = []
    for match in re.finditer(r'\{[^{}]+\}', text):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    
    return objects


