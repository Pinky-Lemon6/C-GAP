"""Phase I: Parser

Cleans raw log text into I-A-O-T (Instruction / Action / Observation / Thought).

This module defines Agent A (Parser) system prompt as a constant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import json

from src.llm_client import LLMClient


PARSER_SYSTEM_PROMPT_AGENT_A = """You are Agent A (Parser) in a Multi-Agent Failure Attribution System.
Your task is to convert raw, heterogeneous log data into a structured I-A-O-T JSON object.

**CORE MISSION:** De-noise the log while **PRESERVING CRITICAL DEBUGGING SIGNALS**.
You must adapt your extraction strategy based on the content type (HTML vs. Code vs. Text).

**STRICT OUTPUT FORMAT (JSON ONLY):**
{
  "instruction": string | null,
  "action": string | null,
  "observation": string | null,
  "thought": string | null
}

**FIELD EXTRACTION RULES:**

### 1. INSTRUCTION (Strict Anti-Hallucination)
- **Definition**: A new command or goal assignment from a User or Orchestrator.
- **Rule**: Set to `null` unless you see an explicit imperative command (e.g., "Search for X", "Run script Y").
- **Artifact Filter**: IGNORE text that looks like system prompts (e.g., "You are an AI...", "Extract I-A-O-T..."). These are log artifacts, NOT the instruction.

### 2. ACTION
- Extract specific tool calls, API requests, code execution commands, or clicks.

### 3. OBSERVATION (Adaptive Strategy - CRITICAL)
Analyze the raw content type and apply the matching rule:

**Type A: Web Content / Screenshots / OCR**
- **Action**: Summarize & Extract Entities.
- **Do**: Extract titles, specific names, dates, numbers, and interactive elements (buttons/links).
- **Do**: Explicitly mention **Distractions/Blockers** (e.g., "Popup ads covering content", "CAPTCHA challenge", "Content Filter warning").
- **Don't**: Copy raw HTML tags (`<div>`, `<span>`) or CSS.

**Type B: Code Execution / Tracebacks / System Errors**
- **Action**: **PRESERVE THE ERROR DETAILS.**
- **Do**: Extract the **Exception Type** (e.g., `ValueError`, `openai.BadRequestError`) and the **Error Message**.
- **Do**: Keep the **Last 3 Frames** of a Traceback to show where it crashed.
- **Don't**: Summarize an error into "An error occurred". We need the *specific* error code.
- **Example**: "Azure Content Filter triggered: 'ResponsibleAIPolicyViolation' regarding 'violence'."

**Type C: Structured Data (JSON / API Responses)**
- **Action**: Summarize Structure & Samples.
- **Do**: Report the structure (e.g., "List of 50 items").
- **Do**: Extract the first 2-3 items as examples.
- **Don't**: Dump a massive JSON object verbatim.

### 4. THOUGHT
- Extract the agent's internal reasoning, planning, or self-correction.

**FALLBACK:**
If a field is not present or cannot be inferred, set it to `null`.

"""


class LogParser:
    """LLM-backed log parser producing an I-A-O-T dict."""

    def __init__(self, llm: LLMClient, model_name: str, temperature: float = 0.0) -> None:
        self.llm = llm
        self.model_name = model_name
        self.temperature = temperature

    def parse_log_segment(self, raw_log: str) -> Dict[str, Optional[str]]:
        """Parse a raw log segment into an IAOT dict.

        Returns:
            Dict with keys: instruction, action, observation, thought.
        """

        if raw_log is None or not str(raw_log).strip():
            return {"instruction": None, "action": None, "observation": None, "thought": None}

        messages = [
            {"role": "system", "content": PARSER_SYSTEM_PROMPT_AGENT_A},
            {
                "role": "user",
                "content": (
                    "Extract I-A-O-T from the following raw log text.\n\n"
                    "RAW_LOG_START\n"
                    f"{raw_log}\n"
                    "RAW_LOG_END"
                ),
            },
        ]

        text = self.llm.one_step_chat(
            messages=messages,
            model_name=self.model_name,
            json_mode=True,
            temperature=self.temperature,
        )

        data = _safe_parse_iaot_json(text)
        return data


def _safe_parse_iaot_json(text: str) -> Dict[str, Optional[str]]:
    """Parse and sanitize the IAOT JSON returned by an LLM."""

    if not text or not text.strip():
        return {"instruction": None, "action": None, "observation": None, "thought": None}

    # In json_mode, models should return pure JSON, but keep a defensive parse.
    try:
        obj: Any = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to salvage by extracting the first JSON object substring.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            raise

    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object, got {type(obj)}")

    def norm(v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return str(v).strip() or None

    out: Dict[str, Optional[str]] = {
        "instruction": norm(obj.get("instruction")),
        "action": norm(obj.get("action")),
        "observation": norm(obj.get("observation")),
        "thought": norm(obj.get("thought")),
    }

    return out
