"""Phase I: Parser

Cleans raw log text into I-A-O-T (Instruction / Action / Observation / Thought).

This module defines Agent A (Parser) system prompt as a constant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import json

from src.llm_client import LLMClient


PARSER_SYSTEM_PROMPT_AGENT_A = """You are Agent A (Parser) in a Multi-Agent Failure Attribution System (C-GAP).

Task:
- Convert raw agent log text into a structured I-A-O-T JSON object.
- **CRITICAL:** Perform SEMANTIC SUMMARIZATION and NOISE FILTERING.

Output requirements (STRICT):
- Output a single JSON object (keys: instruction, action, observation, thought).
- Set fields to null if not present.

Processing Rules:
1. **Instruction**: Extract the core directive. If the raw text is verbose, summarize the intent.
2. **Action**: Extract the specific tool call or command.
3. **Observation (HEAVY DE-NOISING REQUIRED)**:
   - **DO NOT** copy raw HTML, CSS, base64 images, or massive JSON dumps.
   - **DO** summarize the key information content (e.g., "Bing search returned 10 results about Ted Danson, including links to IMDb and Wikipedia").
   - If the observation is a webpage screenshot or OCR, summarize the readable text relevant to the query.
   - **Remove** all layout tags (<div>, <span>), style definitions, and irrelevant metadata.
4. **Thought**: Extract the agent's internal reasoning or plan.

Example:
Raw: "<html><body><div class='nav'>...</div><div class='content'>Found 3 items...</div></body></html>"
Observation: "Web page loaded successfully. Content shows 3 items found related to the search query."

If a field is not explicitly present, set it to null.
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
