from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IAOTContent(BaseModel):
    """I-A-O-T container.

    Notes:
    - Field non-empty => store; do not force classification.
    - Summary can be used to replace long Action/Observation.
    """

    instruction: Optional[str] = None
    action: Optional[str] = None
    observation: Optional[str] = None
    thought: Optional[str] = None

    summary: Optional[str] = None


class StandardLogItem(BaseModel):
    """C-GAP minimal data unit (one step)."""

    id: int = Field(..., ge=0)
    role: str = Field(default="Unknown")
    content: IAOTContent = Field(default_factory=IAOTContent)

    # metadata for later stages
    causal_parents: Dict[int, float] = Field(default_factory=dict)
    relevance_score: Optional[float] = None

    def to_node_text(self) -> str:
        parts: List[str] = [f"[Step {self.id}] (Role: {self.role})"]
        if self.content.instruction:
            parts.append(f"<Instruction> {self.content.instruction}")
        if self.content.action:
            parts.append(f"<Action> {self.content.action}")
        if self.content.observation:
            parts.append(f"<Observation> {self.content.observation}")
        if self.content.thought:
            parts.append(f"<Thought> {self.content.thought}")
        if self.content.summary:
            parts.append(f"<Summary> {self.content.summary}")
        return "\n".join(parts)


class NodeStore(BaseModel):
    items: List[StandardLogItem] = Field(default_factory=list)

    def by_id(self) -> Dict[int, StandardLogItem]:
        return {it.id: it for it in self.items}

    def max_id(self) -> int:
        return max((it.id for it in self.items), default=0)
