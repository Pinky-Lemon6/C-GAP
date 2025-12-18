from cgap.schema import IAOTContent, StandardLogItem


def test_to_node_text_contains_fields() -> None:
    item = StandardLogItem(
        id=1,
        role="Orchestrator",
        content=IAOTContent(instruction="Do X", action="A", observation="O", thought="T", summary="S"),
    )
    text = item.to_node_text()
    assert "[Step 1]" in text
    assert "<Instruction> Do X" in text
    assert "<Action> A" in text
    assert "<Observation> O" in text
    assert "<Thought> T" in text
    assert "<Summary> S" in text
