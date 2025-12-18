from cgap.pipeline import CGAPPipeline


def test_pipeline_runs() -> None:
    raw = """
<Instruction> Do task.

<Action> Step1
<Observation> ok

<Action> Step2
<Observation> boom: error
""".strip()

    p = CGAPPipeline()
    artifacts = p.run(raw_text=raw, error_info="boom: error", error_step_id=3, role="WebSurfer")
    assert artifacts.diagnosis.root_cause_step_id in {1, 2, 3}
    assert len(artifacts.store.items) >= 2
