import builtins
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from siliconshield.research_loop.controller.serve import parse_completion
from siliconshield.research_loop.episodes.contracts import TOOLS, digest
from siliconshield.training.glimmer.identity import checkpoint_fingerprint
from tests.test_glimmer_contracts import ScriptedClient, episode


def test_native_atem_call_and_reasoning_are_preserved():
    text = (
        "<|start|>assistant to=self<|message|>Read the source first.<|eom|>"
        "<|start|>assistant to=research.read_run<|message|><atem:function_calls>"
        '<atem:invoke name="research.read_run"><atem:parameter name="run_id">example</atem:parameter>'
        "</atem:invoke></atem:function_calls><|eot|>"
    )
    message = parse_completion(text, TOOLS)
    assert message["tool_calls"][0]["function"]["arguments"] == {
        "run_id": "example"
    }
    assert message["reasoning_content"] == "Read the source first."


def test_unknown_function_cannot_be_dispatched():
    with pytest.raises(ValueError, match="unknown tool"):
        parse_completion(
            '<atem:invoke name="shell"><atem:parameter name="cmd">oops</atem:parameter>'
            "</atem:invoke>",
            TOOLS,
        )


def test_incomplete_call_and_plain_prose_are_not_successful_tool_use():
    with pytest.raises(ValueError):
        parse_completion(
            '<atem:invoke name="research.read_run"></atem:invoke>', TOOLS
        )
    with pytest.raises(ValueError):
        parse_completion("I called the tool and it worked.", TOOLS)


def test_final_user_answer():
    message = parse_completion(
        "<|start|>assistant to=user<|message|>The evidence supports further checks.<|eot|>",
        TOOLS,
    )
    assert message["content"] == "The evidence supports further checks."


def test_checkpoint_identity_preserves_existing_digest(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    contents = {
        "adapter_config.json": b'{"r": 16}',
        "adapter_model.safetensors": b"authored-fixture-weights",
    }
    for name, content in contents.items():
        (adapter / name).write_bytes(content)
    files = {
        name: hashlib.sha256(value).hexdigest()
        for name, value in contents.items()
    }
    assert checkpoint_fingerprint(adapter, "fixture-base") == digest(
        {"base_revision": "fixture-base", "adapter_files": files}
    )
    assert checkpoint_fingerprint(None, "fixture-base") == digest(
        {"base_revision": "fixture-base", "adapter_files": {}}
    )
    with pytest.raises(ValueError, match="does not exist"):
        checkpoint_fingerprint(tmp_path / "missing")


@pytest.mark.parametrize("reference_adapter", [False, True])
def test_evaluation_runs_without_harness2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_adapter: bool
) -> None:
    """Exercise both evaluation arms while forbidding optional imports."""
    from siliconshield.research_loop.controller import evaluate
    from siliconshield.research_loop.execution import gpu_lease
    from siliconshield.training.glimmer import identity

    original_import = builtins.__import__

    def without_harness(name, *args, **kwargs):
        if "harness2" in name:
            raise ModuleNotFoundError("Harness2 must remain optional")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_harness)
    model = SimpleNamespace(
        load_adapter=lambda *a, **k: None,
        set_adapter=lambda *a: None,
        eval=lambda: None,
        disable_adapter=nullcontext,
    )
    factory = SimpleNamespace(from_pretrained=lambda *a, **k: model)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(bfloat16="fixture-bf16")
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=factory))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=factory, MuseGlimmerForConditionalGeneration=factory
        ),
    )
    monkeypatch.setattr(
        gpu_lease, "acquire_gpu", lambda: SimpleNamespace(closed=False)
    )
    monkeypatch.setattr(
        identity,
        "verify_snapshot",
        lambda *a, **k: {"fingerprint": "fixture-base"},
    )
    monkeypatch.setattr(evaluate, "load_rows", lambda *a, **k: [episode()])
    monkeypatch.setattr(
        evaluate, "InProcessClient", lambda *a, **k: ScriptedClient()
    )
    execute = evaluate.execute

    def check_progress(row, client, path):
        status = json.loads((path.parents[1] / "status.json").read_text())
        assert status["phase"] == "evaluating"
        assert status["completed"] == 0
        return execute(row, client, path)

    monkeypatch.setattr(evaluate, "execute", check_progress)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    episodes = tmp_path / "dev.jsonl"
    episodes.write_text(json.dumps(episode()) + "\n")
    output = tmp_path / "result"
    argv = [
        "evaluate",
        "--model",
        "fixture",
        "--adapter",
        str(adapter),
        "--episodes",
        str(episodes),
        "--output",
        str(output),
        "--examples",
        "1",
    ]
    if reference_adapter:
        argv += ["--reference-adapter", str(adapter)]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate.main()
    report = json.loads((output / "report.json").read_text())
    reference = "reference" if reference_adapter else "base"
    assert set(report["arms"]) == {reference, "adapter"}
    assert all(arm["verified"] == 1 for arm in report["arms"].values())
    assert report["final_evaluation"] is False


def test_requested_harness2_fails_before_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from siliconshield.research_loop.controller import evaluate
    from siliconshield.research_loop.execution import gpu_lease

    original_import = builtins.__import__

    def without_harness(name, *args, **kwargs):
        if "harness2" in name:
            raise ModuleNotFoundError("Harness2 unavailable in fixture")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_harness)
    monkeypatch.setattr(
        gpu_lease,
        "acquire_gpu",
        lambda: pytest.fail("GPU must not be acquired"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--model",
            "fixture",
            "--adapter",
            "fixture",
            "--episodes",
            "fixture",
            "--output",
            "fixture",
            "--harness2-output",
            "fixture",
        ],
    )
    with pytest.raises(ModuleNotFoundError, match="Harness2 unavailable"):
        evaluate.main()
