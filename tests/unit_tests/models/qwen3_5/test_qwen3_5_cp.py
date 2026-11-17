from pathlib import Path


def test_qwen3_5_cp_adapter_uses_override_refactor_paths():
    root = Path(__file__).resolve().parents[4]
    parallelize_source = (root / "torchtitan_npu/override/qwen3_5/parallelize.py").read_text(encoding="utf-8")
    gated_delta_source = (root / "torchtitan_npu/override/qwen3_5/gated_delta.py").read_text(encoding="utf-8")

    assert "QwenCPMetadata" in parallelize_source
    assert "prepare_sequence_metadata" in parallelize_source
    assert "exchange_sequence_heads" in gated_delta_source
    assert "context_parallel" in gated_delta_source
