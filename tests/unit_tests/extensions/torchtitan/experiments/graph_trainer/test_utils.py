"""Tests for graph metadata and profiling/debug utilities."""

import os
from pathlib import Path

import pytest
import torch

from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.utils import (
    assign_stable_node_tags,
    canonical_order,
    dump_fx_graph,
    is_metadata_only_compute_node,
    isolated_cann_profiler_work_path,
    node_id,
)


def test_stable_node_ids_preserve_custom_metadata_and_original_order():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["custom"] = {"module_fqn": "layers.0"}
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    graph.output(neg)
    gm = torch.fx.GraphModule({}, graph)

    assign_stable_node_tags(gm)

    assert x.meta["custom"]["module_fqn"] == "layers.0"
    assert [node_id(node) for node in gm.graph.nodes] == [0, 1, 2]
    assert canonical_order(gm) == {node: ordinal for ordinal, node in enumerate(gm.graph.nodes)}


def test_stable_node_ids_survive_reordering_and_repeated_assignment():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    first = graph.call_function(torch.ops.aten.neg.default, (x,))
    second = graph.call_function(torch.ops.aten.relu.default, (x,))
    output = graph.output((first, second))
    gm = torch.fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)

    first.prepend(second)
    assign_stable_node_tags(gm)

    assert list(gm.graph.nodes) == [x, second, first, output]
    assert [node_id(node) for node in gm.graph.nodes] == [0, 2, 1, 3]


def test_canonical_order_rejects_untagged_graph():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(x)
    gm = torch.fx.GraphModule({}, graph)

    with pytest.raises(RuntimeError, match="has no auto-overlap node ID"):
        canonical_order(gm)


@pytest.mark.parametrize("requires_copy", [False, True])
def test_reshape_is_metadata_only_only_when_storage_is_shared(requires_copy):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    value = torch.empty((2, 3))
    if requires_copy:
        value = value.t()
    x.meta["val"] = value
    reshape = graph.call_function(torch.ops.aten.reshape.default, (x, (6,)))
    reshape.meta["val"] = value.reshape(6)

    assert is_metadata_only_compute_node(reshape) is not requires_copy


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("original", [None, "training/profiling"])
@pytest.mark.parametrize("fail", [False, True])
def test_profile_directory_retention_and_environment_restore(monkeypatch, tmp_path, caplog, retain, original, fail):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    if original is None:
        monkeypatch.delenv("ASCEND_WORK_PATH", raising=False)
    else:
        monkeypatch.setenv("ASCEND_WORK_PATH", original)
    caplog.set_level("INFO")

    def collect():
        nonlocal work_path
        with isolated_cann_profiler_work_path("test_cann_profile_", retain=retain) as work_path:
            assert work_path.is_absolute()
            assert Path(os.environ["ASCEND_WORK_PATH"]) == work_path
            (work_path / "raw_data").mkdir()
            (work_path / "trace.json").write_text("{}", encoding="utf-8")
            if fail:
                raise RuntimeError("parse failed")

    work_path = None
    if fail:
        with pytest.raises(RuntimeError, match="parse failed"):
            collect()
    else:
        collect()
    assert os.environ.get("ASCEND_WORK_PATH") == original
    assert work_path.exists() is retain
    if retain:
        assert (work_path / "trace.json").read_text(encoding="utf-8") == "{}"
        assert (work_path / "raw_data").is_dir()
        assert "rank3_pid" in work_path.name
        assert str(work_path) in caplog.text
        assert "profiling retained" in caplog.text


def test_dump_fx_graph_is_self_contained(monkeypatch, tmp_path):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(x)
    gm = torch.fx.GraphModule({}, graph)
    monkeypatch.chdir(tmp_path)

    output = dump_fx_graph(gm, prefix="auto_overlap_before")

    assert output == tmp_path / "fx_graphs/auto_overlap_before_rank0.py"
    assert output.is_file()
    assert "def forward" in output.read_text(encoding="utf-8")
