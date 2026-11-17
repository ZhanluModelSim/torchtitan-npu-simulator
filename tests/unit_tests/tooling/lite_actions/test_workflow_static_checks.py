#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static checks for the GitHub Actions CI orchestration.

The workflows under ``.github/workflows`` are the control entrypoint for the
internal Lite Actions dispatcher. These checks assert orchestration invariants
that a wrong or stale configuration would otherwise only reveal at dispatch
time: YAML must parse, jobs must be structurally runnable, reusable workflows
must declare their inputs, callers must agree with the workflow they reuse,
and every configured test case must resolve to a registered nightly case.

The checks read the YAML and the model-owned case catalog generically; they do
not pin individual filenames, pipeline names or branch literals, so an
unrelated reordering does not fail them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.integration_tests.tools.lite_actions import entrypoint

pytestmark = pytest.mark.tooling

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def workflow_files() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def load_workflow(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path.name}: workflow root must be a mapping"
    return document


def event_section(workflow: dict):
    """Return the ``on:`` mapping.

    PyYAML parses the bare key ``on`` as the boolean ``True``; accept either
    spelling so the check does not depend on that quoting accident.
    """
    for key in ("on", True):
        if key in workflow:
            return workflow[key]
    return None


def reusable_inputs(workflow: dict) -> dict | None:
    event = event_section(workflow)
    if not isinstance(event, dict):
        return None
    call = event.get("workflow_call")
    if not isinstance(call, dict):
        return None
    inputs = call.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _selector(value) -> dict | None:
    """Return a single ``{suite|test_id: name}`` selector when ``value`` is one."""
    if not isinstance(value, dict) or len(value) != 1:
        return None
    key = next(iter(value))
    if key not in ("suite", "test_id") or not isinstance(value[key], str):
        return None
    return value


def configured_selectors(workflow: dict):
    """Yield ``(location, selector_list)`` for every JSON case list in the YAML.

    Selectors appear as JSON strings (e.g. ``default_cases``); any string that
    does not parse into a list of ``{suite|test_id}`` entries is ignored.
    """
    def walk(node, location):
        if isinstance(node, dict):
            for key, child in node.items():
                yield from walk(child, f"{location}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                yield from walk(child, f"{location}[{index}]")
        elif isinstance(node, str):
            try:
                parsed = json.loads(node)
            except (ValueError, TypeError):
                return
            if isinstance(parsed, list) and parsed:
                selectors = [_selector(item) for item in parsed]
                if all(selector is not None for selector in selectors):
                    yield location, selectors

    yield from walk(workflow, "")


def test_workflows_are_parseable_yaml():
    files = workflow_files()
    assert files, "no workflows discovered under .github/workflows"
    for path in files:
        load_workflow(path)


def test_every_job_declares_runner_or_uses():
    for path in workflow_files():
        jobs = load_workflow(path).get("jobs") or {}
        assert jobs, f"{path.name}: no jobs declared"
        for name, job in jobs.items():
            assert isinstance(job, dict), f"{path.name}: job {name!r} must be a mapping"
            assert "runs-on" in job or "uses" in job, (
                f"{path.name}: job {name!r} declares neither 'runs-on' nor 'uses'")


def test_every_workflow_declares_permissions():
    for path in workflow_files():
        assert "permissions" in load_workflow(path), (
            f"{path.name}: missing top-level 'permissions'")


def test_reusable_workflow_inputs_declare_type():
    for path in workflow_files():
        inputs = reusable_inputs(load_workflow(path))
        if inputs is None:
            continue
        for name, spec in inputs.items():
            assert isinstance(spec, dict), (
                f"{path.name}: workflow_call input {name!r} must be a mapping")
            assert spec.get("type"), (
                f"{path.name}: workflow_call input {name!r} missing 'type'")
            if "required" in spec:
                assert isinstance(spec["required"], bool), (
                    f"{path.name}: workflow_call input {name!r} 'required' must be boolean")


def test_local_reusable_callers_match_callee_inputs():
    for path in workflow_files():
        jobs = load_workflow(path).get("jobs") or {}
        for name, job in jobs.items():
            uses = job.get("uses")
            if not isinstance(uses, str) or not uses.startswith("./"):
                continue
            target = (REPO_ROOT / uses).resolve()
            assert target.is_file(), (
                f"{path.name}: job {name!r} reuses missing workflow {uses!r}")
            declared = reusable_inputs(load_workflow(target)) or {}
            provided = job.get("with") or {}
            assert isinstance(provided, dict), (
                f"{path.name}: job {name!r} 'with' must be a mapping")
            unknown = set(provided) - set(declared)
            assert not unknown, (
                f"{path.name}: job {name!r} passes undeclared inputs {sorted(unknown)}")
            missing = {key for key, spec in declared.items()
                       if isinstance(spec, dict) and spec.get("required")} - set(provided)
            assert not missing, (
                f"{path.name}: job {name!r} omits required inputs {sorted(missing)}")


def test_configured_test_cases_resolve_to_registered_cases():
    catalogue = entrypoint.catalog()
    for path in workflow_files():
        for location, selectors in configured_selectors(load_workflow(path)):
            for selector in selectors:
                entrypoint.select(catalogue, **selector)
