# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ratchet direct imports that cross the torchtitan-npu architecture layers."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = "torchtitan_npu"
LAYER_NAMES = frozenset({"extensions", "models", "ops", "override", "patches"})
FORBIDDEN_TARGETS = {
    "extensions": frozenset({"models", "ops", "override"}),
    "patches": frozenset({"extensions"}),
}


@dataclass(frozen=True, order=True)
class ImportEdge:
    """One direct import between top-level torchtitan-npu layers."""

    source: str
    target_module: str

    @property
    def target_layer(self) -> str:
        return self.target_module.split(".", 2)[1]


def _layer_for_path(path: Path) -> str | None:
    try:
        parts = path.parts
        start = parts.index(PACKAGE_ROOT)
    except ValueError:
        return None
    return parts[start + 1] if len(parts) > start + 1 and parts[start + 1] in LAYER_NAMES else None


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _modules_for_file(path: Path) -> set[str]:
    """Extract imports with AST, falling back for newer syntax on older Python."""
    source = path.read_text(encoding="utf-8")
    try:
        return _imported_modules(ast.parse(source, filename=str(path)))
    except SyntaxError:
        # Local environments can be older than the repository Python target.
        # Ruff and Pyrefly remain responsible for reporting syntax errors.
        return {
            module
            for match in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", source, re.MULTILINE)
            for module in match.groups()
            if module is not None
        }


def collect_edges(root: Path) -> set[ImportEdge]:
    """Return forbidden direct imports found below ``root/torchtitan_npu``."""
    edges: set[ImportEdge] = set()
    package = root / PACKAGE_ROOT
    for path in package.rglob("*.py"):
        source_layer = _layer_for_path(path)
        if source_layer is None:
            continue
        relative = path.relative_to(root).as_posix()
        source = relative
        for module in _modules_for_file(path):
            parts = module.split(".")
            if len(parts) < 2 or parts[0] != PACKAGE_ROOT or parts[1] not in LAYER_NAMES:
                continue
            target_layer = parts[1]
            if target_layer in FORBIDDEN_TARGETS.get(source_layer, frozenset()):
                edges.add(ImportEdge(source=source, target_module=module))
    return edges


def _header_has_upstream_metadata(path: Path) -> bool:
    """Accept legacy upstream markers in the fixed header region only."""
    header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:40])
    return "Upstream-PR:" in header or "upstream PR:" in header or "Upstream PR:" in header


def collect_missing_patch_metadata(root: Path) -> set[str]:
    """Find actual TorchTitan patch modules without header metadata."""
    patch_root = root / PACKAGE_ROOT / "patches" / "torchtitan"
    return {
        path.relative_to(root).as_posix()
        for path in patch_root.rglob("*.py")
        if path.name != "__init__.py" and not _header_has_upstream_metadata(path)
    }


def _imports_from_init(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        prefix = f".{node.module}." if node.module is not None else "."
        imports.extend(f"{prefix}{alias.name}" for alias in node.names)
    return imports


def _baseline_entries(raw_entries: list[dict[str, str]]) -> set[ImportEdge]:
    return {ImportEdge(source=entry["source"], target_module=entry["target_module"]) for entry in raw_entries}


def load_baseline(path: Path) -> tuple[set[ImportEdge], set[str], dict[str, list[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    imports = _baseline_entries(data.get("import", []))
    metadata = set(data.get("patch_metadata", {}).get("missing", []))
    activation = {name: list(order) for name, order in data.get("activation_order", {}).items()}
    return imports, metadata, activation


def check(root: Path, baseline_path: Path) -> list[str]:
    """Return violations; baseline entries may disappear but may not grow."""
    baseline_imports, baseline_metadata, activation = load_baseline(baseline_path)
    current_imports = collect_edges(root)
    current_metadata = collect_missing_patch_metadata(root)
    violations = [
        f"new forbidden import: {edge.source} -> {edge.target_module}"
        for edge in sorted(current_imports - baseline_imports)
    ]
    violations.extend(
        f"new patch missing upstream metadata: {path}" for path in sorted(current_metadata - baseline_metadata)
    )
    for relative_path, expected in sorted(activation.items()):
        actual = _imports_from_init(root / relative_path)
        if actual != expected:
            violations.append(f"activation order changed: {relative_path}; expected {expected}, got {actual}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    baseline = args.baseline.resolve() if args.baseline is not None else root / ".ci/architecture_import_baseline.json"
    violations = check(root, baseline)
    if violations:
        print("Architecture boundary check failed:")
        print("\n".join(f"- {violation}" for violation in violations))
        return 1
    print("Architecture boundary check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
