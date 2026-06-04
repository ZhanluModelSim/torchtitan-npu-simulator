#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generate a PDF numerical stability report from JSON data and an HTML template.

Usage:
    python generate_report.py \
        --report report.json \
        --reproduce reproduce.json \
        --template premerge-accuracy-check.html \
        --output premerge_accuracy_check.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def render_infra_changes(infra_changes: list[dict]) -> str:
    """Render infra changes as comment lines for the bash code block."""
    if not infra_changes:
        return "# （无 infra 变更）"
    lines = []
    for item in infra_changes:
        lines.append(f"#   {item['file']}（{item['reason']}）")
    return "\n".join(lines)


def classify_diff(value: float, tolerance: float) -> str:
    """Return CSS class for diff values."""
    if value <= tolerance:
        return "metric-pass"
    return "metric-fail"


def pass_text(value: float, tolerance: float) -> str:
    return "通过" if value <= tolerance else "未通过"


def fill_template(template: str, data: dict) -> str:
    """Simple {{mustache}}-style template rendering.

    Supports:
      - {{key}} for simple substitution
      - {{#list}}...{{/list}} for iterating over lists of dicts or strings
      - {{#bool}}...{{/bool}} for conditional sections (truthy=show, falsy=hide)
    """
    result = template

    import re

    for key, value in data.items():
        if isinstance(value, list):
            # Handle list sections: {{#list_name}}...{{key}}...{{/list_name}}
            pattern = re.compile(
                r"\{\{#"
                + re.escape(key)
                + r"\}\}(.*?)\{\{/"
                + re.escape(key)
                + r"\}\}",
                re.DOTALL,
            )
            for match in pattern.finditer(result):
                block_template = match.group(1)
                rendered_items = []
                for item in value:
                    rendered = block_template
                    if isinstance(item, dict):
                        for item_key, item_val in item.items():
                            rendered = rendered.replace(
                                "{{" + item_key + "}}", str(item_val)
                            )
                    else:
                        # String or scalar item — {{.}} renders the value itself
                        rendered = rendered.replace("{{.}}", str(item))
                    rendered_items.append(rendered)
                result = result.replace(match.group(0), "\n".join(rendered_items))
        else:
            # Handle boolean/conditional sections: {{#bool_key}}...{{/bool_key}}
            pattern = re.compile(
                r"\{\{#"
                + re.escape(key)
                + r"\}\}(.*?)\{\{/"
                + re.escape(key)
                + r"\}\}",
                re.DOTALL,
            )
            if value:
                result = pattern.sub(r"\1", result)  # truthy: keep content, remove tags
            else:
                result = pattern.sub("", result)  # falsy: remove entire block
            # Simple placeholder substitution
            placeholder = "{{" + key + "}}"
            result = result.replace(
                placeholder, str(value) if value is not None else ""
            )

    # Clean up any remaining {{mustache}} placeholders
    result = re.sub(r"\{\{.*?\}\}", "", result)
    return result


def _extract_case_metrics(
    report: dict, case_label: str, tolerance: float, report_path: str = ""
) -> dict:
    """Extract per-case metrics, plot paths, and pass/fail from a single report.json."""
    loss_metric = None
    grad_metric = None
    for m in report.get("metrics", []):
        tag = m.get("metric", m.get("tag", ""))
        if "loss" in tag.lower():
            loss_metric = m
        elif "grad_norm" in tag:
            grad_metric = m

    case = {"label": case_label}
    if loss_metric:
        case.update(
            {
                "loss_max_abs_diff": f"{loss_metric['max_abs_diff']:.6f}",
                "loss_mean_abs_diff": f"{loss_metric['mean_abs_diff']:.6f}",
                "loss_max_rel_diff": f"{loss_metric['max_rel_diff']:.6f}",
                "loss_mean_rel_diff": f"{loss_metric['mean_rel_diff']:.6f}",
                "loss_common_steps": str(loss_metric.get("num_common_steps", "")),
                "loss_max_abs_diff_class": classify_diff(
                    loss_metric["max_abs_diff"], tolerance
                ),
                "loss_pass_class": classify_diff(
                    loss_metric["max_abs_diff"], tolerance
                ),
                "loss_pass_text": pass_text(loss_metric["max_abs_diff"], tolerance),
            }
        )
    if grad_metric:
        case.update(
            {
                "grad_norm_max_abs_diff": f"{grad_metric['max_abs_diff']:.6f}",
                "grad_norm_mean_abs_diff": f"{grad_metric['mean_abs_diff']:.6f}",
                "grad_norm_max_rel_diff": f"{grad_metric['max_rel_diff']:.6f}",
                "grad_norm_mean_rel_diff": f"{grad_metric['mean_rel_diff']:.6f}",
                "grad_norm_common_steps": str(grad_metric.get("num_common_steps", "")),
                "grad_norm_max_abs_diff_class": classify_diff(
                    grad_metric["max_abs_diff"], tolerance
                ),
                "grad_norm_pass_class": classify_diff(
                    grad_metric["max_abs_diff"], tolerance
                ),
                "grad_norm_pass_text": pass_text(
                    grad_metric["max_abs_diff"], tolerance
                ),
            }
        )
    case["passed"] = report.get("passed", True)

    # Plot paths relative to report output dir
    case_subdir = ""
    if report_path:
        report_dir = os.path.dirname(os.path.abspath(report_path))
        case_subdir = os.path.basename(report_dir) + "/"
    case["loss_plot_path"] = case_subdir + "loss_comparison.png"
    case["grad_norm_plot_path"] = case_subdir + "grad_norm_comparison.png"
    return case


def build_template_data(
    reports: list[dict],
    reproduce: dict,
    tolerance: float,
    report_paths: list[str] | None = None,
) -> dict:
    """Merge multiple report.json metrics into template variables."""
    if report_paths is None:
        report_paths = []

    cases = []
    for i, report in enumerate(reports):
        path = report_paths[i] if i < len(report_paths) else ""
        label = (
            os.path.basename(os.path.dirname(os.path.abspath(path)))
            if path
            else f"case_{i}"
        )
        case = _extract_case_metrics(report, label, tolerance, path)
        case["case_index"] = str(i + 1)  # 5.1, 5.2, ...
        cases.append(case)

    # diff section comes after all per-case loss+grad_norm blocks
    # base sections: 1=环境 2=运行参数 3=分支变更 4=复现 → each case gets 2 blocks (loss + grad_norm)
    diff_sec = 4 + 1 + len(cases) * 2  # 4 base + 1 diff section itself
    verdict_sec = diff_sec + 1
    suggest_sec = diff_sec + 2

    all_passed = all(c["passed"] for c in cases)
    short_len = 7

    data = {
        "timestamp": reproduce.get("timestamp", ""),
        "npu_count": reproduce.get("environment", {}).get("npu_count", ""),
        "cann_version": reproduce.get("environment", {}).get("cann_version", ""),
        "torch_version": reproduce.get("environment", {}).get("torch_version", ""),
        "torch_npu_version": reproduce.get("environment", {}).get(
            "torch_npu_version", ""
        ),
        "baseline_branch": reproduce.get("baseline_branch", ""),
        "baseline_commit": reproduce.get("baseline_commit", ""),
        "baseline_commit_short": reproduce.get("baseline_commit", "")[:short_len],
        "candidate_branch": reproduce.get("candidate_branch", ""),
        "candidate_commit": reproduce.get("candidate_commit", ""),
        "candidate_commit_short": reproduce.get("candidate_commit", "")[:short_len],
        "model_name": reproduce.get("model_name", ""),
        "config_name": _config_name(reproduce),
        "training_steps": reproduce.get("training_steps", ""),
        "parallelism_summary": reproduce.get("parallelism_summary", ""),
        "branch_diff": reproduce.get("branch_diff", "(未记录)"),
        "repo_url": reproduce.get(
            "repo_url", "https://gitcode.com/cann/torchtitan-npu.git"
        ),
        "infra_steps": reproduce.get("infra_steps", []),
        "training_runs": reproduce.get("training_runs", []),
        "compare_commands": reproduce.get("compare_commands", []),
        "compare_section_num": str(len(reproduce.get("training_runs", [])) + 2 + 1),
        "results_section_num": str(len(reproduce.get("training_runs", [])) + 3 + 1),
        "diff_section_num": str(diff_sec),
        "verdict_section_num": str(verdict_sec),
        "suggest_section_num": str(suggest_sec),
        "tolerance": tolerance,
        "cases": cases,
        # PR summary specific
        "single_case": len(cases) == 1,
        "multi_case": len(cases) > 1,
        "single_chart": False,
        "loss_plot_path": cases[0].get("loss_plot_path", "") if cases else "",
        "grad_norm_plot_path": cases[0].get("grad_norm_plot_path", "") if cases else "",
    }

    if all_passed:
        data.update(
            {
                "verdict_class": "verdict-pass",
                "verdict_text": "通过 — 所有 case 的所有指标在容差范围内，未检测到数值精度回退。",
                "conclusion_class": "conclusion-pass",
                "conclusion_detail": f"所有差异 ≤ {tolerance}，代码变更未引入数值精度回退，可以合并。",
                "passed": True,
                "failed": None,
                "failed_details": None,
                "failed_items": [],
            }
        )
    else:
        failed_items = []
        for c, report in zip(cases, reports, strict=True):
            if c["passed"]:
                continue
            for item in _build_failed_items(report, tolerance):
                item["case"] = c["label"]
                failed_items.append(item)
        data.update(
            {
                "verdict_class": "verdict-fail",
                "verdict_text": f"未通过 — {len([c for c in cases if not c['passed']])}/{len(cases)} 个 case 超出容差范围。",
                "conclusion_class": "conclusion-fail",
                "conclusion_detail": f"{len([c for c in cases if not c['passed']])}/{len(cases)} 个 case 超出容差，需调查根因后重新验证。",
                "passed": None,
                "failed": True,
                "failed_details": bool(failed_items),
                "failed_items": failed_items,
            }
        )

    return data


def _config_name(reproduce: dict) -> str:
    """Handle both singular string and plural list formats."""
    val = reproduce.get("config_name", "")
    if val:
        return val if isinstance(val, str) else ", ".join(val)
    names = reproduce.get("config_names", [])
    return ", ".join(names) if names else ""


def _build_failed_items(report: dict, tolerance: float) -> list[dict]:
    items = []
    for m in report.get("metrics", []):
        if m["max_abs_diff"] > tolerance:
            mtag = m.get("metric", m.get("tag", ""))
            tag_label = "Loss" if "loss" in mtag.lower() else "Grad Norm"
            first_abs_diff = m.get("first_exceed_abs_diff", m["max_abs_diff"])
            items.append(
                {
                    "tag": tag_label,
                    "step": str(
                        m.get("first_exceed_step", m.get("max_abs_diff_step", "?"))
                    ),
                    "diff": f"{first_abs_diff:.6f}",
                }
            )
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Generate numerical stability PDF report"
    )
    parser.add_argument(
        "--report",
        required=True,
        action="append",
        help="Path to report.json from loss_compare.py (pass --report multiple times for multiple cases)",
    )
    parser.add_argument("--reproduce", required=True, help="Path to reproduce.json")
    parser.add_argument("--template", required=True, help="Path to HTML template")
    parser.add_argument("--output", required=True, help="Output PDF path")
    parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="Tolerance threshold"
    )
    args = parser.parse_args()

    # Validate inputs
    for path in args.report:
        if not os.path.isfile(path):
            print(f"ERROR: report file not found: {path}")
            sys.exit(1)
    for label, path in [("reproduce", args.reproduce), ("template", args.template)]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    reports = [load_json(p) for p in args.report]
    reproduce = load_json(args.reproduce)

    # Read template
    with open(args.template, encoding="utf-8") as f:
        template = f.read()

    # Merge data and fill template
    data = build_template_data(reports, reproduce, args.tolerance, args.report)
    html = fill_template(template, data)

    # Write intermediate HTML for debugging
    html_path = args.output.replace(".pdf", ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML: {html_path}")

    # Generate PR summary (compact one-page version for PR description / email)
    pr_template_path = os.path.join(os.path.dirname(args.template), "pr-summary.html")
    if os.path.isfile(pr_template_path):
        with open(pr_template_path, encoding="utf-8") as f:
            pr_template = f.read()
        pr_html = fill_template(pr_template, data)
        output_dir = os.path.dirname(os.path.abspath(args.output)) or "."

        # Write HTML for local preview
        pr_html_path = os.path.join(output_dir, "pr_summary.html")
        with open(pr_html_path, "w", encoding="utf-8") as f:
            f.write(pr_html)
        print(f"PR Summary (HTML): {pr_html_path}")

        # Render PDF (self-contained, images embedded)
        try:
            from weasyprint import HTML as WHTML

            pr_pdf_path = os.path.join(output_dir, "pr_summary.pdf")
            WHTML(string=pr_html, base_url=output_dir).write_pdf(pr_pdf_path)
            print(f"PR Summary (PDF):  {pr_pdf_path}")
        except Exception as e:
            print(f"WARNING: PR summary PDF generation failed: {e}")
    else:
        print(f"WARNING: PR summary template not found: {pr_template_path}")

    # Generate PDF
    try:
        from weasyprint import HTML

        # Resolve image paths relative to the output directory
        output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        HTML(string=html, base_url=output_dir).write_pdf(args.output)
        print(f"PDF:  {args.output}")
    except ImportError:
        print(
            "WARNING: weasyprint not installed. Open the HTML file in a browser and print to PDF."
        )
        print("  pip install weasyprint  # to enable PDF generation")
    except Exception as e:
        print("WARNING: PDF generation failed:", e)
        print(f"  HTML report is available at: {html_path}")


if __name__ == "__main__":
    main()
