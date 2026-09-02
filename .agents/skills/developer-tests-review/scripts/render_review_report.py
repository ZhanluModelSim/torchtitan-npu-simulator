#!/usr/bin/env python3

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import hashlib
import html
import re
from pathlib import Path

import markdown


STYLE = r"""
:root { --ink:#17233b; --muted:#5c687a; --line:#dbe3ee; --surface:#f6f8fc; --accent:#2457d6; --warn:#9a6700; }
* { box-sizing:border-box; }
html { background:#edf2f8; font-size:15px; }
body { max-width:1120px; margin:0 auto; padding:30px; color:var(--ink); background:#edf2f8;
  font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif; line-height:1.62; overflow-wrap:anywhere; }
header { padding:28px 32px; color:#fff; background:linear-gradient(135deg,#172a57,#2457d6); border-radius:16px; }
header h1 { margin-top:0; color:#fff; }
header table { color:#fff; background:rgba(255,255,255,.08); }
header th { color:#fff; background:rgba(255,255,255,.1); }
header code { color:#fff; background:rgba(8,22,54,.3); }
main > section { margin:18px 0; padding:20px 24px 24px; border:1px solid #cfd8e6; border-radius:14px; background:#fff; }
main > section:nth-child(1) { border-top:5px solid var(--warn); }
main > section:nth-child(2), main > section:nth-child(3) { border-top:5px solid var(--accent); }
h1,h2,h3 { line-height:1.28; break-after:avoid-page; }
h2 { margin:0 0 14px; padding-left:10px; border-left:5px solid var(--accent); }
h3 { color:#20345f; }
p,ul,ol { margin:.45rem 0 .8rem; }
code { padding:.08em .28em; color:#263b64; background:#edf2f8; border-radius:4px; }
pre { max-width:100%; padding:.85rem; border:1px solid var(--line); border-radius:8px; background:#f4f7fb;
  white-space:pre-wrap; overflow-wrap:anywhere; }
table { width:100%; border-collapse:separate; border-spacing:0; table-layout:fixed; font-size:.88rem; }
th,td { padding:.58rem .62rem; border-right:1px solid var(--line); border-bottom:1px solid var(--line);
  text-align:left; vertical-align:top; overflow-wrap:anywhere; }
tr > :first-child { border-left:1px solid var(--line); }
thead tr:first-child > * { border-top:1px solid var(--line); }
th { color:#304469; background:#edf2f8; }
tbody tr:nth-child(even) { background:#fafbfd; }
tr.partial-coverage > td { background:#fff1e8 !important; border-bottom-color:#efb38d; border-top:1px solid #efb38d; }
tr.partial-coverage > td:first-child { border-left:5px solid #d05a28; }
.status-partial { display:inline-block; padding:.18rem .55rem; color:#9a3412; background:#ffedd5;
  border:1px solid #fdba74; border-radius:999px; font-weight:700; }
footer { color:#738096; text-align:center; }
@media print {
  @page { size:A4 landscape; margin:10mm; }
  html,body { max-width:none; padding:0; color:var(--ink); background:#fff; font-size:8.7pt; line-height:1.48; }
  header { padding:0 0 7mm; color:var(--ink); background:#fff; border:0; border-radius:0; }
  header h1,header th,header td,header code { color:var(--ink); }
  header table { background:#fff; }
  header th { background:#edf2f8; }
  main > section { margin:0 0 5mm; padding:0; border:0; }
  main > section:nth-child(2), main > section:nth-child(3), main > section:nth-child(5) { break-before:page; }
  table { font-size:7.3pt; }
  th,td { padding:.4rem .44rem; }
  pre { font-size:7.2pt; line-height:1.38; }
  footer { display:none; }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a compact UT/ST review report.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def highlight_partial_rows(rendered: str) -> str:
    def add_class(match: re.Match[str]) -> str:
        row = match.group(0)
        if "部分覆盖" not in row:
            return row
        row = row.replace("<tr>", '<tr class="partial-coverage">', 1)
        return row.replace("<strong>部分覆盖</strong>", '<span class="status-partial">部分覆盖</span>')

    return re.sub(r"<tr>.*?</tr>", add_class, rendered, flags=re.DOTALL)


def main() -> None:
    args = parse_args()
    source = args.source.read_text(encoding="utf-8")
    rendered = markdown.markdown(
        source,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    rendered = highlight_partial_rows(rendered)
    parts = re.split(r"(?=<h2>)", rendered)
    header = parts[0]
    sections = "".join(f"<section>{part}</section>" for part in parts[1:] if part.strip())
    title_match = re.search(r"<h1>(.*?)</h1>", header, flags=re.DOTALL)
    title = re.sub(r"<[^>]+>", "", title_match.group(1)) if title_match else args.source.stem
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="report-source" content="{html.escape(str(args.source))}"><meta name="report-source-sha256" content="{digest}">
<title>{html.escape(title)}</title><style>{STYLE}</style></head>
<body><header>{header}</header><main>{sections}</main><footer><p>TorchTitan-NPU · UT/ST 静态审查 · 测试未执行</p></footer></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
