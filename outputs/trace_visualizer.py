#!/usr/bin/env python3
"""Generate readable HTML reports for chain and tree speculative trace files."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PHASE_CHAIN = {"bridge", "draft", "verify"}
PHASE_TREE = {"bridge", "draft_tree", "verify_tree"}


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def fmt_prob(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    if abs(number) < 0.0001 and number != 0:
        return f"{number:.2e}"
    return f"{number:.4f}"


def clean_token(value: Any) -> str:
    if value is None:
        return "<missing>"
    text = str(value)
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if text == "":
        return "<empty>"
    if text == " ":
        return "<space>"
    return text


def compact_topk(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "token_id": item.get("token_id"),
                "token": clean_token(item.get("token")),
                "prob": as_float(item.get("prob")),
                "prob_text": fmt_prob(item.get("prob")),
            }
        )
    return out


def group_trace_by_iteration(trace: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in trace:
        iteration = event.get("iteration")
        if isinstance(iteration, int):
            grouped[iteration].append(event)
    return dict(sorted(grouped.items()))


def detect_mode(trace: list[dict[str, Any]]) -> str:
    phases = {event.get("phase") for event in trace}
    if "draft_tree" in phases or "verify_tree" in phases:
        return "tree"
    return "chain"


def base_report(data: dict[str, Any], input_path: Path, sample_index: int, mode: str) -> dict[str, Any]:
    results = data.get("results") or []
    sample = results[sample_index]
    metrics = sample.get("metrics") or {}
    return {
        "mode": mode,
        "source_file": str(input_path),
        "source_name": input_path.name,
        "dataset": data.get("dataset"),
        "total": data.get("total"),
        "accuracy": data.get("accuracy"),
        "avg_acceptance_rate": data.get("avg_acceptance_rate"),
        "sample_index": sample_index,
        "sample_id": sample.get("sample_id"),
        "sample_source": sample.get("source"),
        "is_correct": sample.get("is_correct"),
        "question": sample.get("question"),
        "reference": sample.get("reference"),
        "prediction": sample.get("prediction"),
        "metrics": metrics,
        "tree_args": data.get("tree_args"),
        "iterations": [],
    }


def normalize_chain(data: dict[str, Any], input_path: Path, sample_index: int) -> dict[str, Any]:
    report = base_report(data, input_path, sample_index, "chain")
    trace = data["results"][sample_index].get("trace") or []
    grouped = group_trace_by_iteration(trace)

    for iteration, events in grouped.items():
        bridge = next((e for e in events if e.get("phase") == "bridge"), None)
        drafts = sorted(
            [e for e in events if e.get("phase") == "draft"],
            key=lambda e: (e.get("draft_step", 0), e.get("elapsed_s", 0)),
        )
        verify = next((e for e in events if e.get("phase") == "verify"), None)
        per_position = verify.get("per_position", []) if isinstance(verify, dict) else []
        verify_by_pos = {
            item.get("pos"): item
            for item in per_position
            if isinstance(item, dict) and item.get("pos") is not None
        }

        nodes: list[dict[str, Any]] = []
        if bridge:
            nodes.append(
                {
                    "id": f"{iteration}:bridge",
                    "kind": "bridge",
                    "status": "bridge",
                    "token_id": bridge.get("base_token_id"),
                    "token": clean_token(bridge.get("base_token")),
                    "base_prob": as_float(bridge.get("base_prob")),
                    "base_prob_text": fmt_prob(bridge.get("base_prob")),
                    "base_topk": compact_topk(bridge.get("base_topk")),
                    "label": "bridge",
                    "raw": small_raw(bridge),
                }
            )

        for index, draft in enumerate(drafts):
            pos = draft.get("draft_step", index + 1)
            verify_item = verify_by_pos.get(index) or verify_by_pos.get(pos)
            accepted = verify_item.get("accepted") if isinstance(verify_item, dict) else None
            first_reject = (
                isinstance(verify, dict)
                and verify.get("first_reject_pos") is not None
                and verify.get("first_reject_pos") == index
            )
            status = "unverified"
            if accepted is True:
                status = "accepted"
            elif accepted is False:
                status = "rejected"
            node = {
                "id": f"{iteration}:draft:{index}",
                "kind": "draft",
                "status": status,
                "first_reject": first_reject,
                "draft_step": draft.get("draft_step", index + 1),
                "token_id": draft.get("draft_token_id"),
                "token": clean_token(draft.get("draft_token")),
                "draft_prob": as_float(draft.get("draft_prob")),
                "draft_prob_text": fmt_prob(draft.get("draft_prob")),
                "target_prob": None,
                "target_prob_text": "n/a",
                "acceptance_ratio": None,
                "acceptance_ratio_text": "n/a",
                "accepted": accepted,
                "draft_entropy": as_float(draft.get("draft_entropy")),
                "draft_hidden_norm": as_float(draft.get("draft_hidden_norm")),
                "draft_topk": compact_topk(draft.get("draft_topk")),
                "target_topk": [],
                "kl_draft_target": None,
                "target_hidden_norm": None,
                "label": f"draft {draft.get('draft_step', index + 1)}",
                "raw": small_raw(draft),
            }
            if isinstance(verify_item, dict):
                node.update(
                    {
                        "target_prob": as_float(verify_item.get("target_prob")),
                        "target_prob_text": fmt_prob(verify_item.get("target_prob")),
                        "acceptance_ratio": as_float(verify_item.get("acceptance_ratio")),
                        "acceptance_ratio_text": fmt_prob(verify_item.get("acceptance_ratio")),
                        "target_topk": compact_topk(verify_item.get("target_topk")),
                        "kl_draft_target": as_float(verify_item.get("kl_draft_target")),
                        "target_hidden_norm": as_float(verify_item.get("target_hidden_norm")),
                        "verify_raw": small_raw(verify_item),
                    }
                )
            nodes.append(node)

        report["iterations"].append(
            {
                "iteration": iteration,
                "summary": {
                    "block_size_proposed": value_or_none(verify, "block_size_proposed"),
                    "num_accepted": value_or_none(verify, "num_accepted"),
                    "num_rejected": value_or_none(verify, "num_rejected"),
                    "first_reject_pos": value_or_none(verify, "first_reject_pos"),
                    "acceptance_rate_this_block": as_float(value_or_none(verify, "acceptance_rate_this_block")),
                    "mean_kl_draft_target": as_float(value_or_none(verify, "mean_kl_draft_target")),
                    "max_kl_draft_target": as_float(value_or_none(verify, "max_kl_draft_target")),
                    "effective_kv_len_after": value_or_none(verify, "effective_kv_len_after"),
                },
                "nodes": nodes,
            }
        )

    return report


def normalize_tree(data: dict[str, Any], input_path: Path, sample_index: int) -> dict[str, Any]:
    report = base_report(data, input_path, sample_index, "tree")
    trace = data["results"][sample_index].get("trace") or []
    grouped = group_trace_by_iteration(trace)

    for iteration, events in grouped.items():
        draft_nodes = sorted(
            [e for e in events if e.get("phase") == "draft_tree"],
            key=lambda e: (e.get("depth", 0), e.get("tree_node_id", 0)),
        )
        verify = next((e for e in events if e.get("phase") == "verify_tree"), None)
        per_position = verify.get("per_position", []) if isinstance(verify, dict) else []
        verify_by_node_id = {
            item.get("tree_node_id"): item
            for item in per_position
            if isinstance(item, dict) and item.get("tree_node_id") is not None
        }

        nodes: list[dict[str, Any]] = []
        for draft in draft_nodes:
            node_id = draft.get("tree_node_id")
            verify_item = verify_by_node_id.get(node_id)
            status = "selected" if draft.get("selected_for_verify") else "unselected"
            accepted = None
            first_reject = False
            if isinstance(verify_item, dict):
                accepted = verify_item.get("accepted")
                candidate_status = verify_item.get("candidate_status")
                if candidate_status == "accepted_path":
                    status = "accepted"
                elif candidate_status == "rejected_at_position":
                    status = "rejected"
                    first_reject = True
                elif candidate_status == "sibling_not_chosen":
                    status = "sibling"
                elif accepted is True:
                    status = "accepted"
                elif accepted is False and draft.get("selected_for_verify"):
                    status = "selected"
            node = {
                "id": node_id,
                "kind": "tree_node",
                "status": status,
                "first_reject": first_reject,
                "tree_node_id": node_id,
                "parent_id": draft.get("parent_id"),
                "depth": draft.get("depth"),
                "draft_step": draft.get("draft_step"),
                "selected_for_verify": bool(draft.get("selected_for_verify")),
                "token_id": draft.get("draft_token_id"),
                "token": clean_token(draft.get("draft_token")),
                "draft_prob": as_float(draft.get("draft_prob")),
                "draft_prob_text": fmt_prob(draft.get("draft_prob")),
                "cumulative_draft_prob": as_float(draft.get("cumulative_draft_prob")),
                "cumulative_draft_prob_text": fmt_prob(draft.get("cumulative_draft_prob")),
                "target_prob": None,
                "target_prob_text": "n/a",
                "acceptance_ratio": None,
                "acceptance_ratio_text": "n/a",
                "accepted": accepted,
                "draft_entropy": as_float(draft.get("draft_entropy")),
                "draft_hidden_norm": as_float(draft.get("draft_hidden_norm")),
                "draft_topk": compact_topk(draft.get("draft_topk")),
                "target_topk": [],
                "kl_draft_target": None,
                "target_hidden_norm": None,
                "candidate_status": None,
                "raw": small_raw(draft),
            }
            if isinstance(verify_item, dict):
                node.update(
                    {
                        "target_prob": as_float(verify_item.get("target_prob")),
                        "target_prob_text": fmt_prob(verify_item.get("target_prob")),
                        "acceptance_ratio": as_float(verify_item.get("acceptance_ratio")),
                        "acceptance_ratio_text": fmt_prob(verify_item.get("acceptance_ratio")),
                        "target_topk": compact_topk(verify_item.get("target_topk")),
                        "kl_draft_target": as_float(verify_item.get("kl_draft_target")),
                        "target_hidden_norm": as_float(verify_item.get("target_hidden_norm")),
                        "candidate_status": verify_item.get("candidate_status"),
                        "verify_raw": small_raw(verify_item),
                    }
                )
            nodes.append(node)

        nodes = add_tree_layout(nodes)
        report["iterations"].append(
            {
                "iteration": iteration,
                "summary": {
                    "block_size_proposed": value_or_none(verify, "block_size_proposed"),
                    "num_accepted": value_or_none(verify, "num_accepted"),
                    "num_rejected": value_or_none(verify, "num_rejected"),
                    "first_reject_pos": value_or_none(verify, "first_reject_pos"),
                    "acceptance_rate_this_block": as_float(value_or_none(verify, "acceptance_rate_this_block")),
                    "mean_kl_draft_target": as_float(value_or_none(verify, "mean_kl_draft_target")),
                    "max_kl_draft_target": as_float(value_or_none(verify, "max_kl_draft_target")),
                    "effective_kv_len_after": value_or_none(verify, "effective_kv_len_after"),
                    "tree_metadata": value_or_none(verify, "tree_metadata") or {},
                },
                "nodes": nodes,
            }
        )

    return report


def value_or_none(mapping: Any, key: str) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key)
    return None


def small_raw(event: dict[str, Any]) -> dict[str, Any]:
    skip = {"draft_topk", "target_topk", "base_topk", "prefill_topk", "per_position"}
    out: dict[str, Any] = {}
    for key, value in event.items():
        if key in skip:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def add_tree_layout(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    id_to_node = {node["tree_node_id"]: node for node in nodes}
    children: dict[Any, list[Any]] = defaultdict(list)
    roots: list[Any] = []
    for node in nodes:
        node_id = node["tree_node_id"]
        parent_id = node.get("parent_id")
        if parent_id is None or parent_id not in id_to_node:
            roots.append(node_id)
        else:
            children[parent_id].append(node_id)

    for child_list in children.values():
        child_list.sort(key=lambda node_id: (id_to_node[node_id].get("draft_prob") is None, -float(id_to_node[node_id].get("draft_prob") or 0), node_id))
    roots.sort(key=lambda node_id: (id_to_node[node_id].get("draft_prob") is None, -float(id_to_node[node_id].get("draft_prob") or 0), node_id))

    y_positions: dict[Any, float] = {}
    cursor = 0.0

    def assign(node_id: Any) -> float:
        nonlocal cursor
        if node_id in y_positions:
            return y_positions[node_id]
        child_ids = children.get(node_id, [])
        if not child_ids:
            y_positions[node_id] = cursor
            cursor += 1.0
            return y_positions[node_id]
        child_ys = [assign(child_id) for child_id in child_ids]
        y_positions[node_id] = sum(child_ys) / len(child_ys)
        return y_positions[node_id]

    for root_id in roots:
        assign(root_id)
        cursor += 0.6

    max_depth = max((int(node.get("depth") or 1) for node in nodes), default=1)
    for node in nodes:
        depth = int(node.get("depth") or 1)
        node["x"] = 60 + (depth - 1) * 245
        node["y"] = 52 + y_positions.get(node["tree_node_id"], 0.0) * 112
        node["layout_depth"] = depth
        node["layout_max_depth"] = max_depth

    return nodes


def make_report(data: dict[str, Any], input_path: Path, sample_index: int, requested_mode: str) -> dict[str, Any]:
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Input JSON has no results array.")
    if sample_index < 0 or sample_index >= len(results):
        raise ValueError(f"sample index {sample_index} is out of range for {len(results)} result(s).")
    trace = results[sample_index].get("trace")
    if not isinstance(trace, list):
        raise ValueError("Selected result has no trace array.")

    detected = detect_mode(trace)
    mode = detected if requested_mode == "auto" else requested_mode
    if mode == "chain":
        if requested_mode != "auto" and detected != "chain":
            raise ValueError(f"Requested chain mode, but trace looks like {detected}.")
        return normalize_chain(data, input_path, sample_index)
    if mode == "tree":
        if requested_mode != "auto" and detected != "tree":
            raise ValueError(f"Requested tree mode, but trace looks like {detected}.")
        return normalize_tree(data, input_path, sample_index)
    raise ValueError(f"Unsupported mode: {mode}")


def render_html(report: dict[str, Any]) -> str:
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trace Visualizer - {report['source_name']}</title>
<style>
:root {{
  --bg: #f6f8fb;
  --panel: #ffffff;
  --ink: #172033;
  --muted: #64748b;
  --line: #d7dde8;
  --blue: #2563eb;
  --green: #059669;
  --red: #e11d48;
  --amber: #d97706;
  --slate: #475569;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  letter-spacing: 0;
}}
header {{
  background: #ffffff;
  border-bottom: 1px solid var(--line);
  padding: 18px 22px 14px;
}}
h1 {{
  margin: 0 0 6px;
  font-size: 22px;
  line-height: 1.2;
  font-weight: 720;
}}
.subtle {{ color: var(--muted); }}
.top-grid {{
  display: grid;
  grid-template-columns: repeat(6, minmax(120px, 1fr));
  gap: 10px;
  margin-top: 14px;
}}
.metric {{
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 9px 10px;
  min-height: 58px;
}}
.metric-label {{
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
}}
.metric-value {{
  margin-top: 5px;
  font-size: 16px;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: end;
  padding: 14px 22px;
  background: #fbfcfe;
  border-bottom: 1px solid var(--line);
}}
label {{
  display: grid;
  gap: 5px;
  color: var(--muted);
  font-size: 12px;
}}
select, input, button {{
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 0 10px;
  background: #ffffff;
  color: var(--ink);
  font: inherit;
}}
button {{
  cursor: pointer;
  color: #ffffff;
  border-color: var(--blue);
  background: var(--blue);
  font-weight: 650;
}}
main {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 14px;
  padding: 14px;
}}
.graph-panel, .side-panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  min-height: 520px;
}}
.graph-panel {{
  overflow: auto;
  position: relative;
}}
.side-panel {{
  overflow: auto;
  max-height: calc(100vh - 190px);
  padding: 14px;
}}
.panel-title {{
  margin: 0 0 10px;
  font-size: 14px;
  font-weight: 760;
}}
.summary-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 14px;
}}
.summary-item {{
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 8px;
}}
.summary-item span {{
  display: block;
  color: var(--muted);
  font-size: 11px;
  margin-bottom: 4px;
}}
.summary-item strong {{
  font-size: 13px;
}}
#graph {{
  display: block;
  min-width: 100%;
  min-height: 520px;
}}
.edge {{
  stroke: #bac4d4;
  stroke-width: 1.6;
  fill: none;
}}
.edge-path {{
  stroke: #64748b;
  stroke-width: 2.2;
}}
.node rect {{
  rx: 8;
  ry: 8;
  stroke-width: 1.5;
  filter: drop-shadow(0 1px 1px rgba(15, 23, 42, 0.08));
}}
.node text {{
  pointer-events: none;
  fill: #172033;
}}
.node .token {{
  font-weight: 720;
  font-size: 13px;
}}
.node .meta {{
  font-size: 11px;
  fill: #475569;
}}
.node.accepted rect {{ fill: #ecfdf5; stroke: var(--green); }}
.node.rejected rect {{ fill: #fff1f2; stroke: var(--red); stroke-width: 2.6; }}
.node.bridge rect {{ fill: #eff6ff; stroke: var(--blue); }}
.node.selected rect {{ fill: #eef6ff; stroke: #0284c7; }}
.node.sibling rect {{ fill: #f8fafc; stroke: #94a3b8; }}
.node.unselected rect {{ fill: #f8fafc; stroke: #cbd5e1; opacity: 0.72; }}
.node.unverified rect {{ fill: #fff7ed; stroke: var(--amber); }}
.node.highlight rect {{ stroke: #111827; stroke-width: 3; }}
.overview-row rect {{
  fill: #ffffff;
  stroke: var(--line);
  rx: 8;
  ry: 8;
  filter: drop-shadow(0 1px 1px rgba(15, 23, 42, 0.06));
}}
.overview-row:hover rect {{
  stroke: var(--blue);
  stroke-width: 2;
}}
.overview-row text {{
  fill: var(--ink);
}}
.overview-row .row-meta {{
  fill: var(--muted);
  font-size: 12px;
}}
.mini-dot {{
  stroke: rgba(15, 23, 42, 0.18);
  stroke-width: 1;
}}
.mini-dot.accepted {{ fill: #10b981; }}
.mini-dot.rejected {{ fill: #fb7185; }}
.mini-dot.selected {{ fill: #38bdf8; }}
.mini-dot.sibling {{ fill: #94a3b8; }}
.mini-dot.unselected {{ fill: #cbd5e1; }}
.mini-dot.bridge {{ fill: #60a5fa; }}
.mini-dot.unverified {{ fill: #f59e0b; }}
.legend {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  color: var(--muted);
  font-size: 12px;
}}
.chip {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
}}
.swatch {{
  width: 11px;
  height: 11px;
  border-radius: 3px;
  border: 1px solid #94a3b8;
  background: #fff;
}}
.swatch.accepted {{ background: #ecfdf5; border-color: var(--green); }}
.swatch.rejected {{ background: #fff1f2; border-color: var(--red); }}
.swatch.selected {{ background: #eef6ff; border-color: #0284c7; }}
.swatch.unselected {{ background: #f8fafc; border-color: #cbd5e1; }}
.swatch.bridge {{ background: #eff6ff; border-color: var(--blue); }}
.detail-block {{
  border-top: 1px solid var(--line);
  padding-top: 12px;
  margin-top: 12px;
}}
.kv {{
  display: grid;
  grid-template-columns: 145px minmax(0, 1fr);
  gap: 7px 10px;
  margin-top: 8px;
}}
.kv div:nth-child(odd) {{
  color: var(--muted);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 8px;
  font-size: 12px;
}}
th, td {{
  border-bottom: 1px solid #e5eaf2;
  padding: 6px 4px;
  text-align: left;
  vertical-align: top;
}}
th {{
  color: var(--muted);
  font-weight: 650;
}}
.token-cell {{
  max-width: 150px;
  overflow-wrap: anywhere;
}}
.question {{
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 9px;
  max-height: 160px;
  overflow: auto;
  white-space: pre-wrap;
  line-height: 1.35;
}}
@media (max-width: 980px) {{
  main {{ grid-template-columns: 1fr; }}
  .side-panel {{ max-height: none; }}
  .top-grid {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
}}
</style>
</head>
<body>
<header>
  <h1 id="title">Trace Visualizer</h1>
  <div class="subtle" id="subtitle"></div>
  <div class="top-grid" id="topMetrics"></div>
</header>
<section class="controls">
  <label>View
    <select id="iterationSelect"></select>
  </label>
  <label>Status
    <select id="statusFilter">
      <option value="all">all nodes</option>
      <option value="accepted">accepted</option>
      <option value="rejected">rejected</option>
      <option value="selected">selected/verified</option>
      <option value="unselected">unselected</option>
    </select>
  </label>
  <label>Probability label
    <select id="probMode">
      <option value="both">draft + target</option>
      <option value="draft">draft only</option>
      <option value="target">target only</option>
      <option value="cumulative">cumulative draft</option>
    </select>
  </label>
  <label>Search token
    <input id="searchBox" type="search" placeholder="token text or id">
  </label>
  <button id="saveSvg">Save SVG</button>
</section>
<main>
  <section class="graph-panel">
    <div class="legend">
      <span class="chip"><span class="swatch bridge"></span>bridge/base</span>
      <span class="chip"><span class="swatch accepted"></span>accepted</span>
      <span class="chip"><span class="swatch rejected"></span>rejected</span>
      <span class="chip"><span class="swatch selected"></span>selected/verified</span>
      <span class="chip"><span class="swatch unselected"></span>unselected</span>
    </div>
    <svg id="graph" xmlns="http://www.w3.org/2000/svg"></svg>
  </section>
  <aside class="side-panel">
    <h2 class="panel-title">Iteration summary</h2>
    <div class="summary-grid" id="iterationSummary"></div>
    <div class="detail-block">
      <h2 class="panel-title">Selected node</h2>
      <div id="nodeDetails" class="subtle">Click a token node to inspect probabilities and top-k lists.</div>
    </div>
    <div class="detail-block">
      <h2 class="panel-title">Question</h2>
      <div class="question" id="questionBox"></div>
    </div>
  </aside>
</main>
<script id="report-data" type="application/json">{payload}</script>
<script>
const REPORT = JSON.parse(document.getElementById("report-data").textContent);
const SVG_NS = "http://www.w3.org/2000/svg";
let selectedNode = null;

const els = {{
  title: document.getElementById("title"),
  subtitle: document.getElementById("subtitle"),
  topMetrics: document.getElementById("topMetrics"),
  iterationSelect: document.getElementById("iterationSelect"),
  statusFilter: document.getElementById("statusFilter"),
  probMode: document.getElementById("probMode"),
  searchBox: document.getElementById("searchBox"),
  graph: document.getElementById("graph"),
  iterationSummary: document.getElementById("iterationSummary"),
  nodeDetails: document.getElementById("nodeDetails"),
  questionBox: document.getElementById("questionBox"),
  saveSvg: document.getElementById("saveSvg"),
}};

function formatValue(value) {{
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "number") {{
    if (Math.abs(value) < 0.0001 && value !== 0) return value.toExponential(2);
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }}
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}}

function metric(label, value) {{
  return `<div class="metric"><div class="metric-label">${{escapeHtml(label)}}</div><div class="metric-value" title="${{escapeHtml(formatValue(value))}}">${{escapeHtml(formatValue(value))}}</div></div>`;
}}

function escapeHtml(text) {{
  return String(text ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
}}

function makeSvg(tag) {{
  return document.createElementNS(SVG_NS, tag);
}}

function init() {{
  els.title.textContent = `${{REPORT.mode.toUpperCase()}} Trace Visualizer`;
  const totalNodes = REPORT.iterations.reduce((n, item) => n + (item.nodes ? item.nodes.length : 0), 0);
  els.subtitle.textContent = `${{REPORT.source_name}} | sample ${{REPORT.sample_id ?? REPORT.sample_index}} | full trace: ${{REPORT.iterations.length}} iteration(s), ${{totalNodes}} node(s)`;
  els.topMetrics.innerHTML = [
    metric("acceptance", REPORT.metrics.acceptance_rate ?? REPORT.avg_acceptance_rate),
    metric("path acceptance", REPORT.metrics.path_acceptance_rate),
    metric("node acceptance", REPORT.metrics.node_acceptance_rate),
    metric("tokens emitted", REPORT.metrics.tokens_emitted),
    metric("iterations", REPORT.metrics.iteration_count ?? REPORT.iterations.length),
    metric("visualized nodes", totalNodes),
  ].join("");
  els.questionBox.textContent = REPORT.question || "n/a";
  const allOpt = document.createElement("option");
  allOpt.value = "overview";
  allOpt.textContent = `all iterations overview (${{REPORT.iterations.length}})`;
  els.iterationSelect.appendChild(allOpt);
  for (const item of REPORT.iterations) {{
    const opt = document.createElement("option");
    opt.value = String(item.iteration);
    const count = item.nodes ? item.nodes.length : 0;
    opt.textContent = `iteration ${{item.iteration}} (${{count}} nodes)`;
    els.iterationSelect.appendChild(opt);
  }}
  els.iterationSelect.addEventListener("change", render);
  els.statusFilter.addEventListener("change", render);
  els.probMode.addEventListener("change", render);
  els.searchBox.addEventListener("input", render);
  els.saveSvg.addEventListener("click", saveSvg);
  render();
}}

function currentIteration() {{
  if (els.iterationSelect.value === "overview") return null;
  const wanted = Number(els.iterationSelect.value);
  return REPORT.iterations.find(item => item.iteration === wanted) || REPORT.iterations[0];
}}

function passesFilter(node) {{
  const filter = els.statusFilter.value;
  if (filter === "all") return true;
  if (filter === "selected") return Boolean(node.selected_for_verify) || ["selected", "accepted", "rejected", "sibling"].includes(node.status);
  if (filter === "unselected") return node.status === "unselected";
  return node.status === filter;
}}

function isSearchHit(node) {{
  const q = els.searchBox.value.trim().toLowerCase();
  if (!q) return false;
  return String(node.token ?? "").toLowerCase().includes(q) || String(node.token_id ?? "").includes(q) || String(node.tree_node_id ?? "").includes(q);
}}

function render() {{
  const item = currentIteration();
  selectedNode = null;
  if (!item) {{
    renderOverviewSummary();
    renderOverview();
    els.nodeDetails.textContent = "This overview includes every iteration. Click a row to open that iteration's detailed graph.";
    return;
  }}
  renderSummary(item);
  renderGraph(item);
  els.nodeDetails.textContent = "Click a token node to inspect probabilities and top-k lists.";
}}

function renderOverviewSummary() {{
  const totalNodes = REPORT.iterations.reduce((n, item) => n + (item.nodes ? item.nodes.length : 0), 0);
  const totals = REPORT.iterations.reduce((acc, item) => {{
    for (const node of item.nodes || []) {{
      acc[node.status] = (acc[node.status] || 0) + 1;
      if (node.selected_for_verify) acc.selectedForVerify += 1;
    }}
    return acc;
  }}, {{accepted: 0, rejected: 0, selected: 0, sibling: 0, unselected: 0, unverified: 0, bridge: 0, selectedForVerify: 0}});
  const rows = [
    ["iterations", REPORT.iterations.length],
    ["all nodes", totalNodes],
    ["accepted nodes", totals.accepted],
    ["rejected nodes", totals.rejected],
    ["selected/verified", totals.selectedForVerify || totals.selected],
    ["unselected", totals.unselected],
    ["acceptance", REPORT.metrics.acceptance_rate ?? REPORT.avg_acceptance_rate],
    ["path acceptance", REPORT.metrics.path_acceptance_rate],
    ["node acceptance", REPORT.metrics.node_acceptance_rate],
    ["tokens emitted", REPORT.metrics.tokens_emitted],
  ];
  els.iterationSummary.innerHTML = rows.map(([k, v]) =>
    `<div class="summary-item"><span>${{escapeHtml(k)}}</span><strong>${{escapeHtml(formatValue(v))}}</strong></div>`
  ).join("");
}}

function renderSummary(item) {{
  const s = item.summary || {{}};
  const md = s.tree_metadata || {{}};
  const rows = [
    ["proposed", s.block_size_proposed],
    ["accepted", s.num_accepted],
    ["rejected", s.num_rejected],
    ["first reject", s.first_reject_pos],
    ["block acceptance", s.acceptance_rate_this_block],
    ["mean KL", s.mean_kl_draft_target],
    ["max KL", s.max_kl_draft_target],
    ["kv len after", s.effective_kv_len_after],
  ];
  if (REPORT.mode === "tree") {{
    rows.push(["tree nodes", md.tree_nodes_generated]);
    rows.push(["selected", md.tree_nodes_selected]);
    rows.push(["path accepted", md.path_tokens_accepted]);
    rows.push(["path checked", md.path_tokens_checked]);
    rows.push(["verify backend", md.verify_backend]);
    rows.push(["accept mode", md.tree_accept_mode]);
  }}
  els.iterationSummary.innerHTML = rows.map(([k, v]) =>
    `<div class="summary-item"><span>${{escapeHtml(k)}}</span><strong>${{escapeHtml(formatValue(v))}}</strong></div>`
  ).join("");
}}

function renderGraph(item) {{
  els.graph.innerHTML = "";
  if (REPORT.mode === "chain") renderChain(item);
  else renderTree(item);
}}

function renderOverview() {{
  els.graph.innerHTML = "";
  const rowHeight = 54;
  const left = 24;
  const labelWidth = 300;
  const usableWidth = Math.max(740, 18 * Math.max(...REPORT.iterations.map(item => (item.nodes || []).length), 12));
  const w = left + labelWidth + usableWidth + 44;
  const h = 58 + REPORT.iterations.length * rowHeight;
  els.graph.setAttribute("width", w);
  els.graph.setAttribute("height", h);
  els.graph.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);

  const heading = makeSvg("text");
  heading.setAttribute("x", left);
  heading.setAttribute("y", 26);
  heading.setAttribute("class", "token");
  heading.textContent = `Full trace overview: ${{REPORT.iterations.length}} iterations, ${{REPORT.iterations.reduce((n, item) => n + (item.nodes ? item.nodes.length : 0), 0)}} nodes. Click a row for details.`;
  els.graph.appendChild(heading);

  REPORT.iterations.forEach((item, idx) => {{
    const y = 44 + idx * rowHeight;
    const nodes = item.nodes || [];
    const row = makeSvg("g");
    row.setAttribute("class", "overview-row");
    row.setAttribute("transform", `translate(${{left}}, ${{y}})`);
    row.setAttribute("tabindex", "0");
    row.style.cursor = "pointer";
    const rect = makeSvg("rect");
    rect.setAttribute("width", w - left * 2);
    rect.setAttribute("height", rowHeight - 8);
    row.appendChild(rect);

    const accepted = nodes.filter(n => n.status === "accepted").length;
    const rejected = nodes.filter(n => n.status === "rejected").length;
    const selected = nodes.filter(n => n.selected_for_verify || ["selected", "accepted", "rejected", "sibling"].includes(n.status)).length;
    const text = makeSvg("text");
    text.setAttribute("x", 12);
    text.setAttribute("y", 19);
    text.textContent = `iteration ${{item.iteration}}`;
    row.appendChild(text);
    const meta = makeSvg("text");
    meta.setAttribute("x", 12);
    meta.setAttribute("y", 36);
    meta.setAttribute("class", "row-meta");
    meta.textContent = `${{nodes.length}} nodes | accepted ${{accepted}} | rejected ${{rejected}} | selected/verified ${{selected}} | block acc ${{formatValue(item.summary?.acceptance_rate_this_block)}}`;
    row.appendChild(meta);

    const dotGap = Math.max(9, Math.min(18, usableWidth / Math.max(nodes.length, 1)));
    nodes.forEach((node, nodeIdx) => {{
      if (!passesFilter(node)) return;
      const circle = makeSvg("circle");
      circle.setAttribute("class", `mini-dot ${{node.status || "unverified"}}${{isSearchHit(node) ? " highlight" : ""}}`);
      circle.setAttribute("cx", labelWidth + nodeIdx * dotGap);
      circle.setAttribute("cy", 23);
      circle.setAttribute("r", isSearchHit(node) ? 6 : 4.5);
      const title = makeSvg("title");
      title.textContent = `${{node.token}} | ${{node.status}} | draft ${{node.draft_prob_text ?? "n/a"}} | target ${{node.target_prob_text ?? "n/a"}}`;
      circle.appendChild(title);
      row.appendChild(circle);
    }});

    row.addEventListener("click", () => {{
      els.iterationSelect.value = String(item.iteration);
      render();
    }});
    row.addEventListener("keydown", event => {{
      if (event.key === "Enter" || event.key === " ") {{
        els.iterationSelect.value = String(item.iteration);
        render();
      }}
    }});
    els.graph.appendChild(row);
  }});
}}

function renderChain(item) {{
  const nodes = item.nodes.filter(passesFilter);
  const w = Math.max(900, 90 + nodes.length * 225);
  const h = 360;
  els.graph.setAttribute("width", w);
  els.graph.setAttribute("height", h);
  els.graph.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  const y = 115;
  const placed = nodes.map((node, i) => ({{node, x: 55 + i * 220, y}}));
  for (let i = 0; i < placed.length - 1; i++) {{
    drawEdge(placed[i].x + 176, y + 56, placed[i + 1].x, y + 56, false);
  }}
  for (const p of placed) {{
    drawNode(p.node, p.x, p.y, 176, 120);
  }}
}}

function renderTree(item) {{
  const allNodes = item.nodes || [];
  const visibleIds = new Set(allNodes.filter(passesFilter).map(n => n.tree_node_id));
  const visible = allNodes.filter(n => visibleIds.has(n.tree_node_id));
  const maxX = Math.max(...visible.map(n => n.x || 0), 800);
  const maxY = Math.max(...visible.map(n => n.y || 0), 460);
  const w = Math.max(980, maxX + 260);
  const h = Math.max(560, maxY + 165);
  els.graph.setAttribute("width", w);
  els.graph.setAttribute("height", h);
  els.graph.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  const byId = new Map(allNodes.map(n => [n.tree_node_id, n]));
  for (const node of visible) {{
    if (node.parent_id !== null && node.parent_id !== undefined && visibleIds.has(node.parent_id)) {{
      const parent = byId.get(node.parent_id);
      const pathEdge = node.status === "accepted" || node.status === "rejected";
      drawEdge((parent.x || 0) + 180, (parent.y || 0) + 48, node.x || 0, (node.y || 0) + 48, pathEdge);
    }}
  }}
  for (const node of visible) {{
    drawNode(node, node.x || 0, node.y || 0, 180, 104);
  }}
}}

function drawEdge(x1, y1, x2, y2, pathEdge) {{
  const path = makeSvg("path");
  const mid = (x1 + x2) / 2;
  path.setAttribute("d", `M ${{x1}} ${{y1}} C ${{mid}} ${{y1}}, ${{mid}} ${{y2}}, ${{x2}} ${{y2}}`);
  path.setAttribute("class", pathEdge ? "edge edge-path" : "edge");
  els.graph.appendChild(path);
}}

function drawNode(node, x, y, w, h) {{
  const g = makeSvg("g");
  const classes = ["node", node.status || "unverified"];
  if (isSearchHit(node)) classes.push("highlight");
  g.setAttribute("class", classes.join(" "));
  g.setAttribute("transform", `translate(${{x}}, ${{y}})`);
  g.setAttribute("tabindex", "0");
  g.style.cursor = "pointer";
  const rect = makeSvg("rect");
  rect.setAttribute("width", w);
  rect.setAttribute("height", h);
  g.appendChild(rect);
  const title = makeSvg("title");
  title.textContent = `${{node.token}} | status: ${{node.status}}`;
  g.appendChild(title);
  const lines = nodeLabelLines(node);
  lines.forEach((line, i) => {{
    const text = makeSvg("text");
    text.setAttribute("x", 12);
    text.setAttribute("y", 22 + i * 17);
    text.setAttribute("class", i === 0 ? "token" : "meta");
    text.textContent = trimLine(line, i === 0 ? 23 : 28);
    g.appendChild(text);
  }});
  g.addEventListener("click", () => showNode(node));
  g.addEventListener("keydown", event => {{
    if (event.key === "Enter" || event.key === " ") showNode(node);
  }});
  els.graph.appendChild(g);
}}

function nodeLabelLines(node) {{
  const mode = els.probMode.value;
  const head = node.kind === "bridge" ? `base ${{node.token}}` : String(node.token);
  const lines = [head];
  if (node.kind === "bridge") {{
    lines.push(`p(base) ${{node.base_prob_text ?? "n/a"}}`);
    lines.push(`token_id ${{node.token_id ?? "n/a"}}`);
    return lines;
  }}
  if (REPORT.mode === "tree") lines.push(`#${{node.tree_node_id}} d${{node.depth}} ${{node.status}}`);
  else lines.push(`${{node.label ?? ""}} ${{node.status}}`);
  if (mode === "draft" || mode === "both") lines.push(`draft ${{node.draft_prob_text ?? "n/a"}}`);
  if (mode === "target" || mode === "both") lines.push(`target ${{node.target_prob_text ?? "n/a"}}`);
  if (mode === "cumulative") lines.push(`cum ${{node.cumulative_draft_prob_text ?? "n/a"}}`);
  if (node.acceptance_ratio_text && node.acceptance_ratio_text !== "n/a") lines.push(`ratio ${{node.acceptance_ratio_text}}`);
  return lines.slice(0, 6);
}}

function trimLine(text, maxLen) {{
  const s = String(text ?? "");
  return s.length > maxLen ? s.slice(0, maxLen - 1) + "..." : s;
}}

function showNode(node) {{
  selectedNode = node;
  const rows = [
    ["status", node.status],
    ["token", node.token],
    ["token_id", node.token_id],
    ["draft_prob", node.draft_prob_text],
    ["target_prob", node.target_prob_text],
    ["cumulative_draft_prob", node.cumulative_draft_prob_text],
    ["acceptance_ratio", node.acceptance_ratio_text],
    ["accepted", node.accepted],
    ["kl_draft_target", node.kl_draft_target],
    ["draft_entropy", node.draft_entropy],
    ["draft_hidden_norm", node.draft_hidden_norm],
    ["target_hidden_norm", node.target_hidden_norm],
  ];
  if (REPORT.mode === "tree") {{
    rows.unshift(["tree_node_id", node.tree_node_id], ["parent_id", node.parent_id], ["depth", node.depth], ["selected_for_verify", node.selected_for_verify], ["candidate_status", node.candidate_status]);
  }}
  let html = `<div class="kv">${{rows.map(([k, v]) => `<div>${{escapeHtml(k)}}</div><div>${{escapeHtml(formatValue(v))}}</div>`).join("")}}</div>`;
  html += topkTable("target_topk", node.target_topk);
  html += topkTable("draft_topk", node.draft_topk);
  html += topkTable("base_topk", node.base_topk);
  html += rawBlock("raw", node.raw);
  if (node.verify_raw) html += rawBlock("verify", node.verify_raw);
  els.nodeDetails.innerHTML = html;
}}

function topkTable(title, rows) {{
  if (!rows || rows.length === 0) return "";
  const body = rows.map(row => `<tr><td>${{escapeHtml(row.token_id)}}</td><td class="token-cell">${{escapeHtml(row.token)}}</td><td>${{escapeHtml(row.prob_text)}}</td></tr>`).join("");
  return `<div class="detail-block"><h2 class="panel-title">${{escapeHtml(title)}}</h2><table><thead><tr><th>id</th><th>token</th><th>prob</th></tr></thead><tbody>${{body}}</tbody></table></div>`;
}}

function rawBlock(title, obj) {{
  if (!obj || Object.keys(obj).length === 0) return "";
  const rows = Object.entries(obj).map(([k, v]) => `<div>${{escapeHtml(k)}}</div><div>${{escapeHtml(formatValue(v))}}</div>`).join("");
  return `<div class="detail-block"><h2 class="panel-title">${{escapeHtml(title)}}</h2><div class="kv">${{rows}}</div></div>`;
}}

function saveSvg() {{
  const source = new XMLSerializer().serializeToString(els.graph);
  const blob = new Blob([source], {{type: "image/svg+xml;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${{REPORT.mode}}_iteration_${{els.iterationSelect.value}}.svg`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}

init();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-file HTML visualization for chain/tree speculative decoding trace JSON."
    )
    parser.add_argument("mode_or_input", help="'auto', 'chain', 'tree', or the input JSON path")
    parser.add_argument("input_path", nargs="?", help="Input JSON path when mode is provided")
    parser.add_argument("--out", help="Output HTML path. Defaults next to the input JSON.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index into results[]. Defaults to 0.")
    return parser.parse_args()


def resolve_mode_and_path(args: argparse.Namespace) -> tuple[str, Path]:
    if args.input_path is None:
        return "auto", Path(args.mode_or_input)
    if args.mode_or_input not in {"auto", "chain", "tree"}:
        raise SystemExit("First argument must be auto, chain, tree, or omit mode and pass only the JSON path.")
    return args.mode_or_input, Path(args.input_path)


def default_out_path(input_path: Path, mode: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_{mode}_trace_report.html")


def main() -> None:
    args = parse_args()
    requested_mode, input_path = resolve_mode_and_path(args)
    input_path = input_path.expanduser().resolve()
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    report = make_report(data, input_path, args.sample_index, requested_mode)
    out_path = Path(args.out).expanduser().resolve() if args.out else default_out_path(input_path, report["mode"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(report), encoding="utf-8")

    node_count = sum(len(item.get("nodes", [])) for item in report["iterations"])
    print(f"mode: {report['mode']}")
    print(f"iterations: {len(report['iterations'])}")
    print(f"nodes: {node_count}")
    print(f"wrote: {out_path}")
    print(f"open: file://{out_path}")


if __name__ == "__main__":
    main()
