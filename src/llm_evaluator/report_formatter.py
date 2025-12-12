# src/llm_evaluator/report_formatter.py
"""
Utilities to convert the 'combined' evaluator output into
a compact, human-friendly 'clean' report (json + optional html).
"""

from typing import Dict, Any, List
import html
import json
from datetime import datetime

def _nl_summary(aggregates: Dict[str, Any]) -> str:
    mr = aggregates.get("mean_relevance", 0.0)
    mc = aggregates.get("mean_completeness", 0.0)
    mh = aggregates.get("mean_hallucination_ratio", 0.0)
    return (
        f"Across {aggregates.get('num_turns', 0)} evaluated replies: "
        f"mean relevance = {mr:.2f}, mean completeness = {mc:.2f}, "
        f"mean hallucination ratio = {mh:.2f}."
    )

def make_clean_report(combined: Dict[str, Any]) -> Dict[str, Any]:
    per_turn_scores = []
    hallucination_findings = []
    warnings = []

    metadata = combined.get("metadata", {})
    conv_summary = combined.get("conversation_summary", {})
    aggregates = combined.get("aggregates", {})

    for tr in combined.get("turn_reports", []):
        pair_index = tr.get("pair_index")
        ai_meta = tr.get("ai_meta", {}) or {}
        turn_id = ai_meta.get("turn") or ai_meta.get("id") or pair_index

        hall = tr.get("hallucination", {})
        hall_ratio = float(hall.get("hallucination_ratio", 0.0))

        manual = bool(tr.get("requires_manual_review"))
        per_turn_scores.append({
            "pair_index": pair_index,
            "turn_id": turn_id,
            "user_preview": tr.get("user_text_preview","")[:200],
            "ai_preview": tr.get("ai_text_preview","")[:300],
            "relevance": round(float(tr.get("relevance",0.0)), 3),
            "completeness": round(float(tr.get("completeness",0.0)), 3),
            "hallucination_ratio": round(hall_ratio, 3),
            "requires_manual_review": manual,
        })

        for c in hall.get("claims", []):
            if c.get("evidence_confidence") in ("weak", "medium"):
                evidence = c.get("evidence", [])
                top_e = evidence[0] if evidence else {}
                snippet = top_e.get("snippet") or top_e.get("text") or ""
                source = top_e.get("source")

                hallucination_findings.append({
                    "pair_index": pair_index,
                    "turn_id": turn_id,
                    "claim": c.get("claim"),
                    "claim_type": c.get("claim_type"),
                    "evidence_confidence": c.get("evidence_confidence"),
                    "best_support_score": c.get("best_support_score"),
                    "top_evidence_snippet": snippet[:280],
                    "top_evidence_source": source
                })

    total_cost = aggregates.get("total_estimated_cost_usd")
    total_input = aggregates.get("total_input_tokens")
    total_output = aggregates.get("total_output_tokens")

    if any(x["requires_manual_review"] for x in per_turn_scores):
        warnings.append("Some responses have weak/medium evidence — manual review recommended.")
    if aggregates.get("mean_completeness", 0.0) < 0.35:
        warnings.append("Completeness is low (mean < 0.35).")

    clean = {
        "generated_at": combined.get("generated_at"),
        "conversation_summary": conv_summary,
        "summary": {
            "mean_relevance": round(float(aggregates.get("mean_relevance", 0.0)), 3),
            "mean_completeness": round(float(aggregates.get("mean_completeness", 0.0)), 3),
            "mean_hallucination_ratio": round(float(aggregates.get("mean_hallucination_ratio", 0.0)), 3),
            "evaluated_responses": len(per_turn_scores),
        },
        "per_turn_scores": per_turn_scores,
        "hallucination_findings": hallucination_findings,
        "costs": {
            "total_input_tokens": int(total_input or 0),
            "total_output_tokens": int(total_output or 0),
            "estimated_cost_usd": float(total_cost or 0.0),
        },
        "warnings": warnings,
        "natural_language_summary": _nl_summary(aggregates),
    }
    return clean

def make_html_report(clean_report: Dict[str, Any], out_path: str):
    parts = []
    parts.append("<html><head><meta charset='utf-8'><title>Clean LLM Report</title></head><body>")
    parts.append(f"<h1>LLM Evaluation Clean Report</h1>")
    parts.append(f"<p>Generated: {html.escape(clean_report['generated_at'])}</p>")

    parts.append("<h2>Summary</h2><ul>")
    for k,v in clean_report["summary"].items():
        parts.append(f"<li><b>{k.replace('_',' ').title()}:</b> {v}</li>")
    parts.append("</ul>")

    parts.append("<h2>Per-turn Scores</h2>")
    parts.append("<table border='1' style='border-collapse:collapse'><tr>"
                 "<th>Pair</th><th>Turn ID</th><th>User</th><th>AI</th>"
                 "<th>Rel</th><th>Comp</th><th>Hall</th><th>Review?</th></tr>")
    for t in clean_report["per_turn_scores"]:
        parts.append("<tr>")
        parts.append(f"<td>{t['pair_index']}</td>")
        parts.append(f"<td>{t['turn_id']}</td>")
        parts.append(f"<td>{html.escape(t['user_preview'])}</td>")
        parts.append(f"<td>{html.escape(t['ai_preview'])}</td>")
        parts.append(f"<td>{t['relevance']}</td>")
        parts.append(f"<td>{t['completeness']}</td>")
        parts.append(f"<td>{t['hallucination_ratio']}</td>")
        parts.append(f"<td>{'YES' if t['requires_manual_review'] else 'NO'}</td>")
        parts.append("</tr>")
    parts.append("</table>")

    parts.append("<h2>Warnings</h2>")
    for w in clean_report["warnings"]:
        parts.append(f"<p style='color:red'>{html.escape(w)}</p>")

    parts.append("<h2>Summary (Text)</h2>")
    parts.append(f"<p>{html.escape(clean_report['natural_language_summary'])}</p>")

    parts.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

def write_clean_and_html(combined: Dict[str, Any], out_json: str, out_clean_json: str, out_html: str):
    # write full combined
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    # make clean
    clean = make_clean_report(combined)

    # write clean json
    with open(out_clean_json, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    # write html
    make_html_report(clean, out_html)

    return clean
