# src/llm_evaluator/aggregate_cli.py
"""
aggregate_cli.py

Produce a single combined JSON report that evaluates EVERY user->assistant pair
in a conversation JSON (conversation_turns) using the same metrics as the single-run CLI.

This enhanced version:
 - accepts --hallucination_threshold to control hallucination detection
 - records source_id_map to resolve numeric source ids
 - computes top-k input token & cost estimates (top_k=5)
 - collects requires_manual_review items for medium/weak evidence
"""
import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Import parser + metrics from your module
from .parser import load_json  # defensive loader
from .parser import flatten_context_vectors
from .embeddings import embed_texts  # to detect backend presence
from .metrics import (
    relevance_score,
    completeness_score,
    hallucination_report,
    estimate_latency,
    estimate_cost,
)
# If names differ in your repo, adapt imports accordingly.

def load_conversation_raw(path: str) -> Dict[str, Any]:
    """Load conversation JSON raw dict (defensive)."""
    return load_json(path)

def extract_user_assistant_pairs(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Walk conversation_turns (or turns) and extract sequential user->assistant pairs.
    Returns a list of { 'user': turn_obj, 'assistant': turn_obj, 'pair_index': i }.
    If a user turn has no following assistant turn, it's skipped.
    """
    turns = conv.get("conversation_turns") or conv.get("turns") or []
    pairs = []
    i = 0
    while i < len(turns):
        t = turns[i]
        role = str(t.get("role","")).lower()
        if "user" in role:
            # find next assistant turn
            j = i + 1
            while j < len(turns):
                r2 = str(turns[j].get("role","")).lower()
                if "ai" in r2 or "assistant" in r2 or "chatbot" in r2:
                    pairs.append({"user": turns[i], "assistant": turns[j], "pair_index": len(pairs)+1})
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return pairs

def evaluate_pair(pair: Dict[str, Any], context_items: List[Dict[str, Any]], pricing: Dict[str, float], support_threshold: float, top_k: int = 5) -> Dict[str, Any]:
    """
    Evaluate a single user->assistant pair and return the report for that turn.
    Returns both the turn_report and a small 'manual_review' list for claims needing human check.
    """
    user_text = (pair["user"].get("message") or pair["user"].get("text") or "").strip()
    ai_text = (pair["assistant"].get("message") or pair["assistant"].get("text") or "").strip()
    ai_meta = pair["assistant"] or {}

    # compute metrics (pass support_threshold to hallucination detector)
    rel = relevance_score(ai_text, [c.get("text","") for c in context_items])
    comp = completeness_score(user_text, ai_text, [c.get("text","") for c in context_items])
    hall = hallucination_report(ai_text, context_items, support_threshold=support_threshold)
    latency = estimate_latency(ai_meta)
    # full-context cost
    cost_full = estimate_cost(user_text, [c.get("text","") for c in context_items], ai_text, pricing=pricing)
    # top-k cost (more realistic prompt cost reporting)
    topk_texts = [c.get("text","") for c in context_items[:top_k]]
    cost_topk = estimate_cost(user_text, topk_texts, ai_text, pricing=pricing)

    # collect manual review items for medium/weak evidence
    manual_review = []
    for claim in hall.get("claims", []):
        conf = claim.get("evidence_confidence", "").lower()
        if conf in ("weak", "medium"):
            manual_review.append({
                "claim": claim.get("claim"),
                "claim_type": claim.get("claim_type"),
                "evidence_confidence": claim.get("evidence_confidence"),
                "best_support_score": claim.get("best_support_score")
            })

    turn_report = {
        "pair_index": pair.get("pair_index"),
        "user_text_preview": user_text.replace("\n"," ")[:400],
        "ai_text_preview": ai_text.replace("\n"," ")[:600],
        "relevance": rel,
        "completeness": comp,
        "hallucination": hall,
        "latency_seconds": latency,
        "token_estimates": {
            "input_tokens_all_contexts": cost_full["input_tokens"],
            "output_tokens": cost_full["output_tokens"]
        },
        "costs": {
            "estimated_cost_usd_all_contexts": cost_full["estimated_cost_usd"],
            "estimated_cost_usd_topk": cost_topk["estimated_cost_usd"]
        },
        "token_estimates_topk": {
            "input_tokens_topk": cost_topk["input_tokens"]
        },
        "estimated_cost_usd": cost_full["estimated_cost_usd"],
        "estimated_cost_usd_topk": cost_topk["estimated_cost_usd"],
        "ai_meta": ai_meta,
        "requires_manual_review": manual_review
    }
    return turn_report

def aggregate_reports(turn_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute simple aggregates across turn_reports."""
    n = len(turn_reports)
    if n == 0:
        return {
            "num_turns": 0,
            "mean_relevance": 0.0,
            "mean_completeness": 0.0,
            "mean_hallucination_ratio": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_estimated_cost_usd": 0.0
        }
    mean_relevance = sum(t["relevance"] for t in turn_reports) / n
    mean_completeness = sum(t["completeness"] for t in turn_reports) / n
    mean_hall_ratio = sum(t["hallucination"].get("hallucination_ratio",0.0) for t in turn_reports) / n
    total_input = sum(t["token_estimates"]["input_tokens_all_contexts"] for t in turn_reports)
    total_output = sum(t["token_estimates"]["output_tokens"] for t in turn_reports)
    total_cost = sum(float(t.get("estimated_cost_usd", 0.0)) for t in turn_reports)
    return {
        "num_turns": n,
        "mean_relevance": mean_relevance,
        "mean_completeness": mean_completeness,
        "mean_hallucination_ratio": mean_hall_ratio,
        "total_input_tokens": int(total_input),
        "total_output_tokens": int(total_output),
        "total_estimated_cost_usd": float(total_cost)
    }

def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluator: evaluate ALL user->assistant pairs in a conversation JSON")
    parser.add_argument("--conv", required=True, help="Path to conversation JSON (contains conversation_turns)")
    parser.add_argument("--context", required=True, help="Path to context vectors JSON")
    parser.add_argument("--out", default="combined_report.json", help="Path to write combined report")
    parser.add_argument("--input_price", type=float, default=0.03, help="USD per 1k input tokens")
    parser.add_argument("--output_price", type=float, default=0.06, help="USD per 1k output tokens")
    parser.add_argument("--hallucination_threshold", type=float, default=0.28, help="Support score threshold below which claims are flagged as hallucinations")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K context snippets to use for top-k token/cost estimates")
    args = parser.parse_args()

    conv = load_conversation_raw(args.conv)
    context_json = load_json(args.context)
    context_items = flatten_context_vectors(context_json)

    pricing = {"input_per_1k_tokens_usd": args.input_price, "output_per_1k_tokens_usd": args.output_price}

    pairs = extract_user_assistant_pairs(conv)
    turn_reports = []
    start_all = time.time()
    for p in pairs:
        tr = evaluate_pair(p, context_items, pricing=pricing, support_threshold=args.hallucination_threshold, top_k=args.top_k)
        turn_reports.append(tr)

    aggregates = aggregate_reports(turn_reports)
    end_all = time.time()

    # detect embedding backend used (best-effort)
    try:
        backend = "sentence-transformers" if embed_texts(["a"]).shape[1] != 256 else "fallback"
    except Exception:
        backend = "unknown"

    # build source_id_map for numeric evidence sources (helps graders resolve numeric ids)
    source_id_map = {str(it.get("id")): it.get("source") for it in context_items if it.get("id")}

    # gather all manual review items across turns
    all_manual = []
    for t in turn_reports:
        if t.get("requires_manual_review"):
            for r in t["requires_manual_review"]:
                r_copy = r.copy()
                r_copy["pair_index"] = t["pair_index"]
                all_manual.append(r_copy)

    combined = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "conversation_file": args.conv,
        "context_file": args.context,
        "metadata": {
            "num_context_items": len(context_items),
            "num_pairs_evaluated": len(turn_reports),
            "embedding_backend": backend,
            "eval_duration_seconds": round(end_all - start_all, 6),
            "hallucination_threshold": args.hallucination_threshold,
            "top_k_for_token_estimates": args.top_k,
        },
        "aggregates": aggregates,
        "turn_reports": turn_reports
    }

    if source_id_map:
        combined["metadata"]["source_id_map"] = source_id_map
    if all_manual:
        combined["metadata"]["requires_manual_review"] = all_manual

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Combined report written to {args.out} — evaluated {len(turn_reports)} pair(s).")

if __name__ == "__main__":
    main()

