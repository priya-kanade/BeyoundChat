# src/llm_evaluator/api.py
"""
Final FastAPI wrapper for the LLM evaluator.

- Exposes /health and /evaluate/combined
- Lazy imports heavy modules to avoid import-time failures
- Accepts conversation + context JSON payloads
- Can save combined + clean + html files on server when save=True
- Returns combined (full) and clean (compact) reports in response
"""
from fastapi import FastAPI, HTTPException, Request
from typing import Any, Dict, Optional
from datetime import datetime
import time
import json
import traceback
import os

app = FastAPI(title="LLM Evaluator API (final)")

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

def _safe_flatten_context(ctx_raw):
    """Ensure ctx_raw is a dict (parse JSON string if needed) and call flatten_context_vectors."""
    if isinstance(ctx_raw, str):
        try:
            ctx_raw = json.loads(ctx_raw)
        except Exception:
            pass
    # lazy import
    from .parser import flatten_context_vectors
    items = flatten_context_vectors(ctx_raw)
    return items

def _detect_embedding_backend():
    """Best-effort detection of embedding backend. Returns string or 'unknown'."""
    try:
        from .embeddings import embed_texts
        emb = embed_texts(["a"])
        shape = getattr(emb, "shape", None)
        if shape is not None:
            try:
                d = int(emb.shape[1])
                return "sentence-transformers" if d != 256 else "fallback"
            except Exception:
                return "unknown"
        return "unknown"
    except Exception:
        return "unknown"

@app.post("/evaluate/combined")
async def evaluate_combined(request: Request):
    """
    POST body (JSON):
    {
      "conversation": { ... },
      "context": { ... },
      "hallucination_threshold": 0.28,
      "top_k": 5,
      "input_price": 0.03,
      "output_price": 0.06,
      "debug_mode": false,
      "save": false,
      "save_basename": "combined_report.json"
    }

    Response:
    {
      "combined": { ... },   # full combined report
      "clean": { ... },      # compact clean report
      "saved_paths": {...} or null,
      "generated_at": "..."
    }
    """
    # parse body
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    conv = payload.get("conversation") or payload.get("conv")
    ctx_json = payload.get("context") or payload.get("contexts")
    if conv is None or ctx_json is None:
        raise HTTPException(status_code=400, detail="Payload must include 'conversation' and 'context'")

    hallucination_threshold = float(payload.get("hallucination_threshold", 0.28))
    top_k = int(payload.get("top_k", 5))
    input_price = float(payload.get("input_price", 0.03))
    output_price = float(payload.get("output_price", 0.06))
    debug_mode = bool(payload.get("debug_mode", False))
    save_flag = bool(payload.get("save", False))
    save_basename = payload.get("save_basename")  # can be None

    pricing = {"input_per_1k_tokens_usd": input_price, "output_per_1k_tokens_usd": output_price}

    start_all = time.time()
    try:
        # flatten context
        context_items = _safe_flatten_context(ctx_json)

        # extract user->assistant pairs
        turns = conv.get("conversation_turns") or conv.get("turns") or []
        pairs = []
        i = 0
        while i < len(turns):
            t = turns[i]
            role = str(t.get("role", "")).lower()
            if "user" in role:
                j = i + 1
                while j < len(turns):
                    r2 = str(turns[j].get("role", "")).lower()
                    if "ai" in r2 or "assistant" in r2 or "chatbot" in r2:
                        pairs.append({"user": turns[i], "assistant": turns[j], "pair_index": len(pairs) + 1})
                        break
                    j += 1
                i = j + 1
            else:
                i += 1

        # lazy import metrics
        from .metrics import relevance_score, completeness_score, hallucination_report, estimate_latency, estimate_cost

        turn_reports = []
        all_manual = []
        for p in pairs:
            user_text = (p["user"].get("message") or p["user"].get("text") or "").strip()
            ai_text = (p["assistant"].get("message") or p["assistant"].get("text") or "").strip()
            ai_meta = p["assistant"] or {}

            rel = relevance_score(ai_text, [c.get("text", "") for c in context_items])
            comp = completeness_score(user_text, ai_text, [c.get("text", "") for c in context_items])
            hall = hallucination_report(ai_text, context_items, support_threshold=hallucination_threshold)
            latency = estimate_latency(ai_meta)

            cost_full = estimate_cost(user_text, [c.get("text", "") for c in context_items], ai_text, pricing=pricing)
            topk_texts = [c.get("text", "") for c in context_items[:top_k]]
            cost_topk = estimate_cost(user_text, topk_texts, ai_text, pricing=pricing)

            manual_review = []
            for claim in hall.get("claims", []):
                conf = claim.get("evidence_confidence", "").lower()
                if conf in ("weak", "medium"):
                    mr = {
                        "claim": claim.get("claim"),
                        "claim_type": claim.get("claim_type"),
                        "evidence_confidence": claim.get("evidence_confidence"),
                        "best_support_score": claim.get("best_support_score"),
                        "top_evidence": claim.get("evidence", [])[:1],
                        "pair_index": p.get("pair_index"),
                    }
                    manual_review.append(mr)
                    all_manual.append(mr)

            tr = {
                "pair_index": p.get("pair_index"),
                "user_text_preview": user_text.replace("\n", " ")[:400],
                "ai_text_preview": ai_text.replace("\n", " ")[:600],
                "relevance": rel,
                "completeness": comp,
                "hallucination": hall,
                "latency_seconds": latency,
                "token_estimates": {
                    "input_tokens_all_contexts": cost_full["input_tokens"],
                    "output_tokens": cost_full["output_tokens"],
                },
                "token_estimates_topk": {"input_tokens_topk": cost_topk["input_tokens"]},
                "estimated_cost_usd": cost_full["estimated_cost_usd"],
                "estimated_cost_usd_topk": cost_topk["estimated_cost_usd"],
                "ai_meta": ai_meta,
                "requires_manual_review": manual_review,
            }
            turn_reports.append(tr)

        # aggregates
        aggregates = {
            "num_turns": len(turn_reports),
            "mean_relevance": sum(t["relevance"] for t in turn_reports) / (len(turn_reports) or 1),
            "mean_completeness": sum(t["completeness"] for t in turn_reports) / (len(turn_reports) or 1),
            "mean_hallucination_ratio": sum(t["hallucination"].get("hallucination_ratio", 0) for t in turn_reports)
            / (len(turn_reports) or 1),
            "total_input_tokens": sum(t["token_estimates"]["input_tokens_all_contexts"] for t in turn_reports),
            "total_output_tokens": sum(t["token_estimates"]["output_tokens"] for t in turn_reports),
            "total_estimated_cost_usd": sum(float(t.get("estimated_cost_usd", 0.0)) for t in turn_reports),
        }

        backend = _detect_embedding_backend()
        end_all = time.time()

        source_id_map = {str(it.get("id")): it.get("source") for it in context_items if it.get("id")}

        combined = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "conversation_summary": {
                "num_turns_in_conversation": len(turns),
                "num_pairs_evaluated": len(turn_reports),
            },
            "metadata": {
                "num_context_items": len(context_items),
                "embedding_backend": backend,
                "eval_duration_seconds": round(end_all - start_all, 6),
                "hallucination_threshold": hallucination_threshold,
                "top_k_for_token_estimates": top_k,
            },
            "aggregates": aggregates,
            "turn_reports": turn_reports,
        }
        if source_id_map:
            combined["metadata"]["source_id_map"] = source_id_map
        if all_manual:
            combined["metadata"]["requires_manual_review"] = all_manual

        # Build clean report (lazy import)
        try:
            from .report_formatter import make_clean_report, write_clean_and_html
            clean = make_clean_report(combined)
        except Exception:
            clean = None
            write_clean_and_html = None

        saved_paths = None
        if save_flag:
            # decide safe basename
            basename = save_basename or "combined_report.json"
            basename = os.path.basename(basename)  # basic sanitization
            combined_out = basename
            clean_out = combined_out.replace(".json", ".clean.json")
            html_out = combined_out.replace(".json", ".clean.html")
            try:
                if write_clean_and_html is None:
                    # lazy import if not available earlier
                    from .report_formatter import write_clean_and_html
                write_clean_and_html(combined, out_json=combined_out, out_clean_json=clean_out, out_html=html_out)
                saved_paths = {"combined": combined_out, "clean_json": clean_out, "clean_html": html_out}
            except Exception as e:
                tb = traceback.format_exc()
                print("Error saving reports:", tb)
                # don't fail entire evaluation on save failure: include save error in response if debug_mode
                if debug_mode:
                    return {"combined": combined, "clean": clean, "saved_paths": None, "save_error": str(e), "traceback": tb}
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to save reports: {e}")

        return {"combined": combined, "clean": clean, "saved_paths": saved_paths, "generated_at": combined.get("generated_at")}

    except Exception as e:
        tb = traceback.format_exc()
        print("Exception in /evaluate/combined:", tb)
        if debug_mode:
            return {"error": str(e), "traceback": tb}
        raise HTTPException(status_code=500, detail="Internal server error while evaluating conversation")


