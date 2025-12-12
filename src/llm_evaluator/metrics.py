# src/llm_evaluator/metrics.py
from typing import List, Dict, Any
from .embeddings import embed_texts, cosine_sim_matrix
import re
import time

try:
    from rouge_score import rouge_scorer
    _ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
except Exception:
    _ROUGE = None

def claim_type(text: str) -> str:
    """Heuristic claim type tagging to help graders interpret claims."""
    if not text:
        return "UNKNOWN"
    if re.search(r'https?://|www\.|\[http', text, flags=re.I):
        return "URL"
    if re.search(r'\d{1,3}(,\d{3})?|\bRs\b|\bUSD\b|\$|\bkg\b|\bpercent\b', text, flags=re.I):
        return "NUMERIC"
    if re.search(r'\b(should|must|recommend|advice|best)\b', text, flags=re.I):
        return "RECOMMENDATION"
    return "ASSERTION"


def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)

def extract_keywords(text: str, topk: int = 12) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9\-']{2,}", text.lower())
    freq = {}
    for t in toks:
        if len(t) <= 2: continue
        freq[t] = freq.get(t, 0) + 1
    return [k for k,_ in sorted(freq.items(), key=lambda x:-x[1])[:topk]]

def relevance_score(ai_text: str, context_texts: List[str], top_k: int = 5) -> float:
    if not context_texts or not ai_text:
        return 0.0
    k = min(top_k, len(context_texts))
    emb = embed_texts([ai_text] + context_texts[:k])
    ai_emb = emb[0]
    ctx_embs = emb[1:]
    sims = cosine_sim_matrix(ai_emb, ctx_embs).flatten()
    return float(sims.mean()) if sims.size else 0.0

def completeness_score(user_text: str, ai_text: str, context_texts: List[str], top_k: int = 5) -> float:
    if not context_texts or not ai_text:
        return 0.0
    k = min(top_k, len(context_texts))
    concat_ctx = "\n".join(context_texts[:k])
    if _ROUGE:
        r = _ROUGE.score(concat_ctx, ai_text)
        completeness = float(r["rougeL"].fmeasure)
    else:
        emb = embed_texts([ai_text, concat_ctx])
        completeness = float(cosine_sim_matrix(emb[0], emb[1])[0][0])
    u_keys = set(extract_keywords(user_text, topk=12))
    a_keys = set(extract_keywords(ai_text, topk=30))
    intent_cov = float(len(u_keys & a_keys) / len(u_keys)) if u_keys else 1.0
    return 0.6 * completeness + 0.4 * intent_cov

def extract_candidate_claims(ai_text: str) -> List[str]:
    sents = re.split(r'(?<=[\.\?\!])\s+', ai_text.strip())
    claims = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if re.search(r'\d', s) or re.search(r'\b(is|are|was|were|has|have|had|will|should|must)\b', s, flags=re.I):
            claims.append(s)
        elif re.search(r'\b[A-Z][a-z]{2,}\b', s):
            claims.append(s)
    out = []
    for c in claims:
        if c not in out:
            out.append(c)
    return out

def evidence_search_for_claim(claim: str, context_items: List[Dict[str, Any]], top_n: int = 3) -> Dict[str, Any]:
    texts = [ci["text"] for ci in context_items]
    if not texts:
        return {"claim": claim, "evidence": [], "best_score": 0.0}
    emb = embed_texts([claim] + texts)
    c_emb = emb[0]
    ctx_embs = emb[1:]
    sims = cosine_sim_matrix(c_emb, ctx_embs).flatten()
    idxs = list(sims.argsort()[-top_n:][::-1])
    evidence = []
    for i in idxs:
        evidence.append({"source": context_items[i].get("source"), "snippet": context_items[i].get("text")[:800], "score": float(sims[i])})
    return {"claim": claim, "evidence": evidence, "best_score": float(sims.max())}

def hallucination_report(ai_text: str, context_items: List[Dict[str, Any]], support_threshold: float = 0.28) -> Dict[str, Any]:
    """
    Build hallucination report including:
      - claim_type (NUMERIC / URL / RECOMMENDATION / ASSERTION)
      - evidence_confidence (weak/medium/strong) derived from best_support_score
    """
    claims = extract_candidate_claims(ai_text)
    if not claims:
        return {"claims": [], "hallucination_ratio": 0.0, "num_claims": 0}

    results = []
    flagged = 0
    for c in claims:
        ev = evidence_search_for_claim(c, context_items, top_n=3)
        best = float(ev.get("best_score", 0.0))
        is_hall = best < support_threshold
        if is_hall:
            flagged += 1

        # evidence confidence buckets
        if best >= 0.55:
            confidence = "strong"
        elif best >= 0.35:
            confidence = "medium"
        else:
            confidence = "weak"

        results.append({
            "claim": c,
            "claim_type": claim_type(c),
            "best_support_score": best,
            "evidence_confidence": confidence,
            "is_hallucination": bool(is_hall),
            "evidence": ev.get("evidence", [])
        })

    ratio = flagged / len(claims)
    return {"claims": results, "hallucination_ratio": float(ratio), "num_claims": len(claims)}


def estimate_latency(ai_meta: Dict[str, Any]) -> float:
    if not ai_meta:
        return 0.0
    latency = ai_meta.get("latency_seconds") or ai_meta.get("response_time") or ai_meta.get("response_time_seconds") or 0.0
    try:
        latency = float(latency)
    except Exception:
        latency = 0.0
    if latency <= 0:
        t0 = time.time()
        time.sleep(0.01)
        latency = round(time.time() - t0, 6)
    return float(latency)

def estimate_cost(user_text: str, top_contexts: List[str], ai_text: str, pricing: Dict[str, float] = None) -> Dict[str, Any]:
    pricing = pricing or {"input_per_1k_tokens_usd": 0.03, "output_per_1k_tokens_usd": 0.06}
    input_text = user_text + "\n" + "\n".join(top_contexts[:5])
    input_tokens = approx_token_count(input_text)
    output_tokens = approx_token_count(ai_text)
    cost = (input_tokens/1000.0) * pricing["input_per_1k_tokens_usd"] + (output_tokens/1000.0) * pricing["output_per_1k_tokens_usd"]
    return {"input_tokens": int(input_tokens), "output_tokens": int(output_tokens), "estimated_cost_usd": float(cost)}
