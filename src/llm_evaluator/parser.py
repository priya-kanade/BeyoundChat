# src/llm_evaluator/parser.py
from typing import Dict, Any, List
import json

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_last_user_and_ai(conv_json: Dict[str, Any]) -> Dict[str, Any]:
    turns = conv_json.get("conversation_turns") or conv_json.get("turns") or []
    last_user = {}
    last_ai = {}
    for t in turns:
        role = str(t.get("role", "")).lower()
        if "user" in role:
            last_user = t
        if "ai" in role or "assistant" in role or "chatbot" in role:
            last_ai = t
    user_text = (last_user.get("message") or last_user.get("text") or "").strip()
    ai_text = (last_ai.get("message") or last_ai.get("text") or "").strip()
    return {"user_text": user_text, "ai_text": ai_text, "ai_meta": last_ai}

def flatten_context_vectors(ctx_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    data = ctx_json.get("data", {}) if isinstance(ctx_json, dict) else {}
    vector_list = data.get("vector_data") or ctx_json.get("vector_data") or ctx_json.get("data")
    if isinstance(vector_list, list):
        for v in vector_list:
            items.append({
                "id": v.get("id"),
                "text": v.get("text") or v.get("snippet") or "",
                "source": v.get("source_url") or v.get("source") or v.get("id"),
                "meta": v
            })
    else:
        def find_vectors(obj):
            found = []
            if isinstance(obj, dict):
                for k, val in obj.items():
                    if k == "vector_data" and isinstance(val, list):
                        found.extend(val)
                    elif isinstance(val, (dict, list)):
                        found.extend(find_vectors(val))
            elif isinstance(obj, list):
                for it in obj:
                    found.extend(find_vectors(it))
            return found
        nested = find_vectors(ctx_json)
        for v in nested:
            items.append({
                "id": v.get("id"),
                "text": v.get("text") or v.get("snippet") or "",
                "source": v.get("source_url") or v.get("source") or v.get("id"),
                "meta": v
            })
    try:
        sources = data.get("sources") or ctx_json.get("sources")
        vi = (sources or {}).get("vectors_info") or []
        info_map = {str(x.get("vector_id") or x.get("id")): x for x in vi if isinstance(x, dict)}
        for it in items:
            vid = str(it.get("id") or "")
            if vid in info_map:
                it["score"] = info_map[vid].get("score")
                it["tokens_count"] = info_map[vid].get("tokens_count") or info_map[vid].get("tokens")
    except Exception:
        pass
    return items
