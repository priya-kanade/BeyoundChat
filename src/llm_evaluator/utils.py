# src/llm_evaluator/utils.py
from typing import Dict, Any
from .parser import load_json, extract_last_user_and_ai, flatten_context_vectors

def load_inputs(conv_path: str, ctx_path: str) -> Dict[str, Any]:
    conv = load_json(conv_path)
    ctx = load_json(ctx_path)
    user_ai = extract_last_user_and_ai(conv)
    context_items = flatten_context_vectors(ctx)
    return {"user_text": user_ai["user_text"], "ai_text": user_ai["ai_text"], "ai_meta": user_ai.get("ai_meta", {}), "context_items": context_items}
