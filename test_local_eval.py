# test_local_eval.py
import json
import traceback
from src.llm_evaluator.parser import load_json, flatten_context_vectors
from src.llm_evaluator.metrics import relevance_score, completeness_score, hallucination_report, estimate_latency, estimate_cost

def main():
    try:
        conv = load_json("data/sample-chat-conversation-01.json")
        ctx = load_json("data/sample_context_vectors-01.json")
        items = flatten_context_vectors(ctx)
        print("Loaded conversation turns:", len(conv.get("conversation_turns", conv.get("turns", []))))
        print("Num context items:", len(items))
        # pick the last user->ai pair like aggregate_cli would
        turns = conv.get("conversation_turns") or conv.get("turns") or []
        # find last user->assistant pair (simple)
        user_text = ""
        ai_text = ""
        for i in range(len(turns)-1):
            r = str(turns[i].get("role","")).lower()
            if "user" in r:
                nr = str(turns[i+1].get("role","")).lower()
                if "ai" in nr or "assistant" in nr or "chatbot" in nr:
                    user_text = turns[i].get("message") or turns[i].get("text") or ""
                    ai_text = turns[i+1].get("message") or turns[i+1].get("text") or ""
        print("User preview:", user_text[:120])
        print("AI preview:", ai_text[:120])
        # run the same metrics
        rel = relevance_score(ai_text, [c.get("text","") for c in items])
        comp = completeness_score(user_text, ai_text, [c.get("text","") for c in items])
        hall = hallucination_report(ai_text, items, support_threshold=0.28)
        print("relevance:", rel)
        print("completeness:", comp)
        print("hallucination ratio:", hall.get("hallucination_ratio"))
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
