# save_report.py
import json, requests

conv = json.load(open("data/sample-chat-conversation-01.json", encoding="utf-8"))
ctx  = json.load(open("data/sample_context_vectors-01.json", encoding="utf-8"))

body = {
    "conversation": conv,
    "context": ctx,
    "debug_mode": False,   # ensure debug off for final report
    "hallucination_threshold": 0.28,
    "top_k": 5
}

r = requests.post("http://127.0.0.1:8000/evaluate/combined", json=body, timeout=120)
r.raise_for_status()  # raise if not 2xx
with open("combined_report.json", "w", encoding="utf-8") as f:
    f.write(r.text)

print("Saved combined_report.json (status =", r.status_code, ")")
