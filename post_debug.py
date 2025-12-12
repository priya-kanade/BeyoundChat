import json, requests

# Load JSON files directly
conv = json.load(open("data/sample-chat-conversation-01.json"))
ctx  = json.load(open("data/sample_context_vectors-01.json"))

body = {
    "conversation": conv,
    "context": ctx,
    "debug_mode": True,   # IMPORTANT: this makes API return traceback
    "hallucination_threshold": 0.28,
    "top_k": 5
}

print("Sending POST request...")
r = requests.post("http://127.0.0.1:8000/evaluate/combined", json=body)

print("\nSTATUS:", r.status_code)
print("\nRESPONSE HEADERS:\n", r.headers)
print("\nRESPONSE BODY:\n", r.text[:10000])
