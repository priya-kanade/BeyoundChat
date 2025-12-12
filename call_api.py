import json, requests

# Load your input files
conv = json.load(open("data/sample-chat-conversation-01.json"))
ctx  = json.load(open("data/sample_context_vectors-01.json"))

# Build request body
body = {
    "conversation": conv,
    "context": ctx,
    "hallucination_threshold": 0.28,
    "top_k": 5,
    "save": True,                          # <-- SAVE FILES ON SERVER IN 1 RUN
    "save_basename": "combined_report.json"
}

# Call API
response = requests.post("http://127.0.0.1:8000/evaluate/combined", json=body)
response.raise_for_status()

# Print response keys
data = response.json()
print("Saved files on server:", data.get("saved_paths"))


# Save returned JSON locally too (optional)
json.dump(data["combined"], open("combined_from_api.json","w"), indent=2)
json.dump(data["clean"], open("combined_from_api.clean.json","w"), indent=2)
