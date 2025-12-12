# LLM Evaluation Pipeline — BeyoundChat

A compact, production-minded evaluation pipeline for assessing LLM responses against user queries and contextual evidence.  
This project automatically scores responses for **Relevance**, **Completeness**, **Hallucination / Factual Accuracy**, and reports **Latency & Token Cost**. It produces both a full debug-ready `combined_report.json` and a concise `combined_report.clean.json` plus a readable HTML summary.

---

## Repository contents

BEYOUNDCHAT/
├── README.md
├── requirements.txt
├── data/ # sample inputs for quick verification
│ ├── sample-chat-conversation-01.json
│ └── sample_context_vectors-01.json
├── call_api.py # example client to call the API
├── evaluate_pipeline.sh / .ps1 (opt) # optional convenience scripts
└── src/
└── llm_evaluator/
├── init.py
├── api.py # FastAPI wrapper (real-time evaluation)
├── evaluate_pipeline.py # CLI entrypoint (batch)
├── aggregate_cli.py # original aggregator (if present)
├── parser.py # JSON parsing & flattening utilities
├── metrics.py # relevance/completeness/hallucination/cost logic
├── embeddings.py # embedding helpers (lazy import)
├── report_formatter.py # produces clean.json + html
└── utils.py


---

## Local setup instructions (exact commands)

> Tested on Windows PowerShell and VS Code. Adjust `python`/path for macOS or Linux.

1. Clone the repo (or ensure you're in the project root):
```powershell
# if you haven't cloned yet:
git clone https://github.com/priya-kanade/BeyoundChat.git
cd BeyoundChat


2. Create and use a Python virtual environment:
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt

3. Run the CLI (batch evaluation — writes files to disk):
python -m src.llm_evaluator.evaluate_pipeline --conv data/sample-chat-conversation-01.json --context data/sample_context_vectors-01.json --out combined_report.json

Output files:

combined_report.json — full detailed report

combined_report.clean.json — compact, human-friendly report

combined_report.clean.html — browser view

4.Run the API (real-time evaluation):
python -m uvicorn src.llm_evaluator.api:app --reload
# In another terminal (project root), run:
python call_api.py

call_api.py demonstrates how to POST the conversation and context and includes "save": true to have the server write the same files as the CLI.

How to reproduce results quickly

Start the server (see step 4 above).

Run python call_api.py — it will POST the sample JSONs and save the same combined_report.json artifacts server-side and optionally locally.

The evaluation pipeline — architecture & flow

Goal: Automatically test LLM answers against an authoritative context and compute metrics used for QA and manual triage.

High-level flow

Input

conversation.json — sequence of timestamped messages (user & assistant).

context_vectors.json — list of context chunks (text, id, source, vector metadata) retrieved from a vector DB for a target user message.

Preprocess

flatten_context_vectors() normalizes context JSON into a list of {id, text, source, ...}.

Pair extraction

Extract all sequential User → Assistant pairs from the conversation to evaluate each assistant reply independently.

Per-pair evaluation

Relevance: similarity between the assistant reply and retrieved context chunks (embedding similarity).

Completeness: whether the assistant reply fully addresses the user question (checks missing sub-questions/steps).

Hallucination / Factual Accuracy: claim extraction from the reply → check each claim against context evidence; compute best_support_score and evidence_confidence (strong/medium/weak).

Latency: estimate or compute from message metadata.

Cost: estimate input/output tokens and cost using configurable input_per_1k_tokens_usd and output_per_1k_tokens_usd. Compute both full-context and top-K estimates.

Aggregation

Compute conversation-level metrics (mean relevance, mean completeness, mean hallucination ratio, total token counts, total estimated cost).

Reporting

Combined report (full) includes everything: per-turn claims, evidence snippets, scores, costs, tokens.

Clean report (concise) summarizes key metrics, highlights manual-review candidates, lists warnings and natural-language summary. Also produces an HTML view for human reviewers.

Interfaces

CLI (evaluate_pipeline.py) — batch artifact generation.

API (api.py) — real-time evaluation with optional server-side saving ("save": true).

Why this design? (choices & reasoning)

Modular, reproducible, and auditable — the pipeline is split into small components (parser, metrics, formatter, API). This provides:

Single source of truth: build_combined() implements evaluation logic used by both CLI and API, guaranteeing identical outputs across backends.

Lazy imports: heavy libraries (embeddings, sentence-transformers) are imported only when needed to reduce API startup time and resource usage.

Two-level reports: full (for audits/debugging) and clean (for graders/stakeholders). This balances traceability vs human readability.

Efficient verification: rather than re-querying the web or calling an LLM for every verification, we rely on vector-retrieved context + lightweight evidence matching. This minimizes calls to expensive models.

Configurable thresholds: hallucination detection and cost estimation parameters are configurable to support experiments.

Fail-safe behavior: the API will not expose internal tracebacks unless debug_mode=True (local debugging only) and sanitizes file paths before saving.

Why not other approaches?

Calling a heavyweight LLM per claim for verification yields higher accuracy but is costly and slow at scale. Our approach aims for an acceptable precision/recall tradeoff using local evidence matching + selective, heavier verification only for flagged cases.

Embedding contexts at evaluation time is wasteful; embeddings should be precomputed in the vector DB. The system assumes contexts are already vectorized.

Scaling & cost/latency controls (million-conversations-per-day scenario)

To support very high throughput while keeping latency and cost manageable, this design advocates the following patterns:

Precompute & store embeddings

Do not embed contexts on-the-fly. Store vectors in a vector database (FAISS, Milvus, Pinecone, Weaviate).

Use ANN indexes (fast nearest-neighbour)

Use HNSW/IVF/PQ to fetch top-K context chunks in sub-millisecond time.

Top-K evaluation & caching

Evaluate primarily against top-K (e.g., 3–5) contexts for real-time responses. Use caches for frequent queries, and cache embeddings for repeated responses.

Asynchronous deep checks

Return a preliminary result quickly (fast similarity + rules). Enqueue heavy claim re-checks (LLM-based verification) for background workers; attach final verdict later.

Sampling & prioritization

Only run full, expensive verification on responses that trigger risk heuristics (numeric claims, medical/legal content, low relevance, low completeness).

Lightweight models in the hot-path

Use small/efficient embedding models and rule-based claim extractors for the hot path. Reserve large models for offline audits.

Autoscaling & horizontalization

Run the stateless evaluator behind autoscaling services; use dedicated vector DB clusters and worker pools for heavy tasks.

Token/cost budgeting

Estimate token cost before calling paid LLMs; implement circuit-breakers and cost-aware throttling when budgets are exceeded.

Monitoring & metrics

Metric keypoints: request latency, manual-review rate, average token cost per eval, queue depth for heavy workers. Alert on drift or budget spikes.

Security & privacy

Sanitize user data before writing to disk.

save_basename is sanitized to avoid path traversal. In production restrict server write directory and enforce authentication.

Don’t commit secrets (API keys) to the repo. Use environment variables or secure vaults.

How to run tests / validate

Unit tests for metrics.py and parser.py should assert deterministic outputs for known inputs.

Integration test: run the CLI on provided data/ and assert the combined_report.clean.json contains expected keys and summary metrics.

End-to-end: start the API and call call_api.py to confirm server-side files are produced.


How to run tests / validate

Unit tests for metrics.py and parser.py should assert deterministic outputs for known inputs.

Integration test: run the CLI on provided data/ and assert the combined_report.clean.json contains expected keys and summary metrics.

End-to-end: start the API and call call_api.py to confirm server-side files are produced.