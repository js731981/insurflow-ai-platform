![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![LLM](https://img.shields.io/badge/LLM-Ollama-orange)
![Vector DB](https://img.shields.io/badge/vector--db-Chroma-purple)

# Insurance AI Decision Platform (Micro-insurance)

Local MVP for **small-ticket claim triage**: **FastAPI** + **Ollama** (LLM + embeddings) + **ChromaDB** (embedded, persistent). No extra infrastructure required.

## What it does

- Ingests a micro-insurance claim and returns **`APPROVED`**, **`REJECTED`**, or **`INVESTIGATE`**.
- Runs **fraud analysis** (LLM, structured JSON) and **policy checks** (rules) **in parallel**, then a **decision** agent.
- **Retrieves similar past claims** from vector memory, applies **weighted ranking** (review signals + coherence with majority reviewed outcomes).
- **Calibrates confidence** against human-reviewed similar cases (keeps raw `confidence_score` and adds `calibrated_confidence`).
- Flags **human-in-the-loop (HITL)** when the decision is `INVESTIGATE`, or when **`calibrated_confidence`** is below **0.75** (see `HitlService` in `app/core/dependencies.py`).
- **Persists** each processed claim (embedding + metadata) for future retrieval and learning.
- Supports an **investigator case workflow** on top of stored claims: list/filter cases, **assign** from `NEW` → `ASSIGNED`, and advance status (`ASSIGNED` → `IN_PROGRESS` → `RESOLVED`) via dedicated endpoints.
- Serves **analytics** over Chroma-backed claims: summary aggregates, anomaly-style alerts, and a fraud-score leaderboard.
- Exposes **structured logs**, in-process **metrics**, and a **minimal web UI** at `/ui`.

## Architecture

```
FastAPI
   └── InsurFlowOrchestrator
          ├── EmbeddingService (Ollama) → VectorStore (Chroma) — similar claims
          ├── FraudAgent (LLM)     ─┐
          ├── PolicyAgent (rules)  ─┼─→ DecisionAgent → HITL
          └── store_claim (single upsert when embedding OK)
```

**LLM routing:** `LLMService` → `LLMRouter` (retries with backoff, timeouts, optional provider fallbacks). Providers: **Ollama** (default), **OpenAI** and **OpenRouter** if API keys are set. Heuristic **USD cost** estimates use `LLM_COST_USD_PER_1K_*` when set.

**Generic inference:** `POST /inference` accepts optional `task`: `cheap` forces **Ollama**, `complex` forces **OpenAI** (if configured).

**Agents (implemented):**

| Agent | Role |
|--------|------|
| **FraudAgent** | LLM JSON: `fraud_score`, fraud-level `decision`/`confidence`, `entities`, structured `explanation` (summary, key factors, optional similar-case reference). |
| **PolicyAgent** | Validates `policy_limit`, `claim_amount`, and amount ≤ limit. |
| **DecisionAgent** | Combines fraud + policy into final triage outcome and `confidence_score`. |

## Tech stack

- Python 3.10+, **FastAPI**, **Pydantic v2**, **httpx**, **python-dotenv**
- **Ollama** — chat completions (`/api/generate`) and embeddings (`/api/embeddings`)
- **ChromaDB** — persistent local vector store

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment

Create a `.env` file (example):

```env
APP_NAME="Insurance AI Decision Platform"
LLM_PROVIDER=ollama
MODEL_NAME=phi3
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=claims
```

Optional: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `LLM_FALLBACK_PROVIDERS`, `LLM_RETRIES`, `LLM_COST_USD_PER_1K_INPUT_TOKENS`, `LLM_COST_USD_PER_1K_OUTPUT_TOKENS`, etc. See `app/core/config.py`.

**LLM timeouts:** Default `LLM_TIMEOUT_S` is **180** seconds so local Ollama (e.g. Phi-3 on CPU) can finish long fraud JSON generations. If you see `LLM timeout ... attempt=1/3`, either wait for retries, raise `LLM_TIMEOUT_S`, or use a smaller/faster model.

### 3. Pull Ollama models

```bash
ollama pull phi3
ollama pull nomic-embed-text
```

Start **Ollama** before the API.

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

- **API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Service info:** `GET /` (links to docs and `/ui`)
- **Health:** `GET /health`
- **Metrics:** `GET /metrics` — process-local counters plus `vector_store_claim_documents` from Chroma; includes `metrics_scope` explaining reset-on-restart behavior
- **Dashboard:** [http://localhost:8000/ui](http://localhost:8000/ui) — claim list, **case** assign/status, and **analytics** (summary, anomalies, leaderboard)

## API overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info and links |
| `GET` | `/health` | Liveness |
| `GET` | `/metrics` | Snapshot: `total_claims_processed`, `hitl_triggered_count`, `reviewed_claims_count` (in-memory, reset when the process restarts), `metrics_scope`, and `vector_store_claim_documents` from Chroma. |
| `POST` | `/claims` | Process a claim (same pipeline as `POST /claim`) |
| `GET` | `/claims` | List stored claims from Chroma (MVP: up to **200** rows, fixed offset **0**) |
| `GET` | `/cases` | List cases derived from stored claims (up to **500**); optional query: `case_status`, `assigned_to`, `unassigned_only` |
| `POST` | `/cases/{claim_id}/assign` | Assign investigator when `case_status` is `NEW` → sets `ASSIGNED`, `assigned_to`, timestamps |
| `POST` | `/cases/{claim_id}/status` | Update workflow: `IN_PROGRESS` (from `ASSIGNED`) or `RESOLVED` (from `IN_PROGRESS`); use assign endpoint for `NEW` → `ASSIGNED` |
| `GET` | `/analytics/summary` | Aggregated stats over stored claims (decisions, review buckets, volumes, etc.) |
| `GET` | `/analytics/anomalies` | Heuristic alerts (e.g. high fraud, investigate decisions, review mismatches) |
| `GET` | `/analytics/leaderboard` | Top claims by fraud score; query: `limit` (1–200), optional `min_fraud_score` |
| `POST` | `/claims/{claim_id}/review` | Human review: `APPROVED` or `REJECTED`; updates metadata in the vector store (restores minimal explanation/entities if missing on legacy rows). |
| `POST` | `/claim` | Process a claim (same as `POST /claims`; lives on the inference router for OpenAPI grouping) |
| `GET` | `/claim/samples` | Sample `ClaimRequest` payloads from `app/data/claims/sample_claims.json` |
| `POST` | `/inference` | Generic LLM completion (`prompt`, optional `context`, `model`, optional `task` hint — see above) |

### Process a claim

`POST /claims` or `POST /claim` — body: `ClaimRequest` (`claim_id`, `claim_amount`, `policy_limit`, plus optional fields such as `description`, `currency`, `product_code`, `incident_date`, `policyholder_id`). **Unknown top-level JSON keys are rejected** (`extra="forbid"`).

If `description` is empty, the orchestrator embeds a **JSON snapshot of the claim** for retrieval so similar-case context still has text to work with.

Response highlights:

- `decision` — `APPROVED` | `REJECTED` | `INVESTIGATE`
- `confidence_score` — raw decision confidence
- `calibrated_confidence` — adjusted using similar claims with human `review_status`
- `hitl_needed` — whether manual review is recommended
- `agent_outputs` — `fraud`, `policy`, `decision` details

### Human review

```http
POST /claims/MIC-2026-00412/review
Content-Type: application/json

{"action": "APPROVED", "reviewed_by": "optional reviewer id"}
```

### Case workflow

Case metadata (`case_status`, `assigned_to`, etc.) lives on the same Chroma document as the claim. Valid statuses: **`NEW`**, **`ASSIGNED`**, **`IN_PROGRESS`**, **`RESOLVED`**.

```http
GET /cases?case_status=NEW&unassigned_only=true
```

```http
POST /cases/MIC-2026-00412/assign
Content-Type: application/json

{"assigned_to": "investigator_1"}
```

```http
POST /cases/MIC-2026-00412/status
Content-Type: application/json

{"case_status": "IN_PROGRESS"}
```

### Analytics

Read-only endpoints over all paginated claim rows in the vector store. OpenAPI: `/docs` under the **analytics** tag.

## Resilience (MVP)

- LLM calls: **timeouts** and **retries** (router); failures logged with **`claim_id`** when provided.
- Embedding or retrieval failure: API **does not crash**; vector **write is skipped** for that request.
- Fraud LLM failure or invalid parse: safe fallback **`INVESTIGATE`** / **0.5** (unless policy invalid → **`REJECTED`**).

## Project layout (main)

- `app/main.py` — FastAPI app, router includes, `/ui` static mount; disables noisy Chroma telemetry in local dev
- `app/agents/` — orchestrator, fraud, policy, decision, base agent
- `app/api/routes/health.py` — `/`, `/health`, `/metrics`
- `app/api/routes/claims.py` — `POST/GET /claims`, `POST /claims/{claim_id}/review`
- `app/api/routes/cases.py` — `GET /cases`, `POST /cases/{claim_id}/assign`, `POST /cases/{claim_id}/status`
- `app/api/routes/analytics.py` — `GET /analytics/summary`, `/analytics/anomalies`, `/analytics/leaderboard`
- `app/api/routes/inference.py` — `POST /claim`, `GET /claim/samples`, `POST /inference`
- `app/services/` — `llm_service`, `llm/router` + providers, `embedding_service`, `vector_store`, `hitl_service`, `metrics`, `claim_samples_service`, `analytics`
- `app/core/` — `config` (`Settings`), `dependencies` (service factories)
- `app/web/` — static dashboard (`index.html`, `app.js`)
- `chroma_db/` (default) — persistent Chroma data; path set by `CHROMA_PERSIST_DIR`

## Author

Jayendran Subramanian — Consulting AI Data Engineer | AI Architect (in progress)

**Note:** This repository is a proof-of-concept and reference implementation. Production systems handling confidential data require additional security, governance, and operations beyond this MVP.

## 📄 License

This project is licensed under the MIT License.
