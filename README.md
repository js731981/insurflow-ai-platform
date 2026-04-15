![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![LLM](https://img.shields.io/badge/LLM-Ollama-orange)
![Vector DB](https://img.shields.io/badge/vector--db-Chroma-purple)

# Insurance AI Decision Platform (Micro-insurance)

Local MVP for **small-ticket claim triage**: **FastAPI** + **Ollama** (LLM + embeddings) + **ChromaDB** (embedded, persistent). No extra infrastructure required.

Key docs:

- `docs/PROJECT_OVERVIEW.md` — concise project overview (what it is, how it works)
- `ARCHITECTURE.md` — deeper architecture narrative and evolution path
- `docs/EXECUTIVE_BRIEF.md` — one-page brief

## What it does

- Ingests a micro-insurance claim and returns **`APPROVED`**, **`REJECTED`**, or **`INVESTIGATE`**.
- Runs **fraud analysis** (LLM, structured JSON) and **policy checks** (rules) **in parallel**, then a **decision** agent.
- **Retrieves similar past claims** from vector memory, applies **weighted ranking** (review signals + coherence with majority reviewed outcomes).
- Builds a **compact RAG context** (token-capped) from similar claims and injects it into the fraud prompt.
- Optionally applies a cheap **reranker** pass to boost similar hits that match the inbound `product_code`.
- **Calibrates confidence** against human-reviewed similar cases (keeps raw `confidence_score` and adds `calibrated_confidence`).
- Flags **human-in-the-loop (HITL)** when the decision is `INVESTIGATE`, or when **`calibrated_confidence`** is below **0.75** (see `HitlService` in `app/core/dependencies.py`).
- **Persists** each processed claim (embedding + metadata) for future retrieval and learning.
- Supports an **investigator case workflow** on top of stored claims: list/filter cases, **assign** from `NEW` → `ASSIGNED`, and advance status (`ASSIGNED` → `IN_PROGRESS` → `RESOLVED`) via dedicated endpoints.
- Serves **analytics** over Chroma-backed claims: summary aggregates, anomaly-style alerts, and a fraud-score leaderboard.
- Exposes **structured logs**, in-process **metrics**, and a **minimal web UI** at `/ui`.
- Supports **multimodal claim triage**: optional **image analysis** signals can be provided via JSON (`image_base64`) or multipart upload (`file`/`image`).
- Optional **CNN damage classifier** (MobileNetV2 + fine-tuned weights) can be enabled for 3-class screen damage: `no_damage`, `minor_crack`, `major_crack`. If the CNN stack/weights are unavailable, the pipeline safely falls back to heuristic image signals.
- Provides **visual explainability** for the CNN path via a **Grad-CAM overlay** endpoint (PNG).

## Architecture

User Claim
   ↓
API (FastAPI)
   ↓
Orchestrator
   ↓
 ┌───────────────┐
 │ Fraud Agent   │ (LLM)
 │ Policy Agent  │ (Rules)
 └───────────────┘
        ↓
Decision Agent
        ↓
HITL Layer
        ↓
Vector DB (Chroma)
        ↓
Analytics + UI

────────────────────────────────────────────
Decision Pipeline (Multi-Agent System)
────────────────────────────────────────────
    ├── Embedding Service (Ollama)
    │       ↓
    │   Vector Store (Chroma)
    │       → Retrieves similar claims (RAG context)
    │
    ├── Fraud Agent (LLM-based reasoning)
    ├── Policy Agent (Rule-based validation)
    │
    └── Decision Agent (Fusion Layer)
            ↓
        HITL (Human-in-the-Loop)
            ↓
────────────────────────────────────────────
Memory Layer (Atomic Write)
────────────────────────────────────────────
    └── store_claim()
          → Embedding + Metadata (single upsert)
          → Enables future retrieval & learning

• Context-aware decisioning using retrieval (RAG)
• Hybrid intelligence (LLM + rules + metadata)
• Feedback-driven learning via HITL
• Atomic memory design for consistency
• Modular and extensible architecture

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
- **Pillow** + **numpy** for image decoding and lightweight vision heuristics
- Optional **torch** + **torchvision** for CNN classification + Grad-CAM
- **python-multipart** for multipart/form-data claim uploads (UI-friendly)
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
LLM_MODEL=phi3:mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=claims
```

Optional: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `LLM_FALLBACK_PROVIDERS`, `LLM_RETRIES`, `LLM_COST_USD_PER_1K_INPUT_TOKENS`, `LLM_COST_USD_PER_1K_OUTPUT_TOKENS`, etc. See `app/core/config.py`.

**Timeouts (interactive defaults):**

- **LLM per-attempt cap**: `LLM_TIMEOUT` / `LLM_TIMEOUT_S` (default **60s**, with a safety floor of 60s even if set lower).
- **Fraud agent hard cap**: `FRAUD_AGENT_TIMEOUT` / `FRAUD_AGENT_TIMEOUT_S` (default **120s**).
- **Whole claim pipeline cap**: `CLAIM_TIMEOUT` / `CLAIM_TIMEOUT_S` (default **35s**). If the LLM is slow/unavailable, the system falls back to a safe `INVESTIGATE` result.

**RAG and optional DL fraud head (recent enhancements):**

- RAG can be toggled and tuned via `RAG_ENABLED`, `RAG_TOP_K`, `RAG_CONTEXT_MAX_TOKENS`, and `RAG_RERANK_ENABLED`.
- Incoming claim payloads may optionally include `rag_filter_decision`, `rag_metadata_filter` (dict), and `product_code` to narrow/shape retrieval.
- A lightweight **DL fraud probability head** can be enabled via `DL_FRAUD_ENABLED` and fused into the decision stage using `DL_FRAUD_FUSION_LLM_WEIGHT` / `DL_FRAUD_FUSION_DL_WEIGHT` (defaults: 0.7 / 0.3). If PyTorch isn’t installed, it falls back to a deterministic logistic scorer.
- Fraud JSON robustness can be tuned via `STRICT_JSON_MODE` and `MAX_LLM_RETRIES` (additional LLM calls when the model returns invalid JSON).

**Image analysis (new):**

- Enable/disable: `ENABLE_IMAGE_ANALYSIS` (default `true`)
- Backend: `IMAGE_MODEL_TYPE=heuristic|cnn` (default `heuristic`)
- Fusion: `IMAGE_FUSION_WEIGHT` (default `0.2`) controls how much image severity influences final triage.
- CNN confidence gating: `CNN_CONF_THRESHOLD` (default `0.7`)
- CNN weights path: `IMAGE_CNN_MODEL_PATH` (optional; if missing/invalid the system falls back safely)

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
| `POST` | `/claims` | Process a claim (JSON or multipart/form-data) |
| `GET` | `/claims` | List stored claims from Chroma (MVP: up to **200** rows, fixed offset **0**) |
| `GET` | `/claims/{claim_id}/image-preview` | Return stored base64 preview for the claim image (if present) |
| `GET` | `/claims/{claim_id}/gradcam` | Return a Grad-CAM heatmap overlay PNG for the stored claim image (503 if CNN/weights unavailable) |
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

**Image input options:**

- **JSON**: include `image_base64` (raw base64 string or a `data:image/...;base64,...` data URL)
- **Multipart**: send `multipart/form-data` with either:
  - `claim` = JSON string + optional file field `file` or `image`, or
  - flat fields `claim_id`, `claim_amount`, `policy_limit`, optional `description`, plus optional file field `file` or `image`

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


## Disclaimer
This project is an independent implementation inspired by real-world systems and does not contain any proprietary information.

## Author

Jayendran Subramanian — Consulting AI Data Engineer | AI Architect (in progress)

**Note:** This repository is a proof-of-concept and reference implementation. Production systems handling confidential data require additional security, governance, and operations beyond this MVP.

## 📄 License

This project is licensed under the MIT License.
