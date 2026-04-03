# InsurFlow AI — Project overview (FINAL)

## Purpose

**InsurFlow AI** is a **local MVP** for **micro-insurance claim triage**: it accepts a small-ticket claim and returns a structured outcome **`APPROVED`**, **`REJECTED`**, or **`INVESTIGATE`**, with fraud signals, policy checks, similarity to past claims, calibrated confidence, and optional human-in-the-loop (HITL) flags. It is built to run **without extra cloud infrastructure** beyond what you choose (default: **Ollama** + **ChromaDB** on disk).

## What the system does

1. **Claim processing pipeline** — An orchestrator runs **fraud analysis** (LLM → structured JSON) and **policy validation** (rules, e.g. amount vs limit) **in parallel**, then a **decision agent** merges them into a final triage decision and confidence.
2. **Vector memory** — Embeddings (Ollama) feed a **persistent ChromaDB** collection so the system can **retrieve similar past claims** and rank them with review-aware weighting.
3. **Calibration** — Raw **`confidence_score`** is kept; **`calibrated_confidence`** adjusts using human-reviewed similar cases when available.
4. **HITL** — Review is recommended when the decision is **`INVESTIGATE`** or when **`calibrated_confidence`** is below **0.75** (see `HitlService` in `app/core/dependencies.py`).
5. **Persistence** — Each successfully embedded claim is **upserted** into the vector store for future retrieval and analytics.
6. **Investigator workflow** — Cases on stored claims: list/filter, **assign** (`NEW` → `ASSIGNED`), and status updates (`ASSIGNED` → `IN_PROGRESS` → `RESOLVED`).
7. **Analytics** — Read-only aggregates: summary stats, heuristic anomaly-style alerts, fraud-score leaderboard.
8. **Observability & UI** — Structured logging, in-process **metrics** (`GET /metrics`), and a **minimal dashboard** at **`/ui`**.

## Architecture (conceptual)

```
FastAPI (v0.1.0)
└── InsurFlowOrchestrator
    ├── EmbeddingService (Ollama) → VectorStore (Chroma) — similar claims
    ├── FraudAgent (LLM)     ─┐
    ├── PolicyAgent (rules)  ─┼─→ DecisionAgent → HITL
    └── store_claim (upsert when embedding succeeds)
```

**LLM execution** goes through **`LLMService` → `LLMRouter`** (timeouts, retries, optional fallbacks). Providers: **Ollama** (default), **OpenAI** and **OpenRouter** if API keys are set. **`POST /inference`** supports optional task hints (`cheap` → Ollama, `complex` → OpenAI when configured). Optional **USD cost** heuristics use `LLM_COST_USD_PER_1K_*` environment variables.

## Technology stack

| Layer | Choice |
|--------|--------|
| API | **FastAPI**, **Pydantic v2**, **httpx**, **python-dotenv** |
| Runtime | **Python 3.10+** |
| LLM / embeddings | **Ollama** (chat + embeddings); optional OpenAI / OpenRouter |
| Vector store | **ChromaDB** (embedded, persistent directory) |

Primary dependencies (see `requirements.txt`): `fastapi`, `uvicorn[standard]`, `httpx`, `python-dotenv`, `pydantic`, `chromadb`.

## API surface (summary)

| Area | Highlights |
|------|------------|
| Service | `GET /`, `GET /health`, `GET /metrics` |
| Claims | `POST /claims`, `POST /claim`, `GET /claims`, `POST /claims/{claim_id}/review` |
| Cases | `GET /cases`, `POST /cases/{claim_id}/assign`, `POST /cases/{claim_id}/status` |
| Analytics | `GET /analytics/summary`, `/analytics/anomalies`, `/analytics/leaderboard` |
| Inference | `GET /claim/samples`, `POST /inference` |

**Claim input** is strict Pydantic (`ClaimRequest`, `extra="forbid"`). Empty `description` still gets a text embedding from a **JSON snapshot** of the claim for retrieval.

## Resilience (MVP)

- **LLM:** timeouts and retries; failures logged with **`claim_id`** when present.
- **Embedding/retrieval failure:** API continues; **vector write skipped** for that request.
- **Fraud LLM failure or bad JSON:** safe fallback toward **`INVESTIGATE`** / moderate confidence unless policy rules force **`REJECTED`**.

## Repository layout (main)

- `app/main.py` — App factory, routers, `/ui` static mount, startup log.
- `app/agents/` — Orchestrator, fraud, policy, decision, base agent.
- `app/api/routes/` — health, inference, claims, cases, analytics.
- `app/services/` — LLM, embeddings, vector store, HITL, metrics, analytics, samples.
- `app/core/` — `Settings`, dependency wiring.
- `app/web/` — Dashboard static assets.
- `chroma_db/` (default) — Chroma persistence (`CHROMA_PERSIST_DIR`).

## Positioning

This repository is a **proof-of-concept / reference implementation** for automated micro-claim triage with local AI and a simple case workflow. **Production** use with confidential data would need stronger security, governance, and operations than this MVP.

For setup, environment variables, and detailed API tables, see the root **`README.md`**.

## Author

Jayendran Subramanian — Consulting AI Data Engineer | AI Architect (in progress)
