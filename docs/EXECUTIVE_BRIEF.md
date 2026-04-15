# Insurance AI Decision Platform — Executive brief (one page)

## In one sentence

Insurance AI Decision Platform is a **local proof-of-concept** that **triages micro-insurance claims** into **approve, reject, or investigate**, using **AI fraud signals**, **rule-based policy checks**, and **memory of similar past claims**—with optional **human review** and a **simple investigator workflow**.

## Problem

Small-ticket claims need **fast, consistent first-pass decisions** without always pulling senior staff. Teams also benefit from **explainable signals**, **audit-friendly outputs**, and a path to **learn from past decisions** when humans review cases.

## Solution

A single **FastAPI** service runs an **orchestrated pipeline**: parallel **LLM fraud analysis** (structured JSON, with strict parsing + retry when enabled) and **policy validation**, then a **decision step** that produces a final outcome and confidence. **Embeddings** store each claim in **ChromaDB** so the system can **retrieve similar cases** using **review-aware weighted ranking**; retrieved hits are converted into a **compact, token-capped context block** for the fraud prompt (with an optional cheap rerank boost for `product_code`). The service also supports **multimodal claim triage**: an optional claim image (JSON base64 or multipart upload) yields lightweight damage/severity signals and an optional CNN-based damage classifier when weights are configured. Confidence is **calibrated** against human-reviewed similar cases when available, and **HITL** is flagged when the outcome is investigate or when calibrated confidence is low. An optional lightweight **DL fraud probability head** can be enabled and fused into the decision stage. **Cases** can be assigned and moved through **NEW → ASSIGNED → IN_PROGRESS → RESOLVED**; **analytics** summarize volumes, risk patterns, and outliers. A Grad-CAM endpoint can provide a visual heatmap overlay for stored claim images when the CNN stack is available.

## Why it matters (MVP value)

- **Speed:** Automated triage on structured + narrative claim input.
- **Consistency:** Same pipeline and rules every time; LLM output is structured (JSON-oriented).
- **Continuity:** Persisted vector memory improves context for future similar claims.
- **Governance hook:** Explicit HITL flag and human **review** endpoint to align automation with oversight.
- **Low footprint:** Default stack runs **locally** (Ollama + on-disk Chroma)—no mandatory cloud for the core demo.

## Technology (at a glance)

**Python / FastAPI / Pydantic** · **Ollama** (LLM + embeddings; optional OpenAI/OpenRouter) · **ChromaDB** (embedded vector store) · **Optional torch/vision CNN** for damage classification + Grad-CAM · **Minimal web UI** at `/ui`.

## Scope and caveat

This is a **reference MVP**, not production-ready for regulated or highly sensitive data. A real deployment needs **security, privacy, model governance, monitoring, and operational controls** beyond this repository.

*For architecture detail and API tables, see [`PROJECT_OVERVIEW.md`](./PROJECT_OVERVIEW.md) and the root [`README.md`](../README.md).*
