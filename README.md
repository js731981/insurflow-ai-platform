InsurFlow AI Platform
Production-Grade Multi-Agent AI Architecture (Insurance Domain)
________________________________________
Overview
InsurFlow AI Platform simulates a real-world insurance decision engine using modern AI architecture patterns:
  •	Multi-LLM routing (Ollama, OpenAI, OpenRouter)
  •	Async agent orchestration
  •	Hybrid AI (Rules + ML + LLM)
  •	Clean modular architecture
  •	Real-time claim decisioning
________________________________________
Key Features
  •	Multi-LLM Routing (OpenAI, Ollama, OpenRouter)
  •	Agent-based architecture (Fraud Agent)
  •	Async execution using asyncio
  •	Clean modular architecture
  •	Cost-efficient local LLM (phi3 via Ollama)
________________________________________
Use Cases
  •	Fraud Detection
  •	Claim Adjudication
  •	Decision Intelligence Systems
  •	AI Agent Orchestration Platforms
________________________________________
Architecture
API → Orchestrator → Agents → LLM Router → Providers

Architecture Diagram
             ┌──────────────┐
             │   FastAPI    │
             │   (API)      │
             └──────┬───────┘
                    │
                    ▼
            ┌──────────────────┐
            │  Orchestrator    │
            │ (Workflow Brain) │
            └──────┬───────────┘
                   │
      ┌────────────┼────────────┐
      ▼            ▼            ▼
   ┌────────┐  ┌──────────┐  ┌──────────┐
   │ Fraud  │  │ Medical  │  │ Policy   │
   │ Agent  │  │ Agent    │  │ Agent    │
   └────┬───┘  └────┬─────┘  └────┬─────┘
        │            │            │
        └────────────┼────────────┘
                     ▼
             ┌──────────────┐
             │ LLM Router   │
             │ (Abstraction)│
             └──────┬───────┘
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
 ┌────────┐   ┌──────────┐   ┌──────────┐
 │ OpenAI │   │ Ollama   │   │OpenRouter│
 └────────┘   └──────────┘   └──────────┘
________________________________________
Tech Stack
  •	FastAPI
  •	Python (Asyncio)
  •	Ollama (Local LLM)
  •	Pydantic
  •	Clean Architecture
________________________________________
Getting Started
1. Install dependencies
  pip install -r requirements.txt
2. Setup environment
  Create .env file:
  LLM_PROVIDER=ollama
  MODEL_NAME=phi3
  OLLAMA_BASE_URL=http://localhost:11434
3. Run server
  uvicorn app.main:app --reload
________________________________________
API Example
POST /claim
{
  "claimant_name": "John Doe",
  "claim_amount": 12345,
  "incident_type": "theft"
}
Response
{
  "fraud_score": 0.2,
  "reason": "The claim amount is high for the reported incident type..."
}
________________________________________
Future Enhancements
  •	Multi-agent orchestration (Fraud, Medical, Policy)
  •	Decision engine
  •	Confidence scoring
  •	Vector memory (ChromaDB)
  •	Observability (logs, metrics)
  •	UI dashboard
________________________________________
Author
Jai – Consulting AI Data Engineer | AI Architect (in progress)

Note:
This project is a proof-of-concept (POC) and reference implementation inspired by real-world enterprise AI systems. Actual production implementations involve confidential data and cannot be publicly shared.

