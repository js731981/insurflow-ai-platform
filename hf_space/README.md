# AI Insurance Claim Decision Demo

Interactive Hugging Face Space for the **Insurance AI Decision Platform**: a product-style, explainable triage experience with fraud visualization, pipeline attribution, and optional live API integration.

## Features

- **Enhanced claim form** — Claim ID, narrative, claim amount, **policy limit**, and drag-and-drop image upload with live preview.
- **Demo scenarios** — One-click presets: low-damage valid claim, no-damage fraud-style narrative, and major-damage high-value claim (loads curated images from `assets/demo_images/`).
- **Fraud score visualization** — Radial gauge plus linear progress bar with **green (0–0.3)**, **yellow (0.3–0.7)**, and **red (0.7–1.0)** risk bands.
- **Decision breakdown** — Horizontal contribution-style bars for **CNN**, **Rules**, and **LLM** (signed-style demo values; richer when backed by API fusion metadata).
- **AI pipeline visual** — Horizontal flow: Image → CNN → Rules → LLM → Decision with **USED** / **SKIPPED** / **FAILED** states.
- **Grad-CAM / attention view** — Toggle **Show AI attention**, opacity slider (0–100%), overlay on the claim image; uses backend Grad-CAM PNG when available, otherwise a lightweight edge-based attention map.
- **Inconsistency detection** — Warning banner when image/text mismatch or unusually high amounts are detected (aligned with backend `fraud_signal` when using the API).
- **Fraud simulation mode** — Extra scrutiny copy and banner styling for demo storytelling.
- **Mini analytics** — Calls **`GET /analytics/summary`** when a backend URL is configured (total claims, approval rate, investigation rate).
- **Download report** — Exports JSON with claim inputs, decision, explanation, CNN label, fraud score, and source.
- **Latency panel** — Total time (client-measured for API), CNN/vision processing from API when present, LLM time estimated from remainder when server omits explicit LLM timing.
- **Dark theme** — Slate background (`#0f172a`), high-contrast text (`#e2e8f0`), accent (`#6366f1`), rounded cards and subtle motion.

## Demo instructions

1. **Local (Spaces default)** — Leave “Call live API” unchecked (or unset env). Click **Analyze claim** after filling the form or use a **demo scenario** button.
2. **With your FastAPI backend** — Set `INSURANCE_API_BASE_URL` (or `BACKEND_URL`) to your API root, e.g. `http://127.0.0.1:8000`. Enable **Call live API (POST /claims)**. Submit a claim; optional Grad-CAM is fetched from `GET /claims/{claim_id}/gradcam` when an image was uploaded.
3. **Grad-CAM** — After analysis, enable **Show AI attention** and adjust **Overlay opacity** to explore the heatmap.
4. **Report** — Click **Download report (JSON)** to save the last run’s structured summary.

## Screenshots

> Placeholder: add Space screenshots here after deployment (form + results + pipeline + fraud gauge).

## Tech stack

- **Gradio** — UI framework.
- **Pillow / NumPy** — Image preview, pseudo–attention maps, and overlays.
- **httpx** — All HTTP calls via `hf_space/utils/api_client.py` (`POST /claims`, `GET /analytics/summary`, optional `GET /claims/{id}/gradcam`).
- **PyTorch / torchvision** (optional) — Loads `hf_space/models/model.pth` when present for CNN inference in local mode.

## Layout

```text
hf_space/
├── app.py
├── ui/
│   ├── components.py
│   ├── styles.css
│   └── scripts.js
├── assets/
│   ├── demo_images/     # generated on first run if missing
│   └── icons/
├── models/
│   └── model.pth        # optional CNN checkpoint
├── utils/
│   ├── api_client.py
│   ├── formatters.py
│   ├── gradcam_utils.py
│   ├── demo_assets.py
│   ├── inference.py
│   └── image_utils.py
└── README.md
```

## Run locally

From the **repository root** (so `import hf_space` resolves):

```bash
python -m hf_space.app
```

Or from `hf_space/` (parent added to `sys.path` by `app.py`):

```bash
cd hf_space
python app.py
```
