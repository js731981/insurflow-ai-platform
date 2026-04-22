---
title: Insurance AI Decision Demo
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.31.0"
python_version: "3.10"
app_file: app.py
pinned: false
short_description: Insurance AI claim decision engine demo
---

# AI Insurance Claim Decision Demo (Baseline)

This Space is intentionally minimal to provide a stable baseline (no schema crashes).

## What it does

- Takes **Description**, **Claim Amount**, and **Policy Limit**
- Returns a simple **Decision**, **Fraud Score**, and **Severity**

## Next steps (planned)

After the baseline is stable, we will re-add features step by step:

1. Decision UI
2. Explanation
3. RAG
4. CNN

## Run locally

From `hf_space/`:

```bash
cd hf_space
python app.py
```
