from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from hf_space import config
from hf_space.utils.image_utils import extract_signals, heuristic_damage_label, load_pil, pil_to_torch_tensor, resize_rgb


@dataclass(frozen=True)
class InferenceResult:
    fraud_score: float
    decision: str
    cnn_label: str
    severity: str
    explanation: str


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _severity_to_risk(severity: str) -> float:
    return float(config.SEVERITY_RISK.get(severity, 0.15))


def _amount_risk(amount: float) -> float:
    if amount <= config.LOW_AMOUNT_USD:
        return 0.05
    if amount >= config.HIGH_AMOUNT_USD:
        return 0.45
    # Linear-ish ramp in the middle
    t = (amount - config.LOW_AMOUNT_USD) / max(1.0, (config.HIGH_AMOUNT_USD - config.LOW_AMOUNT_USD))
    return float(0.05 + 0.40 * float(np.clip(t, 0.0, 1.0)))


def _text_risk(description: str) -> tuple[float, list[str]]:
    d = (description or "").lower()
    flags = []
    risk = 0.10

    kw_high = ["stolen", "theft", "fraud", "fake", "suspicious", "tampered", "multiple claims", "policy lapsed"]
    kw_mid = ["cash", "urgent", "asap", "lost receipt", "no receipt", "unknown", "unwitnessed"]
    kw_damage = ["crack", "cracked", "shattered", "broken", "screen", "impact", "water", "fire"]

    if any(k in d for k in kw_high):
        flags.append("High-risk keywords in description")
        risk += 0.35
    if any(k in d for k in kw_mid):
        flags.append("Ambiguous/urgent wording")
        risk += 0.15
    if any(k in d for k in kw_damage):
        flags.append("Damage-related description")
        risk += 0.05

    return (_clamp01(risk), flags)


def _decision_from_score(score: float) -> str:
    if score <= config.APPROVE_MAX_SCORE:
        return "APPROVE"
    if score <= config.INVESTIGATE_MAX_SCORE:
        return "INVESTIGATE"
    return "REJECT"


def _try_load_cnn_model(model_path: Path):
    """
    Loads a MobileNetV2 checkpoint produced by `train_cnn.py` if present.
    Returns (net, idx_to_class, image_size) or None.
    """
    if not model_path.exists():
        return None

    import torch
    import torch.nn as nn
    import torchvision.models as models

    ckpt = torch.load(str(model_path), map_location="cpu")
    idx_to_class = ckpt.get("idx_to_class") or list(config.TARGET_CLASSES)
    image_size = int(ckpt.get("image_size") or 224)
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        return None

    net = models.mobilenet_v2(weights=None)
    net.classifier[1] = nn.Linear(net.last_channel, len(idx_to_class))
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return (net, list(idx_to_class), image_size)


def _cnn_predict(image: Any) -> tuple[str, str, float, str]:
    """
    Returns (label, severity, confidence, backend_name).
    """
    if not config.USE_CNN:
        return ("unknown", "low", 0.0, "disabled")

    model_path = Path(__file__).resolve().parents[1] / "models" / "model.pth"
    loaded = _try_load_cnn_model(model_path)

    pil = resize_rgb(load_pil(image), 224)
    signals = extract_signals(pil)

    if loaded is None:
        label, severity, conf = heuristic_damage_label(signals)
        return (label, severity, conf, "heuristic")

    net, idx_to_class, image_size = loaded
    x = pil_to_torch_tensor(pil, size=image_size)
    import torch

    with torch.no_grad():
        logits = net(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    label = str(idx_to_class[pred]) if 0 <= pred < len(idx_to_class) else "unknown"
    conf = float(probs[pred]) if 0 <= pred < len(probs) else 0.0

    severity = "low"
    if label == "minor_crack":
        severity = "medium"
    elif label == "major_crack":
        severity = "high"
    return (label, severity, float(_clamp01(conf)), "cnn")


def _mock_llm_explanation(
    *,
    description: str,
    amount: float,
    claim_limit: float,
    cnn_label: str,
    severity: str,
    decision: str,
) -> str:
    text_risk, text_flags = _text_risk(description)
    lines = [
        "This is a demo explanation (no real LLM in Hugging Face Spaces).",
        "",
        f"- Claim amount: ${amount:,.2f}",
        (
            f"- Policy / claim limit: ${claim_limit:,.2f}"
            if claim_limit > 0
            else "- Policy / claim limit: not set (no limit gate applied)"
        ),
        f"- Image assessment: {cnn_label} (severity: {severity})" if cnn_label != "unknown" else "- Image assessment: not provided",
        f"- Text risk signal: {text_risk:.2f}",
    ]
    if text_flags:
        lines.append("- Noted signals:")
        for f in text_flags[:5]:
            lines.append(f"  - {f}")
    lines.extend(
        [
            "",
            f"Decision rationale: Based on amount, image severity, and text signals, the claim is marked **{decision}**.",
            "Disclaimer: This output is for demonstration only and is not an insurance decision.",
        ]
    )
    return "\n".join(lines)


def run_inference(description: str, amount: float, claim_limit: float, image: Any, fraud_simulation: bool = False):
    """
    Returns a dict with core triage fields plus demo-only explainability keys
    (breakdown, pipeline, latency_ms, inconsistency flags, fraud_signal).
    """
    t_all0 = time.perf_counter()
    desc = (description or "").strip()
    amt = float(amount or 0.0)
    try:
        limit = float(claim_limit)
    except (TypeError, ValueError):
        limit = 0.0

    if image is None:
        raise ValueError("No image provided")

    cnn_label = "unknown"
    severity = "low"
    cnn_conf = 0.0
    backend = "none"
    cnn_ms = 0.0

    has_image = image is not None
    if has_image:
        t_c0 = time.perf_counter()
        cnn_label, severity, cnn_conf, backend = _cnn_predict(image)
        cnn_ms = (time.perf_counter() - t_c0) * 1000.0

    text_risk, text_flags = _text_risk(desc)
    text_flags = list(text_flags)
    amt_risk = _amount_risk(amt)
    img_risk = _severity_to_risk(severity) if has_image else 0.0

    # Weighted blend for fast demo. Image has strong impact when present.
    fraud_score = _clamp01(0.45 * text_risk + 0.30 * amt_risk + (0.25 * img_risk if has_image else 0.0))

    # Simple rules override (demo-friendly).
    if amt <= config.LOW_AMOUNT_USD and (not has_image or severity == "low") and fraud_score < 0.40:
        fraud_score = min(fraud_score, 0.25)

    fraud_signal: Optional[str] = None
    mismatch = False
    if has_image and cnn_label == "no_damage" and cnn_conf >= 0.85:
        d = desc.lower()
        if any(k in d for k in ["crack", "cracked", "shattered", "broken"]):
            fraud_score = max(fraud_score, 0.60)
            text_flags.append("Image/text mismatch: description implies damage but image looks undamaged")
            mismatch = True
            fraud_signal = "image_text_mismatch"

    unusually_high = bool(amt > 7500.0 or (limit > 0 and amt > 0.85 * limit))
    if unusually_high:
        text_flags.append("Claim amount is unusually high relative to typical micro-claims")

    if fraud_simulation:
        fraud_score = _clamp01(fraud_score + 0.12)
        text_flags.append("[Fraud simulation] Inconsistency highlighting enabled")

    decision = _decision_from_score(fraud_score)
    if limit > 0 and amt > limit:
        decision = "REJECT"

    t_ll0 = time.perf_counter()
    explanation = _mock_llm_explanation(
        description=desc,
        amount=amt,
        claim_limit=limit if limit > 0 else 0.0,
        cnn_label=(cnn_label if has_image else "unknown"),
        severity=(severity if has_image else "n/a"),
        decision=decision,
    )
    llm_ms = (time.perf_counter() - t_ll0) * 1000.0
    if limit > 0 and amt > limit:
        explanation += "\n\n- Claim amount exceeds the stated policy / claim limit (automatic REJECT)."
    if has_image:
        explanation += f"\n\n(vision backend: {backend}, confidence: {cnn_conf:.2f})"
    if fraud_simulation and text_flags:
        explanation += "\n\n- Fraud simulation notes:"
        for f in text_flags[-4:]:
            explanation += f"\n  - {f}"

    total_ms = (time.perf_counter() - t_all0) * 1000.0

    cnn_b = round((img_risk * 0.55) if has_image else 0.0, 3)
    if has_image and cnn_label == "major_crack":
        cnn_b = round(min(1.0, cnn_b + 0.15), 3)
    rules_b = round(0.08 + 0.35 * amt_risk + (0.28 if limit > 0 and amt > limit else 0.0), 3)
    llm_b = round((text_risk - 0.5) * 0.6, 3)

    llm_status = "used" if config.USE_LLM else "skipped"

    pipeline = [
        {"id": "image", "label": "Image", "status": "used" if has_image else "skipped"},
        {"id": "cnn", "label": "CNN", "status": "used" if has_image else "skipped"},
        {"id": "rules", "label": "Rules", "status": "used"},
        {"id": "llm", "label": "LLM", "status": llm_status},
        {"id": "decision", "label": "Decision", "status": "used"},
    ]

    inconsistent = bool(mismatch or unusually_high)
    inc_msgs: list[str] = []
    if mismatch:
        inc_msgs.append("Image suggests no/minimal damage but the narrative describes substantial damage.")
    if unusually_high:
        inc_msgs.append("Claim amount is unusually high — verify documentation and policy alignment.")

    return {
        "source": "local",
        "claim_id": "",
        "fraud_score": float(fraud_score),
        "decision": str(decision),
        "cnn_label": str(cnn_label if has_image else "n/a"),
        "severity": str(severity if has_image else "n/a"),
        "explanation": str(explanation),
        "breakdown": {"cnn": cnn_b, "rules": rules_b, "llm": llm_b},
        "pipeline": pipeline,
        "latency_ms": {"total": round(total_ms, 2), "cnn": round(cnn_ms, 2), "llm": round(llm_ms, 4)},
        "inconsistent_claim": inconsistent,
        "inconsistency_messages": inc_msgs,
        "fraud_signal": fraud_signal,
    }

