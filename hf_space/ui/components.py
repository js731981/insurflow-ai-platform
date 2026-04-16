from __future__ import annotations

import html
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from config import DEMO_IMAGES
from utils import api_client
from utils.formatters import (
    analytics_panel_html,
    breakdown_panel_html,
    decision_card_html,
    explanation_card_html,
    format_explanation,
    fraud_score_panel_html,
    inconsistency_banner_html,
    latency_panel_html,
    model_insights_html,
    pipeline_panel_html,
    severity_from_cnn_label,
)
from utils.gradcam_utils import (
    composite_gradcam_view,
    compute_edge_attention_map,
    pil_to_png_bytes,
)
from utils.image_utils import load_image_from_url, load_pil, resolve_claim_image
from utils.inference import run_inference

ROOT = Path(__file__).resolve().parents[1]

REPORT_HINT_IDLE = (
    "<div class='report-hint'><strong>Click to download JSON report</strong> (generates a file from the latest analysis).</div>"
)
REPORT_HINT_READY = (
    "<div class='report-hint'><strong>Click to download JSON report</strong> (uses the most recent result).</div>"
)
REPORT_HINT_BUSY = "<div class='report-hint'>Building results…</div>"


def _report_download_meta(path: str) -> str:
    name = Path(path).name
    esc = html.escape(name, quote=True)
    return (
        f"<div class='report-file-meta' title=\"{esc}\">"
        f"<span class='report-file-name'>{esc}</span>"
        f"<span class='report-file-hint'>Click to download JSON report</span>"
        "</div>"
    )


def _load_css() -> str:
    p = ROOT / "ui" / "styles.css"
    return p.read_text(encoding="utf-8")


def load_demo_case(case_type: str) -> dict[str, Any]:
    if case_type == "low":
        return {
            "claim_id": "CLM-DEMO-LOW-001",
            "description": (
                "Minor scuff on bumper after a parking lot tap. Authorized shop quoted $185 for paint touch-up."
            ),
            "amount": 185.0,
            "policy_limit": 5000.0,
            "image_url": DEMO_IMAGES["minor_crack"],
        }
    if case_type == "fraud":
        return {
            "claim_id": "CLM-DEMO-FRAUD-002",
            "description": (
                "Phone screen is badly cracked and shattered after a drop on concrete; urgent replacement needed."
            ),
            "amount": 420.0,
            "policy_limit": 500.0,
            "image_url": DEMO_IMAGES["no_damage"],
        }
    if case_type == "major":
        return {
            "claim_id": "CLM-DEMO-MAJOR-003",
            "description": "Mobile screen major crack after a drop; display failure and urgent replacement required.",
            "amount": 1000.0,
            "policy_limit": 6000.0,
            "image_url": DEMO_IMAGES["major_crack"],
        }
    raise ValueError(case_type)


def apply_scenario(which: str):
    try:
        case = load_demo_case(which)
    except ValueError:
        return (
            "",
            "",
            0.0,
            1000.0,
            None,
            None,
            None,
            "",
            "<div class='caption-text'>Invalid demo scenario.</div>",
        )

    url = str(case["image_url"])
    try:
        pil = load_image_from_url(url)
    except Exception:
        return (
            case["claim_id"],
            case["description"],
            case["amount"],
            case["policy_limit"],
            None,
            None,
            None,
            "",
            "<div class='caption-text'>Demo image failed to load. You can upload an image manually.</div>",
        )

    return (
        case["claim_id"],
        case["description"],
        case["amount"],
        case["policy_limit"],
        pil,
        pil,
        url,
        "",
        "",
    )


def mirror_preview(im: Any) -> Any:
    return im


def on_image_change(img: Any, _demo_url: Optional[str]) -> tuple[Any, Optional[str]]:
    if img is None:
        return None, None
    return mirror_preview(img), gr.update()


def _analysis_error_response(message: str) -> tuple[Any, ...]:
    empty = "<div class='ds-card'><div class='section-h'>{title}</div><div class='caption-text'>{body}</div></div>"
    return (
        f"<div class='status-line status-line--warn'>{html.escape(message)}</div>",
        empty.format(title="Fraud score", body="Run an analysis."),
        empty.format(title="Decision", body="—"),
        empty.format(title="Model insights", body="—"),
        empty.format(title="Explanation", body="—"),
        empty.format(title="Breakdown", body="—"),
        empty.format(title="Pipeline", body="—"),
        empty.format(title="Latency", body="—"),
        "",
        None,
        None,
        False,
        None,
        None,
        REPORT_HINT_IDLE,
    )


def refresh_analytics_html() -> str:
    if not api_client.is_configured():
        return analytics_panel_html(
            None,
            "Connect a backend (set INSURANCE_API_BASE_URL) to load live analytics.",
        )
    try:
        s = api_client.get_analytics_summary()
        return analytics_panel_html(s, None)
    except Exception as exc:
        return analytics_panel_html(None, f"Analytics request failed: {exc}")


def _run_backend(
    claim_id: str,
    description: str,
    amount: float,
    policy_limit: float,
    image: Any,
    fraud_simulation: bool,
) -> tuple[dict[str, Any], Optional[bytes], bool]:
    buf: Optional[bytes] = None
    if image is not None:
        b = io.BytesIO()
        try:
            load_pil(image).save(b, format="JPEG", quality=90)
            buf = b.getvalue()
        except Exception:
            buf = None
    cid = (claim_id or "").strip() or "CLM-HF-ANON"
    raw, elapsed = api_client.submit_claim_multipart(
        claim_id=cid,
        description=description or "",
        claim_amount=float(amount or 0.0),
        policy_limit=float(policy_limit or 0.0),
        image_bytes=buf,
    )
    unified = api_client.parse_claim_api_json(
        raw,
        client_latency_ms=elapsed,
        had_image_upload=buf is not None,
        claim_amount=float(amount or 0.0),
        policy_limit=float(policy_limit or 0.0),
    )
    if fraud_simulation:
        unified["explanation"] = (
            (unified.get("explanation") or "")
            + "\n\n- [Fraud simulation] Additional scrutiny mode is enabled for this demo request."
        )
    gc: Optional[bytes] = None
    is_server = False
    if buf and unified.get("claim_id"):
        gc = api_client.fetch_gradcam_png(str(unified["claim_id"]))
        is_server = gc is not None
    return unified, gc, is_server


def _unified_to_views(
    res: dict[str, Any],
    *,
    backend_note: str,
    fraud_simulation: bool,
) -> tuple[str, ...]:
    fs = float(res.get("fraud_score") or 0.0)
    fraud_html = fraud_score_panel_html(fs)
    pipeline = list(res.get("pipeline") or [])
    llm_status = ""
    for st in pipeline:
        if isinstance(st, dict) and str(st.get("id") or "").lower() == "llm":
            llm_status = str(st.get("status") or "").lower().strip()
            break
    if llm_status == "used":
        ctx = "AI evaluated (LLM)"
    elif llm_status == "failed":
        ctx = "Fallback decision"
    else:
        ctx = "Auto-triaged (Rules)"

    decision_html = decision_card_html(str(res.get("decision") or ""), ctx)

    cnn_label = str(res.get("cnn_label") or "unknown")
    sev_raw = str(res.get("cnn_severity") or res.get("severity") or "").strip()
    sev_ui = sev_raw.upper() if sev_raw else severity_from_cnn_label(cnn_label)
    cnn_html = model_insights_html(cnn_label, sev_ui, backend_note)

    expl_bullets = format_explanation(
        {
            "claim_amount": res.get("claim_amount"),
            "policy_limit": res.get("policy_limit"),
            "cnn_label": cnn_label,
            "fraud_score": fs,
            "pipeline": pipeline,
        }
    )
    expl_html = explanation_card_html(expl_bullets)
    breakdown_html = breakdown_panel_html(res.get("breakdown") or {"cnn": 0.0, "rules": 0.0, "llm": 0.0})
    pipeline_html = pipeline_panel_html(pipeline)
    lat_html = latency_panel_html(res.get("latency_ms") or {"total": 0.0, "cnn": 0.0, "llm": 0.0}, pipeline)

    banner = ""
    if res.get("inconsistent_claim"):
        parts = list(res.get("inconsistency_messages") or [])
        if res.get("fraud_signal") == "image_text_mismatch" and not parts:
            parts.append("Backend flagged a mismatch between narrative damage and visual evidence.")
        msg = " ".join(parts) if parts else "Review image evidence and narrative for alignment."
        banner = inconsistency_banner_html(msg)
    elif fraud_simulation:
        banner = inconsistency_banner_html(
            "Test fraud scenario mode is ON — amplified warnings for demo storytelling.",
        )

    return (
        fraud_html,
        decision_html,
        cnn_html,
        expl_html,
        breakdown_html,
        pipeline_html,
        lat_html,
        banner,
    )


def analyze_claim(
    claim_id: str,
    description: str,
    amount: float,
    policy_limit: float,
    image: Any,
    image_url_typed: str,
    image_url_state: Optional[str],
    fraud_simulation: bool,
    use_backend: bool,
    gradcam_opacity: float,
    show_attention: bool,
):
    combined_url = ((image_url_typed or "").strip() or (image_url_state or "").strip() or None)
    eff, url_err = resolve_claim_image(image, combined_url)
    if eff is None:
        msg = url_err or "No image provided"
        return _analysis_error_response(msg)

    err: Optional[str] = None
    unified: dict[str, Any]
    gc_bytes: Optional[bytes] = None
    gc_server = False

    try:
        if use_backend and api_client.is_configured():
            unified, gc_bytes, gc_server = _run_backend(
                claim_id, description, amount, policy_limit, eff, fraud_simulation
            )
        else:
            unified = run_inference(
                description,
                float(amount or 0.0),
                float(policy_limit or 0.0),
                eff,
                fraud_simulation=fraud_simulation,
            )
            try:
                heat = compute_edge_attention_map(load_pil(eff))
                gc_bytes = pil_to_png_bytes(heat)
            except Exception:
                gc_bytes = None
            gc_server = False
    except Exception as exc:
        err = str(exc)
        unified = run_inference(
            description,
            float(amount or 0.0),
            float(policy_limit or 0.0),
            eff,
            fraud_simulation=fraud_simulation,
        )
        unified["explanation"] = (
            (unified.get("explanation") or "")
            + f"\n\n- API unavailable; using local demo instead. Reason: {err}"
        )
        gc_bytes = None
        gc_server = False
        try:
            heat = compute_edge_attention_map(load_pil(eff))
            gc_bytes = pil_to_png_bytes(heat)
        except Exception:
            pass

    note = (
        "Live API response"
        if unified.get("source") == "api" and not err
        else ("Local inference (API error)" if err else "Local inference")
    )
    # Ensure UI explainability has access to user inputs (even when backend omits them).
    unified["claim_amount"] = float(amount or 0.0)
    unified["policy_limit"] = float(policy_limit or 0.0)
    views = _unified_to_views(unified, backend_note=note, fraud_simulation=fraud_simulation)

    gview = composite_gradcam_view(
        eff,
        gc_bytes,
        gradcam_opacity,
        show_attention,
        server_overlay=gc_server,
    )

    report = build_report(
        claim_id=claim_id or str(unified.get("claim_id") or ""),
        description=description or "",
        claim_amount=float(amount or 0.0),
        policy_limit=float(policy_limit or 0.0),
        unified=unified,
    )

    status = "<div class='status-line'>Done.</div>"
    return (
        status,
        *views,
        gview,
        gc_bytes,
        gc_server,
        report,
        None,
        REPORT_HINT_READY,
    )


def update_gradcam(image: Any, heat_bytes: Optional[bytes], is_server: bool, opacity: float, show: bool) -> Any:
    return composite_gradcam_view(image, heat_bytes, opacity, show, server_overlay=bool(is_server))


def _pipeline_bools_from_steps(pipeline: Any) -> tuple[bool, bool, bool]:
    cnn_used = rules_used = llm_used = False
    if not isinstance(pipeline, list):
        return cnn_used, rules_used, llm_used
    for st in pipeline:
        if not isinstance(st, dict):
            continue
        pid = str(st.get("id") or "").lower()
        status = str(st.get("status") or "").lower().strip()
        if status != "used":
            continue
        if pid == "cnn":
            cnn_used = True
        elif pid == "rules":
            rules_used = True
        elif pid == "llm":
            llm_used = True
    return cnn_used, rules_used, llm_used


def build_report(
    *,
    claim_id: str,
    description: str,
    claim_amount: float,
    policy_limit: float,
    unified: dict[str, Any],
) -> dict[str, Any]:
    cnn_label = str(unified.get("cnn_label") or "")
    sev_raw = str(unified.get("cnn_severity") or unified.get("severity") or "").strip()
    severity = sev_raw.upper() if sev_raw else severity_from_cnn_label(cnn_label)
    confidence = float(unified.get("cnn_confidence") or 0.0)
    pipeline_steps = unified.get("pipeline") if isinstance(unified.get("pipeline"), list) else []
    cnn_used, rules_used, llm_used = _pipeline_bools_from_steps(pipeline_steps)
    cid = (claim_id or "").strip() or str(unified.get("claim_id") or "").strip() or "CLM-HF-ANON"
    return {
        "claim_id": cid,
        "description": description or "",
        "claim_amount": float(claim_amount or 0.0),
        "policy_limit": float(policy_limit or 0.0),
        "fraud_score": float(unified.get("fraud_score") or 0.0),
        "decision": str(unified.get("decision") or ""),
        "severity": severity,
        "cnn_label": cnn_label,
        "confidence": confidence,
        "pipeline": {
            "cnn_used": cnn_used,
            "rules_used": rules_used,
            "llm_used": llm_used,
        },
    }


def generate_report(data: dict) -> str:
    raw_id = str(data.get("claim_id") or "claim").strip() or "claim"
    safe = re.sub(r'[<>:"/\\|?*\s]+', "_", raw_id)[:80] or "claim"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{safe}.json")
    tmp.close()
    with open(tmp.name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return tmp.name


def on_download_click(report: Optional[dict[str, Any]]):
    if not report:
        return None, REPORT_HINT_IDLE
    try:
        path = generate_report(report)
        return path, _report_download_meta(path)
    except Exception as e:
        print("Report generation failed:", e)
        return None, f"<div class='report-hint report-hint--warn'>Could not create file: {html.escape(str(e))}</div>"


def clear_all():
    empty = "<div class='ds-card'><div class='section-h'>{t}</div><div class='caption-text'>{b}</div></div>"
    return (
        "",
        "",
        0.0,
        1000.0,
        None,
        None,
        "",
        "",
        False,
        api_client.is_configured(),
        45.0,
        False,
        "<div class='status-line'>Ready.</div>",
        empty.format(t="Fraud score", b="Run an analysis."),
        empty.format(t="Decision", b="—"),
        empty.format(t="Model insights", b="—"),
        empty.format(t="Explanation", b="—"),
        empty.format(t="Breakdown", b="—"),
        empty.format(t="Pipeline", b="—"),
        empty.format(t="Latency", b="—"),
        "",
        None,
        None,
        False,
        None,
        None,
        REPORT_HINT_IDLE,
        refresh_analytics_html(),
        None,
    )


def create_demo() -> gr.Blocks:
    css = _load_css()
    js_path = ROOT / "ui" / "scripts.js"
    js = js_path.read_text(encoding="utf-8") if js_path.exists() else None

    with gr.Blocks(
        title="AI Insurance Claim Decision Demo",
        theme=gr.themes.Soft(),
        css=css,
    ) as demo:
        # Gradio 6+ expects these on launch(), not Blocks().
        demo._launch_kwargs = {  # type: ignore[attr-defined]
            "js": js,
        }
        report_state = gr.State(value=None)
        heat_state = gr.State(value=None)
        heat_is_server = gr.State(value=False)
        image_url_state = gr.State(value=None)

        gr.HTML(
            """
            <div id="app-header">
              <div>
                <div id="app-title">AI Insurance Claim Decision Demo</div>
                <div id="app-subtitle">Explainable triage • Fraud visualization • Vision + rules + LLM. Optional live API via <code>INSURANCE_API_BASE_URL</code>.</div>
              </div>
            </div>
            """
        )

        with gr.Row(equal_height=False, elem_classes=["main-layout"]):
            with gr.Column(scale=1, min_width=360, elem_classes=["form-panel"]):
                gr.Markdown("### Claim form", elem_classes=["panel-title"])
                claim_id = gr.Textbox(label="Claim ID", placeholder="CLM-2026-000123", container=True)
                description = gr.Textbox(
                    label="Description",
                    placeholder="Describe the incident, damage, and context.",
                    lines=6,
                    elem_id="claim-desc",
                    container=True,
                )
                with gr.Row():
                    amount = gr.Number(
                        label="Claim amount (USD)",
                        value=250.0,
                        minimum=0,
                        precision=2,
                        container=True,
                    )
                    policy_limit = gr.Number(
                        label="Policy limit (USD)",
                        value=1000.0,
                        minimum=0,
                        precision=2,
                        container=True,
                    )

                image = gr.Image(label="Claim image (drag & drop)", type="pil", elem_id="image-uploader", height=240)
                preview = gr.Image(
                    label="Uploaded Image",
                    type="pil",
                    interactive=False,
                    height=160,
                    elem_id="image-preview-box",
                )

                demo_img_status = gr.HTML(value="", elem_classes=["demo-status-slot"])
                image_url_input = gr.Textbox(
                    label="Image URL (optional)",
                    placeholder="https://…",
                    lines=1,
                    max_lines=1,
                    container=True,
                )

                with gr.Row(elem_classes=["scenario-row"]):
                    b_low = gr.Button("Low damage – valid claim", variant="secondary", elem_classes=["scenario-pill"])
                    b_fraud = gr.Button("No damage – fraud case", variant="secondary", elem_classes=["scenario-pill"])
                    b_major = gr.Button("Major damage – high claim", variant="secondary", elem_classes=["scenario-pill"])

                with gr.Row():
                    fraud_sim = gr.Checkbox(label="Test fraud scenario", value=False)
                    use_backend = gr.Checkbox(
                        label="Call live API (POST /claims)",
                        value=api_client.is_configured(),
                    )

                with gr.Row(elem_classes=["form-actions"]):
                    run_btn = gr.Button("Analyze claim", elem_id="analyze-btn", elem_classes=["primary-btn"])
                    clear_btn = gr.Button("Clear", variant="secondary", elem_classes=["btn-clear-outline"])

            with gr.Column(scale=1, min_width=360, elem_classes=["results-panel"]):
                gr.Markdown("### Results", elem_classes=["panel-title"])
                status = gr.HTML(value="<div class='status-line'>Ready.</div>")
                with gr.Group(elem_classes=["card"]):
                    analytics_box = gr.HTML(value=refresh_analytics_html())
                    refresh_a = gr.Button("Refresh analytics", variant="secondary", scale=0, elem_classes=["btn-ghost"])

                with gr.Column(elem_classes=["result-stack"]):
                    banner_box = gr.HTML()
                    with gr.Group(elem_classes=["card"]):
                        fraud_kpi = gr.HTML()
                    with gr.Group(elem_classes=["card"]):
                        decision_badge = gr.HTML()
                    with gr.Group(elem_classes=["card"]):
                        cnn_box = gr.HTML()
                    with gr.Group(elem_classes=["card"]):
                        explanation_box = gr.HTML()
                    with gr.Group(elem_classes=["card"]):
                        breakdown_box = gr.HTML()
                    with gr.Group(elem_classes=["card"]):
                        pipeline_box = gr.HTML()
                    with gr.Group(elem_classes=["card"]):
                        latency_box = gr.HTML()

                with gr.Group(elem_classes=["card", "gradcam-stack"]):
                    gr.Markdown("#### Grad-CAM / attention view", elem_classes=["panel-title", "panel-title--sm"])
                    with gr.Row():
                        show_cam = gr.Checkbox(label="Show AI attention", value=False)
                        cam_opacity = gr.Slider(label="Overlay opacity (%)", minimum=0, maximum=100, value=45, step=1)
                    gradcam_img = gr.Image(label="Attention overlay", type="pil", height=280)

                report_hint = gr.HTML(value=REPORT_HINT_IDLE)
                with gr.Row(elem_classes=["report-row"]):
                    dl_btn = gr.Button("Download report (JSON)", variant="secondary", elem_classes=["btn-report"])
                    report_file = gr.File(
                        label="JSON report",
                        interactive=False,
                        visible=True,
                    )

        image.change(fn=on_image_change, inputs=[image, image_url_state], outputs=[preview, image_url_state])

        def _clear_demo_url_on_upload() -> Optional[str]:
            return None

        if hasattr(image, "upload"):
            image.upload(fn=_clear_demo_url_on_upload, inputs=[], outputs=[image_url_state])

        refresh_a.click(fn=refresh_analytics_html, inputs=[], outputs=[analytics_box])

        busy_targets = [
            status,
            fraud_kpi,
            decision_badge,
            cnn_box,
            explanation_box,
            breakdown_box,
            pipeline_box,
            latency_box,
            banner_box,
            gradcam_img,
            heat_state,
            heat_is_server,
        ]

        outs = busy_targets + [report_state, report_file, report_hint]

        def _busy_shell():
            return "<div class='ds-card'><div class='caption-text'>Updating…</div></div>"

        def _busy(current_report: Optional[dict[str, Any]]):
            sh = _busy_shell()
            return (
                "<div class='status-line'>Analyzing claim…</div>",
                sh,
                sh,
                sh,
                sh,
                sh,
                sh,
                sh,
                "",
                None,
                None,
                False,
                current_report,
                None,
                REPORT_HINT_BUSY,
            )

        run_evt = run_btn.click(fn=_busy, inputs=[report_state], outputs=outs)
        run_evt.then(
            fn=analyze_claim,
            inputs=[
                claim_id,
                description,
                amount,
                policy_limit,
                image,
                image_url_input,
                image_url_state,
                fraud_sim,
                use_backend,
                cam_opacity,
                show_cam,
            ],
            outputs=outs,
            show_progress="full",
        )

        for btn, key in ((b_low, "low"), (b_fraud, "fraud"), (b_major, "major")):
            btn.click(
                fn=lambda k=key: apply_scenario(k),
                inputs=[],
                outputs=[
                    claim_id,
                    description,
                    amount,
                    policy_limit,
                    image,
                    preview,
                    image_url_state,
                    image_url_input,
                    demo_img_status,
                ],
            )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                claim_id,
                description,
                amount,
                policy_limit,
                image,
                image_url_state,
                demo_img_status,
                image_url_input,
                fraud_sim,
                use_backend,
                cam_opacity,
                show_cam,
                status,
                fraud_kpi,
                decision_badge,
                cnn_box,
                explanation_box,
                breakdown_box,
                pipeline_box,
                latency_box,
                banner_box,
                gradcam_img,
                heat_state,
                heat_is_server,
                report_state,
                report_file,
                report_hint,
                analytics_box,
                preview,
            ],
        )

        for evt in [cam_opacity.change, show_cam.change]:
            evt(
                fn=update_gradcam,
                inputs=[image, heat_state, heat_is_server, cam_opacity, show_cam],
                outputs=[gradcam_img],
            )

        dl_btn.click(fn=on_download_click, inputs=[report_state], outputs=[report_file, report_hint])

        demo.load(fn=refresh_analytics_html, inputs=[], outputs=[analytics_box])
        demo.queue(default_concurrency_limit=16)

    return demo
