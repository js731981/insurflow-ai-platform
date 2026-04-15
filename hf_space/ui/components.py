from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from hf_space.utils import api_client
from hf_space.utils.demo_assets import ensure_demo_images
from hf_space.utils.formatters import (
    analytics_panel_html,
    breakdown_panel_html,
    build_report_dict,
    decision_card_html,
    explanation_card_html,
    format_explanation,
    fraud_score_panel_html,
    inconsistency_banner_html,
    latency_panel_html,
    model_insights_html,
    report_json_bytes,
    pipeline_panel_html,
    severity_from_cnn_label,
)
from hf_space.utils.gradcam_utils import (
    composite_gradcam_view,
    compute_edge_attention_map,
    pil_to_png_bytes,
)
from hf_space.utils.image_utils import load_pil
from hf_space.utils.inference import run_inference

ROOT = Path(__file__).resolve().parents[1]


def _load_css() -> str:
    p = ROOT / "ui" / "styles.css"
    return p.read_text(encoding="utf-8")


def _load_scenario(which: str):
    ensure_demo_images(ROOT)
    demo_dir = ROOT / "assets" / "demo_images"
    if which == "low":
        return (
            "CLM-DEMO-LOW-001",
            "Minor scuff on bumper after a parking lot tap. Authorized shop quoted $185 for paint touch-up.",
            185.0,
            5000.0,
            str(demo_dir / "low_damage_valid.jpg"),
        )
    if which == "fraud":
        return (
            "CLM-DEMO-FRAUD-002",
            "Phone screen is badly cracked and shattered after a drop on concrete; urgent replacement needed.",
            420.0,
            500.0,
            str(demo_dir / "no_damage_fraud.jpg"),
        )
    if which == "major":
        return (
            "CLM-DEMO-MAJOR-003",
            "Vehicle collision caused severe panel damage and shattered glass; repair center total quote pending.",
            8500.0,
            6000.0,
            str(demo_dir / "major_damage_high.jpg"),
        )
    raise ValueError(which)


def apply_scenario(which: str):
    cid, desc, amt, lim, rel = _load_scenario(which)
    from PIL import Image

    p = Path(rel)
    img = Image.open(p).convert("RGB") if p.exists() else None
    return cid, desc, amt, lim, img


def mirror_preview(im: Any) -> Any:
    return im


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
    fraud_simulation: bool,
    use_backend: bool,
    gradcam_opacity: float,
    show_attention: bool,
):
    err: Optional[str] = None
    unified: dict[str, Any]
    gc_bytes: Optional[bytes] = None
    gc_server = False

    try:
        if use_backend and api_client.is_configured():
            unified, gc_bytes, gc_server = _run_backend(
                claim_id, description, amount, policy_limit, image, fraud_simulation
            )
        else:
            unified = run_inference(
                description,
                float(amount or 0.0),
                float(policy_limit or 0.0),
                image,
                fraud_simulation=fraud_simulation,
            )
            if image is not None:
                try:
                    heat = compute_edge_attention_map(load_pil(image))
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
            image,
            fraud_simulation=fraud_simulation,
        )
        unified["explanation"] = (
            (unified.get("explanation") or "")
            + f"\n\n- API unavailable; using local demo instead. Reason: {err}"
        )
        gc_bytes = None
        gc_server = False
        if image is not None:
            try:
                heat = compute_edge_attention_map(load_pil(image))
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
        image,
        gc_bytes,
        gradcam_opacity,
        show_attention,
        server_overlay=gc_server,
    )

    report = build_report_dict(
        claim_id=claim_id or str(unified.get("claim_id") or ""),
        description=description or "",
        claim_amount=float(amount or 0.0),
        policy_limit=float(policy_limit or 0.0),
        decision=str(unified.get("decision") or ""),
        explanation=str(unified.get("explanation") or ""),
        cnn_label=str(unified.get("cnn_label") or ""),
        fraud_score=float(unified.get("fraud_score") or 0.0),
        source=str(unified.get("source") or "local"),
        breakdown=unified.get("breakdown") if isinstance(unified.get("breakdown"), dict) else None,
        pipeline=list(unified.get("pipeline") or []) if unified.get("pipeline") else None,
        latency_ms=unified.get("latency_ms") if isinstance(unified.get("latency_ms"), dict) else None,
        fraud_signal=str(unified.get("fraud_signal") or "").strip() or None,
    )

    status = "<div class='help-text'>Done.</div>"
    return (
        status,
        *views,
        gview,
        gc_bytes,
        gc_server,
        report,
    )


def update_gradcam(image: Any, heat_bytes: Optional[bytes], is_server: bool, opacity: float, show: bool) -> Any:
    return composite_gradcam_view(image, heat_bytes, opacity, show, server_overlay=bool(is_server))


def build_report_file(report: Optional[dict[str, Any]]):
    if not report:
        return None
    p = Path(tempfile.gettempdir()) / "insurance_claim_report.json"
    p.write_bytes(report_json_bytes(report))
    return str(p)


def clear_all():
    return (
        "",
        "",
        0.0,
        1000.0,
        None,
        False,
        api_client.is_configured(),
        45.0,
        False,
        "<div class='help-text'>Ready.</div>",
        "<div class='soft-card'><div class='section-h'>Fraud score</div><div class='help-text'>Run an analysis.</div></div>",
        "<div class='soft-card'><div class='section-h'>Decision</div><div class='help-text'>—</div></div>",
        "<div class='soft-card'><div class='section-h'>Model insights</div><div class='help-text'>—</div></div>",
        "<div class='soft-card'><div class='section-h'>Explanation</div><div class='help-text'>—</div></div>",
        "<div class='soft-card'><div class='section-h'>Breakdown</div><div class='help-text'>—</div></div>",
        "<div class='soft-card'><div class='section-h'>Pipeline</div><div class='help-text'>—</div></div>",
        "<div class='soft-card'><div class='section-h'>Latency</div><div class='help-text'>—</div></div>",
        "",
        None,
        None,
        False,
        None,
        refresh_analytics_html(),
        None,
    )


def create_demo() -> gr.Blocks:
    ensure_demo_images(ROOT)
    css = _load_css()
    js_path = ROOT / "ui" / "scripts.js"
    js = js_path.read_text(encoding="utf-8") if js_path.exists() else None

    with gr.Blocks(title="AI Insurance Claim Decision Demo") as demo:
        # Gradio 6+ expects these on launch(), not Blocks().
        demo._launch_kwargs = {  # type: ignore[attr-defined]
            "css": css,
            "js": js,
            "theme": gr.themes.Base(),
        }
        report_state = gr.State(value=None)
        heat_state = gr.State(value=None)
        heat_is_server = gr.State(value=False)

        gr.HTML(
            """
            <div id="app-header">
              <div>
                <div id="app-title">AI Insurance Claim Decision Demo</div>
                <div id="app-subtitle">Explainable triage • Fraud visualization • Vision + rules + LLM attribution</div>
              </div>
              <div class="help-text" style="font-size:12px;max-width:420px;text-align:right;">
                Premium demo UI — configure <code>INSURANCE_API_BASE_URL</code> to stream results from your FastAPI backend.
              </div>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=360, elem_classes=["card"]):
                gr.Markdown("### Claim form")
                claim_id = gr.Textbox(label="Claim ID", placeholder="CLM-2026-000123")
                description = gr.Textbox(
                    label="Description",
                    placeholder="Describe the incident, damage, and context.",
                    lines=6,
                    elem_id="claim-desc",
                )
                with gr.Row():
                    amount = gr.Number(label="Claim amount (USD)", value=250.0, minimum=0, precision=2)
                    policy_limit = gr.Number(label="Policy limit (USD)", value=1000.0, minimum=0, precision=2)

                image = gr.Image(label="Claim image (drag & drop)", type="pil", elem_id="image-uploader", height=240)
                preview = gr.Image(
                    label="Upload preview",
                    type="pil",
                    interactive=False,
                    height=140,
                    elem_id="image-preview-box",
                )

                with gr.Row(elem_classes=["scenario-row"]):
                    b_low = gr.Button("Low damage – valid claim", variant="secondary")
                    b_fraud = gr.Button("No damage – fraud case", variant="secondary")
                    b_major = gr.Button("Major damage – high claim", variant="secondary")

                with gr.Row():
                    fraud_sim = gr.Checkbox(label="Test fraud scenario", value=False)
                    use_backend = gr.Checkbox(
                        label="Call live API (POST /claims)",
                        value=api_client.is_configured(),
                    )

                with gr.Row():
                    run_btn = gr.Button("Analyze claim", elem_id="analyze-btn", elem_classes=["submit-btn"])
                    clear_btn = gr.Button("Clear", variant="secondary")

            with gr.Column(scale=1, min_width=360, elem_classes=["card"]):
                gr.Markdown("### Results")
                status = gr.HTML(value="<div class='help-text'>Ready.</div>")
                analytics_box = gr.HTML(value=refresh_analytics_html())
                refresh_a = gr.Button("Refresh analytics", variant="secondary", scale=0)

                fraud_kpi = gr.HTML()
                decision_badge = gr.HTML()
                banner_box = gr.HTML()
                cnn_box = gr.HTML()
                breakdown_box = gr.HTML()
                pipeline_box = gr.HTML()
                latency_box = gr.HTML()
                explanation_box = gr.HTML()

                gr.Markdown("#### Grad-CAM / attention view")
                with gr.Row():
                    show_cam = gr.Checkbox(label="Show AI attention", value=False)
                    cam_opacity = gr.Slider(label="Overlay opacity (%)", minimum=0, maximum=100, value=45, step=1)
                gradcam_img = gr.Image(label="Attention overlay", type="pil", height=280)

                with gr.Row():
                    dl_btn = gr.Button("Download report (JSON)", variant="secondary")
                    file_out = gr.File(label="Report file", interactive=False, visible=True)

        image.change(fn=mirror_preview, inputs=[image], outputs=[preview])

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

        outs = busy_targets + [report_state]

        def _busy():
            return (
                "<div class='help-text'>Analyzing claim with AI…</div>",
                "<div class='soft-card'><div class='help-text'>Updating…</div></div>",
                "<div class='soft-card'><div class='help-text'>Updating…</div></div>",
                "<div class='soft-card'><div class='help-text'>Updating…</div></div>",
                "<div class='soft-card'><div class='help-text'>Updating…</div></div>",
                "<div class='soft-card'><div class='help-text'>Updating…</div></div>",
                "<div class='soft-card'><div class='help-text'>Updating…</div></div>",
                "<div class='soft-card'><div class='help-text'>Updating…</div></div>",
                "",
                None,
                None,
                False,
            )

        run_evt = run_btn.click(fn=_busy, inputs=[], outputs=busy_targets)
        run_evt.then(
            fn=analyze_claim,
            inputs=[
                claim_id,
                description,
                amount,
                policy_limit,
                image,
                fraud_sim,
                use_backend,
                cam_opacity,
                show_cam,
            ],
            outputs=outs,
            show_progress="full",
        )

        for btn, key in ((b_low, "low"), (b_fraud, "fraud"), (b_major, "major")):
            btn.click(fn=lambda k=key: apply_scenario(k), inputs=[], outputs=[claim_id, description, amount, policy_limit, image])

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                claim_id,
                description,
                amount,
                policy_limit,
                image,
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

        dl_btn.click(fn=build_report_file, inputs=[report_state], outputs=[file_out])

        demo.load(fn=refresh_analytics_html, inputs=[], outputs=[analytics_box])
        demo.queue(default_concurrency_limit=16)

    return demo
