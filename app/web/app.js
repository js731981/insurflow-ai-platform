/* Insurance AI Decision Platform — vanilla JS, no frameworks */

const statusEl = document.getElementById("status");
const rowsEl = document.getElementById("rows");
const refreshBtn = document.getElementById("refreshBtn");
const globalLoadingEl = document.getElementById("globalLoading");
const toastHost = document.getElementById("toastHost");
const claimForm = document.getElementById("claimForm");
const claimSubmitBtn = document.getElementById("claimSubmitBtn");
const claimSubmitSpinner = document.getElementById("claimSubmitSpinner");
const fieldFile = document.getElementById("fieldFile");
const filePreview = document.getElementById("filePreview");
const filePreviewImg = document.getElementById("filePreviewImg");
const dropzone = document.getElementById("dropzone");
const uploadFilenameEl = document.getElementById("uploadFilename");
const imageModal = document.getElementById("imageModal");
const imageModalOriginalImg = document.getElementById("imageModalOriginalImg");
const imageModalEvidenceImg = document.getElementById("imageModalEvidenceImg");
const imageModalMeta = document.getElementById("imageModalMeta");
const imageModalClose = document.getElementById("imageModalClose");
const imageModalHint = document.getElementById("imageModalHint");
const sysHealth = document.getElementById("sysHealth");
const sysHealthText = document.getElementById("sysHealthText");

const decisionEmpty = document.getElementById("decisionEmpty");
const decisionContent = document.getElementById("decisionContent");
const decisionClaimRef = document.getElementById("decisionClaimRef");
const decisionFraud = document.getElementById("decisionFraud");
const decisionDl = document.getElementById("decisionDl");
const decisionConf = document.getElementById("decisionConf");
const decisionBadge = document.getElementById("decisionBadge");
const decisionHitl = document.getElementById("decisionHitl");
const decisionMismatch = document.getElementById("decisionMismatch");
const decisionFallback = document.getElementById("decisionFallback");
const aiModelConfidenceEl = document.getElementById("aiModelConfidence");
const aiImageSeverityEl = document.getElementById("aiImageSeverity");
const aiCnnLabelEl = document.getElementById("aiCnnLabel");
const aiCnnConfidenceEl = document.getElementById("aiCnnConfidence");
const aiCnnSeverityEl = document.getElementById("aiCnnSeverity");
const aiAnalysisBadgeEl = document.getElementById("aiAnalysisBadge");
const aiAnalysisLoadingEl = document.getElementById("aiAnalysisLoading");
const aiAnalysisErrorEl = document.getElementById("aiAnalysisError");
const viewAiEvidenceBtn = document.getElementById("viewAiEvidenceBtn");
const aiPipelineSummaryEl = document.getElementById("aiPipelineSummary");
const aiPipelineChecklistEl = document.getElementById("aiPipelineChecklist");

let latestDecisionClaimId = null;

const intelInsightsEl = document.getElementById("intelInsights");
const intelErrorEl = document.getElementById("intelError");
const intelCardsEl = document.getElementById("intelCards");
const intelFraudScoresEl = document.getElementById("intelFraudScores");
const intelProductsEl = document.getElementById("intelProducts");
const intelBandsEl = document.getElementById("intelBands");
const intelTimelineEl = document.getElementById("intelTimeline");
const intelReviewStripEl = document.getElementById("intelReviewStrip");
const fraudAlertsListEl = document.getElementById("fraudAlertsList");
const fraudAlertsEmptyEl = document.getElementById("fraudAlertsEmpty");
const fraudAlertsErrorEl = document.getElementById("fraudAlertsError");
const leaderboardRowsEl = document.getElementById("leaderboardRows");
const leaderboardErrorEl = document.getElementById("leaderboardError");
const leaderboardLimitEl = document.getElementById("leaderboardLimit");
const leaderboardHighOnlyEl = document.getElementById("leaderboardHighOnly");
const leaderboardUnreviewedOnlyEl = document.getElementById("leaderboardUnreviewedOnly");
const caseMgmtRowsEl = document.getElementById("caseMgmtRows");
const caseMgmtErrorEl = document.getElementById("caseMgmtError");
const caseMgmtUnassignedOnlyEl = document.getElementById("caseMgmtUnassignedOnly");
const caseMgmtMyIdEl = document.getElementById("caseMgmtMyId");
const caseMgmtMyOnlyEl = document.getElementById("caseMgmtMyOnly");

function setLoading(on) {
  const v = Boolean(on);
  if (globalLoadingEl) {
    globalLoadingEl.hidden = !v;
    globalLoadingEl.setAttribute("aria-hidden", v ? "false" : "true");
  }
  if (refreshBtn) refreshBtn.disabled = v;
}

function setFormSubmitting(on) {
  const v = Boolean(on);
  if (claimSubmitBtn) claimSubmitBtn.disabled = v;
  if (claimSubmitSpinner) claimSubmitSpinner.classList.toggle("hidden-inline", !v);
}

function setHealth(state) {
  if (!sysHealth) return;
  sysHealth.classList.remove("health-pill--degraded", "health-pill--error");
  if (state === "error") {
    sysHealth.classList.add("health-pill--error");
    if (sysHealthText) sysHealthText.textContent = "System error";
  } else if (state === "degraded") {
    sysHealth.classList.add("health-pill--degraded");
    if (sysHealthText) sysHealthText.textContent = "Degraded";
  } else {
    if (sysHealthText) sysHealthText.textContent = "System Healthy";
  }
}

function showToast(message, isError = false) {
  if (!toastHost) return;
  const el = document.createElement("div");
  el.className = `toast ${isError ? "error" : "success"}`;
  el.textContent = message;
  toastHost.appendChild(el);
  setTimeout(() => el.remove(), 4800);
}

function isDelayedOrUnavailableAnalysis(out) {
  const sum = out?.agent_outputs?.fraud?.explanation?.summary;
  const src = String(out?.decision_source || "").toLowerCase();
  if (src === "fallback") {
    const hitl = Boolean(out?.hitl_needed);
    const dec = String(out?.decision || "").toUpperCase();
    return hitl || dec === "INVESTIGATE";
  }
  return String(sum || "").toLowerCase().includes("ai analysis delayed or unavailable");
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result || ""));
    fr.onerror = () => reject(fr.error || new Error("Failed to read file."));
    fr.readAsDataURL(file);
  });
}

function setUploadFilename(name) {
  if (!uploadFilenameEl) return;
  if (name) {
    uploadFilenameEl.textContent = name;
    uploadFilenameEl.hidden = false;
  } else {
    uploadFilenameEl.textContent = "";
    uploadFilenameEl.hidden = true;
  }
}

function clearFilePreview() {
  if (filePreview) filePreview.classList.add("hidden");
  if (filePreviewImg) filePreviewImg.removeAttribute("src");
  if (fieldFile) fieldFile.value = "";
  setUploadFilename("");
}

function assignImageFile(file) {
  if (!fieldFile || !file) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  fieldFile.files = dt.files;
  fieldFile.dispatchEvent(new Event("change", { bubbles: true }));
}

function closeImageModal() {
  if (!imageModal) return;
  imageModal.classList.remove("open");
  imageModal.hidden = true;
  if (imageModalOriginalImg) imageModalOriginalImg.removeAttribute("src");
  if (imageModalEvidenceImg) imageModalEvidenceImg.removeAttribute("src");
  if (imageModalHint) imageModalHint.textContent = "";
}

async function openImageModal(claimId) {
  if (!imageModal || !imageModalOriginalImg || !imageModalEvidenceImg || !imageModalMeta) return;
  imageModal.hidden = false;
  imageModalMeta.textContent = `Claim ${claimId} — loading image…`;
  imageModalOriginalImg.removeAttribute("src");
  imageModalEvidenceImg.removeAttribute("src");
  if (imageModalHint) {
    imageModalHint.textContent = "Highlighted regions indicate damage areas influencing AI decision.";
  }
  requestAnimationFrame(() => imageModal.classList.add("open"));
  let evidenceUrl = null;
  try {
    const data = await apiJson(`/claims/${encodeURIComponent(claimId)}/image-preview`);
    const mime = data.mime_type || "image/jpeg";
    const b64 = data.image_base64 || "";
    if (!b64) throw new Error("Empty preview payload.");
    imageModalMeta.textContent = `Claim ${claimId}`;
    imageModalOriginalImg.src = `data:${mime};base64,${b64}`;

    // Best-effort: fetch Grad-CAM overlay; fail-safe if unavailable.
    const res = await fetch(`/claims/${encodeURIComponent(claimId)}/gradcam`);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      if (imageModalHint) {
        imageModalHint.textContent =
          `⚠️ AI Evidence unavailable (${res.status}). ` +
          (text ? `Details: ${text.slice(0, 240)}` : "The CNN explainability stack may be disabled.");
      }
      return;
    }
    const blob = await res.blob();
    evidenceUrl = URL.createObjectURL(blob);
    imageModalEvidenceImg.src = evidenceUrl;
    const lbl = res.headers.get("X-Gradcam-Label");
    const conf = res.headers.get("X-Gradcam-Confidence");
    if (imageModalHint && (lbl || conf)) {
      const bits = [];
      if (lbl) bits.push(`pred=${lbl}`);
      if (conf) bits.push(`conf=${Number(conf).toFixed(3)}`);
      imageModalHint.textContent = bits.length ? `AI Evidence: ${bits.join(" · ")}` : "";
    }
  } catch (err) {
    imageModalMeta.textContent = String(err?.message || err);
  } finally {
    // Revoke the object URL shortly after image loads (or immediately on close).
    if (evidenceUrl && imageModalEvidenceImg) {
      imageModalEvidenceImg.addEventListener(
        "load",
        () => {
          try {
            URL.revokeObjectURL(evidenceUrl);
          } catch {}
        },
        { once: true },
      );
    }
  }
}

function extractFraudFromSubmit(out) {
  const fused = out?.agent_outputs?.decision?.fused_fraud_score;
  if (fused !== undefined && fused !== null && Number.isFinite(Number(fused))) return Number(fused);
  const f = out?.agent_outputs?.fraud?.fraud_score;
  if (f !== undefined && f !== null && Number.isFinite(Number(f))) return Number(f);
  return null;
}

function extractDlFromSubmit(out) {
  const p = out?.agent_outputs?.dl_fraud?.fraud_probability;
  if (p === null || p === undefined) return null;
  const n = Number(p);
  return Number.isFinite(n) ? n : null;
}

function decisionPillClass(d) {
  const u = String(d || "").toUpperCase();
  if (u === "APPROVED") return "badge-pill badge-pill--approved";
  if (u === "REJECTED") return "badge-pill badge-pill--rejected";
  if (u === "INVESTIGATE") return "badge-pill badge-pill--investigate";
  return "badge-pill badge-pill--neutral";
}

function showDecisionEmpty() {
  if (decisionEmpty) decisionEmpty.classList.remove("hidden");
  if (decisionContent) decisionContent.classList.add("hidden");
}

function fillDecisionPanel({
  claimId,
  fraudScore,
  dlScore,
  confidence,
  decision,
  hitl,
  decisionSource,
  cnnFallbackUsed,
  fraudSignal,
}) {
  if (!decisionEmpty || !decisionContent) return;
  decisionEmpty.classList.add("hidden");
  decisionContent.classList.remove("hidden");
  if (decisionClaimRef) {
    decisionClaimRef.textContent = claimId ? `Claim ${claimId}` : "";
  }
  if (decisionFraud) {
    const fs = fraudScore;
    decisionFraud.textContent = fs === null || fs === undefined || Number.isNaN(Number(fs)) ? "—" : fmt3(fs);
    decisionFraud.className = `decision-metric__value fraud-score-lg ${fraudClass(fs)}`;
  }
  if (decisionDl) {
    decisionDl.textContent = dlScore === null || dlScore === undefined ? "—" : fmt3(dlScore);
  }
  if (decisionConf) {
    const c = confidence;
    decisionConf.textContent = c === null || c === undefined || Number.isNaN(Number(c)) ? "—" : fmt3(c);
  }
  if (decisionBadge) {
    const d = String(decision || "").trim();
    if (d) {
      decisionBadge.textContent = d.toUpperCase();
      decisionBadge.className = decisionPillClass(d);
      decisionBadge.hidden = false;
    } else {
      decisionBadge.textContent = "";
      decisionBadge.hidden = true;
    }
  }
  if (decisionHitl) {
    decisionHitl.hidden = !hitl;
  }
  if (decisionMismatch) {
    decisionMismatch.hidden = String(fraudSignal || "").trim() !== "image_text_mismatch";
  }
  if (decisionFallback) {
    const src = String(decisionSource || "").toLowerCase();
    const dec = String(decision || "").toUpperCase();
    if (cnnFallbackUsed) {
      decisionFallback.hidden = false;
      decisionFallback.textContent = "CNN low confidence — fallback used";
    } else if (!hitl && dec === "APPROVED" && (src === "fallback" || src === "rule")) {
      decisionFallback.hidden = false;
      decisionFallback.textContent = src === "fallback" ? "Auto-approved (Fallback)" : "Auto-approved (Rules)";
    } else if (src === "fallback") {
      decisionFallback.hidden = false;
      decisionFallback.textContent = "Fallback decision (AI unavailable)";
    } else {
      decisionFallback.hidden = true;
      decisionFallback.textContent = "";
    }
  }
}

function updateDecisionPanelFromSubmit(out) {
  const fraud = extractFraudFromSubmit(out);
  const dl = extractDlFromSubmit(out);
  const conf = out?.calibrated_confidence ?? out?.confidence_score;
  const imgSignals = out?.agent_outputs?.image?.features?.signals;
  const cnnLabel = out?.cnn_label ?? imgSignals?.cnn_label ?? out?.agent_outputs?.image?.features?.damage_type ?? "";
  const cnnFallbackUsed =
    Boolean(imgSignals && typeof imgSignals === "object" && imgSignals.cnn_used === false) ||
    String(imgSignals?.fallback_reason || "").toLowerCase() === "cnn_low_confidence";
  const pipeline = normalizePipelineMetadata(out);
  const modelUsed = inferModelUsed(out);
  const severity = out?.agent_outputs?.image?.features?.severity ?? out?.image_severity ?? imgSignals?.severity;
  const cnnConfidence = out?.cnn_confidence ?? imgSignals?.cnn_confidence ?? out?.agent_outputs?.image?.features?.confidence ?? null;
  const cnnSeverity = out?.cnn_severity ?? imgSignals?.cnn_severity ?? out?.agent_outputs?.image?.features?.severity ?? null;
  const fbReason = String(imgSignals?.fallback_reason || "").trim();
  const cnnUnavailable = fbReason.toLowerCase().includes("cnn_unavailable") || fbReason.toLowerCase().includes("cnn unavailable");
  const cnnUnknown = String(cnnLabel || "").trim().toLowerCase() === "unknown";
  const badgeText = cnnFallbackUsed ? "CNN Low Confidence → Fallback Used" : pipelineBadgeText(pipeline);
  const badgeTone =
    cnnFallbackUsed || pipeline.llmStatus === "failed" || String(out?.decision_source || "").toLowerCase() === "fallback"
      ? "warn"
      : pipeline.llmStatus === "used"
        ? "info"
        : "neutral";
  const hasImage = Boolean(out?.agent_outputs?.image || out?.image_base64 || out?.has_image);
  fillDecisionPanel({
    claimId: out?.claim_id,
    fraudScore: fraud,
    dlScore: dl,
    confidence: conf,
    decision: out?.decision,
    hitl: Boolean(out?.hitl_needed),
    decisionSource: out?.decision_source,
    cnnFallbackUsed,
    fraudSignal: out?.fraud_signal,
  });
  setAiAnalysisState({
    claimId: out?.claim_id,
    modelUsed,
    confidence: conf,
    severity,
    cnnLabel,
    cnnConfidence,
    cnnSeverity,
    badgeText,
    badgeTone,
    loading: false,
    error: cnnUnavailable ? "CNN unavailable — fallback used" : cnnUnknown ? "Fallback used" : "",
    hasImage,
    pipeline,
  });
}

function updateDecisionPanelFromClaim(it) {
  if (!it) {
    showDecisionEmpty();
    return;
  }
  const pipeline = normalizePipelineMetadata(it);
  const modelUsed = inferModelUsed(it);
  const conf = it.confidence;
  const severity = it.image_severity;
  const cnnLabel = it.image_damage_type || "";
  const badgeText = pipelineBadgeText(pipeline);
  const badgeTone = pipeline.llmStatus === "failed" ? "warn" : pipeline.llmStatus === "used" ? "info" : "neutral";
  fillDecisionPanel({
    claimId: it.claim_id,
    fraudScore: it.fraud_score,
    dlScore: null,
    confidence: it.confidence,
    decision: it.decision,
    hitl: Boolean(it.hitl_needed),
    decisionSource: it.decision_source,
    cnnFallbackUsed: false,
    fraudSignal: it.fraud_signal,
  });
  setAiAnalysisState({
    claimId: it.claim_id,
    modelUsed,
    confidence: conf,
    severity,
    cnnLabel,
    cnnConfidence: it.cnn_confidence ?? null,
    cnnSeverity: it.cnn_severity ?? it.image_severity ?? null,
    badgeText,
    badgeTone,
    loading: false,
    error: "",
    hasImage: Boolean(it.has_image),
    pipeline,
  });
}

async function postClaimWithFallback() {
  const claimId = String(document.getElementById("fieldClaimId")?.value || "").trim();
  const amtRaw = document.getElementById("fieldAmount")?.value;
  const limRaw = document.getElementById("fieldLimit")?.value;
  const claimAmount = Number(amtRaw);
  const policyLimit = Number(limRaw);
  const description = String(document.getElementById("fieldDesc")?.value || "").trim();
  const file = fieldFile?.files?.[0] || null;

  if (!claimId) throw new Error("Claim ID is required.");
  if (!Number.isFinite(claimAmount)) throw new Error("Claim amount must be a valid number.");
  if (!Number.isFinite(policyLimit)) throw new Error("Policy limit must be a valid number.");

  const fd = new FormData();
  fd.append("claim_id", claimId);
  fd.append("claim_amount", String(claimAmount));
  fd.append("policy_limit", String(policyLimit));
  if (description) fd.append("description", description);
  if (file) fd.append("file", file);

  let res = await fetch("/claims", { method: "POST", body: fd });
  if (!res.ok && file && (res.status === 422 || res.status === 415)) {
    const dataUrl = await readFileAsDataUrl(file);
    const jsonBody = {
      claim_id: claimId,
      claim_amount: claimAmount,
      policy_limit: policyLimit,
      ...(description ? { description } : {}),
      image_base64: dataUrl,
    };
    res = await fetch("/claims", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(jsonBody),
    });
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText}${text ? ` — ${text}` : ""}`);
  }
  return await res.json();
}

function setStatus(text, isError = false) {
  if (!statusEl) return;
  statusEl.textContent = text;
  statusEl.style.color = isError ? "#991b1b" : "";
}

function fmt3(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return Number(x).toFixed(3);
}

function fmt2(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return Number(x).toFixed(2);
}

function normalizeSeverity(s) {
  const u = String(s || "").trim().toUpperCase();
  if (!u) return "";
  if (u === "HIGH" || u === "SEVERE") return "HIGH";
  if (u === "MEDIUM" || u === "MODERATE") return "MEDIUM";
  if (u === "LOW" || u === "MINOR") return "LOW";
  return u;
}

function setSeverityPill(el, sevRaw) {
  if (!el) return;
  const sev = normalizeSeverity(sevRaw);
  el.classList.remove(
    "severity-pill--low",
    "severity-pill--medium",
    "severity-pill--high",
    "severity-pill--neutral",
  );
  if (sev === "LOW") el.classList.add("severity-pill--low");
  else if (sev === "MEDIUM") el.classList.add("severity-pill--medium");
  else if (sev === "HIGH") el.classList.add("severity-pill--high");
  else el.classList.add("severity-pill--neutral");
  el.textContent = sev ? sev.charAt(0) + sev.slice(1).toLowerCase() : "—";
}

function inferModelUsed(payload) {
  const ds = String(payload?.decision_source || "").trim().toLowerCase();
  if (ds === "cnn") return "CNN";
  if (ds === "llm") return "LLM";
  if (ds === "rule" || ds === "rules") return "Rules";
  if (ds === "fallback") return "Fallback";

  const sig =
    payload?.agent_outputs?.image?.features?.signals ||
    payload?.agent_outputs?.image?.signals ||
    payload?.image_signals ||
    null;
  if (sig && typeof sig === "object") {
    if (sig.cnn_used === true) return "CNN";
    if (sig.cnn_used === false) return "Fallback";
  }

  if (String(payload?.image_severity || "").trim() || String(payload?.image_damage_type || "").trim()) return "CNN";

  if (payload?.llm_used === true) return "LLM";
  if (payload?.llm_used === false) return "Rules";
  return "Fallback";
}

function normalizePipelineMetadata(payload) {
  const md = payload?.metadata && typeof payload.metadata === "object" ? payload.metadata : null;
  const decisionSource = String(md?.decision_source || payload?.decision_source || "").toLowerCase() || "fallback";
  const cnnUsedRaw =
    payload?.cnn_used ??
    md?.cnn_used ??
    (payload?.cnn_label && String(payload.cnn_label).toLowerCase() !== "unknown"
      ? true
      : false);
  const cnnUsed = Boolean(cnnUsedRaw);

  const llmUsedRaw =
    payload?.llm_used ??
    md?.llm_used ??
    (payload?.agent_outputs?.fraud?.source === "LLM" ? true : undefined);
  const llmUsed = Boolean(llmUsedRaw);

  // Rules are always part of the pipeline (policy + deterministic logic).
  const rulesUsed = true;

  const fallbackUsedRaw =
    payload?.fallback_used ??
    (decisionSource === "rule" || decisionSource === "fallback" ? true : false);
  const fallbackUsed = Boolean(fallbackUsedRaw);

  const llmStatus = llmUsed ? "used" : "skipped";
  const llmFailureReason = String(md?.llm_failure_reason || "");
  const contributorsRaw = Array.isArray(md?.contributors) ? md.contributors : [];
  const contributors = contributorsRaw.map((x) => String(x || "").toLowerCase()).filter(Boolean);
  const explanation = String(md?.explanation || "");
  return {
    decisionSource,
    llmUsed,
    cnnUsed,
    rulesUsed,
    fallbackUsed,
    llmStatus,
    llmFailureReason,
    contributors,
    explanation,
  };
}

function pipelineBadgeText(p) {
  const parts = [];
  if (p.cnnUsed) parts.push("CNN");
  if (p.rulesUsed) parts.push("Rules");
  const base = parts.length ? parts.join(" + ") : "Fallback";
  if (!p.llmUsed) return `${base} (LLM skipped)`;
  if (p.llmUsed) return parts.length ? `${base} + LLM` : "LLM";
  return base;
}

function setPipelineChecklist(p) {
  if (!aiPipelineChecklistEl) return;
  const llmLine = p.llmUsed ? "USED" : "SKIPPED";
  const rows = [
    { name: "CNN", ok: Boolean(p.cnnUsed), status: p.cnnUsed ? "USED" : "NOT USED" },
    { name: "LLM", ok: Boolean(p.llmUsed), status: llmLine, warn: !p.llmUsed },
    { name: "Rules", ok: true, status: p.fallbackUsed ? "USED (fallback)" : "USED" },
  ];
  aiPipelineChecklistEl.innerHTML = rows
    .map((r) => {
      const iconClass = r.ok ? "ok" : r.warn ? "no" : "muted";
      const icon = r.ok ? "✔" : r.warn ? "✖" : "—";
      const statusClass = r.ok ? "ok" : r.warn ? "warn" : "";
      return `<div class="ai-pipe-row">
        <div class="ai-pipe-left">
          <span class="ai-pipe-icon ${iconClass}" aria-hidden="true">${icon}</span>
          <span class="ai-pipe-name">${esc(r.name)}</span>
        </div>
        <span class="ai-pipe-status ${statusClass}">${esc(r.status)}</span>
      </div>`;
    })
    .join("");
}

function setAiAnalysisState({
  claimId,
  modelUsed,
  confidence,
  severity,
  cnnLabel,
  cnnConfidence,
  cnnSeverity,
  badgeText,
  badgeTone,
  loading,
  error,
  hasImage,
  pipeline,
}) {
  latestDecisionClaimId = claimId || null;

  if (aiModelConfidenceEl) {
    const c = confidence;
    aiModelConfidenceEl.textContent = c === null || c === undefined || Number.isNaN(Number(c)) ? "—" : fmt2(c);
  }
  if (aiImageSeverityEl) setSeverityPill(aiImageSeverityEl, severity);
  if (aiCnnLabelEl) {
    const s = String(cnnLabel || "").trim();
    aiCnnLabelEl.textContent = s ? s : "—";
  }
  if (aiCnnConfidenceEl) {
    const v = cnnConfidence;
    aiCnnConfidenceEl.textContent =
      v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : fmt3(Number(v));
  }
  if (aiCnnSeverityEl) {
    const s = String(cnnSeverity || "").trim();
    aiCnnSeverityEl.textContent = s ? s : "—";
  }
  if (aiPipelineSummaryEl) {
    const line = String(pipeline?.explanation || "").trim();
    aiPipelineSummaryEl.textContent = line;
    aiPipelineSummaryEl.hidden = !line;
  }
  if (pipeline) setPipelineChecklist(pipeline);

  if (aiAnalysisLoadingEl) aiAnalysisLoadingEl.hidden = !Boolean(loading);
  if (aiAnalysisErrorEl) {
    aiAnalysisErrorEl.hidden = !String(error || "").trim();
    aiAnalysisErrorEl.textContent = String(error || "");
  }

  if (aiAnalysisBadgeEl) {
    const t = String(badgeText || "").trim();
    if (!t) {
      aiAnalysisBadgeEl.hidden = true;
      aiAnalysisBadgeEl.textContent = "";
      aiAnalysisBadgeEl.className = "ai-badge";
    } else {
      aiAnalysisBadgeEl.hidden = false;
      aiAnalysisBadgeEl.textContent = t;
      aiAnalysisBadgeEl.className = `ai-badge${badgeTone ? ` ai-badge--${badgeTone}` : ""}`;
    }
  }

  if (viewAiEvidenceBtn) {
    viewAiEvidenceBtn.disabled = !Boolean(hasImage) || !Boolean(claimId);
    viewAiEvidenceBtn.setAttribute("data-claim", claimId ? String(claimId) : "");
  }
}

function claimDomId(claimId) {
  return `claim-${String(claimId || "").replace(/[^a-zA-Z0-9_-]/g, "_")}`;
}

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fraudClass(score) {
  const s = Number(score);
  if (Number.isNaN(s)) return "score-mid";
  if (s >= 0.66) return "score-high";
  if (s >= 0.33) return "score-mid";
  return "score-low";
}

function claimStatusHtml(it) {
  const reviewed = it.review_status || it.reviewed_action;
  if (reviewed) {
    return `<span class="status-chip status-chip--done">${esc(reviewed)}</span>`;
  }
  if (it.hitl_needed) {
    return `<span class="status-chip status-chip--pending">Review required</span>`;
  }
  const dec = String(it.decision || "").toUpperCase();
  const src = String(it.decision_source || "").toLowerCase();
  const sev = normalizeSeverity(it.image_severity);
  const hasCnnSignals = Boolean(String(it.image_severity || "").trim() || String(it.image_damage_type || "").trim());
  const isCnn = src === "cnn" || hasCnnSignals;
  if (dec === "APPROVED" && src === "fallback") {
    return `<span class="status-chip status-chip--auto">Auto-approved (Fallback)</span>`;
  }
  if (dec === "APPROVED" && (src === "rule" || it.llm_used === false)) {
    return `<span class="status-chip status-chip--auto">Auto-approved (Rules)</span>`;
  }
  if (isCnn) {
    const sevTag = sev ? ` · ${esc(sev.charAt(0) + sev.slice(1).toLowerCase())}` : "";
    return `<span class="status-chip status-chip--auto">CNN evaluated${sevTag}</span>`;
  }
  if (src === "llm" || it.llm_used === true) {
    return `<span class="status-chip status-chip--auto">AI evaluated (LLM)</span>`;
  }
  if (src === "fallback") {
    return `<span class="status-chip status-chip--auto">Fallback decision (AI unavailable)</span>`;
  }
  return `<span class="status-chip status-chip--auto">Auto triaged</span>`;
}

function decisionDisplayHtml(decRaw) {
  if (!decRaw) return `<span class="muted">—</span>`;
  const pill = decisionPillClass(decRaw);
  return `<span class="${pill}">${esc(String(decRaw).toUpperCase())}</span>`;
}

function renderExplanation(text) {
  const t = String(text || "").trim();
  if (!t) return '<div class="explanation muted">No explanation stored.</div>';
  try {
    const j = JSON.parse(t);
    if (j && typeof j === "object" && (j.summary || j.key_factors)) {
      const sum = j.summary ? `<div class="expl-summary">${esc(j.summary)}</div>` : "";
      const factors = Array.isArray(j.key_factors)
        ? `<ul class="expl-factors">${j.key_factors.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`
        : "";
      const cnnLabel = j.cnn_label ? `<div class="expl-kv"><strong>cnn_label</strong> ${esc(j.cnn_label)}</div>` : "";
      const cnnConf =
        j.cnn_confidence !== undefined && j.cnn_confidence !== null
          ? `<div class="expl-kv"><strong>cnn_confidence</strong> ${esc(fmt3(j.cnn_confidence))}</div>`
          : "";
      const fb = j.fallback_reason
        ? `<div class="expl-kv expl-kv--warn"><strong>fallback_reason</strong> ${esc(j.fallback_reason)}</div>`
        : "";
      const ref = j.similar_case_reference
        ? `<div class="muted expl-ref">${esc(j.similar_case_reference)}</div>`
        : "";
      const extras = cnnLabel || cnnConf || fb ? `<div class="expl-extras">${cnnLabel}${cnnConf}${fb}</div>` : "";
      return `<div class="explanation expl-structured">${sum}${factors}${extras}${ref}</div>`;
    }
  } catch {
    /* fall through */
  }
  return `<div class="explanation">${esc(t)}</div>`;
}

function renderEntities(obj) {
  if (!obj || typeof obj !== "object") return "";
  try {
    const s = JSON.stringify(obj).slice(0, 400);
    return `<div class="entities">${esc(s)}</div>`;
  } catch {
    return "";
  }
}

async function apiJson(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText}${text ? ` — ${text}` : ""}`);
  }
  return await res.json();
}

async function review(claimId, action) {
  setLoading(true);
  setStatus("");
  try {
    await apiJson(`/claims/${encodeURIComponent(claimId)}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    showToast(`Review saved: ${action}`, false);
    await refresh(false);
  } catch (err) {
    setStatus(String(err?.message || err), true);
    showToast(String(err?.message || err), true);
  } finally {
    setLoading(false);
  }
}

function computeInsights(s) {
  const insights = [];
  const total = Number(s?.total_claims) || 0;
  const dd = s?.decision_distribution || {};
  const decTotal = (Number(dd.APPROVED) || 0) + (Number(dd.REJECTED) || 0) + (Number(dd.INVESTIGATE) || 0);
  if (decTotal >= 3 && (Number(dd.REJECTED) || 0) / decTotal >= 0.35) {
    insights.push("High rejection trend detected");
  }
  const rd = s?.review_distribution || {};
  const reviewed = (Number(rd.APPROVED) || 0) + (Number(rd.REJECTED) || 0);
  if (total >= 3 && reviewed >= 5 && reviewed / total >= 0.2) {
    insights.push("Frequent human intervention required");
  }
  const topProd = s?.top_entities?.product?.[0];
  if (topProd && total >= 5 && Number(topProd.count) / total >= 0.4) {
    insights.push(`Pattern: High claims for ${String(topProd.value)}`);
  }
  return insights;
}

function renderInsightList(messages) {
  if (!messages.length) {
    intelInsightsEl.innerHTML = "";
    intelInsightsEl.hidden = true;
    return;
  }
  intelInsightsEl.innerHTML = messages.map((m) => `<li>${esc(m)}</li>`).join("");
  intelInsightsEl.hidden = false;
}

function maxCountInList(rows) {
  let m = 0;
  for (const r of rows) {
    const c = Number(r.count) || 0;
    if (c > m) m = c;
  }
  return m || 1;
}

function renderBarRows(rows, className) {
  const mx = maxCountInList(rows);
  return rows
    .map((r) => {
      const c = Number(r.count) || 0;
      const pct = Math.round((c / mx) * 100);
      return `<div class="intel-bar-row">
        <div class="intel-bar-meta"><span>${esc(r.value)}</span><span>${c}</span></div>
        <div class="intel-bar-track"><div class="intel-bar-fill ${className}" style="width:${pct}%"></div></div>
      </div>`;
    })
    .join("");
}

function showIntelError(msg) {
  intelErrorEl.textContent = msg || "";
  intelErrorEl.style.display = msg ? "block" : "none";
}

function fraudAlertSeverityClass(sev) {
  const u = String(sev || "").toUpperCase();
  if (u === "HIGH") return "severity-high";
  if (u === "MEDIUM") return "severity-medium";
  return "severity-low";
}

function showFraudAlertsError(msg) {
  fraudAlertsErrorEl.textContent = msg || "";
  fraudAlertsErrorEl.style.display = msg ? "block" : "none";
}

function showLeaderboardError(msg) {
  leaderboardErrorEl.textContent = msg || "";
  leaderboardErrorEl.style.display = msg ? "block" : "none";
}

function riskLevelClass(level) {
  const u = String(level || "").toUpperCase();
  if (u === "HIGH") return "high";
  if (u === "MEDIUM") return "medium";
  if (u === "LOW") return "low";
  return "";
}

function caseStatusBadgeClass(status) {
  const u = String(status || "").toUpperCase();
  if (u === "NEW") return "new";
  if (u === "ASSIGNED") return "assigned";
  if (u === "IN_PROGRESS") return "in_progress";
  if (u === "RESOLVED") return "resolved";
  return "new";
}

function caseMgmtQueryUrl() {
  const params = new URLSearchParams();
  if (caseMgmtUnassignedOnlyEl?.checked) params.set("unassigned_only", "true");
  const myId = String(caseMgmtMyIdEl?.value || "").trim();
  if (caseMgmtMyOnlyEl?.checked && myId) params.set("assigned_to", myId);
  const qs = params.toString();
  return qs ? `/cases?${qs}` : "/cases";
}

function showCaseMgmtError(msg) {
  if (!caseMgmtErrorEl) return;
  caseMgmtErrorEl.textContent = msg || "";
  caseMgmtErrorEl.style.display = msg ? "block" : "none";
}

function renderCaseMgmtRows(payload) {
  if (!caseMgmtRowsEl) return;
  caseMgmtRowsEl.innerHTML = "";
  const list = payload && Array.isArray(payload.cases) ? payload.cases : [];
  if (!list.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="6" class="muted">No cases match the current filters.</td>`;
    caseMgmtRowsEl.appendChild(tr);
    return;
  }
  for (const it of list) {
    const cid = String(it.claim_id || "");
    const cs = String(it.case_status || "NEW").toUpperCase();
    const badge = esc(cs.replace(/_/g, " "));
    const assigned = esc(it.assigned_to || "—");
    const st = caseStatusBadgeClass(cs);
    let actionsHtml = `<span class="muted">—</span>`;
    if (cs === "NEW") {
      actionsHtml = `<div class="case-mgmt-actions">
        <input type="text" class="case-assign-input" placeholder="Investigator" maxlength="200" />
        <button type="button" class="btn btn--primary btn--sm" data-case-assign="${esc(cid)}">Assign</button>
      </div>`;
    } else if (cs === "ASSIGNED") {
      actionsHtml = `<div class="case-mgmt-actions">
        <button type="button" class="btn btn--secondary btn--sm" data-case-status="${esc(
          cid,
        )}" data-next-status="IN_PROGRESS">Start</button>
      </div>`;
    } else if (cs === "IN_PROGRESS") {
      actionsHtml = `<div class="case-mgmt-actions">
        <button type="button" class="btn btn--primary btn--sm" data-case-status="${esc(
          cid,
        )}" data-next-status="RESOLVED">Resolve</button>
      </div>`;
    }
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="claim-id">${esc(cid)}</span></td>
      <td><span class="case-badge ${st}">${badge}</span></td>
      <td>${assigned}</td>
      <td>${decisionDisplayHtml(it.decision)}</td>
      <td><span class="score-pill ${fraudClass(it.fraud_score)}">${fmt3(it.fraud_score)}</span></td>
      <td>${actionsHtml}</td>
    `;
    caseMgmtRowsEl.appendChild(tr);
  }
}

async function loadCaseMgmt() {
  if (!caseMgmtRowsEl) return;
  showCaseMgmtError("");
  try {
    const data = await apiJson(caseMgmtQueryUrl());
    renderCaseMgmtRows(data);
  } catch (err) {
    showCaseMgmtError(String(err?.message || err));
    caseMgmtRowsEl.innerHTML = "";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="6" class="muted">Could not load cases.</td>`;
    caseMgmtRowsEl.appendChild(tr);
  }
}

async function caseAssign(claimId, assignedTo) {
  setLoading(true);
  setStatus("");
  try {
    await apiJson(`/cases/${encodeURIComponent(claimId)}/assign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ assigned_to: assignedTo }),
    });
    showToast(`Assigned ${claimId} to ${assignedTo}`, false);
    await refresh(false);
  } catch (err) {
    setStatus(String(err?.message || err), true);
    showToast(String(err?.message || err), true);
  } finally {
    setLoading(false);
  }
}

async function caseStatusUpdate(claimId, caseStatus) {
  setLoading(true);
  setStatus("");
  try {
    await apiJson(`/cases/${encodeURIComponent(claimId)}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ case_status: caseStatus }),
    });
    showToast(`Case ${claimId} → ${caseStatus}`, false);
    await refresh(false);
  } catch (err) {
    setStatus(String(err?.message || err), true);
    showToast(String(err?.message || err), true);
  } finally {
    setLoading(false);
  }
}

function renderLeaderboardRows(items) {
  leaderboardRowsEl.innerHTML = "";
  const list = Array.isArray(items) ? items : [];
  if (!list.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="8" class="muted">No claims match the current filters.</td>`;
    leaderboardRowsEl.appendChild(tr);
    return;
  }
  list.forEach((it, idx) => {
    const rank = idx + 1;
    const rl = String(it.risk_level || "—");
    const rlClass = riskLevelClass(rl);
    const highBadge =
      rl.toUpperCase() === "HIGH"
        ? `<span class="risk-badge" title="High priority">High</span>`
        : "";
    const reviewed = String(it.review_status || "").trim();
    const reviewCell = reviewed
      ? `<span class="review-status done"><strong>${esc(reviewed)}</strong></span>`
      : `<span class="muted">—</span>`;
    const anchor = claimDomId(it.claim_id);
    const actions = reviewed
      ? `<div class="lb-actions"><a href="#${esc(anchor)}">Open claim</a></div>`
      : `<div class="lb-actions">
           <a href="#${esc(anchor)}">Open claim</a>
           <button type="button" class="btn btn--secondary btn--sm" data-action="APPROVED" data-claim="${esc(
             it.claim_id,
           )}">Approve</button>
           <button type="button" class="btn btn--secondary btn--sm" data-action="REJECTED" data-claim="${esc(
             it.claim_id,
           )}">Reject</button>
         </div>`;
    const tr = document.createElement("tr");
    if (rank <= 3) tr.classList.add("lb-top3");
    tr.innerHTML = `
      <td>${rank}</td>
      <td><span class="claim-id">${esc(it.claim_id)}</span></td>
      <td><span class="score-pill ${fraudClass(it.fraud_score)}">${fmt3(it.fraud_score)}</span></td>
      <td><span class="risk-level ${rlClass}">${esc(rl)}</span>${highBadge}</td>
      <td>${decisionDisplayHtml(it.decision)}</td>
      <td>${fmt3(it.confidence)}</td>
      <td>${reviewCell}</td>
      <td>${actions}</td>
    `;
    leaderboardRowsEl.appendChild(tr);
  });

  leaderboardRowsEl.querySelectorAll("button[data-action][data-claim]").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const action = e.currentTarget.getAttribute("data-action");
      const claimId = e.currentTarget.getAttribute("data-claim");
      if (!action || !claimId) return;
      await review(claimId, action);
    });
  });
}

async function loadLeaderboard() {
  showLeaderboardError("");
  const want = Math.max(1, Math.min(100, Number(leaderboardLimitEl.value) || 10));
  leaderboardLimitEl.value = String(want);
  const highOnly = leaderboardHighOnlyEl.checked;
  const unrevOnly = leaderboardUnreviewedOnlyEl.checked;
  const needWide = highOnly || unrevOnly;
  const fetchLimit = needWide ? Math.min(200, Math.max(want, 64)) : want;
  try {
    const data = await apiJson(`/analytics/leaderboard?limit=${fetchLimit}`);
    let rows = Array.isArray(data?.top_risky_claims) ? data.top_risky_claims : [];
    if (highOnly) rows = rows.filter((r) => String(r.risk_level || "").toUpperCase() === "HIGH");
    if (unrevOnly) rows = rows.filter((r) => !String(r.review_status || "").trim());
    renderLeaderboardRows(rows.slice(0, want));
  } catch (err) {
    showLeaderboardError(String(err?.message || err));
    leaderboardRowsEl.innerHTML = "";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="8" class="muted">Could not load leaderboard.</td>`;
    leaderboardRowsEl.appendChild(tr);
  }
}

function renderFraudAlerts(payload) {
  showFraudAlertsError("");
  const raw = payload && typeof payload === "object" ? payload.alerts : null;
  const alerts = Array.isArray(raw) ? raw : [];
  if (!alerts.length) {
    fraudAlertsListEl.innerHTML = "";
    fraudAlertsListEl.hidden = true;
    fraudAlertsEmptyEl.style.display = "block";
    return;
  }
  fraudAlertsEmptyEl.style.display = "none";
  fraudAlertsListEl.hidden = false;
  fraudAlertsListEl.innerHTML = alerts
    .map((a) => {
      const sev = fraudAlertSeverityClass(a.severity);
      const msg = esc(a.message);
      const typ = esc(a.type);
      const cnt = Number(a.count);
      const countStr = Number.isFinite(cnt) ? String(cnt) : "—";
      return `<li class="fraud-alert-item ${sev}">
        <div>${msg}</div>
        <div class="fraud-alert-meta">
          <span><strong>Type</strong> ${typ}</span>
          <span><strong>Count</strong> ${esc(countStr)}</span>
          <span><strong>Severity</strong> ${esc(a.severity || "—")}</span>
        </div>
      </li>`;
    })
    .join("");
}

function renderAnalytics(s) {
  showIntelError("");
  if (!s || typeof s !== "object") {
    intelCardsEl.innerHTML = "";
    if (intelReviewStripEl) intelReviewStripEl.textContent = "";
    intelFraudScoresEl.innerHTML = "";
    intelProductsEl.innerHTML = "";
    intelBandsEl.innerHTML = "";
    intelTimelineEl.style.display = "none";
    renderInsightList([]);
    return;
  }

  const total = Number(s.total_claims) || 0;
  const dd = s.decision_distribution || {};
  const ap = Number(dd.APPROVED) || 0;
  const rj = Number(dd.REJECTED) || 0;
  const inv = Number(dd.INVESTIGATE) || 0;
  const llmPct = Number(s.llm_usage_percentage);
  const llmUsedPct = Number.isFinite(llmPct) ? llmPct : 0;
  const rulePct = total > 0 ? Math.max(0, Math.min(100, 100 - llmUsedPct)) : 0;

  intelCardsEl.innerHTML = `
    <div class="intel-card"><div class="label">Total claims</div><div class="value">${total}</div></div>
    <div class="intel-card approved"><div class="label">Approved</div><div class="value">${ap}</div></div>
    <div class="intel-card rejected"><div class="label">Rejected</div><div class="value">${rj}</div></div>
    <div class="intel-card investigate"><div class="label">Investigate</div><div class="value">${inv}</div></div>
    <div class="intel-card"><div class="label">AI Usage</div><div class="value">${llmUsedPct.toFixed(
      1,
    )}%</div><div class="muted" style="font-size:0.75rem;margin-top:4px;">Rule-based ${rulePct.toFixed(
      1,
    )}%</div></div>
  `;

  const rd = s.review_distribution || {};
  const rAp = Number(rd.APPROVED) || 0;
  const rRj = Number(rd.REJECTED) || 0;
  const rPd = Number(rd.PENDING) || 0;
  if (intelReviewStripEl) {
    intelReviewStripEl.textContent = `Human review: ${rAp} approved · ${rRj} rejected · ${rPd} pending`;
  }

  const fs = s.fraud_score_stats || {};
  intelFraudScoresEl.innerHTML = `
    <h3>Fraud score</h3>
    <div class="intel-stat-row"><span>Average</span><strong>${fmt3(fs.avg)}</strong></div>
    <div class="intel-stat-row"><span>Minimum</span><strong>${fmt3(fs.min)}</strong></div>
    <div class="intel-stat-row"><span>Maximum</span><strong>${fmt3(fs.max)}</strong></div>
  `;

  const products = Array.isArray(s.top_entities?.product) ? s.top_entities.product : [];
  const bands = Array.isArray(s.top_entities?.amount_band) ? s.top_entities.amount_band : [];

  intelProductsEl.innerHTML = products.length
    ? `<h3>Top products</h3>${renderBarRows(products, "")}`
    : `<h3>Top products</h3><p class="intel-muted">No product entity data in stored claims.</p>`;

  intelBandsEl.innerHTML = bands.length
    ? `<h3>Top amount bands</h3>${renderBarRows(bands, "band")}`
    : `<h3>Top amount bands</h3><p class="intel-muted">No amount band entity data in stored claims.</p>`;

  const overTime = Array.isArray(s.claims_over_time) ? s.claims_over_time : [];
  if (overTime.length) {
    intelTimelineEl.style.display = "block";
    const mapped = overTime.map((x) => ({ value: x.date, count: x.count }));
    intelTimelineEl.innerHTML = `<h3>Claims over time</h3>${renderBarRows(mapped, "intel-bar-timeline")}`;
  } else {
    intelTimelineEl.style.display = "none";
    intelTimelineEl.innerHTML = "";
  }

  renderInsightList(computeInsights(s));
}

function render(items) {
  rowsEl.innerHTML = "";
  for (const it of items) {
    const hitlNeeded = Boolean(it.hitl_needed);
    const reviewed = it.review_status || it.reviewed_action;
    const canAct = hitlNeeded && !reviewed;
    const hasImg = Boolean(it.has_image);

    const preview = (it.claim_description || "").slice(0, 120);
    const expl = it.explanation || "";
    const decRaw = it.decision || "";

    const viewBtn = hasImg
      ? `<button type="button" class="btn btn--secondary btn--sm" data-view-image="${esc(it.claim_id)}">View image</button>`
      : `<button type="button" class="btn btn--secondary btn--sm" disabled title="No image on file">View image</button>`;

    const evidenceBtn = hasImg
      ? `<button type="button" class="btn btn--secondary btn--sm" data-view-evidence="${esc(
          it.claim_id,
        )}">View AI Evidence</button>`
      : "";

    const actionsHtml = canAct
      ? `<div class="cell-actions">
           ${viewBtn}
           ${evidenceBtn}
           <button type="button" class="btn btn--primary btn--sm" data-action="APPROVED" data-claim="${esc(
             it.claim_id,
           )}">Approve</button>
           <button type="button" class="btn btn--secondary btn--sm" data-action="REJECTED" data-claim="${esc(
             it.claim_id,
           )}">Reject</button>
         </div>`
      : `<div class="cell-actions">${viewBtn}${evidenceBtn ? ` ${evidenceBtn}` : ""}</div>`;

    const tr = document.createElement("tr");
    tr.id = claimDomId(it.claim_id);
    const modelUsed = inferModelUsed(it);
    const sev = normalizeSeverity(it.image_severity);
    const sevLabel = sev ? sev.charAt(0) + sev.slice(1).toLowerCase() : "—";
    const sevClass =
      sev === "LOW"
        ? "severity-pill--low"
        : sev === "MEDIUM"
          ? "severity-pill--medium"
          : sev === "HIGH"
            ? "severity-pill--high"
            : "severity-pill--neutral";
    const mismatchBadge =
      String(it.fraud_signal || "").trim() === "image_text_mismatch"
        ? `<span class="inline-badge inline-badge--warn" title="Fraud signal: description-image mismatch">⚠️ Inconsistent claim</span>`
        : "";
    tr.innerHTML = `
      <td>
        <div class="claim-id">${esc(it.claim_id)}</div>
        ${mismatchBadge}
        ${
          preview
            ? `<div class="claim-preview-line" title="${esc(it.claim_description || "")}">${esc(preview)}${
                (it.claim_description || "").length > 120 ? "…" : ""
              }</div>`
            : ""
        }
        <details class="explain-details">
          <summary>Explanation &amp; signals</summary>
          <div class="ai-inline">
            <div class="ai-inline__head">
              <span class="ai-inline__title"><span aria-hidden="true">🔍</span> AI Analysis</span>
              <span class="ai-inline__badge">${esc(modelUsed)}</span>
            </div>
            <div class="ai-inline__grid">
              <div class="ai-inline__kv"><span class="muted">confidence</span><strong>${fmt3(it.confidence)}</strong></div>
              <div class="ai-inline__kv"><span class="muted">severity</span><strong class="severity-pill ${sevClass}">${esc(sevLabel)}</strong></div>
              <div class="ai-inline__kv"><span class="muted">cnn_label</span><strong>${esc(it.image_damage_type || "—")}</strong></div>
            </div>
          </div>
          ${renderExplanation(expl)}
          ${renderEntities(it.entities)}
        </details>
      </td>
      <td><span class="score-pill ${fraudClass(it.fraud_score)}">${fmt3(it.fraud_score)}</span></td>
      <td>${decisionDisplayHtml(decRaw)}</td>
      <td><span class="muted">${fmt3(it.confidence)}</span></td>
      <td>${claimStatusHtml(it)}</td>
      <td>${actionsHtml}</td>
    `;
    rowsEl.appendChild(tr);
  }

  rowsEl.querySelectorAll("button[data-action][data-claim]").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const action = e.currentTarget.getAttribute("data-action");
      const claimId = e.currentTarget.getAttribute("data-claim");
      if (!action || !claimId) return;
      await review(claimId, action);
    });
  });

  rowsEl.querySelectorAll("button[data-view-image]").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const claimId = e.currentTarget.getAttribute("data-view-image");
      if (!claimId || e.currentTarget.disabled) return;
      await openImageModal(claimId);
    });
  });

  rowsEl.querySelectorAll("button[data-view-evidence]").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const claimId = e.currentTarget.getAttribute("data-view-evidence");
      if (!claimId || e.currentTarget.disabled) return;
      await openImageModal(claimId);
    });
  });
}

async function refresh(showLoading = true) {
  if (showLoading) setLoading(true);
  setStatus("");
  let health = "ok";
  try {
    const [items, summaryResult, anomaliesResult] = await Promise.all([
      apiJson("/claims"),
      apiJson("/analytics/summary").catch((e) => ({ __error: e })),
      apiJson("/analytics/anomalies").catch((e) => ({ __error: e })),
      loadLeaderboard(),
      loadCaseMgmt(),
    ]);
    render(items);
    setStatus(`Loaded ${items.length} claim(s).`);
    if (items.length) {
      updateDecisionPanelFromClaim(items[0]);
    } else {
      showDecisionEmpty();
    }

    if (summaryResult && summaryResult.__error) {
      showIntelError("Could not load analytics summary.");
      renderAnalytics(null);
      health = "degraded";
    } else {
      renderAnalytics(summaryResult);
    }
    if (anomaliesResult && anomaliesResult.__error) {
      showFraudAlertsError("Could not load fraud alerts.");
      fraudAlertsListEl.innerHTML = "";
      fraudAlertsListEl.hidden = true;
      fraudAlertsEmptyEl.style.display = "none";
      health = "degraded";
    } else {
      renderFraudAlerts(anomaliesResult);
    }
    setHealth(health);
  } catch (err) {
    setStatus(String(err?.message || err), true);
    showToast(String(err?.message || err), true);
    showIntelError("Could not load analytics summary.");
    renderAnalytics(null);
    showFraudAlertsError("Could not load fraud alerts.");
    fraudAlertsListEl.innerHTML = "";
    fraudAlertsListEl.hidden = true;
    fraudAlertsEmptyEl.style.display = "none";
    showCaseMgmtError("Could not load case list.");
    if (caseMgmtRowsEl) {
      caseMgmtRowsEl.innerHTML = "";
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="6" class="muted">Could not load cases.</td>`;
      caseMgmtRowsEl.appendChild(tr);
    }
    showDecisionEmpty();
    setHealth("error");
  } finally {
    if (showLoading) setLoading(false);
  }
}

refreshBtn.addEventListener("click", () => refresh(true));
leaderboardLimitEl.addEventListener("change", () => loadLeaderboard());
leaderboardHighOnlyEl.addEventListener("change", () => loadLeaderboard());
leaderboardUnreviewedOnlyEl.addEventListener("change", () => loadLeaderboard());

document.querySelectorAll(".tab-bar__btn[data-tab]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.getAttribute("data-tab");
    document.querySelectorAll(".tab-bar__btn[data-tab]").forEach((b) => b.classList.toggle("is-active", b === btn));
    const dash = document.getElementById("panel-dashboard");
    const ops = document.getElementById("panel-operations");
    if (!dash || !ops) return;
    if (tab === "operations") {
      dash.hidden = true;
      ops.hidden = false;
    } else {
      ops.hidden = true;
      dash.hidden = false;
    }
  });
});

if (caseMgmtRowsEl) {
  caseMgmtRowsEl.addEventListener("click", async (e) => {
    const assignBtn = e.target.closest("button[data-case-assign]");
    if (assignBtn && caseMgmtRowsEl.contains(assignBtn)) {
      e.preventDefault();
      const claimId = assignBtn.getAttribute("data-case-assign");
      const tr = assignBtn.closest("tr");
      const inp = tr?.querySelector(".case-assign-input");
      const name = String(inp?.value || "").trim();
      if (!claimId) return;
      if (!name) {
        showToast("Enter an investigator name or ID.", true);
        return;
      }
      await caseAssign(claimId, name);
      return;
    }
    const stBtn = e.target.closest("button[data-case-status]");
    if (stBtn && caseMgmtRowsEl.contains(stBtn)) {
      e.preventDefault();
      const claimId = stBtn.getAttribute("data-case-status");
      const next = stBtn.getAttribute("data-next-status");
      if (!claimId || !next) return;
      await caseStatusUpdate(claimId, next);
    }
  });
}
caseMgmtUnassignedOnlyEl?.addEventListener("change", () => loadCaseMgmt());
caseMgmtMyOnlyEl?.addEventListener("change", () => loadCaseMgmt());
caseMgmtMyIdEl?.addEventListener("change", () => {
  if (caseMgmtMyOnlyEl?.checked) loadCaseMgmt();
});

function wireFilePreview() {
  if (!fieldFile || !filePreview || !filePreviewImg) return;
  fieldFile.addEventListener("change", () => {
    const f = fieldFile.files?.[0];
    if (!f) {
      clearFilePreview();
      return;
    }
    setUploadFilename(f.name);
    const fr = new FileReader();
    fr.onload = () => {
      filePreviewImg.src = String(fr.result || "");
      filePreview.classList.remove("hidden");
    };
    fr.onerror = () => {
      showToast("Could not read the selected image.", true);
      clearFilePreview();
    };
    fr.readAsDataURL(f);
  });
}

wireFilePreview();

if (viewAiEvidenceBtn) {
  viewAiEvidenceBtn.addEventListener("click", async (e) => {
    const claimId = e.currentTarget.getAttribute("data-claim") || latestDecisionClaimId;
    if (!claimId || e.currentTarget.disabled) return;
    await openImageModal(claimId);
  });
}

if (dropzone && fieldFile) {
  ["dragenter", "dragover"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("is-dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("is-dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (!file || !file.type.startsWith("image/")) {
      showToast("Please drop a PNG or JPEG image.", true);
      return;
    }
    assignImageFile(file);
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fieldFile.click();
    }
  });
}

if (claimForm) {
  claimForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (!claimForm.reportValidity()) return;
    setFormSubmitting(true);
    setStatus("");
    let submittedClaimId = latestDecisionClaimId;
    setAiAnalysisState({
      claimId: latestDecisionClaimId,
      modelUsed: "CNN",
      confidence: null,
      severity: "",
      cnnLabel: "",
      cnnConfidence: null,
      cnnSeverity: "",
      badgeText: "CNN Evaluated",
      badgeTone: "info",
      loading: true,
      error: "",
      hasImage: true,
      pipeline: null,
    });
    try {
      const out = await postClaimWithFallback();
      console.log("API response:", out);
      submittedClaimId = out?.claim_id || submittedClaimId;
      const dec = out.decision || "";
      showToast(`Claim ${out.claim_id} submitted · decision: ${dec}`, false);
      const sig = out?.agent_outputs?.image?.features?.signals;
      const cnnFb =
        (sig && typeof sig === "object" && sig.cnn_used === false) ||
        String(sig?.fallback_reason || "").toLowerCase() === "cnn_low_confidence";
      if (cnnFb) {
        showToast("CNN low confidence — fallback used", false);
      }
      if (isDelayedOrUnavailableAnalysis(out)) {
        showToast("AI analysis delayed, escalated for human review", false);
        setHealth("degraded");
      }

      // Update Decision Output immediately (don't wait on refresh()).
      updateDecisionPanelFromSubmit(out);

      claimForm.reset();
      clearFilePreview();
      if (dropzone) dropzone.classList.remove("is-dragover");
      refresh(false).catch((err) => {
        setStatus(String(err?.message || err), true);
        showToast(String(err?.message || err), true);
      });
    } catch (err) {
      let msg = String(err?.message || err);
      if (msg.toLowerCase().includes("timeout")) {
        msg = "AI analysis delayed, escalated for human review";
      }
      setStatus(msg, true);
      showToast(msg, true);
      setAiAnalysisState({
        claimId: submittedClaimId || latestDecisionClaimId,
        modelUsed: "Fallback",
        confidence: null,
        severity: "",
        cnnLabel: "",
        cnnConfidence: null,
        cnnSeverity: "",
        badgeText: "Fallback Decision",
        badgeTone: "warn",
        loading: false,
        error: msg.toLowerCase().includes("cnn") ? "CNN unavailable — fallback used" : "",
        hasImage: Boolean(fieldFile?.files?.[0]),
        pipeline: null,
      });
    } finally {
      setFormSubmitting(false);
      // Hard guarantee: never leave the AI analysis loader stuck on.
      if (aiAnalysisLoadingEl && !aiAnalysisLoadingEl.hidden) {
        setAiAnalysisState({
          claimId: submittedClaimId || latestDecisionClaimId,
          modelUsed: "—",
          confidence: null,
          severity: "",
          cnnLabel: aiCnnLabelEl?.textContent || "",
          cnnConfidence: aiCnnConfidenceEl?.textContent || "",
          cnnSeverity: aiCnnSeverityEl?.textContent || "",
          badgeText: aiAnalysisBadgeEl && !aiAnalysisBadgeEl.hidden ? aiAnalysisBadgeEl.textContent : "",
          badgeTone: "",
          loading: false,
          error: "",
          hasImage: true,
          pipeline: null,
        });
      }
    }
  });
}

imageModalClose?.addEventListener("click", () => closeImageModal());
imageModal?.addEventListener("click", (e) => {
  if (e.target === imageModal) closeImageModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeImageModal();
});

refresh(true);
