/* Insurance AI Decision Platform — vanilla JS, no frameworks */

const statusEl = document.getElementById("status");
const rowsEl = document.getElementById("rows");
const refreshBtn = document.getElementById("refreshBtn");
const loadingEl = document.getElementById("loading");
const toastEl = document.getElementById("toast");
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
  loadingEl.classList.toggle("show", Boolean(on));
  refreshBtn.disabled = Boolean(on);
}

function showToast(message, isError = false) {
  toastEl.textContent = message;
  toastEl.classList.add("show");
  toastEl.classList.toggle("error", Boolean(isError));
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toastEl.classList.remove("show"), 4500);
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? "#991b1b" : "var(--muted)";
}

function fmt3(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return Number(x).toFixed(3);
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

function decisionClass(d) {
  const u = String(d || "").toUpperCase();
  if (u === "APPROVED") return "approved";
  if (u === "REJECTED") return "rejected";
  if (u === "INVESTIGATE") return "investigate";
  return "";
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
      const ref = j.similar_case_reference
        ? `<div class="muted expl-ref">${esc(j.similar_case_reference)}</div>`
        : "";
      return `<div class="explanation expl-structured">${sum}${factors}${ref}</div>`;
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
        <button type="button" class="primary" data-case-assign="${esc(cid)}">Assign</button>
      </div>`;
    } else if (cs === "ASSIGNED") {
      actionsHtml = `<div class="case-mgmt-actions">
        <button type="button" data-case-status="${esc(cid)}" data-next-status="IN_PROGRESS">Start</button>
      </div>`;
    } else if (cs === "IN_PROGRESS") {
      actionsHtml = `<div class="case-mgmt-actions">
        <button type="button" class="primary" data-case-status="${esc(cid)}" data-next-status="RESOLVED">Resolve</button>
      </div>`;
    }
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="claim-id">${esc(cid)}</span></td>
      <td><span class="case-badge ${st}">${badge}</span></td>
      <td>${assigned}</td>
      <td><span class="decision ${decisionClass(it.decision)}">${esc(it.decision || "—")}</span></td>
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
        ? `<span class="risk-badge" title="High priority">⚠ High</span>`
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
           <button type="button" data-action="APPROVED" data-claim="${esc(it.claim_id)}">Approve</button>
           <button type="button" data-action="REJECTED" data-claim="${esc(it.claim_id)}">Reject</button>
         </div>`;
    const tr = document.createElement("tr");
    if (rank <= 3) tr.classList.add("lb-top3");
    tr.innerHTML = `
      <td>${rank}</td>
      <td><span class="claim-id">${esc(it.claim_id)}</span></td>
      <td><span class="score-pill ${fraudClass(it.fraud_score)}">${fmt3(it.fraud_score)}</span></td>
      <td><span class="risk-level ${rlClass}">${esc(rl)}</span>${highBadge}</td>
      <td><span class="decision ${decisionClass(it.decision)}">${esc(it.decision || "—")}</span></td>
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

  intelCardsEl.innerHTML = `
    <div class="intel-card"><div class="label">Total claims</div><div class="value">${total}</div></div>
    <div class="intel-card approved"><div class="label">Approved</div><div class="value">${ap}</div></div>
    <div class="intel-card rejected"><div class="label">Rejected</div><div class="value">${rj}</div></div>
    <div class="intel-card investigate"><div class="label">Investigate</div><div class="value">${inv}</div></div>
  `;

  const rd = s.review_distribution || {};
  const rAp = Number(rd.APPROVED) || 0;
  const rRj = Number(rd.REJECTED) || 0;
  const rPd = Number(rd.PENDING) || 0;
  if (intelReviewStripEl) {
    intelReviewStripEl.textContent =
      `Human review: ${rAp} approved · ${rRj} rejected · ${rPd} pending`;
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
    const reviewStatusHtml = reviewed
      ? `<div class="review-status done"><strong>${esc(reviewed)}</strong>`
        + (it.reviewed_at ? `<div class="muted">${esc(it.reviewed_at)}</div>` : "")
        + (it.reviewed_by ? `<div class="muted">by ${esc(it.reviewed_by)}</div>` : "")
        + `</div>`
      : `<span class="muted">—</span>`;

    const hitlPill = hitlNeeded
      ? `<span class="pill hitl">Needs review</span>`
      : `<span class="pill ok">OK</span>`;

    const canAct = hitlNeeded && !reviewed;
    const actionsHtml = canAct
      ? `<div class="cell-actions">
           <button type="button" data-action="APPROVED" data-claim="${esc(it.claim_id)}">Approve</button>
           <button type="button" data-action="REJECTED" data-claim="${esc(it.claim_id)}">Reject</button>
         </div>`
      : `<span class="muted">—</span>`;

    const preview = (it.claim_description || "").slice(0, 160);
    const expl = it.explanation || "";
    const tr = document.createElement("tr");
    tr.id = claimDomId(it.claim_id);
    tr.innerHTML = `
      <td>
        <div class="claim-id">${esc(it.claim_id)}</div>
        <div class="preview">${esc(preview)}${(it.claim_description || "").length > 160 ? "…" : ""}</div>
        <details class="explain-details">
          <summary>Why flagged / full explanation</summary>
          ${renderExplanation(expl)}
          ${renderEntities(it.entities)}
        </details>
      </td>
      <td>
        <span class="score-pill ${fraudClass(it.fraud_score)}">${fmt3(it.fraud_score)}</span>
        <div class="muted" style="margin-top:6px">${hitlPill}</div>
      </td>
      <td>
        <span class="decision ${decisionClass(it.decision)}">${esc(it.decision || "—")}</span>
      </td>
      <td>${fmt3(it.confidence)}</td>
      <td>${reviewStatusHtml}</td>
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
}

async function refresh(showLoading = true) {
  if (showLoading) setLoading(true);
  setStatus("");
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
    if (summaryResult && summaryResult.__error) {
      showIntelError("Could not load analytics summary.");
      renderAnalytics(null);
    } else {
      renderAnalytics(summaryResult);
    }
    if (anomaliesResult && anomaliesResult.__error) {
      showFraudAlertsError("Could not load fraud alerts.");
      fraudAlertsListEl.innerHTML = "";
      fraudAlertsListEl.hidden = true;
      fraudAlertsEmptyEl.style.display = "none";
    } else {
      renderFraudAlerts(anomaliesResult);
    }
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
  } finally {
    if (showLoading) setLoading(false);
  }
}

refreshBtn.addEventListener("click", () => refresh(true));
leaderboardLimitEl.addEventListener("change", () => loadLeaderboard());
leaderboardHighOnlyEl.addEventListener("change", () => loadLeaderboard());
leaderboardUnreviewedOnlyEl.addEventListener("change", () => loadLeaderboard());

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

refresh(true);
