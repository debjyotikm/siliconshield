"use strict";
const $ = (s, root = document) => root.querySelector(s);
const esc = (value) =>
  String(value ?? "")
    // Strip invisible bidi controls: receipts preserve backend bytes
    // exactly, but rendered text must not reorder visually.
    .replace(/[\u202A-\u202E\u2066-\u2069\u200E\u200F\u061C]/g, "")
    .replace(
      /[&<>"']/g,
      (c) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
          c
        ],
    );
const labels = {
  satisfies_requirement: "Satisfies requirement",
  violates_requirement: "Violates requirement",
  insufficient_context: "Insufficient context",
  unresolved: "Unresolved",
};
const shortLabels = {
  satisfies_requirement: "Satisfies",
  violates_requirement: "Violates",
  insufficient_context: "Insufficient context",
};
const paths = {
  chip: "M9 3v2m6-2v2M9 19v2m6-2v2M3 9h2m-2 6h2m14-6h2m-2 6h2M7 5h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Zm2 4h6v6H9Z",
  play: "m9 5 11 7-11 7V5Z",
  upload: "M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5",
  arrow: "M4 12h16m-6-6 6 6-6 6",
  check: "m5 12 4 4L19 6",
  scan: "M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M7 12h10",
  network: "M12 8v4m-7 4v-4h14v4M9 3h6v5H9ZM2 16h6v5H2Zm14 0h6v5h-6Z",
  train: "M4 19h16M6 16V9m6 7V4m6 12v-6",
  download: "M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5",
  clock: "M12 7v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
};
function icon(name, size = 18) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${paths[name] || paths.chip}"/></svg>`;
}
const app = {
  token: "",
  examples: [],
  view: "contract-1",
  status: null,
  forms: {},
  results: {},
  busy: {},
  messages: {},
  errors: {},
};
function toast(message) {
  const t = $("#toast");
  t.textContent = message;
  t.classList.add("show");
  clearTimeout(app.toastTimer);
  app.toastTimer = setTimeout(() => t.classList.remove("show"), 5000);
}
async function api(path, body) {
  const opts =
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Demo-Token": app.token,
          },
          body: JSON.stringify(body),
        };
  const response = await fetch(path, opts);
  const data = await response.json();
  if (!response.ok) {
    const error = Error(data.error || "Request failed");
    error.code = response.status;
    throw error;
  }
  return data;
}
function download(value, name) {
  const link = document.createElement("a");
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
function capture() {
  if (!$("#rtl")) return;
  app.forms[app.view] = {
    rtl: $("#rtl").value,
    requirement: $("#requirement").value,
    assumptions: $("#assumptions").value,
    example: $("#example").value,
    scope: $("#scope")?.value,
    arbitrate: $("#arbitrate")?.checked,
  };
}
function titleFor(view) {
  return {
    "contract-1": "Contract-1",
    "contract-2": "Contract-2",
    router: "Expert Analysis",
  }[view];
}
function chosenModel() {
  return app.status?.models.find((m) => m.id === app.view);
}
function selectView(view) {
  if (!["contract-1", "contract-2", "router"].includes(view))
    view = "contract-1";
  capture();
  app.view = view;
  history.replaceState(null, "", "#" + view);
  document
    .querySelectorAll("nav button")
    .forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $("#breadcrumb-page").textContent = titleFor(view);
  render();
}
function initialForm() {
  const e = (app.examples || []).find(
    (x) =>
      x.id ===
      (app.view === "contract-1"
        ? "privilege-register"
        : "fifo-simultaneous-guarded"),
  );
  if (!e)
    return {
      example: "custom",
      rtl: "",
      requirement: "",
      assumptions: "",
      arbitrate: true,
    };
  return { ...e, example: e.id, arbitrate: true };
}
function render() {
  $("#main").dataset.view = app.view;
  if (!app.forms[app.view]) app.forms[app.view] = initialForm();
  const router = app.view === "router",
    c1 = app.view === "contract-1",
    f = app.forms[app.view];
  const examples = (app.examples || []).filter(
    (e) => router || e.model === app.view,
  );
  $("#main").innerHTML =
    `<section class="page-heading"><div><div class="eyebrow">${router ? "RTL REQUIREMENT ASSESSMENT" : "RTL REQUIREMENT ASSESSMENT"}</div><h1>${router ? "Expert Analysis" : c1 ? "Contract-1" : "Contract-2"}</h1><p>${router ? "Select a property domain, compare specialist predictions, and optionally request a Muse Spark final assessment." : c1 ? "Evaluate authorization, lock, and write-enable requirements against the supplied RTL." : "Evaluate FIFO occupancy, counter, and arbitration requirements against the supplied RTL."}</p></div><div class="badge-icon">${icon(router ? "network" : "chip", 28)}</div></section>
${router ? `<div class="routing-flow"><div class="flow-box">RTL + requirement<small>Selected property domain</small></div><span class="flow-arrow">${icon("arrow")}</span><div class="flow-box">Capability router<small>Contract-1 · Contract-2</small></div><span class="flow-arrow">${icon("arrow")}</span><div class="flow-box">Muse Spark<small>Final assessment</small></div></div>` : `<div class="capability-strip"><div class="strip-cell"><label>Design focus</label><strong>${c1 ? "Local guards & register writes" : "FIFO · Counter · Arbitration"}</strong></div><div class="strip-cell"><label>Decision contract</label><strong>Three outcomes</strong><div class="sub">Satisfies · Violates · Context</div></div><div class="strip-cell"><label>Consulted by</label><strong id="skill-engine">Checking</strong><div class="sub" id="skill-status">Checking the specialist service</div></div></div>`}
<div class="workspace-grid"><section class="panel design-panel"><div class="panel-head"><div class="panel-title">Design under review <small>SystemVerilog / Verilog</small></div><span class="pill neutral">EDITABLE INPUT</span></div><div class="panel-body"><div class="toolbar"><select id="example" aria-label="Load an authored example">${examples.map((e) => `<option value="${esc(e.id)}" ${f.example === e.id ? "selected" : ""}>${esc(e.title)}</option>`).join("")}<option value="custom" ${f.example === "custom" ? "selected" : ""}>Custom design</option></select><button class="btn small" id="upload">${icon("upload", 15)} Open file</button><input class="file-input" id="file" type="file" accept=".sv,.v,.txt"></div>
<label class="field-label" for="rtl">RTL region <span class="field-hint" id="file-name">Self-contained design</span></label><div class="code-wrap"><div id="line-numbers" class="line-numbers" aria-hidden="true"></div><textarea id="rtl" spellcheck="false" aria-label="RTL source code">${esc(f.rtl)}</textarea></div><div class="code-caption"><span id="line-count"></span><span>Full input is sent · oversize inputs are rejected</span></div>
<div class="input-context"><div class="field"><label class="field-label" for="requirement">Requirement <span class="field-hint">Plain English</span></label><textarea id="requirement" rows="2">${esc(f.requirement)}</textarea></div><div class="field"><label class="field-label" for="assumptions">Analysis assumptions</label><textarea id="assumptions" rows="2">${esc(f.assumptions)}</textarea></div></div>
${
  router
    ? `<div class="route-options"><div class="field"><label class="field-label" for="scope">Routing domain</label><select id="scope">${[
        ["fifo", "FIFO occupancy & backpressure"],
        ["counter", "Counter control & saturation"],
        ["arbitration", "Arbitration & grants"],
        ["ot-access-control", "Register access control"],
      ]
        .map(
          ([v, l]) =>
            `<option value="${v}" ${f.scope === v ? "selected" : ""}>${l}</option>`,
        )
        .join(
          "",
        )}</select></div><label class="toggle-row"><input type="checkbox" id="arbitrate" ${f.arbitrate ? "checked" : ""}>Ask Muse Spark for the final assessment</label></div><div class="note route-privacy">When selected, input and evidence go to Muse Spark for final assessment.</div>`
    : ""
}</div>
<div class="button-row">${router ? '<span class="model-load-state">Measured profiles · skill-backed specialists</span>' : '<span class="model-load-state">Skill-backed specialist · no local model to load</span>'}<button class="btn primary" id="run">${icon("play", 15)} ${router ? "Route & assess" : "Run analysis"}</button></div></section>
<div class="right-stack assessment-stack"><section class="panel result-panel"><div class="panel-head"><span class="panel-title">${router ? "Assessment & arbitration" : "Specialist assessment"}</span><span class="pill neutral" id="result-state">NO RESULT</span></div><div id="result-content"></div></section><section class="panel activity-panel"><div class="panel-head"><span class="panel-title">Execution activity</span>${icon("clock", 15)}</div><div class="panel-body"><ul class="activity" id="activity"></ul></div></section><div class="note">${router ? "Routing uses measured development profiles, not the highest raw confidence. A missing specialist or failed agent call is reported as an execution error." : "Authored examples, not an independent benchmark. Verdicts come from the configured specialist service."}</div></div></div>`;
  function invalidate() {
    capture();
    app.results[app.view] = null;
    app.errors[app.view] = null;
    renderResult();
  }
  $("#rtl").addEventListener("input", () => {
    lineNumbers();
    invalidate();
  });
  for (const field of ["#requirement", "#assumptions"])
    $(field).addEventListener("input", invalidate);
  // Router controls change the assessment too: a verdict computed
  // for another domain must not stay on screen as current.
  if ($("#scope")) $("#scope").addEventListener("change", invalidate);
  if ($("#arbitrate"))
    $("#arbitrate").addEventListener("change", invalidate);
  $("#rtl").addEventListener("scroll", () => {
    $("#line-numbers").scrollTop = $("#rtl").scrollTop;
  });
  $("#example").onchange = () => {
    const e = (app.examples || []).find((e) => e.id === $("#example").value);
    if (e) {
      app.forms[app.view] = {
        ...e,
        example: e.id,
        arbitrate: $("#arbitrate")?.checked ?? f.arbitrate,
      };
      app.results[app.view] = null;
      app.errors[app.view] = null;
      render();
    }
  };
  $("#upload").onclick = () => $("#file").click();
  $("#file").onchange = async () => {
    const file = $("#file").files[0];
    if (!file) return;
    if (file.size > 50000) {
      toast("Choose a self-contained RTL file smaller than 50 KB.");
      return;
    }
    $("#rtl").value = await file.text();
    $("#example").value = "custom";
    $("#file-name").textContent = file.name;
    capture();
    app.results[app.view] = null;
    app.errors[app.view] = null;
    lineNumbers();
    renderResult();
  };
  $("#run").onclick = () => {
    capture();
    const body = { ...app.forms[app.view], model: app.view };
    startJob(router ? "/api/route" : "/api/assess", body, app.view);
  };
  lineNumbers();
  renderResult();
  renderActivity();
  syncStatus();
}
function lineNumbers() {
  const el = $("#rtl");
  if (!el) return;
  const lines = el.value.split("\n").length;
  $("#line-numbers").textContent = Array.from(
    { length: lines },
    (_, i) => i + 1,
  ).join("\n");
  $("#line-count").textContent =
    `${lines} lines · ${el.value.length.toLocaleString()} characters`;
}
function activityLines(entries, base) {
  if (!entries.length) return "";
  base = base || entries[0].time || 0;
  return entries
    .map((m) => {
      const dt =
        m.time && base ? Math.max(0, Math.round(m.time - base)) : null;
      const prefix =
        dt === null ? "" : `<span class="t">+${dt}s</span> `;
      return `<li>${prefix}${esc(m.text)}</li>`;
    })
    .join("");
}
function renderActivity() {
  if (!$("#activity")) return;
  const entries = app.messages[app.view] || [];
  $("#activity").innerHTML = entries.length
    ? activityLines(entries)
    : "<li>Load an example or open your own RTL file.</li><li>Results will come from the configured specialist service.</li>";
}
async function startJob(endpoint, body, view) {
  if (app.busy[view]) return;
  app.busy[view] = true;
  app.errors[view] = null;
  app.messages[view] = [{ text: "Request accepted by the local demo." }];
  syncStatus();
  renderActivity();
  renderResult();
  try {
    const response = await api(endpoint, body);
    let job,
      failures = 0;
    for (;;) {
      await new Promise((r) => setTimeout(r, 900));
      try {
        job = await api("/api/jobs/" + response.job_id);
      } catch (e) {
        if (e.code === 404 || e.code === 410 || ++failures > 20) throw e;
        continue;
      }
      failures = 0;
      app.messages[view] = job.messages;
      if (app.view === view) {
        renderActivity();
        if (app.busy[view]) renderResult();
      }
      if (job.status !== "running") break;
    }
    if (job.status === "error") throw Error(job.error);
    app.results[view] = job.result;
    toast("Assessment complete.");
  } catch (e) {
    app.errors[view] = e.message;
    toast(e.message);
  } finally {
    app.busy[view] = false;
    await refreshStatus();
    if (app.view === view) {
      renderResult();
      renderActivity();
    }
    syncStatus();
  }
}
function verdictClass(label) {
  return label === "violates_requirement"
    ? "violation"
    : label === "insufficient_context"
      ? "context"
      : label === "unresolved"
        ? "unresolved"
        : "";
}
function scores(result) {
  return `<div>${Object.keys(shortLabels)
    .map(
      (k, i) =>
        `<div class="score-row"><div class="score-caption"><span>${shortLabels[k]}</span><strong>${((result.scores?.[k] || 0) * 100).toFixed(2)}%</strong></div><div class="bar ${["", "vio", "mc"][i]}"><i style="width:${Math.max(0, Math.min(100, (result.scores?.[k] || 0) * 100))}%"></i></div></div>`,
    )
    .join("")}</div>`;
}
function renderResult() {
  const box = $("#result-content");
  if (!box) return;
  const view = app.view,
    error = app.errors[view],
    result = app.results[view],
    busy = app.busy[view];
  $("#result-state").textContent = busy
    ? "WORKING"
    : error
      ? "EXECUTION ERROR"
      : result
        ? "LIVE RESULT"
        : "NO RESULT";
  $("#result-state").className =
    "pill " + (error ? "bad" : result ? "" : "neutral");
  if (error) {
    box.innerHTML = `<div class="panel-body"><div class="error-box"><strong>Unable to complete this request</strong>${esc(error)}</div><p class="note">No new model verdict was issued for this failed request.</p></div>`;
    return;
  }
  if (busy) {
    const entries = app.messages[view] || [];
    const lines = activityLines(
      entries.slice(-8),
      entries[0] && entries[0].time,
    );
    box.innerHTML = `<div class="empty-result"><div class="empty-symbol"><span class="spinner"></span></div><h3>Request in progress</h3><p>Live specialist and Muse calls appear below as they run.</p></div>${lines ? `<div class="panel-body"><ul class="activity">${lines}</ul></div>` : ""}`;
    return;
  }
  if (!result) {
    box.innerHTML = `<div class="empty-result"><div class="empty-symbol">${icon(view === "router" ? "network" : "scan", 26)}</div><h3>${view === "router" ? "No expert analysis" : "No specialist assessment"}</h3><p>${view === "router" ? "Run an assessment to compare specialist predictions and domain support." : "Run the supplied RTL and requirement through the Contract skill."}</p></div><div class="panel-body"><div class="note box">${view === "router" ? "The router uses measured capability profiles. Muse receives specialist results, profile support, and the supplied requirement." : "Predictions depend on the RTL, requirement, and stated assumptions. Scores are not calibrated correctness probabilities."}</div></div>`;
    return;
  }
  if (view === "router") {
    renderRouterResult(box, result);
    return;
  }
  box.innerHTML = `<div class="result-verdict ${verdictClass(result.label)}"><div class="caption">${esc(titleFor(view))} · SPECIALIST VERDICT</div><div class="verdict-line">${icon(result.label === "satisfies_requirement" ? "check" : "scan", 21)}<h2>${esc(labels[result.label] || result.label)}</h2></div></div><div class="panel-body score-body">${scores(result)}<p class="note">${esc(result.glimmer_model || "")} · ${Number(result.wall_seconds || 0).toFixed(3)} s · not independently verified</p></div>${glimmerCalls(result)}${evidenceControls(result, view + "-assessment.json")}`;
  bindEvidence(result, view + "-assessment.json");
}
function callPath(endpoint) {
  try {
    return new URL(endpoint).pathname;
  } catch {
    return endpoint || "";
  }
}
function glimmerCalls(result) {
  const calls = result.glimmer_calls || [];
  if (!calls.length) return "";
  const cards = calls
    .map((c, i) => {
      if (c.error)
        return `<div class="expert-card"><div class="error-box"><strong>Call ${i + 1} failed</strong>${esc(c.error)}</div></div>`;
      const sent = (c.request_messages || [])
        .map(
          (m) =>
            `${esc(m.role)} ${Number(m.chars || 0).toLocaleString()} chars`,
        )
        .join(" · ");
      const responseLabel =
        c.kind === "score" ? "Exact score response (JSON)" : "Exact response text";
      return `<div class="expert-card"><div class="expert-head"><strong>Call ${i + 1} · POST ${esc(callPath(c.endpoint))}</strong></div><div class="expert-verdict">${esc(c.model || "")} · ${Number(c.seconds || 0).toFixed(1)} s · ${Number(c.response_chars || 0).toLocaleString()} chars</div><div class="expert-scores">Skill ${(c.skill_sha256 || "").slice(0, 12)}… · sent ${sent}</div><details class="details"><summary>${responseLabel}</summary><pre class="raw-evidence">${esc(c.response || "")}</pre></details></div>`;
    })
    .join("");
  return `<div class="panel-body"><div class="subheading">Live specialist calls (${calls.length})</div>${cards}</div>`;
}
function evidenceControls(result, name) {
  return `<div class="button-row result-actions"><button class="btn small" id="inspect-evidence">Inspect evidence</button><button class="btn small" id="download-result">${icon("download", 14)} Download</button></div><dialog id="evidence-dialog" class="info-dialog" aria-labelledby="evidence-title"><div class="panel-head"><strong id="evidence-title">Exact response & artifact identity</strong><button class="btn small" id="close-evidence">Close</button></div><div class="panel-body"><p class="note">The unchanged response includes the consulted artifacts, inference settings, and any execution errors. Model predictions are not formal proof.</p><pre class="raw-evidence">${esc(JSON.stringify(result, null, 2))}</pre></div></dialog>`;
}
function bindEvidence(result, name) {
  $("#inspect-evidence").onclick = () => $("#evidence-dialog").showModal();
  $("#close-evidence").onclick = () => $("#evidence-dialog").close();
  $("#download-result").onclick = () => download(result, name);
}
function renderRouterResult(box, result) {
  const a = result.assessment,
    ar = result.arbitration;
  let summary = "";
  if (ar)
    summary = `<div class="result-verdict ${verdictClass(ar.conclusion)}"><div class="caption">${esc((result.arbitration_model || "Muse Spark").toUpperCase())} · FINAL ASSESSMENT</div><h2>${esc(labels[ar.conclusion] || ar.conclusion)}</h2></div><div class="panel-body"><div class="arbitration-text">${esc(ar.explanation)}</div>${ar.open_questions?.length ? `<p class="note"><strong>Open questions</strong><br>${ar.open_questions.map(esc).join("<br>")}</p>` : ""}<p class="note">${ar.evidence?.length ? "Cited: " + ar.evidence.map((e) => esc(e.expert_id)).join(", ") + ". " : ""}Model assessment, not independent verification.</p></div>`;
  else if (result.arbitration_error)
    summary =
      '<div class="panel-body"><p class="note">Arbitration was not completed. Inspect the specialist responses and execution errors.</p></div>';
  else if (result.arbitration_requested)
    summary =
      '<div class="panel-body"><p class="note">Arbitration was requested but returned no verdict. Inspect the specialist responses and execution errors.</p></div>';
  else
    summary =
      '<div class="panel-body"><p class="note">Arbitration was not requested. Check the option above and re-run to ask Muse Spark for the final assessment of the specialist evidence.</p></div>';
  if (result.arbitration_error)
    summary += `<div class="panel-body"><div class="error-box"><strong>Final assessment unavailable</strong>${esc(result.arbitration_error)}</div></div>`;
  let experts = '<div class="panel-body">';
  for (const r of a.responses) {
    const tier = a.routing.tiers[r.expert_id] || "unrated";
    experts += `<div class="expert-card"><div class="expert-head"><strong>${esc(titleFor(r.expert_id) || r.expert_id)}</strong><span class="pill ${tier === "strong" ? "" : tier === "partial" ? "warning" : "neutral"}">${esc(tier.toUpperCase())} SUPPORT</span></div><div class="expert-verdict">${esc(labels[r.label])}</div><div class="expert-scores">${Object.entries(
      shortLabels,
    )
      .map(
        ([k, v]) =>
          `<span>${v}: <b>${(r.scores[k] * 100).toFixed(1)}%</b></span>`,
      )
      .join("")}</div></div>`;
  }
  for (const e of a.errors)
    experts += `<div class="error-box"><strong>${esc(e.expert_id)} failed</strong>${esc(e.detail)}</div>`;
  for (const [id, why] of Object.entries(result.broken_experts || {}))
    experts += `<div class="error-box"><strong>${esc(id)} excluded</strong>${esc(why)}</div>`;
  for (const id of result.unprofiled_experts || [])
    experts += `<div class="error-box"><strong>${esc(id)} excluded</strong>no capability profile; not consulted</div>`;
  experts += `<p class="note">${esc(a.routing.rationale.join(" "))}${a.routing.thin_evidence ? " Coverage is thin. Additional verification is required." : ""}</p></div>`;
  box.innerHTML = `<div class="panel-tabs" role="tablist" aria-label="Router result"><button id="summary-tab" role="tab" aria-controls="router-summary" aria-selected="true">Muse conclusion</button><button id="experts-tab" role="tab" aria-controls="router-experts" aria-selected="false">Specialists (${a.responses.length})${a.errors.length ? " · " + a.errors.length + " errors" : ""}</button></div><div id="router-summary" class="result-tab" role="tabpanel" aria-labelledby="summary-tab">${summary}</div><div id="router-experts" class="result-tab" role="tabpanel" aria-labelledby="experts-tab" hidden>${experts}</div>${evidenceControls(result, "router-assessment.json")}`;
  for (const [button, panel] of [
    ["summary-tab", "router-summary"],
    ["experts-tab", "router-experts"],
  ])
    $("#" + button).onclick = () => {
      for (const id of ["summary-tab", "experts-tab"])
        $("#" + id).setAttribute("aria-selected", String(id === button));
      for (const id of ["router-summary", "router-experts"])
        $("#" + id).hidden = id !== panel;
    };
  bindEvidence(result, "router-assessment.json");
}
function syncStatus() {
  const status = app.status;
  if (!status) return;
  const glimmer = status.health.glimmer,
    linked = glimmer.status === "ready";
  $("#side-models").textContent = status.models.length + " / 2 registered";
  $("#side-glimmer").textContent = linked
    ? "Connected"
    : status.glimmer_configured
      ? "Unavailable"
      : "Not configured";
  $("#connection-dot").classList.toggle("green", linked);
  $("#top-glimmer").textContent = linked
    ? `Engine · ${glimmer.model}`
    : status.glimmer_configured
      ? "Engine unavailable"
      : "Engine not configured";
  const busy = app.busy[app.view],
    model = chosenModel();
  for (const el of document.querySelectorAll(
    "#rtl,#requirement,#assumptions,#scope,#arbitrate,#example,#upload",
  ))
    el.disabled = !!busy;
  const run = $("#run");
  if (run)
    run.disabled =
      !!busy ||
      !linked ||
      (app.view !== "router" && !model) ||
      (app.view === "router" && !status.models.length);
  const adapterOnly =
    status.models.length > 0 &&
    status.models.every((m) => m.kind === "adapter");
  if ($("#arbitrate"))
    $("#arbitrate").disabled =
      !!busy ||
      !(app.arbiter_configured || (app.glimmer_configured && !adapterOnly));
  if ($("#skill-status")) {
    if ($("#skill-engine"))
      $("#skill-engine").textContent = !model
        ? "Unregistered"
        : "Meta Muse Glimmer";
    $("#skill-status").textContent = !model
      ? "Register the specialist in the specialist registry"
      : linked
        ? "Consulted through Meta Muse Glimmer"
        : "Specialist service unavailable";
  }
}
async function refreshStatus() {
  try {
    app.status = await api("/api/status");
    syncStatus();
  } catch (e) {
    $("#side-models").textContent = "Disconnected";
    $("#side-glimmer").textContent = "Disconnected";
    $("#top-glimmer").textContent = "Demo unreachable";
    $("#connection-dot").classList.remove("green");
    if (!app.status)
      $("#main").innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
  }
}
async function boot() {
  try {
    const config = await api("/api/bootstrap");
    Object.assign(app, config);
    document
      .querySelectorAll("nav button")
      .forEach((b) => (b.onclick = () => selectView(b.dataset.view)));
    $("#refresh-health").onclick = refreshStatus;
    window.addEventListener("hashchange", () =>
      selectView(location.hash.slice(1)),
    );
    await refreshStatus();
    selectView(location.hash.slice(1) || "contract-1");
    setInterval(refreshStatus, 2200);
  } catch (e) {
    $("#main").innerHTML =
      `<div class="error-box"><strong>Unable to connect to the demo</strong>${esc(e.message)}</div>`;
  }
}
boot();
