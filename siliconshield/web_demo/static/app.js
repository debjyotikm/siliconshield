"use strict";
const $ = (s, root = document) => root.querySelector(s);
const esc = (value) =>
  String(value ?? "").replace(
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
  if (!response.ok) throw Error(data.error || "Request failed");
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
    device:
      $("#device")?.value || app.forms[app.view]?.device || app.default_device,
  };
}
function titleFor(view) {
  return {
    "contract-1": "Contract-1",
    "contract-2": "Contract-2",
    router: "Expert router",
    training: "Train with Muse",
  }[view];
}
function displayDevice(value) {
  return value === "cpu"
    ? "CPU"
    : value?.startsWith("cuda:")
      ? "GPU " + value.split(":")[1]
      : "Not loaded";
}
function chosenModel() {
  return app.status?.models.find((m) => m.id === app.view);
}
function selectView(view) {
  if (!["contract-1", "contract-2", "router", "training"].includes(view))
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
  const e = app.examples.find(
    (x) =>
      x.id ===
      (app.view === "contract-1"
        ? "privilege-register"
        : "fifo-simultaneous-guarded"),
  );
  return { ...e, example: e.id, arbitrate: false, device: app.default_device };
}
function render() {
  $("#main").dataset.view = app.view;
  if (app.view === "training") {
    renderTraining();
    return;
  }
  if (!app.forms[app.view]) app.forms[app.view] = initialForm();
  const router = app.view === "router",
    c1 = app.view === "contract-1",
    f = app.forms[app.view];
  const examples = app.examples.filter((e) => router || e.model === app.view);
  $("#main").innerHTML =
    `<section class="page-heading"><div><div class="eyebrow">${router ? "RTL REQUIREMENT ASSESSMENT" : "RTL REQUIREMENT ASSESSMENT"}</div><h1>${router ? "Expert routing" : c1 ? "Contract-1" : "Contract-2"}</h1><p>${router ? "Select a property domain, compare specialist predictions, and optionally request a Muse Glimmer assessment." : c1 ? "Evaluate authorization, lock, and write-enable requirements against the supplied RTL." : "Evaluate FIFO occupancy, counter, and arbitration requirements against the supplied RTL."}</p></div><div class="badge-icon">${icon(router ? "network" : "chip", 28)}</div></section>
${router ? `<div class="routing-flow"><div class="flow-box">RTL + requirement<small>Selected property domain</small></div><span class="flow-arrow">${icon("arrow")}</span><div class="flow-box">Capability router<small>Contract-1 · Contract-2</small></div><span class="flow-arrow">${icon("arrow")}</span><div class="flow-box">Muse Glimmer<small>Review specialist outputs</small></div></div>` : `<div class="capability-strip"><div class="strip-cell"><label>Design focus</label><strong>${c1 ? "Local guards & register writes" : "FIFO · Counter · Arbitration"}</strong></div><div class="strip-cell"><label>Decision contract</label><strong>Three outcomes</strong><div class="sub">Satisfies · Violates · Context</div></div><div class="strip-cell"><label>Execution</label><strong>CPU or GPU inference</strong><div class="sub" id="model-device-status">Select a device and load the model</div></div></div>`}
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
        )}</select></div><label class="toggle-row"><input type="checkbox" id="arbitrate" ${f.arbitrate ? "checked" : ""}>Ask local Muse Glimmer to arbitrate</label></div><div class="note route-privacy">When selected, input and evidence go to your configured local Glimmer service.</div>`
    : ""
}</div>
<div class="button-row">${router ? '<span class="model-load-state">Registry devices · measured profiles</span>' : `<div class="model-load-group"><label class="sr-only" for="device">Inference device</label><select id="device" title="Inference runs on the workstation, not in this browser">${app.devices.map((d) => `<option value="${esc(d.id)}" ${f.device === d.id ? "selected" : ""}>${esc(d.label)}</option>`).join("")}</select><button class="btn" id="load-model">${icon("chip", 16)} Load model</button></div>`}<button class="btn primary" id="run">${icon("play", 15)} ${router ? "Route & assess" : "Run analysis"}</button></div></section>
<div class="right-stack assessment-stack"><section class="panel result-panel"><div class="panel-head"><span class="panel-title">${router ? "Assessment & arbitration" : "Model assessment"}</span><span class="pill neutral" id="result-state">NO RESULT</span></div><div id="result-content"></div></section><section class="panel activity-panel"><div class="panel-head"><span class="panel-title">Execution activity</span>${icon("clock", 15)}</div><div class="panel-body"><ul class="activity" id="activity"></ul></div></section><div class="note">${router ? "Routing uses measured development profiles, not the highest raw confidence. A missing worker or failed agent call is reported as an execution error." : "Authored examples, not an independent benchmark. CPU and GPU results may differ."}</div></div></div>`;
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
  $("#rtl").addEventListener("scroll", () => {
    $("#line-numbers").scrollTop = $("#rtl").scrollTop;
  });
  $("#example").onchange = () => {
    const e = app.examples.find((e) => e.id === $("#example").value);
    if (e) {
      app.forms[app.view] = {
        ...e,
        example: e.id,
        arbitrate: $("#arbitrate")?.checked ?? f.arbitrate,
        device: $("#device")?.value || f.device,
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
  $("#device")?.addEventListener("change", () => {
    invalidate();
    syncStatus();
  });
  $("#load-model")?.addEventListener("click", () =>
    startJob(
      "/api/models/load",
      { model: app.view, device: $("#device").value },
      app.view,
    ),
  );
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
function renderActivity() {
  if (!$("#activity")) return;
  const entries = app.messages[app.view] || [];
  $("#activity").innerHTML = entries.length
    ? entries.map((m) => `<li>${esc(m.text)}</li>`).join("")
    : app.view === "training"
      ? "<li>When enabled, start a new toy study with the frozen recipe.</li><li>The dashboard reads the saved run state, including jobs started before this page was opened.</li>"
      : "<li>Load an example or open your own RTL file.</li><li>Results will come from the configured infrastructure.</li>";
}
async function startJob(endpoint, body, view) {
  if (app.busy[view]) return;
  app.busy[view] = true;
  app.errors[view] = null;
  app.messages[view] = [{ text: "Request accepted by the local demo." }];
  syncStatus();
  renderActivity();
  if (view !== "training") renderResult();
  try {
    const response = await api(endpoint, body);
    let job;
    do {
      await new Promise((r) => setTimeout(r, 900));
      job = await api("/api/jobs/" + response.job_id);
      app.messages[view] = job.messages;
      if (app.view === view) {
        renderActivity();
        if (view === "training") updateTraining();
      }
    } while (job.status === "running");
    if (job.status === "error") throw Error(job.error);
    if (endpoint.endsWith("/load"))
      toast(
        titleFor(view) + " loaded on " + displayDevice(job.result.device) + ".",
      );
    else if (view === "training") toast("Training controller started.");
    else {
      app.results[view] = job.result;
      toast("Assessment complete.");
    }
  } catch (e) {
    app.errors[view] = e.message;
    toast(e.message);
  } finally {
    app.busy[view] = false;
    await refreshStatus();
    if (app.view === view) {
      if (view === "training") updateTraining();
      else renderResult();
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
    box.innerHTML = `<div class="empty-result"><div class="empty-symbol"><span class="spinner"></span></div><h3>Request in progress</h3><p>Execution activity lists model loading, inference, and agent calls.</p></div>`;
    return;
  }
  if (!result) {
    box.innerHTML = `<div class="empty-result"><div class="empty-symbol">${icon(view === "router" ? "network" : "scan", 26)}</div><h3>${view === "router" ? "No router assessment" : "No model assessment"}</h3><p>${view === "router" ? "Run an assessment to compare specialist predictions and domain support." : "Select CPU or GPU, load the model, then run the supplied RTL and requirement."}</p></div><div class="panel-body"><div class="note box">${view === "router" ? "The router uses profiles matched to the configured devices. Muse receives specialist results, profile support, and the supplied requirement." : "Predictions depend on the RTL, requirement, and stated assumptions. Scores are not calibrated correctness probabilities."}</div></div>`;
    return;
  }
  if (view === "router") {
    renderRouterResult(box, result);
    return;
  }
  box.innerHTML = `<div class="result-verdict ${verdictClass(result.label)}"><div class="caption">${esc(titleFor(view))} · MODEL PREDICTION</div><div class="verdict-line">${icon(result.label === "satisfies_requirement" ? "check" : "scan", 21)}<h2>${esc(labels[result.label] || result.label)}</h2></div></div><div class="panel-body score-body">${scores(result)}<p class="note">${esc(displayDevice(result.device))} · ${esc(result.execution_identity?.inference?.precision?.replace("torch.", "") || "")} · ${Number(result.wall_seconds || 0).toFixed(3)} s · not independently verified</p></div>${evidenceControls(result, view + "-assessment.json")}`;
  bindEvidence(result, view + "-assessment.json");
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
    summary = `<div class="result-verdict ${verdictClass(ar.conclusion)}"><div class="caption">MUSE GLIMMER · PROVISIONAL CONCLUSION</div><h2>${esc(labels[ar.conclusion] || ar.conclusion)}</h2></div><div class="panel-body"><div class="arbitration-text">${esc(ar.explanation)}</div>${ar.open_questions?.length ? `<p class="note"><strong>Open questions</strong><br>${ar.open_questions.map(esc).join("<br>")}</p>` : ""}<p class="note">${ar.evidence?.length ? "Cited: " + ar.evidence.map((e) => esc(e.expert_id)).join(", ") + ". " : ""}Model assessment, not independent verification.</p></div>`;
  else
    summary =
      '<div class="panel-body"><p class="note">Arbitration was not completed. Inspect the specialist responses and execution errors.</p></div>';
  if (result.arbitration_error)
    summary += `<div class="panel-body"><div class="error-box"><strong>Glimmer arbitration unavailable</strong>${esc(result.arbitration_error)}</div></div>`;
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
function renderTraining() {
  $("#main").innerHTML = `
<section class="page-heading"><div><div class="eyebrow">DECISION-MODEL TRAINING</div><h1>Training with Muse Glimmer</h1><p>Train Contract-1 adapters on an authored dataset. Glimmer monitors the job and records permitted actions.</p></div><button class="btn small" id="training-help">Recipe & controls</button></section>
<div class="training-steps">
 <div class="training-step" data-phase="preparing"><span class="step-num">01</span><strong>Preflight</strong><small>Inputs · GPU · sampler</small></div>
 <div class="training-step" data-phase="fitting"><span class="step-num">02</span><strong>Fitting</strong><small>TRAIN only · ${app.toy_updates} updates</small></div>
 <div class="training-step" data-phase="development"><span class="step-num">03</span><strong>Development</strong><small>Fresh parent · ${app.toy_updates} updates</small></div>
 <div class="training-step" data-phase="complete"><span class="step-num">04</span><strong>Closeout</strong><small>Selected export · evidence</small></div>
</div><div id="training-banner"></div>
<div class="training-grid">
<section class="panel recipe-panel"><div class="panel-head"><span class="panel-title">Toy training run</span><span class="pill neutral">FROZEN</span></div>
<div class="panel-body"><div class="recipe-grid">
 <div><label>Decision model</label><strong>Contract-1 child</strong></div><div><label>Local operator</label><strong>Muse Glimmer</strong></div>
 <div><label>Training hardware</label><strong>${app.gpu === null ? "Not configured" : "GPU " + app.gpu}</strong></div><div><label>Learning rate / batch</label><strong>0.00001 / 6</strong></div>
</div><p class="note">Trainable: residual adapters + decision head.</p>
<div class="subheading">Write-authorization fixture</div>
<table class="recipe-table"><thead><tr><th>Split</th><th>Rows</th><th>Per class</th></tr></thead><tbody><tr><td>TRAIN</td><td>18</td><td>6</td></tr><tr><td>Dev</td><td>6</td><td>2</td></tr></tbody></table>
<p class="note box">TRAIN and dev share template patterns. Results test the training workflow, not generalization.</p></div>
<div class="button-row"><button class="btn danger small" id="stop-training" disabled>Stop</button><button class="btn primary" id="start-training">${icon("play", 15)} Start training</button></div></section>
<section class="panel learning-panel"><div class="panel-head"><span class="panel-title">Learning & outcomes</span><span class="pill neutral">MEASURED</span></div><div class="learning-body">
<div id="train-metrics" class="metrics-row"></div><div id="loss-chart" class="chart-container"></div>
<div class="chart-legend"><span><i></i>Fitting</span><span><i class="blue"></i>Development</span><span>Cross-entropy</span></div>
<div id="phase-reports" class="panel-body"></div>
</div><div class="button-row"><span class="note">No calibration or test accessed</span><button class="btn small" id="download-training">${icon("download", 14)} Evidence</button></div></section>
<section class="panel agent-panel"><div class="panel-head"><span class="panel-title">Muse Glimmer</span><span class="pill neutral" id="agent-state">WAITING</span></div>
<div class="panel-tabs" role="tablist" aria-label="Training activity"><button id="agent-tab" role="tab" aria-controls="agent-actions" aria-selected="true">Agent actions</button><button id="launch-tab" role="tab" aria-controls="launch-actions" aria-selected="false">Launch activity</button></div>
<div id="agent-actions" class="panel-body" role="tabpanel" aria-labelledby="agent-tab"></div><div id="launch-actions" class="panel-body" role="tabpanel" aria-labelledby="launch-tab" hidden><ul class="activity" id="activity"></ul></div>
</section></div>
<dialog id="training-info" class="info-dialog" aria-labelledby="training-info-title"><div class="panel-head"><strong id="training-info-title">Recipe & run controls</strong><button class="btn small" id="close-training-help">Close</button></div><div class="panel-body">
<p>The contract allows a register update only for a privileged writer. Each group contains a guarded write, a missing privilege check, and an omitted policy definition.</p>
<p>The frozen encoder stays fixed. The adapters and decision head train at learning rate 0.00001, batch 6, seed 7. The budget is ${app.toy_updates} fitting updates and ${app.toy_updates} development updates.</p>
<p>Development starts again from the original parent using the qualified fitting settings, not the fitting weights. A failed fitting gate ends the study.</p>
<p>Starting releases only this app’s inference workers. Existing GPU jobs are never interrupted. Glimmer uses the configured local endpoint and operates the owned job on the configured GPU.</p>
<p>Stop requests a checkpoint-boundary stop and prevents the next phase. Closing the browser does not stop training.</p>
<p class="note box">TRAIN and development examples have related templates and widths. This is an integration fixture, not an independent benchmark. The released models remain unchanged.</p>
</div></dialog>`;
  $("#start-training").onclick = () =>
    startJob("/api/training/start", { confirm: true }, "training");
  $("#stop-training").onclick = async () => {
    try {
      await api("/api/training/stop", {});
      toast("Stop requested for this demo’s jobs only.");
      await refreshStatus();
    } catch (e) {
      toast(e.message);
    }
  };
  $("#download-training").onclick = async () =>
    download(await api("/api/training/receipt"), "toy-study-evidence.json");
  $("#training-help").onclick = () => $("#training-info").showModal();
  $("#close-training-help").onclick = () => $("#training-info").close();
  for (const [button, activePanel] of [
    ["agent-tab", "agent-actions"],
    ["launch-tab", "launch-actions"],
  ])
    $("#" + button).onclick = () => {
      for (const id of ["agent-tab", "launch-tab"])
        $("#" + id).setAttribute("aria-selected", String(id === button));
      for (const id of ["agent-actions", "launch-actions"])
        $("#" + id).hidden = id !== activePanel;
    };
  renderActivity();
  updateTraining();
}

function lossChart(jobs) {
  const series = jobs
    .map((j) => ({
      name: j.name.includes("-fit-") ? "Fitting" : "Development",
      color: j.name.includes("-fit-") ? "#216bc1" : "#455568",
      points: j.events.filter(
        (e) => e.event === "train" && Number.isFinite(e.loss),
      ),
    }))
    .filter((s) => s.points.length);
  if (!series.length)
    return '<div class="chart-empty">Loss values appear after the first optimizer update.</div>';
  const w = 560,
    h = 200,
    l = 47,
    r = 22,
    t = 15,
    b = 32;
  const max =
    Math.max(1, ...series.flatMap((s) => s.points.map((p) => p.loss))) * 1.1;
  const steps = Math.max(
    app.toy_updates,
    ...series.flatMap((s) => s.points.map((p) => p.step)),
  );
  const x = (v) => l + ((v - 1) / Math.max(1, steps - 1)) * (w - l - r),
    y = (v) => h - b - (v / max) * (h - b - t);
  let s = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Measured training cross-entropy by optimizer update"><rect width="560" height="200" fill="white"/>`;
  for (let i = 0; i < 5; i++) {
    const v = (max * i) / 4,
      yy = y(v);
    s += `<line x1="${l}" y1="${yy}" x2="${w - r}" y2="${yy}" stroke="#e6eaf0"/><text x="${l - 8}" y="${yy + 4}" text-anchor="end" font-size="10" fill="#68778b">${v.toFixed(2)}</text>`;
  }
  for (let i = 1; i <= steps; i++)
    s += `<text x="${x(i)}" y="${h - 13}" text-anchor="middle" font-size="10" fill="#68778b">${i}</text>`;
  for (const line of series) {
    s += `<polyline points="${line.points.map((p) => `${x(p.step)},${y(p.loss)}`).join(" ")}" fill="none" stroke="${line.color}" stroke-width="2"/>`;
    for (const p of line.points)
      s += `<circle cx="${x(p.step)}" cy="${y(p.loss)}" r="3" fill="${line.color}"><title>${line.name}: update ${p.step}, loss ${p.loss.toFixed(4)}</title></circle>`;
  }
  return s + "</svg>";
}
function updateTraining() {
  if (app.view !== "training" || !$("#training-banner")) return;
  const tr = app.status?.training || { status: "idle" },
    busy = app.busy.training,
    active = tr.active || busy;
  let phase = tr.phase || "preparing";
  const jobs = tr.jobs || [];
  if (jobs.some((j) => j.name.endsWith("-development"))) phase = "development";
  else if (jobs.length) phase = "fitting";
  if (tr.status === "complete") phase = "complete";
  document
    .querySelectorAll(".training-step")
    .forEach((e) =>
      e.classList.toggle(
        "on",
        tr.status !== "idle" && e.dataset.phase === phase,
      ),
    );
  const error = app.errors.training || tr.error;
  let banner = error
    ? `<div class="error-box" role="alert"><strong>Run needs attention</strong><span>${esc(error)}</span></div>`
    : tr.status === "idle"
      ? !app.training_enabled
        ? '<div class="note box">Toy training is disabled. Configure Linux training paths, a GPU, and local Glimmer to enable this panel.</div>'
        : ""
      : `<div class="running-banner"><span>${active ? '<span class="spinner"></span> ' : icon("check", 15) + " "}<strong>${tr.status === "complete" ? "Bounded study complete" : tr.stop_requested ? "Stop requested" : tr.status === "stopped" ? "Study stopped" : active ? "Training · " + phase : titleFor(phase) || esc(tr.status)}</strong><br><span class="note">${esc(tr.id || "Preparing configuration")}</span></span><span class="pill ${active ? "warning" : ""}">${active ? "RUNNING" : esc(tr.status?.toUpperCase())}</span></div>`;
  const agentErrors = (tr.operator_cycles || []).filter((c) => c.error).length;
  if (tr.status === "complete" && agentErrors)
    banner += `<div class="agent-warning">${agentErrors} refused agent cycles · inspect the action log</div>`;
  $("#training-banner").innerHTML = banner;
  $("#start-training").disabled = active || !app.training_enabled;
  $("#start-training").innerHTML = active
    ? '<span class="spinner"></span> Study running'
    : icon("play", 15) + " Start training";
  $("#stop-training").disabled = !tr.active || tr.stop_requested;
  $("#download-training").disabled = tr.status === "idle";
  $("#loss-chart").innerHTML = lossChart(jobs);
  const events = jobs.flatMap((j) =>
    j.events.filter((e) => e.event === "train"),
  );
  const last = events.at(-1);
  $("#train-metrics").innerHTML = [
    [
      "Updates",
      String(events.length) +
        " / " +
        2 * (tr.updates_per_phase || app.toy_updates),
    ],
    ["TRAIN loss", last?.loss?.toFixed(3) || "—"],
    [
      "Dev correct",
      (() => {
        const d = jobs.find((j) => j.name.endsWith("-development"))?.report
          ?.development;
        return d?.accuracy === undefined
          ? "—"
          : `${Math.round(d.accuracy * d.n)} / ${d.n}`;
      })(),
    ],
  ]
    .map(
      ([label, value]) =>
        `<div class="metric-box"><label>${label}</label><strong>${value}</strong></div>`,
    )
    .join("");
  const cycles = tr.operator_cycles || [];
  $("#agent-state").textContent = cycles.length
    ? "LATEST FIRST"
    : active
      ? "OBSERVING"
      : "WAITING";
  setPanelHTML(
    "#agent-actions",
    cycles.length
      ? [...cycles]
          .reverse()
          .map(
            (c) =>
              `<div class="action-item" data-key="${esc(c.time)}"><strong>${esc(c.experiment?.includes("-fit-") ? "Fitting" : "Development")} · ${esc(c.phase || "observe")}</strong>${c.error ? `<p class="error-box">${esc(c.error)}</p>` : `<p>${esc(actionSummary(c.executed))}</p>`}<details class="details"><summary>Exact tool calls and evidence</summary><pre>${esc(JSON.stringify(c, null, 2))}</pre></details></div>`,
          )
          .join("")
      : `<div class="empty-result" style="padding:22px 10px"><div class="empty-symbol">${icon("network", 26)}</div><h3>${active ? "Waiting for recorded Glimmer actions" : "No operator actions recorded"}</h3><p>Recorded observations, decisions, and errors appear here as the operator runs.</p></div>`,
  );
  let reports =
    '<div class="subheading">Selected checkpoints</div><table class="outcome-table"><thead><tr><th>Phase</th><th>Correct</th><th>False-safe</th><th>Step</th></tr></thead><tbody>';
  for (const fit of [true, false]) {
    const j = jobs.find((j) => j.name.includes("-fit-") === fit),
      r = j?.report || {},
      m = fit ? r.train : r.development;
    reports += `<tr><th>${fit ? "TRAIN" : "Dev"}</th><td>${m ? Math.round(m.accuracy * m.n) + " / " + m.n : "—"}</td><td>${m ? (m.false_safe_count ?? "—") + " / " + (m.violation_count ?? "—") : "—"}</td><td>${r.selected_checkpoint?.step ?? "—"}</td></tr>`;
  }
  reports +=
    '</tbody></table><p class="note">False-safe counts use violation denominators. Toy results do not qualify a release.</p>';
  setPanelHTML("#phase-reports", reports);
  renderActivity();
}
function setPanelHTML(selector, markup) {
  const el = $(selector);
  if (el._lastMarkup === markup) return;
  const top = el.scrollTop,
    opened = [...el.querySelectorAll("details")]
      .filter((d) => d.open)
      .map((d) => d.closest(".action-item")?.dataset.key);
  el.innerHTML = markup;
  el._lastMarkup = markup;
  for (const d of el.querySelectorAll("details"))
    if (opened.includes(d.closest(".action-item")?.dataset.key)) d.open = true;
  el.scrollTop = top;
}
function actionSummary(executed) {
  if (!executed) return "No executed action in this cycle.";
  const parts = [];
  function walk(x) {
    if (!x || typeof x !== "object") return;
    if (Array.isArray(x)) {
      x.forEach(walk);
      return;
    }
    if (x.name) parts.push(String(x.name));
    if (x.action) parts.push(String(x.action));
    if (x.reason) parts.push(String(x.reason));
    if (x.explanation) parts.push(String(x.explanation));
    for (const [k, v] of Object.entries(x)) {
      if (
        [
          "arguments",
          "function",
          "params",
          "calls",
          "executed",
          "decision",
        ].includes(k)
      )
        walk(v);
    }
  }
  walk(executed);
  return parts.length
    ? [...new Set(parts)].join(" · ")
    : JSON.stringify(executed).slice(0, 1000);
}
function syncStatus() {
  const status = app.status;
  if (!status) return;
  const loaded = status.models.filter((m) => m.loaded).length;
  $("#side-models").textContent = loaded + " / 2 loaded";
  $("#side-glimmer").textContent =
    status.health.glimmer.status === "ready"
      ? "Connected"
      : status.glimmer_configured
        ? "Unavailable"
        : "Not configured";
  $("#connection-dot").classList.toggle(
    "green",
    status.health.glimmer.status === "ready",
  );
  const train = status.training;
  $("#side-training").textContent = train.active
    ? "Running"
    : train.status === "complete"
      ? "Complete"
      : train.status === "error"
        ? "Attention"
        : "Idle";
  const gpu = status.health.gpus.find((g) => Number(g.index) === app.gpu);
  $("#top-gpu").textContent = gpu
    ? `GPU ${gpu.index} · ${(Number(gpu.memory_used) / 1024).toFixed(1)} GB used`
    : app.gpu === null
      ? "CPU inference"
      : "GPU status unavailable";
  if (app.view === "training") {
    updateTraining();
    return;
  }
  const busy = app.busy[app.view],
    model = chosenModel(),
    active = train.active,
    selected = app.forms[app.view]?.device || app.default_device,
    ready = model?.loaded && model.device === selected;
  for (const el of document.querySelectorAll(
    "#rtl,#requirement,#assumptions,#scope,#arbitrate,#example,#upload,#device",
  ))
    el.disabled = !!busy;
  const run = $("#run");
  if (run)
    run.disabled =
      !!busy ||
      !!active ||
      (app.view !== "router" && !ready) ||
      (app.view === "router" && !status.models.length);
  const load = $("#load-model");
  if (load) {
    load.disabled = !!busy || !!active || !!ready || !model;
    load.innerHTML = busy
      ? '<span class="spinner"></span> Loading'
      : ready
        ? icon("check", 15) + " " + displayDevice(selected) + " ready"
        : icon("chip", 15) + " Load model";
  }
  if ($("#arbitrate"))
    $("#arbitrate").disabled = !!busy || !app.glimmer_configured;
  if ($("#model-device-status"))
    $("#model-device-status").textContent = !model
      ? "Register the checkpoint in the local configuration"
      : ready
        ? `${displayDevice(selected)} · loaded`
        : model?.loaded
          ? `${displayDevice(model.device)} loaded · reload for ${displayDevice(selected)}`
          : `${displayDevice(selected)} selected · not loaded`;
}
async function refreshStatus() {
  try {
    app.status = await api("/api/status");
    syncStatus();
  } catch (e) {
    $("#side-models").textContent = "Disconnected";
    $("#side-glimmer").textContent = "Disconnected";
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
