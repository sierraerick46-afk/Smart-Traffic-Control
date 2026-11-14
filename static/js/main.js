
const $ = (id) => document.getElementById(id);
const DIAS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

async function api(url, opts) {
  const r = await fetch(url, opts);
  if (r.status === 401) {
    window.location.href = `/login?next=${encodeURIComponent(window.location.pathname)}`;
    throw new Error("Login required");
  }
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `Error ${r.status}`);
  return data;
}
const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                       body: JSON.stringify(body || {}) });
const slot = () => {
  const j = $("sel-junction").value;
  return `hora=${$("sel-hora").value}&dia=${$("sel-dia").value}` + (j !== "all" ? `&junction=${j}` : "");
};
const setMsg = (el, text, kind = "secondary") => { el.className = `small mb-2 text-${kind}`; el.textContent = text; };

const initHooks = [];
const slotHooks = [];

async function loadClusters() {
  const d = await api(`/api/clusters?${slot()}`);
  const badge = $("cluster-badge");
  badge.className = `badge fs-6 text-bg-${d.current.color}`;
  badge.textContent = d.current.label;
  $("cluster-text").textContent = `${d.current.dia_nombre}, ${String(d.current.hora).padStart(2, "0")}:00`;

  const strip = $("hour-strip");
  strip.innerHTML = "";
  d.profile.forEach((p) => {
    const c = document.createElement("div");
    c.className = `hour-cell c-${p.color}` + (p.hour === d.current.hora ? " sel" : "");
    c.textContent = p.hour;
    c.title = `${p.hour}:00 · ${p.mean} veh/h`;
    c.onclick = () => { $("sel-hora").value = p.hour; refreshSlot(); };
    strip.appendChild(c);
  });

  $("cluster-table").innerHTML = d.summary.map((s) => {
    const hrs = s.weekday_hours.length ? s.weekday_hours.map((h) => h + "h").join(", ") : "–";
    return `<tr><td><span class="badge text-bg-${s.color}">${s.label}</span></td>
            <td class="text-end">${s.mean_vehicles}</td><td class="small">${hrs}</td></tr>`;
  }).join("");
  $("silhouette").textContent = d.silhouette != null ? `Clustering quality (silhouette): ${d.silhouette.toFixed(2)}` : "";
}

function refreshSlot() {
  loadClusters().catch(console.error);
  slotHooks.forEach((h) => Promise.resolve().then(h).catch(console.error));
}

async function initDashboard() {
  const st = await api("/api/status");
  const dia = $("sel-dia"), hora = $("sel-hora"), junc = $("sel-junction");
  DIAS.forEach((n, i) => dia.add(new Option(n, i)));
  junc.add(new Option("All (average)", "all"));
  st.intersections.forEach((x) => junc.add(new Option(`${x.junction} · ${x.address}`, x.junction)));
  for (let h = 0; h < 24; h++) hora.add(new Option(String(h).padStart(2, "0") + ":00", h));
  const now = await api("/api/clusters");
  dia.value = now.current.dia; hora.value = now.current.hora;
  dia.onchange = hora.onchange = junc.onchange = refreshSlot;

  const d = st.data;
  $("status-badges").innerHTML =
    `<span class="badge text-bg-${d.is_synthetic ? "warning" : "success"}">${d.is_synthetic ? "Synthetic data (no CSV in data/raw)" : "Real data"}</span>
     <span class="badge text-bg-light border">${d.rows.toLocaleString("en-US")} records</span>
     <span class="badge text-bg-light border">${d.intersections} intersections</span>`;

  initHooks.forEach((h) => Promise.resolve().then(() => h(st)).catch(console.error));
  refreshSlot();
}

document.addEventListener("DOMContentLoaded", () => {
  if ($("cluster-card")) initDashboard().catch(console.error);
});

function renderMetrics(m) {
  if (!m || !m.rows) return;
  $("metrics-table").innerHTML = m.rows.map((x) => {
    const best = m.best_model === x.key;
    return `<tr class="${best ? "table-success" : ""}"><td>${x.label}${best ? " ✓" : ""}</td>
      <td class="text-end">${x.cv_rmse ?? "–"}</td><td class="text-end">${x.mse}</td>
      <td class="text-end">${x.mae}</td><td class="text-end">${x.r2}</td></tr>`;
  }).join("");
  const t = m.test;
  $("test-result").innerHTML = t ? `<b>Independent test (May–Jun 2017) · ${m.test_label}:</b> R² ${t.r2} · MAE ${t.mae} veh/h · RMSE ${t.rmse}
    <span class="text-secondary">(${m.source === "saved" ? "saved model" : "retrained now"}; train ${m.n_train.toLocaleString()} · validation ${m.n_validation.toLocaleString()} · test ${m.n_test.toLocaleString()} rows)</span>` : "";
}

async function loadPredict() {
  const d = await api(`/api/predict?${slot()}`);
  $("pred-value").textContent = `${d.vehicles} veh/h`;
  $("pred-model").textContent = d.model === "Historical average" ? `model: ${d.model}` : `model: ${d.model} · traffic level of ${d.reference_date}`;
}

async function trainPrediction() {
  const btn = $("btn-train-pred");
  btn.disabled = true; $("spin-pred").classList.remove("d-none");
  setMsg($("pred-msg"), "Retraining the five models with the tuned parameters… this can take a few minutes.");
  try {
    await api("/api/train_prediction", { method: "POST" });
    let s;
    do {
      await new Promise((r) => setTimeout(r, 3000));
      s = await api("/api/prediction_status");
    } while (s.training);
    if (s.error) throw new Error(s.error);
    renderMetrics(s.metrics);
    setMsg($("pred-msg"), "Done: models retrained with the same partitions and parameters.", "success");
    await loadPredict();
  } catch (e) { setMsg($("pred-msg"), e.message, "danger"); }
  btn.disabled = false; $("spin-pred").classList.add("d-none");
}

initHooks.push(async () => {
  $("btn-train-pred").onclick = trainPrediction;
  const s = await api("/api/prediction_status");
  if (s.metrics) renderMetrics(s.metrics);
});
slotHooks.push(loadPredict);

function renderEval(ev) {
  const row = (name, x) => `<tr><td>${name}</td><td class="text-end">${x.avg_queue}</td>
    <td class="text-end">${x.throughput}</td><td class="text-end">${x.overflow}</td></tr>`;
  $("eval-table").innerHTML = row("DQN agent", ev.dqn) + row("Fixed time (30 s)", ev.fixed) +
    `<tr class="table-light"><td colspan="4" class="small">Queue reduction vs. fixed time: <strong>${ev.improvement_pct}%</strong></td></tr>`;
}

async function trainControl() {
  const btn = $("btn-train-ctrl");
  btn.disabled = true; $("spin-ctrl").classList.remove("d-none");
  setMsg($("ctrl-msg"), "Training the agent in the simulator…");
  try {
    const r = await post("/api/train_control", { episodes: parseInt($("episodes").value, 10) || 60 });
    setMsg($("ctrl-msg"), `Reward per episode: ${r.report.reward_first} → ${r.report.reward_last}`, "success");
    renderEval(r.report.evaluation);
  } catch (e) { setMsg($("ctrl-msg"), e.message, "danger"); }
  btn.disabled = false; $("spin-ctrl").classList.add("d-none");
}

initHooks.push(() => { $("btn-train-ctrl").onclick = trainControl; });

let timer = null, busy = false, running = false, options = null;
const FILTERS = { date: "f-date", hour: "f-hour", junction: "f-junction", policy: "f-policy",
                  model: "f-model", source: "f-source", speed: "f-speed" };

function buildSite(x, single) {
  const acc = (i, cls) => `<div class="acc ${cls}">
      <div class="name" id="s${x.junction}-n-${i}"></div>
      <div class="signal red" id="s${x.junction}-sig-${i}"><span class="lamp lamp-r"></span><span class="lamp lamp-y"></span><span class="lamp lamp-g"></span></div>
      <div class="queue-bar"><div class="queue-fill" id="s${x.junction}-qf-${i}"></div></div>
      <div class="small"><span id="s${x.junction}-qn-${i}">0</span> veh.</div>
      <div class="cars" id="s${x.junction}-cars-${i}"></div></div>`;
  return `<div class="${single ? "col-12" : "col-xl-6"}"><div class="card site ${single ? "single" : ""}">
    <div class="card-header d-flex flex-wrap justify-content-between align-items-center gap-2">
      <span>Intersection ${x.junction} · ${x.address}</span>
      <span id="s${x.junction}-cluster" class="badge text-bg-secondary">–</span></div>
    <div class="card-body">
      <div class="intersection">${acc(0, "acc-n")}${acc(3, "acc-w")}
        <div class="crossing"><div id="s${x.junction}-phase" class="small text-center">–</div></div>
        ${acc(2, "acc-e")}${acc(1, "acc-s")}</div>
      <div class="row g-2 text-center mt-3 site-metrics">
        <div class="col-4 col-md-2"><div class="small text-secondary">Forecast</div><div id="s${x.junction}-fc" class="fw-semibold">–</div></div>
        <div class="col-4 col-md-2"><div class="small text-secondary">Observed</div><div id="s${x.junction}-obs" class="fw-semibold">–</div></div>
        <div class="col-4 col-md-3"><div class="small text-secondary">Decision</div><div id="s${x.junction}-act" class="fw-semibold">–</div></div>
        <div class="col-4 col-md-2"><div class="small text-secondary">Queue</div><div id="s${x.junction}-tq" class="fw-semibold">0</div></div>
        <div class="col-4 col-md-1"><div class="small text-secondary">Avg.</div><div id="s${x.junction}-avg" class="fw-semibold">0</div></div>
        <div class="col-4 col-md-2"><div class="small text-secondary">Served</div><div id="s${x.junction}-dep" class="fw-semibold">0</div></div>
      </div>
      <div class="progress mt-3" style="height:6px" title="Time in current phase"><div id="s${x.junction}-bar" class="progress-bar bg-dark" style="width:0%"></div></div>
    </div></div></div>`;
}

function buildSites(cfg) {
  const list = options.intersections.filter((x) => cfg.junction === "all" || x.junction === cfg.junction);
  $("sites").innerHTML = list.map((x) => buildSite(x, list.length === 1)).join("");
}

function renderSite(d) {
  const p = `s${d.junction}-`;
  for (let i = 0; i < 4; i++) {
    $(`${p}n-${i}`).textContent = d.access_names[i];
    $(`${p}sig-${i}`).className = `signal ${d.signals[i]}`;
    const q = d.queues[i], pct = (100 * q) / d.queue_cap, f = $(`${p}qf-${i}`);
    f.style.width = pct + "%";
    f.className = "queue-fill" + (pct > 65 ? " high" : pct > 35 ? " mid" : "");
    $(`${p}qn-${i}`).textContent = q;
    $(`${p}cars-${i}`).innerHTML = '<span class="car"></span>'.repeat(Math.min(q, 18));
  }
  $(`${p}phase`).textContent = `Green: ${d.phase_name}` + (d.yellow ? " (yellow)" : "");
  const c = $(`${p}cluster`); c.className = `badge text-bg-${d.cluster_color}`; c.textContent = d.cluster_label;
  $(`${p}fc`).textContent = d.forecast;
  $(`${p}obs`).textContent = d.observed === null ? "no record" : d.observed;
  $(`${p}act`).textContent = d.action_name;
  $(`${p}tq`).textContent = d.total_queue;
  $(`${p}avg`).textContent = d.avg_queue;
  $(`${p}dep`).textContent = d.departed_total;
  $(`${p}bar`).style.width = Math.min(100, (100 * d.time_in_phase) / 60) + "%";
}

function renderSim(d) {
  d.intersections.forEach(renderSite);
  $("m-time").textContent = `${d.day_name} ${d.datetime}`;
  $("m-model").textContent = d.model_label;
  $("m-next").textContent = `${d.seconds_to_next_hour} s`;
  $("bar-hour").style.width = (100 * (60 - d.seconds_to_next_hour)) / 60 + "%";
  $("m-total").textContent = d.totals.total_queue;
  $("m-avg").textContent = d.totals.avg_queue;
  $("m-dep").textContent = d.totals.departed_total;
  const p = $("sim-policy");
  p.className = `badge text-bg-${d.policy === "DQN agent" ? "success" : d.trained || d.config.policy === "fixed" ? "secondary" : "warning"}`;
  p.textContent = d.policy;
}

async function tick() {
  if (busy) return;
  busy = true;
  try { renderSim(await api("/api/simulation_step")); } catch (e) { console.error(e); }
  busy = false;
}

function setRunning(on) {
  running = on;
  $("btn-start").disabled = on; $("btn-pause").disabled = !on;
  clearInterval(timer); timer = null;
  if (on) { tick(); timer = setInterval(tick, 1000); }
}

function readFilters() {
  const v = (k) => $(FILTERS[k]).value;
  return { date: v("date"), hour: parseInt(v("hour"), 10), junction: v("junction") === "all" ? "all" : parseInt(v("junction"), 10),
           policy: v("policy"), model: v("model"), source: v("source"), speed: parseInt(v("speed"), 10) };
}

function fillFilters(cfg) {
  $("f-date").value = cfg.date; $("f-hour").value = cfg.hour; $("f-junction").value = cfg.junction;
  $("f-policy").value = cfg.policy; $("f-model").value = cfg.model; $("f-source").value = cfg.source;
  $("f-speed").value = cfg.speed;
}

async function applyFilters() {
  const r = await post("/api/simulation_config", readFilters());
  fillFilters(r.config);
  buildSites(r.config);
  ["m-total", "m-avg", "m-dep"].forEach((id) => ($(id).textContent = "0"));
  if (running) tick();
}

async function initSimulation() {
  options = await api("/api/simulation_options");
  $("f-date").min = options.date_min; $("f-date").max = options.date_max;
  for (let h = 0; h < 24; h++) $("f-hour").add(new Option(String(h).padStart(2, "0") + ":00", h));
  $("f-junction").add(new Option("All 4 intersections", "all"));
  options.intersections.forEach((x) => $("f-junction").add(new Option(`${x.junction} · ${x.address}`, x.junction)));
  options.models.forEach((m) => $("f-model").add(new Option(m.label + (m.available ? "" : " (train it first)"), m.key)));
  options.speeds.forEach((s) => $("f-speed").add(new Option(`${s}x`, s)));
  $("f-date-hint").textContent = `Data available: ${options.date_min} to ${options.date_max}`;
  fillFilters(options.config);
  buildSites(options.config);

  $("btn-apply").onclick = () => applyFilters().catch(console.error);
  $("btn-start").onclick = () => setRunning(true);
  $("btn-pause").onclick = () => setRunning(false);
  $("btn-reset").onclick = async () => { setRunning(false); await post("/api/simulation_reset"); await applyFilters(); };
}

document.addEventListener("DOMContentLoaded", () => {
  if ($("sim-page")) initSimulation().catch(console.error);
});
