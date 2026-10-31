const $ = (id) => document.getElementById(id);
const DIAS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

async function api(url, opts) {
  const r = await fetch(url, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `Error ${r.status}`);
  return data;
}
const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                       body: JSON.stringify(body || {}) });
const slot = () => `hora=${$("sel-hora").value}&dia=${$("sel-dia").value}`;
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
  const dia = $("sel-dia"), hora = $("sel-hora");
  DIAS.forEach((n, i) => dia.add(new Option(n, i)));
  for (let h = 0; h < 24; h++) hora.add(new Option(String(h).padStart(2, "0") + ":00", h));
  const now = await api("/api/clusters");         
  dia.value = now.current.dia; hora.value = now.current.hora;
  dia.onchange = hora.onchange = refreshSlot;

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
  const names = { random_forest: "Random Forest", mlp: "MLP (PyTorch)" };
  $("metrics-table").innerHTML = ["random_forest", "mlp"].map((k) => {
    const x = m[k], best = m.best_model === k;
    return `<tr class="${best ? "table-success" : ""}"><td>${names[k]}${best ? " ✓" : ""}</td>
      <td class="text-end">${x.mse}</td><td class="text-end">${x.mae}</td>
      <td class="text-end">${x.r2}</td><td class="text-end">${x.train_seconds}s</td></tr>`;
  }).join("");
}

async function loadPredict() {
  const d = await api(`/api/predict?${slot()}`);
  $("pred-value").textContent = `${d.vehicles} veh/h`;
  $("pred-model").textContent = `model: ${d.model}`;
}

async function trainPrediction() {
  const btn = $("btn-train-pred");
  btn.disabled = true; $("spin-pred").classList.remove("d-none");
  setMsg($("pred-msg"), "Training Random Forest and MLP in parallel…");
  try {
    const r = await api("/api/train_prediction", { method: "POST" });
    renderMetrics(r.metrics);
    setMsg($("pred-msg"), `Done: ${r.metrics.n_train} training samples, ${r.metrics.n_test} test samples.`, "success");
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


let timer = null, busy = false;

function buildAccess(i) {
  $(`acc-${i}`).innerHTML = `<div class="name" id="n-${i}"></div>
    <div class="signal red" id="sig-${i}"><span class="lamp lamp-r"></span><span class="lamp lamp-y"></span><span class="lamp lamp-g"></span></div>
    <div class="queue-bar"><div class="queue-fill" id="qf-${i}"></div></div>
    <div class="small"><span id="qn-${i}">0</span> vehicles</div>
    <div class="cars" id="cars-${i}"></div>`;
}

function renderSim(d) {
  for (let i = 0; i < 4; i++) {
    $(`n-${i}`).textContent = d.access_names[i];
    $(`sig-${i}`).className = `signal ${d.signals[i]}`;
    const q = d.queues[i], pct = (100 * q) / d.queue_cap;
    const f = $(`qf-${i}`);
    f.style.width = pct + "%";
    f.className = "queue-fill" + (pct > 65 ? " high" : pct > 35 ? " mid" : "");
    $(`qn-${i}`).textContent = q;
    $(`cars-${i}`).innerHTML = '<span class="car"></span>'.repeat(Math.min(q, 24));
  }
  $("phase-name").textContent = `Green: ${d.phase_name}`;
  $("m-time").textContent = `${d.day_name} ${String(d.sim_hour).padStart(2, "0")}:00`;
  const c = $("m-cluster"); c.className = `badge text-bg-${d.cluster_color}`; c.textContent = d.cluster_label;
  $("m-pred").textContent = d.predicted;
  $("m-action").textContent = d.action_name;
  $("m-green").textContent = d.yellow ? `${d.phase_name} (yellow)` : d.phase_name;
  $("m-reward").textContent = d.reward;
  $("m-tip").textContent = `${d.time_in_phase} s`;
  $("bar-phase").style.width = Math.min(100, (100 * d.time_in_phase) / 60) + "%";
  $("m-next").textContent = `${d.seconds_to_next_hour} s`;
  $("bar-hour").style.width = (100 * (60 - d.seconds_to_next_hour)) / 60 + "%";
  $("m-total").textContent = d.total_queue;
  $("m-avg").textContent = d.avg_queue;
  $("m-dep").textContent = d.departed_total;
  const p = $("sim-policy"); p.className = `badge text-bg-${d.trained ? "success" : "warning"}`; p.textContent = d.policy;
}

async function tick() {
  if (busy) return;                      
  busy = true;
  try { renderSim(await api("/api/simulation_step")); } catch (e) { console.error(e); }
  busy = false;
}

function setRunning(on) {
  $("btn-start").disabled = on; $("btn-pause").disabled = !on;
  if (on) { tick(); timer = setInterval(tick, 1000); } else { clearInterval(timer); timer = null; }
}

function initSimulation() {
  for (let i = 0; i < 4; i++) buildAccess(i);
  $("btn-start").onclick = () => setRunning(true);
  $("btn-pause").onclick = () => setRunning(false);
  $("btn-reset").onclick = async () => {
    setRunning(false); await post("/api/simulation_reset");
    ["m-total", "m-avg", "m-dep"].forEach((id) => ($(id).textContent = "0"));
    for (let i = 0; i < 4; i++) { $(`qf-${i}`).style.width = "0%"; $(`qn-${i}`).textContent = "0"; $(`cars-${i}`).innerHTML = ""; $(`sig-${i}`).className = "signal red"; }
  };
}

document.addEventListener("DOMContentLoaded", () => {
  if ($("sim-page")) initSimulation();
});