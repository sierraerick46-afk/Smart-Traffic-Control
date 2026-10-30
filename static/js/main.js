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