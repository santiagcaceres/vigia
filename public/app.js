"use strict";

const TYPE_NAMES = { salida_sin_pago: "Salida sin pago", merodeo: "Merodeo", zona_restringida: "Zona restringida", fuera_de_horario: "Fuera de horario" };
const LABEL_NAMES = { hurto: "Hurto", sospechoso: "Sospechoso", normal: "Normal", falso_positivo: "Falso positivo" };
const LABEL_KEYS = { "1": "hurto", "2": "sospechoso", "3": "normal", "4": "falso_positivo" };
const ZONE_NAMES = { shelf: "Góndola", checkout: "Caja", exit: "Salida", restricted: "Restringida" };
const ZONE_COLORS = { shelf: "#ffc800", checkout: "#00c800", exit: "#0096ff", restricted: "#ff3030" };
const STATE_NAMES = { en_linea: "En línea", sin_conexion: "Sin conexión", reconectando: "Reconectando", error: "Error", iniciando: "Iniciando", detenida: "Detenida" };
const CHANNEL_NAMES = { portal: "Portal / link", telegram: "Telegram", whatsapp: "WhatsApp", email: "Email", webhook: "Webhook" };
const REPORT_STATUS = { enviado: ["ok", "Enviado"], parcial: ["warn", "Enviado en parte"], error: ["bad", "No se pudo enviar"] };

const S = { me: null, clients: [], cameras: [], lastList: [], timers: [], keyHandler: null };

// ------------------------------------------------------------------ utilidades
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString("es-AR", { dateStyle: "short", timeStyle: "medium" }) : "—");
const fmtSec = (s) => { if (s == null) return "—"; s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60; return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(x).padStart(2, "0"); };
const pct = (v) => (v == null ? "—" : Math.round(v * 100) + "%");
const isStaff = () => S.me && (S.me.role === "admin" || S.me.role === "entrenador");
const isAdmin = () => S.me && S.me.role === "admin";
const clientName = (id) => (S.clients.find((c) => c.id === id) || {}).name || "—";

async function api(method, path, body, opts = {}) {
  const init = { method, headers: {}, credentials: "same-origin" };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
  const r = await fetch(path, init);
  if (r.status === 401 && !opts.noRedirect) { showLogin(); throw new Error("Sesión vencida"); }
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : (j.detail || []).map((d) => d.msg).join("; ") || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
}

function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (err ? " err" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), 3500);
}
const fail = (e) => toast(e.message || String(e), true);

function every(ms, fn) { const id = setInterval(fn, ms); S.timers.push(id); return id; }
function onKeys(fn) { S.keyHandler = fn; }
document.addEventListener("keydown", (e) => {
  if (!S.keyHandler || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
  S.keyHandler(e);
});

const typeChip = (e) => `<span class="chip ${esc(e.severity)}">${esc(TYPE_NAMES[e.type] || e.type)}</span>`;
const labelChip = (l) => (l ? `<span class="chip label-${esc(l)}">${esc(LABEL_NAMES[l] || l)}</span>` : `<span class="chip">Sin revisar</span>`);
const options = (items, sel, empty) => (empty !== undefined ? `<option value="">${esc(empty)}</option>` : "") +
  items.map(([v, t]) => `<option value="${esc(v)}" ${String(v) === String(sel ?? "") ? "selected" : ""}>${esc(t)}</option>`).join("");

// ------------------------------------------------------------------ sesión y navegación
function showLogin() { $("#app").classList.add("hidden"); $("#login").classList.remove("hidden"); }

$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  try {
    await api("POST", "/api/login", { username: f.get("username"), password: f.get("password") }, { noRedirect: true });
    $("#login-error").textContent = "";
    await boot();
  } catch (err) { $("#login-error").textContent = err.message; }
});
$("#logout").addEventListener("click", async () => { await api("POST", "/api/logout").catch(() => {}); showLogin(); });

async function boot() {
  try { S.me = await api("GET", "/api/me", undefined, { noRedirect: true }); } catch (_) { return showLogin(); }
  S.clients = await api("GET", "/api/clients");
  $("#login").classList.add("hidden");
  $("#app").classList.remove("hidden");
  $("#user-name").textContent = `${S.me.username} · ${S.me.role}`;
  const links = S.me.role === "cliente" ? [["#/avisos", "Incidentes"], ["#/eventos", "Eventos"]] : [["#/eventos", "Eventos"]];
  if (isStaff()) links.push(["#/revisar", "Revisar y etiquetar"], ["#/avisos", "Avisos a clientes"], ["#/videos", "Videos"], ["#/entrenamiento", "Entrenamiento"]);
  links.push(["#/camaras", "Cámaras"]);
  if (isAdmin()) links.push(["#/admin", "Administración"]);
  $("#nav").innerHTML = links.map(([h, t]) => `<a href="${h}">${t}</a>`).join("");
  route();
}

window.addEventListener("hashchange", route);
async function route() {
  if (!S.me) return;
  S.timers.forEach(clearInterval); S.timers = []; S.keyHandler = null;
  const h = location.hash || "#/eventos";
  $$("#nav a").forEach((a) => a.classList.toggle("active", h.startsWith(a.getAttribute("href"))));
  const old = $("#view"), v = old.cloneNode(false); // nodo nuevo: descarta listeners de la vista anterior
  old.replaceWith(v);
  window.scrollTo(0, 0);
  let m;
  try {
    if ((m = h.match(/^#\/evento\/(\d+)/))) return await eventView(v, +m[1]);
    if (h.startsWith("#/revisar") && isStaff()) return await reviewView(v);
    if ((m = h.match(/^#\/camara\/(\d+)\/zonas/)) && isStaff()) return await zonesView(v, +m[1]);
    if (h.startsWith("#/camaras")) return await camerasView(v);
    if ((m = h.match(/^#\/video\/(\d+)/)) && isStaff()) return await importView(v, +m[1]);
    if (h.startsWith("#/videos") && isStaff()) return await importsView(v);
    if (h.startsWith("#/entrenamiento") && isStaff()) return await trainingView(v);
    if (h.startsWith("#/avisos")) return await reportsView(v);
    if (h.startsWith("#/admin") && isAdmin()) return await adminView(v);
    return await eventsView(v);
  } catch (e) { v.innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
}

async function loadCameras() { S.cameras = await api("GET", "/api/cameras"); return S.cameras; }

// ------------------------------------------------------------------ eventos
async function eventsView(v) {
  await loadCameras();
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const stats = await api("GET", "/api/stats?days=30" + (q.get("client_id") ? "&client_id=" + q.get("client_id") : ""));
  v.innerHTML = `
    <h1>Eventos</h1><p class="muted">Últimos 30 días en vivo: ${stats.total} eventos, ${stats.alerts} alertas enviadas, ${stats.by_label.sin_etiqueta || 0} sin revisar.</p>
    <div class="toolbar">
      ${S.me.role !== "cliente" ? `<label>Cliente<select id="f-client">${options(S.clients.map((c) => [c.id, c.name]), q.get("client_id"), "Todos")}</select></label>` : ""}
      <label>Cámara<select id="f-camera">${options(S.cameras.map((c) => [c.id, c.name]), q.get("camera_id"), "Todas")}</select></label>
      <label>Tipo<select id="f-type">${options(Object.entries(TYPE_NAMES), q.get("type"), "Todos")}</select></label>
      <label>Revisión<select id="f-label">${options([["sin_etiqueta", "Sin revisar"], ...Object.entries(LABEL_NAMES)], q.get("label"), "Todas")}</select></label>
      <label>Origen<select id="f-source">${options([["vivo", "En vivo"], ["importado", "Videos importados"]], q.get("source"), "Todos")}</select></label>
      <label>Alerta<select id="f-alert">${options([["true", "Solo alertas"], ["false", "Filtrados por el modelo"]], q.get("alert"), "Todos")}</select></label>
    </div>
    <div id="list" class="grid"></div>
    <div class="row" style="margin-top:14px"><span id="count" class="muted"></span><span class="spacer"></span><button id="more" class="hidden">Cargar más</button></div>`;
  const ids = ["client", "camera", "type", "label", "source", "alert"];
  ids.forEach((k) => $("#f-" + k)?.addEventListener("change", () => {
    const p = new URLSearchParams();
    ids.forEach((x) => { const el = $("#f-" + x); if (el && el.value) p.set(x === "client" ? "client_id" : x === "camera" ? "camera_id" : x, el.value); });
    location.hash = "#/eventos?" + p.toString();
  }));
  let offset = 0;
  S.lastList = [];
  const load = async () => {
    const p = new URLSearchParams(q); p.set("limit", 48); p.set("offset", offset);
    const data = await api("GET", "/api/events?" + p.toString());
    S.lastList.push(...data.items.map((e) => e.id));
    $("#list").insertAdjacentHTML("beforeend", data.items.map(eventCard).join(""));
    offset += data.items.length;
    $("#count").textContent = `${offset} de ${data.total}`;
    $("#more").classList.toggle("hidden", offset >= data.total);
    if (!data.total) $("#list").outerHTML = `<div class="empty">No hay eventos con estos filtros.</div>`;
  };
  $("#more").addEventListener("click", () => load().catch(fail));
  v.addEventListener("click", (e) => { const c = e.target.closest(".card[data-id]"); if (c) location.hash = "#/evento/" + c.dataset.id; });
  await load();
}

function eventCard(e) {
  return `<div class="card" data-id="${e.id}">
    ${e.snapshot_url ? `<img loading="lazy" src="${e.snapshot_url}" alt="">` : ""}
    <div class="body">
      <div class="row">${typeChip(e)}${labelChip(e.label)}${e.alert ? "" : `<span class="chip" title="El modelo lo consideró poco probable">filtrado</span>`}</div>
      <div class="title">${esc(e.camera_name)}${S.me.role !== "cliente" ? ` · <span class="muted">${esc(e.client_name)}</span>` : ""}</div>
      <div class="muted">${fmtDate(e.occurred_at)}${e.source === "importado" ? " · video importado" : ""}${e.score != null ? ` · puntaje ${pct(e.score)}` : ""}</div>
    </div></div>`;
}

function mediaHtml(e, autoplay) {
  if (e.clip_url) return `<div class="media"><video src="${e.clip_url}" controls playsinline ${autoplay ? "autoplay muted loop" : ""} poster="${e.snapshot_url || ""}"></video></div>`;
  if (e.snapshot_url) return `<div class="media"><img src="${e.snapshot_url}" alt=""></div><p class="muted">El clip todavía no llegó (el agente lo sube unos segundos después).</p>`;
  return `<div class="empty">Sin imagen</div>`;
}

function labelPanel(e) {
  if (!isStaff()) return e.label ? `<p>Revisión: ${labelChip(e.label)}</p>` : "";
  return `<h3>¿Qué pasó?</h3>
    <div class="labels">${Object.entries(LABEL_KEYS).map(([k, l]) => `<button data-label="${l}" class="${e.label === l ? "sel-" + l : ""}"><kbd>${k}</kbd> ${LABEL_NAMES[l]}</button>`).join("")}</div>
    <p class="muted" style="margin:8px 0">Hurto o Sospechoso = merece alerta. Normal = persona real sin problema. Falso positivo = error de detección (sombra, reflejo, no es una persona).</p>
    <label>Nota (opcional)<textarea id="note" rows="2">${esc(e.note || "")}</textarea></label>`;
}

function featuresHtml(e) {
  const f = e.features || {};
  const rows = [["Tiempo en escena", fmtSec(f.track_age_s)], ["Tiempo en góndola", fmtSec(f.shelf_total_s)], ["Tiempo en caja", fmtSec(f.checkout_total_s)],
    ["Visitas a góndola", f.shelf_visits ?? "—"], ["Personas en escena", f.people_in_scene ?? "—"]];
  return `<details><summary class="muted">Datos medidos</summary><dl class="kv" style="margin-top:8px">${rows.map(([k, x]) => `<dt>${k}</dt><dd>${esc(x)}</dd>`).join("")}</dl></details>`;
}

async function eventView(v, id) {
  const e = await api("GET", "/api/events/" + id);
  const i = S.lastList.indexOf(id);
  const prev = i > 0 ? S.lastList[i - 1] : null, next = i >= 0 && i < S.lastList.length - 1 ? S.lastList[i + 1] : null;
  v.innerHTML = `
    <div class="row"><a href="#/eventos">← Eventos</a><span class="spacer"></span>
      ${prev ? `<a class="btn" href="#/evento/${prev}">← Anterior</a>` : ""}${next ? `<a class="btn" href="#/evento/${next}">Siguiente →</a>` : ""}</div>
    <div class="two" style="margin-top:12px">
      <div>${mediaHtml(e, false)}</div>
      <div class="panel">
        <div class="row">${typeChip(e)}${labelChip(e.label)}</div>
        <h1 style="margin-top:8px">${esc(e.camera_name)}</h1>
        <p>${esc(e.description)}</p>
        <dl class="kv">
          ${S.me.role !== "cliente" ? `<dt>Cliente</dt><dd>${esc(e.client_name)}</dd>` : ""}
          <dt>Fecha</dt><dd>${fmtDate(e.occurred_at)}</dd>
          <dt>Origen</dt><dd>${e.source === "importado" ? `<a href="#/video/${e.import_id}">Video importado</a> (seg. ${fmtSec(e.video_ts)})` : "En vivo"}</dd>
          <dt>Clip</dt><dd>${e.clip_url ? ({ nvr: "Grabación del NVR", buffer: "Buffer del agente", video: "Del video importado" }[e.clip_source] || "Clip") + ` · <a href="${e.clip_url}" download>Descargar</a>` : "—"}</dd>
          ${e.score != null ? `<dt>Puntaje</dt><dd>${pct(e.score)} ${e.alert ? "" : "(no se envió alerta)"}</dd>` : ""}
        </dl>
        <div style="margin-top:14px">${labelPanel(e)}</div>
        ${isStaff() ? `<div id="report-area" style="margin-top:16px"></div>` : ""}
        <div style="margin-top:12px">${isStaff() ? featuresHtml(e) : ""}</div>
      </div>
    </div>`;
  if (!isStaff()) return;
  renderReportArea($("#report-area"), e);
  const setLabel = async (label) => {
    const r = await api("PATCH", "/api/events/" + id, { label, note: $("#note").value || null });
    toast("Guardado: " + LABEL_NAMES[r.label]);
    if (next) location.hash = "#/evento/" + next; else route();
  };
  $$(".labels button", v).forEach((b) => b.addEventListener("click", () => setLabel(b.dataset.label).catch(fail)));
  onKeys((ev) => {
    if (LABEL_KEYS[ev.key]) setLabel(LABEL_KEYS[ev.key]).catch(fail);
    else if (ev.key === "ArrowRight" && next) location.hash = "#/evento/" + next;
    else if (ev.key === "ArrowLeft" && prev) location.hash = "#/evento/" + prev;
  });
}

// ------------------------------------------------------------------ revisión rápida (entrenamiento)
async function reviewView(v) {
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  v.innerHTML = `
    <h1>Revisar y etiquetar</h1>
    <p class="muted">Cada etiqueta enseña al sistema qué alertas valen la pena. Mirá el clip y apretá <kbd>1</kbd>–<kbd>4</kbd>; <kbd>S</kbd> saltea, <kbd>R</kbd> repite el clip.</p>
    <div class="toolbar">
      <label>Cliente<select id="r-client">${options(S.clients.map((c) => [c.id, c.name]), q.get("client_id"), "Todos")}</select></label>
      <label>Origen<select id="r-source">${options([["vivo", "En vivo"], ["importado", "Videos importados"]], q.get("source"), "Todos")}</select></label>
      <span class="spacer"></span><span id="r-count" class="muted"></span>
    </div>
    <div id="r-body"></div>`;
  ["client", "source"].forEach((k) => $("#r-" + k).addEventListener("change", () => {
    const p = new URLSearchParams();
    if ($("#r-client").value) p.set("client_id", $("#r-client").value);
    if ($("#r-source").value) p.set("source", $("#r-source").value);
    location.hash = "#/revisar?" + p.toString();
  }));
  let queue = [], total = 0, done = 0;
  const skipped = new Set();
  const fill = async () => {
    const p = new URLSearchParams(q); p.set("label", "sin_etiqueta"); p.set("order", "antiguo"); p.set("limit", 100);
    const data = await api("GET", "/api/events?" + p.toString());
    total = data.total;
    queue = data.items.filter((e) => !skipped.has(e.id));
  };
  const show = () => {
    $("#r-count").textContent = `${done} etiquetados en esta sesión · quedan ${Math.max(total - done, 0)}`;
    const e = queue[0];
    if (!e) { $("#r-body").innerHTML = `<div class="empty">No quedan eventos sin revisar con estos filtros. ¡Buen trabajo!<br><br><a href="#/entrenamiento">Ir a Entrenamiento →</a></div>`; S.keyHandler = null; return; }
    $("#r-body").innerHTML = `<div class="two">
      <div>${mediaHtml(e, true)}</div>
      <div class="panel">
        <div class="row">${typeChip(e)}<span class="muted">${esc(e.client_name)} · ${esc(e.camera_name)}</span></div>
        <p>${esc(e.description)}</p><p class="muted">${fmtDate(e.occurred_at)}${e.source === "importado" ? " · video importado" : ""}</p>
        ${labelPanel(e)}
        <div class="row" style="margin-top:10px"><button id="skip"><kbd>S</kbd> Saltear</button><a class="btn" href="#/evento/${e.id}">Abrir detalle</a></div>
        <div style="margin-top:12px">${featuresHtml(e)}</div>
      </div></div>`;
    const label = async (l) => {
      await api("PATCH", "/api/events/" + e.id, { label: l, note: $("#note").value || null });
      done++; queue.shift();
      if (!queue.length) await fill(); else total = Math.max(total, done + queue.length);
      show();
    };
    const skip = () => { skipped.add(e.id); queue.shift(); total--; show(); };
    $$(".labels button").forEach((b) => b.addEventListener("click", () => label(b.dataset.label).catch(fail)));
    $("#skip").addEventListener("click", skip);
    onKeys((ev) => {
      if (LABEL_KEYS[ev.key]) label(LABEL_KEYS[ev.key]).catch(fail);
      else if (ev.key.toLowerCase() === "s") skip();
      else if (ev.key.toLowerCase() === "r") { const vid = $("#r-body video"); if (vid) { vid.currentTime = 0; vid.play(); } }
    });
  };
  await fill();
  show();
}

// ------------------------------------------------------------------ cámaras
function statusChip(st) {
  if (!st) return `<span class="chip">Sin datos del agente</span>`;
  if (!st.agent_online) return `<span class="chip bad">Agente desconectado</span>`;
  const ok = st.state === "en_linea";
  return `<span class="chip ${ok ? "ok" : "bad"}" title="${esc(st.error || "")}">${esc(STATE_NAMES[st.state] || st.state)}${ok && st.fps ? ` · ${st.fps} fps` : ""}</span>`;
}

async function camerasView(v) {
  await loadCameras();
  const agents = isStaff() ? await api("GET", "/api/agents") : [];
  v.innerHTML = `
    <div class="row"><h1>Cámaras</h1><span class="spacer"></span>${isAdmin() ? `<button id="add" class="primary">Agregar cámara</button>` : ""}</div>
    <div id="cam-form"></div>
    <div class="table-wrap panel" style="margin-top:12px;padding:0"><table>
      <thead><tr><th>Cámara</th>${S.me.role !== "cliente" ? "<th>Cliente</th><th>Conexión</th>" : ""}<th>Estado</th><th>Zonas</th><th></th></tr></thead>
      <tbody>${S.cameras.map((c) => `<tr>
        <td><b>${esc(c.name)}</b>${c.enabled ? "" : ` <span class="chip">deshabilitada</span>`}</td>
        ${S.me.role !== "cliente" ? `<td>${esc(clientName(c.client_id))}</td><td class="mono">${c.has_source_url ? "URL manual" : `${esc(c.host)} · canal ${c.channel} · ${c.detect_stream === "sub" ? "secundario" : "principal"}`}${c.detector === "motion" ? " · movimiento" : ""}</td>` : ""}
        <td>${statusChip(c.status)}</td>
        <td>${(c.zones || []).length}</td>
        <td class="row">${isStaff() ? `<a class="btn small" href="#/camara/${c.id}/zonas">Zonas y reglas</a>` : ""}${isAdmin() ? `<button class="small" data-edit="${c.id}">Editar</button>` : ""}</td>
      </tr>`).join("") || `<tr><td colspan="6" class="muted">Todavía no hay cámaras.</td></tr>`}</tbody></table></div>`;
  if (!isAdmin()) return;
  const form = (c = {}) => {
    $("#cam-form").innerHTML = `<div class="panel" style="margin-top:12px">
      <h3>${c.id ? "Editar cámara" : "Nueva cámara"}</h3>
      <p class="muted">Datos del NVR o cámara Hikvision (los mismos que usa iVMS-4200). Si la cámara está en un NVR, la IP es la del NVR y el canal es el número de cámara en el NVR.</p>
      <form class="form" id="cf">
        <label>Cliente<select name="client_id" required>${options(S.clients.map((x) => [x.id, x.name]), c.client_id)}</select></label>
        <label>Agente<select name="agent_id">${options(agents.map((a) => [a.id, `${a.name} (${clientName(a.client_id)})`]), c.agent_id, "Cualquiera del cliente")}</select></label>
        <label>Nombre<input name="name" value="${esc(c.name || "")}" required placeholder="Salón principal"></label>
        <label>IP del NVR o cámara<input name="host" value="${esc(c.host || "")}" placeholder="192.168.1.64"></label>
        <label>Canal<input name="channel" type="number" min="1" value="${c.channel || 1}"></label>
        <label>Usuario<input name="username" value="${esc(c.username || "admin")}"></label>
        <label>Clave<input name="password" type="password" placeholder="${c.has_password ? "(sin cambios)" : ""}" autocomplete="new-password"></label>
        <label>Puerto RTSP<input name="rtsp_port" type="number" value="${c.rtsp_port || 554}"></label>
        <label>Puerto HTTP<input name="http_port" type="number" value="${c.http_port || 80}"></label>
        <label>Stream para analizar<select name="detect_stream">${options([["sub", "Secundario (recomendado)"], ["main", "Principal"]], c.detect_stream || "sub")}</select></label>
        <label>Detector<select name="detector">${options([["yolo", "Personas (YOLO)"], ["motion", "Movimiento (solo pruebas)"]], c.detector || "yolo")}</select></label>
        <label>Cuadros por segundo a analizar<input name="process_fps" type="number" step="0.5" min="1" max="15" value="${c.process_fps || 5}"></label>
        <label>Hora de grabación del NVR<select name="playback_time">${options([["local", "Hora local (habitual)"], ["utc", "UTC"]], c.playback_time || "local")}</select></label>
        <label>URL RTSP manual (opcional)<input name="source_url" placeholder="Solo para otras marcas"></label>
        <label class="inline"><input type="checkbox" name="nvr_clips" ${c.nvr_clips !== false ? "checked" : ""}> Bajar clip de la grabación del NVR</label>
        <label class="inline"><input type="checkbox" name="enabled" ${c.enabled !== false ? "checked" : ""}> Habilitada</label>
        <div class="row"><button class="primary" type="submit">Guardar</button><button type="button" id="cancel">Cancelar</button></div>
      </form></div>`;
    $("#cancel").addEventListener("click", () => ($("#cam-form").innerHTML = ""));
    $("#cf").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const f = new FormData(ev.target);
      const body = {
        client_id: +f.get("client_id"), agent_id: f.get("agent_id") ? +f.get("agent_id") : null, name: f.get("name"),
        host: f.get("host"), channel: +f.get("channel"), username: f.get("username"), password: f.get("password") || null,
        rtsp_port: +f.get("rtsp_port"), http_port: +f.get("http_port"), detect_stream: f.get("detect_stream"),
        detector: f.get("detector"), process_fps: +f.get("process_fps"), playback_time: f.get("playback_time"),
        source_url: f.get("source_url") || null, nvr_clips: !!f.get("nvr_clips"), enabled: !!f.get("enabled"),
      };
      try {
        await api(c.id ? "PUT" : "POST", c.id ? "/api/cameras/" + c.id : "/api/cameras", body);
        toast("Cámara guardada. El agente la toma en menos de un minuto."); route();
      } catch (e) { fail(e); }
    });
  };
  $("#add").addEventListener("click", () => { if (!S.clients.length) return toast("Primero creá un cliente en Administración", true); form(); });
  $$("[data-edit]").forEach((b) => b.addEventListener("click", () => form(S.cameras.find((c) => c.id === +b.dataset.edit))));
}

// ------------------------------------------------------------------ editor de zonas
async function zonesView(v, id) {
  const cam = await api("GET", "/api/cameras/" + id);
  let zones = JSON.parse(JSON.stringify(cam.zones || []));
  const rules = cam.rules || {};
  let draft = [], selected = -1;
  const enabled = new Set(rules.enabled_rules || Object.keys(TYPE_NAMES));
  v.innerHTML = `
    <div class="row"><a href="#/camaras">← Cámaras</a></div>
    <h1>${esc(cam.name)} · zonas y reglas</h1>
    <p class="muted">Hacé clic sobre la imagen para marcar los vértices de cada zona; doble clic o "Cerrar zona" para terminarla. Se evalúan los <b>pies</b> de la persona, así que dibujá las zonas sobre el piso.</p>
    <div class="two">
      <div>
        <div class="zone-editor"><canvas id="cv"></canvas></div>
        <div class="row" style="margin-top:8px">
          <span class="muted" id="bg-info"></span><span class="spacer"></span>
          <label class="btn small">Usar otra imagen<input type="file" id="bg-file" accept="image/*" hidden></label>
        </div>
      </div>
      <div class="panel">
        <h3>Nueva zona</h3>
        <div class="form" style="grid-template-columns:1fr 1fr">
          <label>Nombre<input id="z-name" placeholder="gondola_bebidas"></label>
          <label>Tipo<select id="z-type">${options(Object.entries(ZONE_NAMES), "shelf")}</select></label>
        </div>
        <div class="row" style="margin:8px 0 14px"><button id="z-close" class="small">Cerrar zona</button><button id="z-undo" class="small">Deshacer punto</button><span class="muted" id="z-draft"></span></div>
        <h3>Zonas</h3><div id="z-list"></div>
        <h3 style="margin-top:16px">Reglas</h3>
        <div class="form" style="grid-template-columns:1fr 1fr">
          <label>Seg. en góndola para "tomó algo"<input id="r-shelf" type="number" step="0.5" min="0" value="${rules.shelf_dwell_seconds ?? 3}"></label>
          <label>Seg. en caja para "pagó"<input id="r-checkout" type="number" step="0.5" min="0" value="${rules.checkout_min_seconds ?? 1}"></label>
          <label>Seg. para merodeo<input id="r-loiter" type="number" step="1" min="0" value="${rules.loiter_seconds ?? 45}"></label>
          <label>Seg. en zona restringida<input id="r-restricted" type="number" step="0.5" min="0" value="${rules.restricted_min_seconds ?? 1}"></label>
          <label>Abre (opcional)<input id="r-open" type="time" value="${rules.opening_hours?.start || ""}"></label>
          <label>Cierra<input id="r-close" type="time" value="${rules.opening_hours?.end || ""}"></label>
        </div>
        <div style="margin-top:8px">${Object.entries(TYPE_NAMES).map(([k, t]) => `<label class="inline" style="margin-right:12px"><input type="checkbox" class="r-en" value="${k}" ${enabled.has(k) ? "checked" : ""}> ${t}</label>`).join("")}</div>
        <div class="row" style="margin-top:14px"><button id="save" class="primary">Guardar</button></div>
      </div>
    </div>`;
  const cv = $("#cv"), ctx = cv.getContext("2d");
  const img = new Image();
  let hasImg = false;
  img.onload = () => { hasImg = true; cv.width = img.naturalWidth; cv.height = img.naturalHeight; draw(); };
  img.onerror = () => { cv.width = 1280; cv.height = 720; $("#bg-info").textContent = "Sin vista previa: el agente la envía al conectarse. También podés subir una captura de iVMS."; draw(); };
  if (cam.has_preview) { img.src = `/api/cameras/${id}/preview?t=${Date.now()}`; $("#bg-info").textContent = "Vista previa del " + fmtDate(cam.preview_at); }
  else img.onerror();
  $("#bg-file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) { img.src = URL.createObjectURL(f); $("#bg-info").textContent = "Imagen local: " + f.name; } });

  function draw() {
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    if (hasImg) ctx.drawImage(img, 0, 0, W, H); else { ctx.fillStyle = "#111"; ctx.fillRect(0, 0, W, H); }
    const lw = Math.max(2, W / 400);
    zones.forEach((z, i) => {
      const col = ZONE_COLORS[z.type] || "#fff";
      ctx.beginPath(); z.polygon.forEach(([x, y], k) => (k ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H))); ctx.closePath();
      ctx.fillStyle = col + (i === selected ? "66" : "33"); ctx.fill(); ctx.lineWidth = lw * (i === selected ? 2 : 1); ctx.strokeStyle = col; ctx.stroke();
      const [lx, ly] = z.polygon[0]; ctx.font = `${Math.round(W / 50)}px system-ui`; ctx.fillStyle = col; ctx.fillText(z.name, lx * W + 6, ly * H + W / 45);
    });
    if (draft.length) {
      const col = ZONE_COLORS[$("#z-type").value];
      ctx.beginPath(); draft.forEach(([x, y], k) => (k ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash([lw * 3, lw * 2]); ctx.stroke(); ctx.setLineDash([]);
      draft.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x * W, y * H, lw * 2.5, 0, 7); ctx.fillStyle = col; ctx.fill(); });
    }
    $("#z-draft").textContent = draft.length ? `${draft.length} puntos` : "";
    $("#z-list").innerHTML = zones.map((z, i) => `<div class="zone-item" data-i="${i}" style="${i === selected ? "border-color:var(--muted)" : ""}">
      <span class="swatch" style="background:${ZONE_COLORS[z.type]}"></span><b>${esc(z.name)}</b><span class="muted">${ZONE_NAMES[z.type]}</span><span class="spacer"></span>
      <button class="small ghost" data-del="${i}">Borrar</button></div>`).join("") || `<p class="muted">Sin zonas. Para "salida sin pago" hacen falta al menos una góndola, una caja y una salida.</p>`;
  }
  const pt = (e) => { const r = cv.getBoundingClientRect(); return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))].map((x) => Math.round(x * 1000) / 1000); };
  const close = () => {
    if (draft.length < 3) return toast("Una zona necesita al menos 3 puntos", true);
    const name = ($("#z-name").value || "").trim() || `${$("#z-type").value}_${zones.length + 1}`;
    if (zones.some((z) => z.name === name)) return toast("Ya hay una zona con ese nombre", true);
    zones.push({ name, type: $("#z-type").value, polygon: draft }); draft = []; $("#z-name").value = ""; selected = zones.length - 1; draw();
  };
  cv.addEventListener("click", (e) => { draft.push(pt(e)); draw(); });
  cv.addEventListener("dblclick", (e) => { e.preventDefault(); draft.pop(); close(); });
  $("#z-close").addEventListener("click", close);
  $("#z-undo").addEventListener("click", () => { draft.pop(); draw(); });
  $("#z-type").addEventListener("change", draw);
  $("#z-list").addEventListener("click", (e) => {
    const d = e.target.closest("[data-del]"); if (d) { zones.splice(+d.dataset.del, 1); selected = -1; return draw(); }
    const it = e.target.closest("[data-i]"); if (it) { selected = +it.dataset.i; draw(); }
  });
  onKeys((e) => { if (e.key === "Escape") { draft = []; draw(); } if (e.key === "Enter") close(); });
  $("#save").addEventListener("click", async () => {
    if (draft.length) return toast("Tenés una zona sin cerrar", true);
    const r = { shelf_dwell_seconds: +$("#r-shelf").value, checkout_min_seconds: +$("#r-checkout").value,
      loiter_seconds: +$("#r-loiter").value, restricted_min_seconds: +$("#r-restricted").value,
      enabled_rules: $$(".r-en").filter((x) => x.checked).map((x) => x.value) };
    if ($("#r-open").value && $("#r-close").value) r.opening_hours = { start: $("#r-open").value, end: $("#r-close").value };
    try { await api("PUT", `/api/cameras/${id}/analysis`, { zones, rules: r }); toast("Guardado. El agente aplica los cambios en menos de un minuto."); }
    catch (e) { fail(e); }
  });
}

// ------------------------------------------------------------------ videos importados
async function importsView(v) {
  await loadCameras();
  v.innerHTML = `
    <h1>Videos</h1>
    <p class="muted">Subí grabaciones exportadas desde iVMS-4200 (formato MP4). El agente del cliente las analiza con las zonas de la cámara y genera eventos para etiquetar. Si marcás dónde hubo hurtos reales, te dice cuántos detectó.</p>
    <div class="panel"><form class="form" id="up">
      <label>Cámara<select name="camera_id" required>${options(S.cameras.map((c) => [c.id, `${clientName(c.client_id)} · ${c.name}`]), "")}</select></label>
      <label>Inicio de la grabación (hora local)<input name="recorded_at" type="datetime-local" step="1"></label>
      <label>Archivo<input name="file" type="file" accept="video/*,.mp4,.mkv,.avi,.mov,.ts" required></label>
      <button class="primary" type="submit">Subir y analizar</button>
    </form><div id="up-progress" class="hidden" style="margin-top:10px"><div class="bar"><i style="width:0"></i></div><span class="muted"></span></div></div>
    <div class="table-wrap panel" style="margin-top:14px;padding:0"><table><thead><tr><th>Video</th><th>Cámara</th><th>Duración</th><th>Estado</th><th>Subido</th></tr></thead><tbody id="imp-rows"></tbody></table></div>`;
  const render = async () => {
    const rows = await api("GET", "/api/imports");
    $("#imp-rows").innerHTML = rows.map((i) => `<tr>
      <td><a href="#/video/${i.id}">${esc(i.filename)}</a></td><td>${esc(clientName(i.client_id))} · ${esc(i.camera_name)}</td><td>${fmtSec(i.duration_s)}</td>
      <td>${i.status === "procesando" ? `<div class="bar" style="width:120px"><i style="width:${Math.round(i.progress * 100)}%"></i></div>` : esc({ subiendo: "Subida incompleta", en_cola: i.agent_available ? "En cola" : "Esperando que un agente del cliente esté en línea", listo: "Listo", error: "Error: " + (i.error || "") }[i.status] || i.status)}</td>
      <td>${fmtDate(i.created_at)}</td></tr>`).join("") || `<tr><td colspan="5" class="muted">Todavía no subiste videos.</td></tr>`;
    return rows.some((i) => i.status === "en_cola" || i.status === "procesando");
  };
  await render();
  every(3000, () => render().catch(() => {}));
  $("#up").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target, fd = new FormData(form), file = fd.get("file");
    const btn = $("button[type=submit]", form);
    const box = $("#up-progress");
    const setBar = (p, txt) => { $("i", box).style.width = p * 100 + "%"; $("span", box).textContent = txt; };
    let imp;
    try {
      btn.disabled = true;
      imp = await api("POST", "/api/imports", { camera_id: +fd.get("camera_id"), filename: file.name, size: file.size,
        recorded_at: fd.get("recorded_at") || null });
      box.classList.remove("hidden"); setBar(0, "Subiendo 0%");
      // el archivo va directo al almacenamiento (Supabase), sin pasar por el servidor
      await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) setBar(ev.loaded / ev.total, `Subiendo ${Math.round((ev.loaded / ev.total) * 100)}%`); };
        xhr.onload = () => (xhr.status < 300 ? resolve() : reject(new Error(xhr.status === 413 ? "El archivo supera el límite de tamaño del almacenamiento" : `La subida falló (${xhr.status})`)));
        xhr.onerror = () => reject(new Error("Se cortó la subida"));
        xhr.open("PUT", imp.upload_url);
        xhr.setRequestHeader("Content-Type", file.type || "video/mp4");
        xhr.send(file);
      });
      await api("POST", `/api/imports/${imp.id}/uploaded`);
      toast("Video subido. Un agente del cliente lo va a analizar.");
      form.reset(); render();
    } catch (err) {
      fail(err);
      if (imp) api("DELETE", `/api/imports/${imp.id}`).catch(() => {});
    } finally { btn.disabled = false; box.classList.add("hidden"); }
  });
}

async function importView(v, id) {
  const imp = await api("GET", "/api/imports/" + id);
  const evs = (await api("GET", `/api/events?import_id=${id}&order=antiguo&limit=500`)).items;
  S.lastList = evs.map((e) => e.id);
  v.innerHTML = `
    <div class="row"><a href="#/videos">← Videos</a></div>
    <h1>${esc(imp.filename)}</h1>
    <p class="muted">${esc(imp.camera_name)} · ${fmtSec(imp.duration_s)} · ${esc(imp.status)}${imp.error ? " · " + esc(imp.error) : ""}</p>
    <div class="two">
      <div>
        <div class="media"><video id="vid" src="${imp.video_url}" controls preload="metadata"></video></div>
        <p class="muted">Si el video no se reproduce en el navegador (algunas exportaciones de iVMS usan H.265), el análisis igual funciona; marcá los tiempos a mano.</p>
        <h2>Eventos detectados (${evs.length})</h2>
        <div class="grid">${evs.map((e) => `<div class="card" data-id="${e.id}" data-ts="${e.video_ts}">${e.snapshot_url ? `<img loading="lazy" src="${e.snapshot_url}">` : ""}
          <div class="body"><div class="row">${typeChip(e)}${labelChip(e.label)}</div><div class="muted">Segundo ${fmtSec(e.video_ts)} · <a href="#/evento/${e.id}">abrir</a></div></div></div>`).join("") || `<div class="empty">Sin eventos.</div>`}</div>
      </div>
      <div>
        <div class="panel">
          <h3>Hurtos reales en este video</h3>
          <p class="muted">Marcá cada hurto que viste para medir el sistema. Podés tomar los tiempos del reproductor.</p>
          <div class="form" style="grid-template-columns:1fr 1fr">
            <label>Inicio (m:ss)<input id="i-start" placeholder="1:20"></label><label>Fin (m:ss)<input id="i-end" placeholder="1:45"></label>
          </div>
          <div class="row" style="margin:8px 0"><button class="small" id="from-start">Inicio = ahora</button><button class="small" id="from-end">Fin = ahora</button></div>
          <label>Nota<input id="i-note" placeholder="Guarda producto en mochila"></label>
          <div class="row" style="margin-top:8px"><button class="primary small" id="i-add">Agregar</button></div>
          <div id="i-list" style="margin-top:12px"></div>
        </div>
        <div class="panel" style="margin-top:12px"><h3>Resultado</h3><div id="eval" class="muted">—</div></div>
        ${imp.status === "listo" || imp.status === "error" ? `<div class="row" style="margin-top:12px"><button id="reprocess" class="small">Volver a analizar (después de cambiar zonas)</button></div>` : ""}
      </div>
    </div>`;
  const vid = $("#vid");
  const parse = (s) => { if (!s) return NaN; const p = s.trim().split(":").map(Number); return p.reduce((a, x) => a * 60 + x, 0); };
  $("#from-start").addEventListener("click", () => ($("#i-start").value = fmtSec(vid.currentTime)));
  $("#from-end").addEventListener("click", () => ($("#i-end").value = fmtSec(vid.currentTime)));
  const renderIncidents = async () => {
    const d = await api("GET", "/api/imports/" + id);
    const ev = await api("GET", `/api/imports/${id}/evaluation`);
    const byId = Object.fromEntries(ev.detail.map((r) => [r.id, r]));
    $("#i-list").innerHTML = d.incidents.map((x) => `<div class="zone-item"><span>${fmtSec(x.start_s)}–${fmtSec(x.end_s)}</span>
      ${byId[x.id]?.detected ? `<span class="chip ok">detectado</span>` : `<span class="chip bad">no detectado</span>`}<span class="muted">${esc(x.note || "")}</span><span class="spacer"></span>
      <button class="small ghost" data-seek="${x.start_s}">Ver</button><button class="small ghost" data-del="${x.id}">Borrar</button></div>`).join("") || `<p class="muted">Sin hurtos marcados.</p>`;
    $("#eval").innerHTML = ev.incidents ? `<dl class="kv">
      <dt>Hurtos detectados</dt><dd><b>${ev.detected} de ${ev.incidents}</b> (${pct(ev.detected / ev.incidents)})</dd>
      <dt>Con alerta</dt><dd>${ev.alerted} de ${ev.incidents}</dd>
      <dt>Falsas alarmas</dt><dd>${ev.false_alarms}${ev.false_alarms_per_hour != null ? ` (${ev.false_alarms_per_hour} por hora de video)` : ""}</dd></dl>`
      : `Marcá al menos un hurto real para medir. Eventos detectados: ${ev.events}.`;
  };
  $("#i-add").addEventListener("click", async () => {
    const start = parse($("#i-start").value), end = parse($("#i-end").value);
    if (isNaN(start) || isNaN(end)) return toast("Completá inicio y fin (m:ss)", true);
    try { await api("POST", `/api/imports/${id}/incidents`, { start_s: start, end_s: end, note: $("#i-note").value || null }); $("#i-start").value = $("#i-end").value = $("#i-note").value = ""; renderIncidents(); }
    catch (e) { fail(e); }
  });
  $("#i-list").addEventListener("click", async (e) => {
    const s = e.target.closest("[data-seek]"); if (s) { vid.currentTime = +s.dataset.seek; vid.play(); return; }
    const d = e.target.closest("[data-del]"); if (d) { await api("DELETE", `/api/imports/${id}/incidents/${d.dataset.del}`).catch(fail); renderIncidents(); }
  });
  v.querySelector(".grid")?.addEventListener("click", (e) => {
    if (e.target.closest("a")) return;
    const c = e.target.closest("[data-ts]"); if (c) { vid.currentTime = Math.max(0, +c.dataset.ts - 5); vid.play(); vid.scrollIntoView({ behavior: "smooth" }); }
  });
  $("#reprocess")?.addEventListener("click", async () => {
    if (!confirm("Se borran los eventos de este video (y sus etiquetas) y se vuelve a analizar. ¿Seguir?")) return;
    await api("POST", `/api/imports/${id}/reprocess`).catch(fail); location.hash = "#/videos";
  });
  await renderIncidents();
  if (imp.status === "en_cola" || imp.status === "procesando") every(3000, async () => { const d = await api("GET", "/api/imports/" + id); if (d.status !== imp.status || d.status === "listo") route(); });
}

// ------------------------------------------------------------------ entrenamiento
async function trainingView(v) {
  const [sum, models] = await Promise.all([api("GET", "/api/training/summary"), api("GET", "/api/models")]);
  const ready = (c) => { const l = c.labels; const pos = (l.hurto || 0) + (l.sospechoso || 0), neg = (l.normal || 0) + (l.falso_positivo || 0); return [pos, neg, pos + neg >= sum.min_total && pos >= sum.min_per_class && neg >= sum.min_per_class]; };
  v.innerHTML = `
    <h1>Entrenamiento</h1>
    <p class="muted">El sistema aprende de tus etiquetas qué eventos merecen una alerta. Con el modelo activo, los eventos poco probables se siguen guardando pero no generan alerta.
    Se necesitan al menos ${sum.min_total} etiquetas, con ${sum.min_per_class} de cada tipo (alerta real y sin importancia).</p>
    <div class="table-wrap panel" style="padding:0"><table><thead><tr><th>Cliente</th><th>Hurto</th><th>Sospechoso</th><th>Normal</th><th>Falso positivo</th><th>Sin revisar</th><th>Listo para entrenar</th><th></th></tr></thead>
      <tbody>${sum.clients.map((c) => { const [p, n, ok] = ready(c); return `<tr><td>${esc(c.client_name)}</td>
        ${["hurto", "sospechoso", "normal", "falso_positivo", "sin_etiqueta"].map((k) => `<td>${c.labels[k] || 0}</td>`).join("")}
        <td>${ok ? `<span class="chip ok">Sí</span>` : `<span class="chip">${p} reales / ${n} sin importancia</span>`}</td>
        <td><a class="btn small" href="#/revisar?client_id=${c.client_id}">Etiquetar</a></td></tr>`; }).join("") || `<tr><td colspan="8" class="muted">Todavía no hay eventos.</td></tr>`}</tbody></table></div>
    <div class="panel" style="margin-top:14px">
      <h3>Entrenar un modelo</h3>
      <form class="form" id="tf">
        <label>Alcance<select name="client_id">${options(S.clients.map((c) => [c.id, "Solo " + c.name]), "", "Todos los clientes (global)")}</select></label>
        <label>No perder más del…<select name="target_recall">${options([["0.95", "5% de los hurtos"], ["0.9", "10% de los hurtos"], ["0.8", "20% de los hurtos"]], "0.9")}</select></label>
        <button class="primary" type="submit">Entrenar</button>
      </form>
      <p class="muted">El modelo global aprovecha todas las etiquetas y sirve para clientes nuevos. Uno por cliente se ajusta mejor cuando ese cliente tiene muchas etiquetas propias.</p>
      <div id="tres"></div>
    </div>
    <h2>Modelos</h2>
    <div class="table-wrap panel" style="padding:0"><table><thead><tr><th>#</th><th>Alcance</th><th>Fecha</th><th>Etiquetas</th><th>Hurtos que sigue avisando</th><th>Aciertos de las alertas</th><th>Falsas alarmas eliminadas</th><th>Validación</th><th></th></tr></thead>
      <tbody>${models.map((m) => `<tr><td>${m.id}</td><td>${m.scope === "global" ? "Global" : esc(m.client_name)}</td><td>${fmtDate(m.created_at)}</td><td>${m.n_samples}</td>
        <td>${pct(m.metrics.recall)}</td><td>${pct(m.metrics.precision)} <span class="muted">(sin modelo ${pct(m.metrics.precision_sin_modelo)})</span></td>
        <td>${pct(m.metrics.falsas_alarmas_eliminadas)}</td><td class="muted">${m.metrics.validation === "por_camara" ? "por cámara" : "aleatoria"}</td>
        <td>${m.active ? `<span class="chip ok">Activo</span> <button class="small ghost" data-off="${m.id}">Desactivar</button>` : `<button class="small" data-on="${m.id}">Activar</button>`}</td></tr>`).join("") || `<tr><td colspan="9" class="muted">Sin modelos todavía.</td></tr>`}</tbody></table></div>
    <p><a href="/api/training/export.csv">Descargar etiquetas (CSV)</a></p>`;
  $("#tf").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    try {
      const m = await api("POST", "/api/training/run", { client_id: f.get("client_id") ? +f.get("client_id") : null, target_recall: +f.get("target_recall") });
      const x = m.metrics;
      $("#tres").innerHTML = `<div class="panel" style="margin-top:10px"><b>Modelo #${m.id} entrenado con ${x.n} etiquetas.</b>
        <p>Probado sobre datos que no vio${x.validation === "por_camara" ? " (cámaras separadas)" : ""}: sigue avisando el <b>${pct(x.recall)}</b> de los casos reales
        (${x.reales_perdidas} se habrían perdido) y elimina el <b>${pct(x.falsas_alarmas_eliminadas)}</b> de las falsas alarmas.
        De cada 10 alertas, ${Math.round(x.precision * 10)} serían reales (hoy: ${Math.round(x.precision_sin_modelo * 10)}).</p>
        <button class="primary small" data-on="${m.id}">Activar este modelo</button></div>`;
    } catch (err) { fail(err); }
  });
  v.addEventListener("click", async (e) => {
    const on = e.target.closest("[data-on]"), off = e.target.closest("[data-off]");
    if (on) { await api("POST", `/api/models/${on.dataset.on}/activate`).catch(fail); toast("Modelo activo: los agentes lo toman en menos de un minuto"); route(); }
    if (off) { await api("POST", `/api/models/${off.dataset.off}/deactivate`).catch(fail); route(); }
  });
}

// ------------------------------------------------------------------ avisos al cliente
function defaultReportMessage(e) {
  return `Detectamos una situación sospechosa (posible hurto) en la cámara «${e.camera_name}» el ${fmtDate(e.occurred_at)}. Te compartimos el clip para que lo revises.`;
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast("Link copiado"); }
  catch (_) { const t = document.createElement("textarea"); t.value = text; document.body.appendChild(t); t.select(); document.execCommand("copy"); t.remove(); toast("Link copiado"); }
}

function deliveryHtml(rep) {
  const parts = Object.entries(rep.delivery || {}).map(([ch, r]) =>
    `<span class="chip ${r.ok ? "ok" : "bad"}" title="${esc(r.detail || "")}">${esc(CHANNEL_NAMES[ch] || ch)}</span>`);
  return `<div class="row" style="flex-wrap:wrap;gap:6px">${parts.join("")}</div>
    <div class="row" style="margin-top:8px"><a class="btn small" href="${esc(rep.link)}" target="_blank" rel="noopener">Abrir el link</a>
      <button class="small" data-copy="${esc(rep.link)}">Copiar link</button></div>`;
}

function renderReportArea(area, e) {
  if (!area) return;
  if (!e.clip_url) { area.innerHTML = `<p class="muted">Cuando llegue el clip vas a poder avisarle al cliente desde acá.</p>`; return; }
  const client = S.clients.find((c) => c.id === e.client_id) || {};
  const channels = client.notify_channels || ["portal"];
  area.innerHTML = `<button id="open-report" class="primary">Avisar al cliente</button>
    <span class="muted" style="margin-left:8px">Le manda el clip al dueño del local.</span>`;
  $("#open-report", area).addEventListener("click", () => {
    area.innerHTML = `<div class="panel">
      <h3>Avisar al cliente · ${esc(client.name || "")}</h3>
      <p class="muted">Se genera un link al clip que vence en 14 días. Revisá el texto: describí lo que se ve, sin acusar a una persona concreta.</p>
      <label>Mensaje<textarea id="rep-msg" rows="3">${esc(defaultReportMessage(e))}</textarea></label>
      <div style="margin:8px 0"><span class="muted">Enviar por:</span><br>
        ${channels.map((ch) => `<label class="inline" style="margin-right:12px"><input type="checkbox" class="rep-ch" value="${ch}" checked ${ch === "portal" ? "disabled" : ""}> ${esc(CHANNEL_NAMES[ch] || ch)}</label>`).join("")}
        ${channels.length <= 1 ? `<br><span class="muted">Este cliente solo tiene el link/portal. Agregá Telegram, WhatsApp o email en Administración para enviarle directo.</span>` : ""}
      </div>
      <div class="row"><button id="rep-send" class="primary">Enviar aviso</button><button id="rep-cancel">Cancelar</button></div>
      <div id="rep-result" style="margin-top:12px"></div>
    </div>`;
    $("#rep-cancel", area).addEventListener("click", () => renderReportArea(area, e));
    $("#rep-send", area).addEventListener("click", async () => {
      const btn = $("#rep-send", area); btn.disabled = true;
      const chans = $$(".rep-ch", area).filter((x) => x.checked).map((x) => x.value);
      try {
        const rep = await api("POST", `/api/events/${e.id}/report`, { message: $("#rep-msg", area).value, channels: chans });
        const [cls, txt] = REPORT_STATUS[rep.status] || ["", rep.status];
        $("#rep-result", area).innerHTML = `<div class="row"><span class="chip ${cls}">${esc(txt)}</span></div>
          <div style="margin-top:8px">${deliveryHtml(rep)}</div>
          <p class="muted" style="margin-top:8px">Queda registrado en <a href="#/avisos">Avisos a clientes</a>.</p>`;
        $$("[data-copy]", area).forEach((b) => b.addEventListener("click", () => copyText(b.dataset.copy)));
      } catch (err) { fail(err); btn.disabled = false; }
    });
  });
}

async function reportsView(v) {
  const forClient = S.me.role === "cliente";
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  v.innerHTML = `
    <h1>${forClient ? "Incidentes" : "Avisos a clientes"}</h1>
    <p class="muted">${forClient
      ? "Clips que te enviamos cuando detectamos una situación sospechosa en tus cámaras."
      : "Cada aviso le manda al cliente un link al clip. El texto por defecto dice «posible hurto», no una acusación."}</p>
    ${forClient ? "" : `<div class="toolbar"><label>Cliente<select id="rp-client">${options(S.clients.map((c) => [c.id, c.name]), q.get("client_id"), "Todos")}</select></label></div>`}
    <div id="rp-body"></div>`;
  if (!forClient) $("#rp-client").addEventListener("change", () => {
    const val = $("#rp-client").value;
    location.hash = "#/avisos" + (val ? "?client_id=" + val : "");
  });
  const p = new URLSearchParams(); if (q.get("client_id")) p.set("client_id", q.get("client_id")); p.set("limit", 100);
  const data = await api("GET", "/api/reports?" + p.toString());
  if (!data.items.length) { $("#rp-body").innerHTML = `<div class="empty">${forClient ? "Todavía no recibiste avisos." : "Todavía no enviaste avisos."}</div>`; return; }
  if (forClient) {
    $("#rp-body").innerHTML = `<div class="grid">${data.items.map((r) => `<div class="card">
      <div class="body"><div class="row"><span class="chip bad">Posible hurto</span></div>
        <p>${esc(r.message)}</p>
        <div class="muted">${fmtDate(r.occurred_at || r.sent_at)}</div>
        <div class="row" style="margin-top:8px">${r.has_clip ? `<a class="btn small" href="${esc(r.link)}" target="_blank" rel="noopener">Ver el clip</a>` : `<span class="muted">Clip no disponible</span>`}</div>
      </div></div>`).join("")}</div>`;
    return;
  }
  $("#rp-body").innerHTML = `<div class="table-wrap panel" style="padding:0"><table>
    <thead><tr><th>Cliente</th><th>Cámara</th><th>Enviado</th><th>Estado</th><th>Visto</th><th>Vence</th><th></th></tr></thead>
    <tbody>${data.items.map((r) => { const [cls, txt] = REPORT_STATUS[r.status] || ["", r.status]; return `<tr>
      <td>${esc(r.client_name)}</td><td>${esc(r.camera_name)}</td><td>${fmtDate(r.sent_at)}</td>
      <td><span class="chip ${cls}">${esc(txt)}</span></td>
      <td>${r.viewed_at ? `<span class="chip ok">Sí</span>` : "No"}</td><td>${fmtDate(r.expires_at)}</td>
      <td class="row"><a class="btn small" href="#/evento/${r.event_id}">Evento</a>
        <button class="small" data-copy="${esc(r.link)}">Copiar link</button>
        <button class="small ghost" data-revoke="${r.id}">Revocar</button></td></tr>`; }).join("")}</tbody></table></div>`;
  $$("[data-copy]", v).forEach((b) => b.addEventListener("click", () => copyText(b.dataset.copy)));
  $$("[data-revoke]", v).forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("El link deja de funcionar para el cliente. ¿Revocar este aviso?")) return;
    try { await api("DELETE", "/api/reports/" + b.dataset.revoke); toast("Aviso revocado"); route(); } catch (e) { fail(e); }
  }));
}

// ------------------------------------------------------------------ administración
async function adminView(v) {
  const [users, agents] = await Promise.all([api("GET", "/api/users"), api("GET", "/api/agents")]);
  S.clients = await api("GET", "/api/clients");
  const roleNames = { admin: "Administrador", entrenador: "Entrenador", cliente: "Cliente" };
  v.innerHTML = `
    <h1>Administración</h1>
    <h2>Clientes</h2>
    <p class="muted">El contacto del cliente se usa solo cuando un empleado decide avisarle de un hurto desde el detalle del evento.</p>
    <div class="table-wrap panel" style="padding:0"><table><thead><tr><th>Nombre</th><th>Zona horaria</th><th>WhatsApp</th><th>Email</th><th>Telegram</th><th>Canales de aviso</th></tr></thead>
      <tbody>${S.clients.map((c) => `<tr><td>${esc(c.name)}</td><td>${esc(c.timezone)}</td><td class="mono">${esc(c.contact_whatsapp || "—")}</td><td class="mono">${esc(c.contact_email || "—")}</td><td class="mono">${esc(c.telegram_chat_id || "—")}</td><td>${(c.notify_channels || []).map((x) => esc(CHANNEL_NAMES[x] || x)).join(", ")}</td></tr>`).join("") || `<tr><td colspan="6" class="muted">Sin clientes.</td></tr>`}</tbody></table></div>
    <form class="form panel" id="cf" style="margin-top:10px">
      <label>Nombre<input name="name" required></label>
      <label>Zona horaria<input name="timezone" value="America/Argentina/Buenos_Aires"></label>
      <label>WhatsApp del cliente (opcional)<input name="contact_whatsapp" placeholder="+59891234567"></label>
      <label>Email del cliente (opcional)<input name="contact_email" type="email" placeholder="dueno@local.com"></label>
      <label>Chat de Telegram (opcional)<input name="telegram_chat_id" placeholder="-1001234567890"></label>
      <label>Webhook (opcional)<input name="webhook_url" placeholder="https://…"></label>
      <button class="primary" type="submit">Crear cliente</button>
    </form>

    <h2>Agentes (una PC por local)</h2>
    <div class="table-wrap panel" style="padding:0"><table><thead><tr><th>Agente</th><th>Cliente</th><th>Estado</th><th>Última conexión</th><th>Pendientes de subir</th><th></th></tr></thead>
      <tbody>${agents.map((a) => `<tr><td>${esc(a.name)}</td><td>${esc(clientName(a.client_id))}</td><td>${a.online ? `<span class="chip ok">En línea</span>` : `<span class="chip bad">Desconectado</span>`}</td>
        <td>${fmtDate(a.last_seen)}</td><td>${a.status?.outbox_pending ?? "—"}</td><td><button class="small" data-rotate="${a.id}">Nuevo token</button></td></tr>`).join("") || `<tr><td colspan="6" class="muted">Sin agentes.</td></tr>`}</tbody></table></div>
    <form class="form panel" id="af" style="margin-top:10px">
      <label>Cliente<select name="client_id" required>${options(S.clients.map((c) => [c.id, c.name]), "")}</select></label>
      <label>Nombre<input name="name" required placeholder="PC del local Centro"></label>
      <button class="primary" type="submit">Crear agente</button>
    </form>
    <div id="token"></div>

    <h2>Usuarios</h2>
    <div class="table-wrap panel" style="padding:0"><table><thead><tr><th>Usuario</th><th>Rol</th><th>Cliente</th><th>Activo</th></tr></thead>
      <tbody>${users.map((u) => `<tr><td>${esc(u.username)}</td><td>${roleNames[u.role]}</td><td>${u.client_id ? esc(clientName(u.client_id)) : "—"}</td><td>${u.active ? "Sí" : "No"}</td></tr>`).join("")}</tbody></table></div>
    <form class="form panel" id="uf" style="margin-top:10px">
      <label>Usuario<input name="username" required minlength="3"></label>
      <label>Clave (mín. 8)<input name="password" type="password" required minlength="8" autocomplete="new-password"></label>
      <label>Rol<select name="role">${options(Object.entries(roleNames), "entrenador")}</select></label>
      <label>Cliente (solo rol cliente)<select name="client_id">${options(S.clients.map((c) => [c.id, c.name]), "", "—")}</select></label>
      <button class="primary" type="submit">Crear usuario</button>
    </form>`;
  const showToken = (a) => {
    $("#token").innerHTML = `<div class="panel" style="margin-top:10px"><b>Token del agente "${esc(a.name)}"</b> — copialo ahora, no se vuelve a mostrar.
      <div class="token-box mono" style="margin:8px 0">${esc(a.token)}</div>
      <p class="muted">En la PC del local:</p><div class="token-box mono">python -m agent --server ${esc(location.origin)} --token ${esc(a.token)}</div></div>`;
  };
  $("#cf").addEventListener("submit", async (e) => {
    e.preventDefault(); const f = Object.fromEntries(new FormData(e.target));
    for (const k of ["telegram_chat_id", "webhook_url", "contact_whatsapp", "contact_email"]) if (!f[k]) f[k] = null;
    try { await api("POST", "/api/clients", f); toast("Cliente creado"); route(); } catch (err) { fail(err); }
  });
  $("#af").addEventListener("submit", async (e) => {
    e.preventDefault(); const f = new FormData(e.target);
    try { const a = await api("POST", "/api/agents", { client_id: +f.get("client_id"), name: f.get("name") }); e.target.reset(); showToken(a); } catch (err) { fail(err); }
  });
  $("#uf").addEventListener("submit", async (e) => {
    e.preventDefault(); const f = new FormData(e.target);
    try {
      await api("POST", "/api/users", { username: f.get("username"), password: f.get("password"), role: f.get("role"), client_id: f.get("client_id") ? +f.get("client_id") : null });
      toast("Usuario creado"); route();
    } catch (err) { fail(err); }
  });
  $$("[data-rotate]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("El token anterior deja de funcionar. ¿Generar uno nuevo?")) return;
    try { showToken(await api("POST", `/api/agents/${b.dataset.rotate}/token`)); } catch (err) { fail(err); }
  }));
}

boot();
