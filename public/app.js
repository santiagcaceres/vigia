"use strict";

const TYPE_NAMES = { salida_sin_pago: "Salida sin pago", merodeo: "Merodeo", zona_restringida: "Zona restringida", fuera_de_horario: "Fuera de horario", alarma_dvr: "Alarma con persona",
  mano_bolsillo: "Mano al bolsillo", mano_ropa: "Mano dentro de la ropa", mano_bolsa: "Mano en la bolsa" };
// reglas de zona (las únicas que van en `rules.enabled_rules`) y gestos de ocultamiento (van aparte, en `gestures`:
// un agente viejo rechaza nombres que no conoce dentro de `rules`)
const RULE_KEYS = ["salida_sin_pago", "zona_restringida", "fuera_de_horario"];  // (el merodeo ya no existe)
const GESTURE_KEYS = ["mano_bolsillo", "mano_ropa", "mano_bolsa"];
const ALARM_TYPES = { VMD: "Movimiento", fielddetection: "Intrusión", linedetection: "Cruce de línea", regionEntrance: "Entrada a zona", regionExiting: "Salida de zona" };
const MODE_NAMES = { alarmas_equipo: "Alarmas del grabador", alarmas: "Alarmas HPP", vivo: "En vivo" };
const isAlarmMode = (m) => m === "alarmas" || m === "alarmas_equipo";
const ALARM_STATUS = { pendiente: ["", "Esperando"], procesando: ["", "Analizando"], con_persona: ["ok", "Con persona → a revisar"], sin_persona: ["", "Sin personas (descartada)"], sin_camara: ["warn", "Cámara no cargada"], error: ["bad", "Error"] };
const alarmTypeName = (t) => String(t || "").split("+").map((x) => ALARM_TYPES[x] || x).join(" + ");
const LABEL_NAMES = { hurto: "Hurto", sospechoso: "Sospechoso", normal: "Normal", falso_positivo: "Falso positivo" };
const LABEL_KEYS = { "1": "hurto", "2": "sospechoso", "3": "normal", "4": "falso_positivo" };
const ZONE_NAMES = { shelf: "Góndola", aisle: "Pasillo", checkout: "Caja", exit: "Salida", restricted: "Restringida", entrance: "Entrada (conteo)", pump: "Surtidor", shop: "Tienda", staff: "Personal (empleados)" };
const ZONE_COLORS = { shelf: "#ffc800", aisle: "#78ffff", checkout: "#00c800", exit: "#0096ff", restricted: "#ff3030", entrance: "#c800ff", pump: "#ff7800", shop: "#00c8c8", staff: "#8a8a8a" };
const PLAN_NAMES = { hurto: "Detección de hurtos", comercio: "Métricas de comercio", estacion: "Estación de servicio" };
const BRAND_NAMES = { hikvision: "Hikvision", dahua: "Dahua" };
const STATE_NAMES = { escuchando: "Escuchando alarmas", en_linea: "En línea", sin_conexion: "Sin conexión", reconectando: "Reconectando", error: "Error", iniciando: "Iniciando", detenida: "Detenida", fuera_de_horario: "Local cerrado" };
const CHANNEL_NAMES = { portal: "Portal / link", telegram: "Telegram", whatsapp: "WhatsApp", email: "Email", webhook: "Webhook" };
const REPORT_STATUS = { enviado: ["ok", "Enviado"], parcial: ["warn", "Enviado en parte"], error: ["bad", "No se pudo enviar"] };

const ROLE_NAMES = { admin: "Administrador", entrenador: "Empleado", cliente: "Cliente" };
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
$("#my-password").addEventListener("click", async () => {
  const current = prompt("Tu clave actual:");
  if (!current) return;
  const nueva = prompt("Clave nueva (mínimo 8 caracteres):");
  if (!nueva) return;
  try { await api("POST", "/api/me/password", { current, new: nueva }); toast("Clave cambiada"); } catch (e) { fail(e); }
});

async function boot() {
  try { S.me = await api("GET", "/api/me", undefined, { noRedirect: true }); } catch (_) { return showLogin(); }
  await loadClients(true);
  $("#login").classList.add("hidden");
  $("#app").classList.remove("hidden");
  $("#user-name").textContent = `${S.me.username} · ${ROLE_NAMES[S.me.role] || S.me.role}`;
  // Los empleados trabajan en la app de escritorio; los clientes y los administradores (que miran métricas y datos
  // de todos los clientes, p. ej. desde el celular) usan la web.
  let emergencia = false;
  try { emergencia = sessionStorage.getItem("vigia_web_staff") === "1"; } catch (_) {}
  if (S.me.role === "entrenador" && !emergencia) {
    $("#nav").innerHTML = "";
    $("#view").innerHTML = `<div class="panel" style="max-width:640px;margin:40px auto">
      <h1>Usá la app de escritorio</h1>
      <p>La gestión de clientes, cámaras, zonas, alertas, entrenamiento y usuarios se hace desde la <b>app de escritorio de Analítica Crítica</b>. La web quedó para que los clientes vean su local.</p>
      <p><a class="btn primary" href="/api/app/download">Descargar la app para Windows</a></p>
      <p class="muted">Descomprimila y abrí el programa (la primera vez se renombra solo a «Analítica Crítica.exe»). Entrá con este mismo usuario. Después se actualiza sola con «Buscar actualización».</p>
      <p class="muted" style="margin-top:24px"><a href="#" id="web-emergencia">Entrar igual a la versión web (solo si la app no anda)</a></p>
    </div>`;
    $("#web-emergencia").addEventListener("click", (e) => { e.preventDefault(); try { sessionStorage.setItem("vigia_web_staff", "1"); } catch (_) {} location.reload(); });
    return;
  }
  const links = S.me.role === "cliente"
    ? [["#/resumen", "Resumen"], ["#/alertas", "Alertas"], ["#/avisos", "Incidentes"], ["#/camaras", "Cámaras"]]
    : [["#/resumen", "Resumen"], ["#/clientes", "Clientes"], ["#/metricas", "Métricas"], ["#/alertas", "Alertas"]];
  if (isAdmin()) links.push(["#/admin", "Administración"]);
  $("#nav").innerHTML = links.map(([h, t]) => `<a href="${h}">${t}</a>`).join("");
  route();
}

window.addEventListener("hashchange", route);
async function route() {
  if (!S.me) return;
  S.timers.forEach(clearInterval); S.timers = []; S.keyHandler = null;
  const h = location.hash || "#/resumen";
  const section = h.startsWith("#/evento") || h.startsWith("#/revisar") || h.startsWith("#/alarmas") ? "#/alertas"
    : isStaff() && (h.startsWith("#/camara") || h.startsWith("#/cliente")) ? "#/clientes" : h;
  $$("#nav a").forEach((a) => a.classList.toggle("active", section.startsWith(a.getAttribute("href"))));
  const old = $("#view"), v = old.cloneNode(false); // nodo nuevo: descarta listeners de la vista anterior
  old.replaceWith(v);
  window.scrollTo(0, 0);
  let m;
  try {
    if ((m = h.match(/^#\/evento\/(\d+)/))) return await eventView(v, +m[1]);
    if (h.startsWith("#/revisar") && isStaff()) return await reviewView(v);
    if ((m = h.match(/^#\/camara\/(\d+)\/zonas/)) && isStaff()) return await zonesView(v, +m[1]);
    if ((m = h.match(/^#\/camara\/(\d+)/))) return await cameraView(v, +m[1]);
    if ((m = h.match(/^#\/cliente\/(\d+)/)) && isStaff()) return await clientCamerasView(v, +m[1]);
    if (h.startsWith("#/camaras")) return await (isStaff() ? camerasView(v) : clientCamerasView(v, S.me.client_id));
    if ((m = h.match(/^#\/video\/(\d+)/)) && isStaff()) return await importView(v, +m[1]);
    if (h.startsWith("#/videos") && isStaff()) return await importsView(v);
    if (h.startsWith("#/entrenamiento") && isStaff()) return await trainingView(v);
    if (h.startsWith("#/avisos")) return await reportsView(v);
    if (h.startsWith("#/alarmas") && isStaff()) return await alarmsView(v);
    if (h.startsWith("#/admin") && isAdmin()) return await adminView(v);
    if (h.startsWith("#/clientes") && isStaff()) return await clientsView(v);
    if (h.startsWith("#/metricas") && isStaff()) return await metricsView(v, +(new URLSearchParams(h.split("?")[1] || "").get("client_id") || 0));
    if (h.startsWith("#/resumen")) return await (isStaff() ? summaryView(v) : metricsView(v, S.me.client_id));
    if (h.startsWith("#/prueba") && isStaff()) return await pruebaView(v);
    return await eventsView(v);
  } catch (e) { v.innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
}

const CACHE_MS = 20000;
async function loadCameras(force = false) {
  if (!force && S.cameras.length && Date.now() - (S._camsAt || 0) < CACHE_MS) return S.cameras;
  S.cameras = await api("GET", "/api/cameras"); S._camsAt = Date.now();
  return S.cameras;
}
async function loadClients(force = false) {
  if (!force && S.clients.length && Date.now() - (S._clientsAt || 0) < CACHE_MS) return S.clients;
  S.clients = await api("GET", "/api/clients"); S._clientsAt = Date.now();
  return S.clients;
}

// ------------------------------------------------------------------ eventos
async function eventsView(v) {
  await loadCameras();
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const stats = await api("GET", "/api/stats?days=30" + (q.get("client_id") ? "&client_id=" + q.get("client_id") : ""));
  v.innerHTML = `
    <div class="row"><h1>Alertas</h1><span class="spacer"></span>${isStaff() ? `<a class="btn" href="#/alarmas">Alarmas del grabador</a><a class="btn primary" href="#/revisar">Revisar pendientes (${stats.by_label.sin_etiqueta || 0})</a>` : ""}</div>
    <p class="muted">Últimos 30 días: ${stats.total} alertas, ${stats.by_label.sin_etiqueta || 0} sin revisar. Cada alerta es un indicio para revisar, no una prueba.</p>
    <div class="toolbar">
      ${S.me.role !== "cliente" ? `<label>Cliente<select id="f-client">${options(S.clients.map((c) => [c.id, c.name]), q.get("client_id"), "Todos")}</select></label>` : ""}
      <label>Cámara<select id="f-camera">${options(S.cameras.map((c) => [c.id, c.name]), q.get("camera_id"), "Todas")}</select></label>
      <label>Tipo<select id="f-type">${options(Object.entries(TYPE_NAMES), q.get("type"), "Todos")}</select></label>
      <label>Revisión<select id="f-label">${options([["sin_etiqueta", "Sin revisar"], ...Object.entries(LABEL_NAMES)], q.get("label"), "Todas")}</select></label>
      <label>Origen<select id="f-source">${options([["alarma", "Alarmas del grabador"], ["vivo", "En vivo"], ["prueba", "Prueba (pantalla)"], ["importado", "Videos importados"]], q.get("source"), "Todos")}</select></label>
      <label>Alerta<select id="f-alert">${options([["true", "Solo alertas"], ["false", "Filtrados por el modelo"]], q.get("alert"), "Todos")}</select></label>
    </div>
    <div id="list" class="grid"></div>
    <div class="row" style="margin-top:14px"><span id="count" class="muted"></span><span class="spacer"></span><button id="more" class="hidden">Cargar más</button></div>`;
  const ids = ["client", "camera", "type", "label", "source", "alert"];
  ids.forEach((k) => $("#f-" + k)?.addEventListener("change", () => {
    const p = new URLSearchParams();
    ids.forEach((x) => { const el = $("#f-" + x); if (el && el.value) p.set(x === "client" ? "client_id" : x === "camera" ? "camera_id" : x, el.value); });
    location.hash = "#/alertas?" + p.toString();
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
      <div class="muted">${fmtDate(e.occurred_at)}${e.source === "importado" ? " · video importado" : ""}${e.source === "alarma" ? " · alarma " + esc(alarmTypeName(e.alarm?.dvr_event)) : ""}${e.source === "prueba" ? " · prueba desde la pantalla" : ""}${e.score != null ? ` · puntaje ${pct(e.score)}` : ""}</div>
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

const fmtSec1 = (s) => (s == null ? "—" : Number(s).toFixed(1).replace(".", ",") + " s");

function featuresHtml(e) {
  const f = e.features || {};
  const rows = [["Tiempo en escena", fmtSec(f.track_age_s)], ["Tiempo en góndola", fmtSec(f.shelf_total_s)], ["Tiempo en caja", fmtSec(f.checkout_total_s)],
    ["Visitas a góndola", f.shelf_visits ?? "—"], ["Personas en escena", f.people_in_scene ?? "—"]];
  if (GESTURE_KEYS.includes(e.type) || f.gesture_dwell_s > 0) { // medidas del gesto (posible ocultamiento)
    rows.push(["Mano en el lugar", fmtSec1(f.gesture_dwell_s)],
      ["Desde que tomó algo", f.since_take_s == null ? "—" : f.since_take_s < 0 ? "No se vio que tomara algo" : fmtSec1(f.since_take_s)],
      ["Veces que tomó algo", f.takes ?? "—"]);
    if (e.type === "mano_bolsa") rows.push(["Bolso a la vista", f.bag_seen > 0 ? "Sí" : "No (manos a la cintura)"]);
    if (f.pose_ok_ratio != null) rows.push(["Postura bien vista", pct(f.pose_ok_ratio)]);
  }
  return `<details><summary class="muted">Datos medidos</summary><dl class="kv" style="margin-top:8px">${rows.map(([k, x]) => `<dt>${k}</dt><dd>${esc(x)}</dd>`).join("")}</dl></details>`;
}

async function eventView(v, id) {
  const e = await api("GET", "/api/events/" + id);
  const i = S.lastList.indexOf(id);
  const prev = i > 0 ? S.lastList[i - 1] : null, next = i >= 0 && i < S.lastList.length - 1 ? S.lastList[i + 1] : null;
  v.innerHTML = `
    <div class="row"><a href="#/alertas">← Alertas</a><span class="spacer"></span>
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
          <dt>Origen</dt><dd>${e.source === "importado" ? `<a href="#/video/${e.import_id}">Video importado</a> (seg. ${fmtSec(e.video_ts)})` : e.source === "alarma" ? `Alarma del DVR: ${esc(alarmTypeName(e.alarm?.dvr_event))}${e.alarm?.triggers > 1 ? ` (${e.alarm.triggers} disparos)` : ""}` : e.source === "prueba" ? "Prueba desde la pantalla de HPP" : "En vivo"}</dd>
          <dt>Clip</dt><dd>${e.clip_url ? ({ nvr: "Grabación del grabador", buffer: "Buffer del agente", video: "Del video importado", prueba: "Grabado de la pantalla" }[e.clip_source] || "Clip") + ` · <a href="${e.clip_url}" download>Descargar</a>` : "—"}</dd>
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
  const ok = st.state === "en_linea" || st.state === "escuchando";
  if (st.state === "fuera_de_horario") return `<span class="chip" title="Fuera del horario del local no se mira (ahorra video y análisis)">Local cerrado</span>`;
  return `<span class="chip ${ok ? "ok" : "bad"}" title="${esc(st.error || "")}">${esc(STATE_NAMES[st.state] || st.state)}${ok && st.fps ? ` · ${st.fps} fps` : ""}</span>`;
}

const previewUrl = (c) => (c.has_preview ? `/api/cameras/${c.id}/preview?t=${Math.floor(Date.now() / 30000)}` : null);
const camThumb = (c) => (previewUrl(c) ? `<img loading="lazy" src="${previewUrl(c)}" alt="">` : `<div class="thumb-empty">Sin imagen todavía<br><small>la PC central la manda al conectarse</small></div>`);
const connText = (c) => (c.host === undefined ? esc(MODE_NAMES[c.mode] || "") : `${esc(BRAND_NAMES[c.brand] || c.brand || "")} · ${isAlarmMode(c.mode) ? `${MODE_NAMES[c.mode]} · ${c.mode === "alarmas" ? esc(c.hpp_serial || "") + " · " : ""}canal ${c.channel} · ${esc(c.host)}` : c.has_source_url ? "URL manual" : `${esc(c.host)} · canal ${c.channel} · ${c.detect_stream === "sub" ? "secundario" : "principal"}`}${c.detector === "motion" ? " · movimiento" : ""}`);

/** Formulario de alta/edición de cámara (compartido por la tabla general, la grilla del cliente y la ficha). */
function cameraForm(box, agents, c = {}, opts = {}) {
  const lockClient = !!opts.clientId && !c.id;
  box.innerHTML = `<div class="panel" style="margin-top:12px">
      <h3>${c.id ? "Editar cámara" : "Nueva cámara"}</h3>
      <p class="muted">Datos del grabador (DVR/NVR): los mismos que usás en iVMS-4200 (Hikvision) o SmartPSS Lite (Dahua). La IP es la <b>IP pública o el DDNS</b> del cliente y los puertos son los <b>externos</b> que abriste en su router. Paso a paso en la <a href="guia.html" target="_blank" rel="noopener">guía de conexión</a>.</p>
      <form class="form" id="cf">
        <label class="${lockClient ? "hidden" : ""}">Cliente<select name="client_id" required>${options(S.clients.map((x) => [x.id, x.name]), c.client_id ?? opts.clientId)}</select></label>
        <label>PC central (agente)<select name="agent_id">${options(agents.map((a) => [a.id, a.central ? a.name : `${a.name} (${clientName(a.client_id)})`]), c.agent_id ?? (agents.find((a) => a.central) || {}).id, "—")}</select></label>
        <label>Nombre<input name="name" value="${esc(c.name || "")}" required placeholder="Salón principal"></label>
        <label>Marca<select name="brand">${options(Object.entries(BRAND_NAMES), c.brand || "hikvision")}</select><small id="brand-note" class="muted"></small></label>
        <label>Modo<select name="mode">${options([["alarmas_equipo", "Alarmas del propio grabador (recomendado)"], ["alarmas", "Alarmas por Hik-Partner Pro (solo Hikvision)"], ["vivo", "En vivo 24 h"]], c.mode || "alarmas_equipo")}</select><small id="mode-note" class="muted"></small></label>
        <label class="alarm-only hpp-only">N.º de serie del DVR en Hik-Partner Pro<input name="hpp_serial" value="${esc(c.hpp_serial || "")}" placeholder="E44034229"></label>
        <fieldset class="alarm-only"><legend>Alarmas del grabador que se analizan</legend>${Object.entries(ALARM_TYPES).map(([k, t]) => `<label class="inline"><input type="checkbox" name="alarm_type" value="${k}" ${((c.alarm_cfg || {}).event_types || ["VMD", "fielddetection", "linedetection"]).includes(k) ? "checked" : ""}> ${t}</label>`).join("")}</fieldset>
        <label class="alarm-only">Segundos de video antes de la alarma<input name="pre_seconds" type="number" min="0" max="60" value="${(c.alarm_cfg || {}).pre_seconds ?? 10}"></label>
        <label class="alarm-only">Segundos de video después<input name="post_seconds" type="number" min="3" max="90" value="${(c.alarm_cfg || {}).post_seconds ?? 20}"></label>
        <label>IP pública, DDNS o IP del grabador<input name="host" value="${esc(c.host || "")}" placeholder="almacen.ddns.net o 192.168.1.64"></label>
        <label>Canal<input name="channel" type="number" min="1" value="${c.channel || 1}"></label>
        <label>Usuario<input name="username" value="${esc(c.username || "admin")}"></label>
        <label>Clave<input name="password" type="password" placeholder="${c.has_password ? "(sin cambios)" : ""}" autocomplete="new-password"></label>
        <label>Puerto RTSP (externo si va por router)<input name="rtsp_port" type="number" value="${c.rtsp_port || 554}"></label>
        <label>Puerto HTTP (externo si va por router)<input name="http_port" type="number" value="${c.http_port || 80}"></label>
        <label class="live-only">Stream para analizar<select name="detect_stream">${options([["sub", "Secundario (recomendado)"], ["main", "Principal"]], c.detect_stream || "sub")}</select></label>
        <label>Detector<select name="detector">${options([["yolo", "Personas (YOLO)"], ["motion", "Movimiento (solo pruebas)"]], c.detector || "yolo")}</select></label>
        <label>Cuadros por segundo a analizar<input name="process_fps" type="number" step="0.5" min="1" max="15" value="${c.process_fps || 5}"></label>
        <label>Hora de grabación del equipo<select name="playback_time">${options([["local", "Hora local (habitual)"], ["utc", "UTC"]], c.playback_time || "local")}</select></label>
        <label class="live-only">URL RTSP manual (opcional)<input name="source_url" placeholder="Solo para otras marcas"></label>
        <label class="inline live-only"><input type="checkbox" name="nvr_clips" ${c.nvr_clips !== false ? "checked" : ""}> Bajar clip de la grabación del equipo</label>
        <label class="inline"><input type="checkbox" name="enabled" ${c.enabled !== false ? "checked" : ""}> Habilitada</label>
        <div class="row"><button class="primary" type="submit">Guardar</button><button type="button" id="cancel">Cancelar</button></div>
      </form></div>`;
  $("#cancel", box).addEventListener("click", () => (box.innerHTML = ""));
  const syncMode = () => {
    // el modo "alarmas" usa Hik-Partner Pro: solo existe para Hikvision
    const esHik = $("#cf [name=brand]", box).value === "hikvision";
    const modo = $("#cf [name=mode]", box);
    modo.querySelector('option[value=alarmas]').disabled = !esHik;
    if (!esHik && modo.value === "alarmas") modo.value = "alarmas_equipo";
    const alarms = isAlarmMode(modo.value);
    $$("#cf .alarm-only", box).forEach((el) => el.classList.toggle("hidden", !alarms));
    $$("#cf .hpp-only", box).forEach((el) => el.classList.toggle("hidden", modo.value !== "alarmas"));
    $$("#cf .live-only", box).forEach((el) => el.classList.toggle("hidden", alarms));
    $("#brand-note", box).textContent = esHik ? "" : "Para Dahua: alarmas del propio grabador o en vivo.";
    $("#mode-note", box).textContent = { alarmas_equipo: "El grabador le avisa al agente cada alarma (movimiento, intrusión…) y solo se analiza ese tramo. Sin nube ni API key: activá las detecciones en el grabador.",
      alarmas: "Las alarmas llegan por el webhook de Hik-Partner Pro (hay que conectar HPP en Alarmas).", vivo: "El agente mira el stream las 24 h (más carga de PC e internet)." }[modo.value] || "";
  };
  $("#cf [name=mode]", box).addEventListener("change", syncMode);
  $("#cf [name=brand]", box).addEventListener("change", syncMode);
  syncMode();
  $("#cf", box).addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = new FormData(ev.target);
    const body = {
      client_id: +f.get("client_id"), agent_id: f.get("agent_id") ? +f.get("agent_id") : null, name: f.get("name"),
      brand: f.get("brand"), host: f.get("host"), channel: +f.get("channel"), username: f.get("username"), password: f.get("password") || null,
      rtsp_port: +f.get("rtsp_port"), http_port: +f.get("http_port"), detect_stream: f.get("detect_stream"),
      detector: f.get("detector"), process_fps: +f.get("process_fps"), playback_time: f.get("playback_time"),
      source_url: f.get("source_url") || null, nvr_clips: !!f.get("nvr_clips"), enabled: !!f.get("enabled"),
      mode: f.get("mode"), hpp_serial: f.get("hpp_serial") || null,
      alarm_cfg: { event_types: f.getAll("alarm_type"), pre_seconds: +f.get("pre_seconds"), post_seconds: +f.get("post_seconds") },
    };
    try {
      const saved = await api(c.id ? "PUT" : "POST", c.id ? "/api/cameras/" + c.id : "/api/cameras", body);
      toast("Cámara guardada. La PC central la toma en menos de un minuto."); S._camsAt = 0;
      if (opts.onSaved) opts.onSaved(saved); else route();
    } catch (e) { fail(e); }
  });
}

/** "Probar conexión": el servidor entra al grabador por HTTP y muestra modelo y canales. */
async function probeCamera(c, out) {
  out.innerHTML = `<div class="panel" style="margin-top:12px"><p class="muted">Probando ${esc(c.name)} desde el servidor (hasta 10 segundos)…</p></div>`;
  try {
    const r = await api("POST", `/api/cameras/${c.id}/probe`);
    const chans = (r.channels || []).map((x) => `${x.channel}: ${esc(x.name || "")}${x.channel === c.channel ? " <span class=\"chip ok\">esta cámara</span>" : ""}`).join("<br>");
    out.innerHTML = `<div class="panel" style="margin-top:12px">
      <div class="row"><b>${esc(c.name)}</b> ${r.ok ? `<span class="chip ok">El grabador responde</span>` : `<span class="chip bad">No responde</span>`}</div>
      ${r.ok ? `<p>Modelo: <span class="mono">${esc(r.model || r.deviceType || "—")}</span> · Serie: <span class="mono">${esc(r.serialNumber || "—")}</span></p>
        <p>Canales que ve:<br>${chans || "<span class=\"muted\">no informa canales (en una cámara suelta es normal)</span>"}</p>
        ${!(r.channels || []).some((x) => x.channel === c.channel) && (r.channels || []).length ? `<p class="error">El canal ${c.channel} no está en la lista: revisá el número de canal.</p>` : ""}
        <p class="muted">Esto prueba el puerto HTTP. El video (puerto RTSP) lo prueba el agente: mirá el estado en un minuto, o probá en VLC desde afuera del local:<br><span class="mono">${esc(r.rtsp_url)}</span></p>`
      : `<p class="error">${esc(r.error)}</p><p class="muted">Enlace para probar en VLC desde afuera del local: <span class="mono">${esc(r.rtsp_url)}</span></p>`}
      ${r.public && r.standard_ports ? `<p class="error">Este grabador está expuesto a internet con los puertos estándar (554 / 80). Conviene cambiarlos en el router por puertos altos (ver guía, sección 3 y 8).</p>` : ""}
    </div>`;
  } catch (e) { out.innerHTML = ""; fail(e); }
}

/** Tabla con todas las cámaras de todos los clientes (se llega desde el Resumen; ya no está en el menú). */
async function camerasView(v) {
  await loadCameras();
  const agents = await api("GET", "/api/agents");
  v.innerHTML = `
    <div class="row"><h1>Todas las cámaras</h1><span class="spacer"></span><a class="btn" href="guia.html" target="_blank" rel="noopener">Guía de conexión</a><button id="add" class="primary">Agregar cámara</button></div>
    <p class="muted">Para ver las cámaras de un comercio con su imagen, entrá por <a href="#/clientes">Clientes</a>.</p>
    <div id="probe-out"></div>
    <div id="cam-form"></div>
    <div class="table-wrap panel" style="margin-top:12px;padding:0"><table>
      <thead><tr><th>Cámara</th><th>Cliente</th><th>Conexión</th><th>Agente</th><th>Estado</th><th>Zonas</th><th></th></tr></thead>
      <tbody>${S.cameras.map((c) => { const ag = agents.find((a) => a.id === c.agent_id); return `<tr>
        <td><a href="#/camara/${c.id}"><b>${esc(c.name)}</b></a>${c.enabled ? "" : ` <span class="chip">deshabilitada</span>`}</td>
        <td><a href="#/cliente/${c.client_id}">${esc(clientName(c.client_id))}</a></td><td class="mono">${connText(c)}</td><td>${ag ? esc(ag.name) : `<span class="chip bad">sin agente</span>`}</td>
        <td>${c.mode === "alarmas" ? "" : statusChip(c.status)}${isAlarmMode(c.mode) ? ` <a href="#/alarmas?camera_id=${c.id}">Ver alarmas</a>` : ""}</td>
        <td>${(c.zones || []).length}</td>
        <td class="row"><a class="btn small" href="#/camara/${c.id}/zonas">Zonas y reglas</a><button class="small" data-edit="${c.id}">Editar</button><button class="small" data-probe="${c.id}">Probar conexión</button></td>
      </tr>`; }).join("") || `<tr><td colspan="7" class="muted">Todavía no hay cámaras.</td></tr>`}</tbody></table></div>`;
  $("#add").addEventListener("click", () => { if (!S.clients.length) return toast("Primero creá un cliente en Clientes", true); if (!agents.length) return toast("Primero creá la PC central en Administración", true); cameraForm($("#cam-form"), agents); });
  $$("[data-probe]").forEach((b) => b.addEventListener("click", () => probeCamera(S.cameras.find((x) => x.id === +b.dataset.probe), $("#probe-out"))));
  const pre = new URLSearchParams(location.hash.split("?")[1] || "");
  if (pre.get("nueva")) cameraForm($("#cam-form"), agents, { mode: "alarmas", hpp_serial: pre.get("serie"), channel: +pre.get("canal") || 1, name: pre.get("nombre") || "" });
  $$("[data-edit]").forEach((b) => b.addEventListener("click", () => cameraForm($("#cam-form"), agents, S.cameras.find((c) => c.id === +b.dataset.edit))));
}

/** "Cámaras de <cliente>": miniaturas con la última imagen; un clic abre la ficha de la cámara. */
async function clientCamerasView(v, clientId) {
  await Promise.all([loadClients(), loadCameras()]);
  const client = S.clients.find((c) => c.id === clientId) || (S.me.role === "cliente" ? { id: clientId, name: S.me.client_name || "mi comercio" } : null);
  if (!client) { v.innerHTML = `<div class="empty">Cliente no encontrado.</div>`; return; }
  const agents = isStaff() ? await api("GET", "/api/agents") : [];
  const cams = S.cameras.filter((c) => c.client_id === clientId);
  v.innerHTML = `
    ${isStaff() ? `<div class="row"><a href="#/clientes">← Clientes</a></div>` : ""}
    <div class="row"><h1>Cámaras de ${esc(client.name)}</h1><span class="spacer"></span>${isStaff() ? `<a class="btn" href="#/metricas?client_id=${clientId}">Métricas</a><button id="edit-client" class="small">Editar cliente</button><a class="btn" href="guia.html" target="_blank" rel="noopener">Guía de conexión</a><button id="add" class="primary">Agregar cámara</button>` : ""}</div>
    <div id="cam-form"></div>
    <div class="grid cams" style="margin-top:12px">${cams.map((c) => `<a class="card cam" href="#/camara/${c.id}">${camThumb(c)}
      <div class="body"><div class="row"><span class="title">${esc(c.name)}</span><span class="spacer"></span>${c.enabled ? (c.mode === "alarmas" ? `<span class="chip">${MODE_NAMES[c.mode]}</span>` : statusChip(c.status)) : `<span class="chip">deshabilitada</span>`}</div>
      <span class="muted mono">${connText(c)}</span></div></a>`).join("") || `<div class="empty">Este cliente todavía no tiene cámaras.${isStaff() ? " Tocá <b>Agregar cámara</b>." : ""}</div>`}</div>`;
  if (!isStaff()) { every(30000, () => $$("img", v).forEach((i) => { const c = cams.find((x) => "#/camara/" + x.id === i.closest("a").getAttribute("href")); if (c) i.src = previewUrl(c); })); return; }
  $("#add").addEventListener("click", () => { if (!agents.length) return toast("Primero creá la PC central en Administración", true); cameraForm($("#cam-form"), agents, {}, { clientId }); });
  $("#edit-client").addEventListener("click", () => (location.hash = `#/clientes?editar=${clientId}`));
  // cada 30 s se refresca el estado y las imágenes (la PC central manda una captura cada 5 min)
  every(30000, async () => { S._camsAt = 0; await loadCameras(); if (location.hash.startsWith(`#/cliente/${clientId}`)) clientCamerasView(v, clientId); });
}

/** Ficha de una cámara: imagen grande, estado, conexión y acciones. */
async function cameraView(v, id) {
  const cam = await api("GET", "/api/cameras/" + id);
  await loadClients();
  const agents = isStaff() ? await api("GET", "/api/agents") : [];
  const ag = agents.find((a) => a.id === cam.agent_id);
  const back = isStaff() ? `#/cliente/${cam.client_id}` : "#/camaras";
  v.innerHTML = `
    <div class="row"><a href="${back}">← Cámaras de ${esc(clientName(cam.client_id))}</a></div>
    <div class="row"><h1>${esc(cam.name)}</h1>${cam.enabled ? (cam.mode === "alarmas" ? `<span class="chip">${MODE_NAMES[cam.mode]}</span>` : statusChip(cam.status)) : `<span class="chip">deshabilitada</span>`}<span class="spacer"></span>
      ${isStaff() ? `<a class="btn" href="#/camara/${cam.id}/zonas">Zonas y reglas</a><button id="edit">Editar</button><button id="probe">Probar conexión</button>` : ""}${isAdmin() ? `<button id="delcam" class="ghost">Eliminar</button>` : ""}
      <a class="btn" href="#/alertas?camera_id=${cam.id}">Alertas</a>${isAlarmMode(cam.mode) && isStaff() ? `<a class="btn" href="#/alarmas?camera_id=${cam.id}">Alarmas del grabador</a>` : ""}</div>
    <div id="probe-out"></div>
    <div id="cam-form"></div>
    <div class="two" style="margin-top:12px">
      <div class="panel cam-big"><div class="live-box">${previewUrl(cam) ? `<img id="live-img" src="${previewUrl(cam)}" alt="">` : `<img id="live-img" alt="" class="hidden"><div class="thumb-empty" id="live-empty">Sin imagen todavía</div>`}<span class="live-tag" id="live-state">Conectando con la PC central…</span></div>
        <p class="muted" style="margin:8px 0 0" id="live-note">${cam.has_preview ? `Última imagen: ${fmtDate(cam.preview_at)}.` : ""} La PC central manda un cuadro por segundo con las personas marcadas mientras esta pantalla esté abierta.</p></div>
      <div class="panel"><h3 style="margin-top:0">${isStaff() ? "Conexión" : "Detalle"}</h3>
        <dl class="kv">
          ${isStaff() ? `<dt>Marca</dt><dd>${esc(BRAND_NAMES[cam.brand] || cam.brand || "—")}</dd>
          <dt>Modo</dt><dd>${esc(MODE_NAMES[cam.mode] || cam.mode)}</dd>
          <dt>Grabador</dt><dd class="mono">${cam.has_source_url ? "URL manual" : `${esc(cam.host)} · RTSP ${cam.rtsp_port} · HTTP ${cam.http_port}`}</dd>
          <dt>Canal</dt><dd>${cam.channel}${cam.mode === "alarmas" ? ` · serie ${esc(cam.hpp_serial || "—")}` : ""}</dd>
          <dt>Usuario</dt><dd class="mono">${esc(cam.username || "—")}</dd>
          <dt>PC central</dt><dd>${ag ? esc(ag.name) : `<span class="chip bad">sin agente</span>`}</dd>` : `<dt>Modo</dt><dd>${esc(MODE_NAMES[cam.mode] || cam.mode)}</dd>`}
          <dt>Zonas</dt><dd>${(cam.zones || []).length ? (cam.zones || []).map((z) => `${esc(z.name)} (${ZONE_NAMES[z.type] || z.type})`).join(", ") : `<span class="muted">ninguna: sin zonas no se disparan reglas</span>`}</dd>
          ${cam.status && cam.status.error ? `<dt>Error</dt><dd class="error">${esc(cam.status.error)}</dd>` : ""}
        </dl></div>
    </div>`;
  liveView(v, cam);
  if (!isStaff()) return;
  $("#edit").addEventListener("click", () => cameraForm($("#cam-form"), agents, cam, { onSaved: () => cameraView(v, id) }));
  $("#probe").addEventListener("click", () => probeCamera(cam, $("#probe-out")));
  const del = $("#delcam");
  if (del) del.addEventListener("click", async () => {
    if (!confirm(`¿Eliminar la cámara "${cam.name}"? Se borran también sus métricas. Si tiene alertas, no se puede: deshabilitala.`)) return;
    try { await api("DELETE", "/api/cameras/" + cam.id); toast("Cámara eliminada"); S._camsAt = 0; location.hash = `#/cliente/${cam.client_id}`; } catch (e) { fail(e); }
  });
}

/** Vista en vivo liviana: latido cada 10 s ("alguien mira") y un cuadro por segundo con los recuadros. */
function liveView(v, cam) {
  const img = $("#live-img", v), tag = $("#live-state", v);
  let frameUrl = null, active = false, agentOnline = false, lastOk = 0;
  const setTag = (txt, cls) => { tag.textContent = txt; tag.className = "live-tag " + (cls || ""); };
  const beat = async () => {
    if (document.visibilityState === "hidden") return; // pestaña oculta: no gastar
    try {
      const r = await api("POST", `/api/cameras/${cam.id}/live`);
      frameUrl = r.frame_url; active = r.active; agentOnline = r.agent_online;
      if (!agentOnline) setTag("PC central desconectada", "bad");
      else if (!active) setTag(cam.mode === "vivo" ? "Conectando con la PC central…" : "Abriendo el stream del grabador (hasta 30 s)…", "");
      else setTag("EN VIVO", "ok");
    } catch (e) { setTag("Sin vista en vivo: " + (e.message || e), "bad"); }
  };
  beat();
  setTimeout(beat, 4000);
  every(10000, beat);
  every(1000, () => {
    if (!frameUrl || !active || document.visibilityState === "hidden") return;
    const url = frameUrl + (frameUrl.includes("?") ? "&" : "?") + "t=" + Date.now();
    const probe = new Image();
    probe.onload = () => { img.src = url; img.classList.remove("hidden"); const e = $("#live-empty", v); if (e) e.remove(); lastOk = Date.now(); };
    probe.src = url;
  });
}

// ------------------------------------------------------------------ alarmas de Hik-Partner Pro
async function alarmsView(v) {
  await loadCameras();
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const p = new URLSearchParams(); ["status", "client_id", "camera_id"].forEach((k) => q.get(k) && p.set(k, q.get(k)));
  const data = await api("GET", "/api/alarms?" + p.toString());
  const c = data.counts_7d || {};
  const total = Object.values(c).reduce((a, b) => a + b, 0);
  const tile = (n, t, hint) => `<div class="panel stat"><span class="n">${n}</span><span>${t}</span>${hint ? `<span class="muted">${hint}</span>` : ""}</div>`;
  v.innerHTML = `
    <h1>Alarmas</h1>
    <p class="muted">Cada alarma de un grabador (movimiento, intrusión, cruce de línea) llega acá, avisada por el propio equipo o por Hik-Partner Pro. La PC central baja ese tramo de video y lo analiza:
      si no hay ninguna persona se descarta sola; si hay, pasa a <a href="#/revisar">Revisar y etiquetar</a>.</p>
    <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(170px,1fr))">
      ${tile(total, "Alarmas (7 días)")}
      ${tile(c.sin_persona || 0, "Descartadas solas", total ? `${pct((c.sin_persona || 0) / total)} sin personas` : "")}
      ${tile(c.con_persona || 0, "Con persona", "para revisar")}
      ${tile((c.pendiente || 0) + (c.procesando || 0), "En cola")}
      ${tile((c.error || 0) + (c.sin_camara || 0), "Con problemas", "error o cámara no cargada")}
    </div>
    <div id="hpp"></div>
    <div class="toolbar" style="margin-top:14px">
      <label>Estado<select id="f-status">${options(Object.entries(ALARM_STATUS).map(([k, [, t]]) => [k, t]), q.get("status"), "Todos")}</select></label>
      <label>Cliente<select id="f-client">${options(S.clients.map((x) => [x.id, x.name]), q.get("client_id"), "Todos")}</select></label>
      <label>Cámara<select id="f-camera">${options(S.cameras.filter((x) => isAlarmMode(x.mode)).map((x) => [x.id, x.name]), q.get("camera_id"), "Todas")}</select></label>
    </div>
    <div class="table-wrap panel" style="padding:0"><table>
      <thead><tr><th>Hora</th><th>Cámara</th><th>Alarma</th><th>Resultado</th><th></th></tr></thead>
      <tbody>${data.items.map((a) => { const [cls, txt] = ALARM_STATUS[a.status] || ["", a.status]; return `<tr>
        <td>${fmtDate(a.first_at)}${a.triggers > 1 ? `<div class="muted">${a.triggers} disparos</div>` : ""}</td>
        <td>${a.camera_name ? `${esc(a.camera_name)}<div class="muted">${esc(a.client_name)}</div>` : `<span class="mono">${esc(a.device_serial)} · canal ${a.channel}</span>`}</td>
        <td>${esc(alarmTypeName(a.event_type))}${a.source === "hpp" ? `<div class="muted">por Hik-Partner Pro</div>` : ""}</td>
        <td><span class="chip ${cls}" title="${esc(a.error || "")}">${esc(txt)}</span>${a.error ? `<div class="muted">${esc(a.error)}</div>` : ""}${a.stats && a.stats.frames ? `<div class="muted">persona en ${a.stats.person_frames} de ${a.stats.frames} cuadros</div>` : ""}</td>
        <td class="row">${a.event_id ? `<a class="btn small" href="#/evento/${a.event_id}">Ver clip</a>` : ""}
          ${a.status === "sin_camara" && isStaff() ? `<a class="btn small" href="#/camaras?nueva=1&serie=${encodeURIComponent(a.device_serial)}&canal=${a.channel}">Cargar cámara</a>` : ""}
          ${["error", "sin_camara", "sin_persona"].includes(a.status) ? `<button class="small" data-retry="${a.id}">Reintentar</button>` : ""}</td>
      </tr>`; }).join("") || `<tr><td colspan="5" class="muted">Todavía no llegaron alarmas. Cargá cámaras en modo «alarmas del propio grabador» y activá las detecciones en el equipo.</td></tr>`}</tbody></table></div>`;
  ["status", "client", "camera"].forEach((k) => $("#f-" + k).addEventListener("change", () => {
    const n = new URLSearchParams();
    if ($("#f-status").value) n.set("status", $("#f-status").value);
    if ($("#f-client").value) n.set("client_id", $("#f-client").value);
    if ($("#f-camera").value) n.set("camera_id", $("#f-camera").value);
    location.hash = "#/alarmas?" + n.toString();
  }));
  $$("[data-retry]", v).forEach((b) => b.addEventListener("click", async () => {
    try { await api("POST", `/api/alarms/${b.dataset.retry}/retry`); toast("Vuelve a la cola"); route(); } catch (e) { fail(e); }
  }));
  every(30000, () => { if (!$("#hpp-out")?.innerHTML) route(); });
  await hppPanel($("#hpp"));
}

async function hppPanel(box) {
  const st = await api("GET", "/api/hpp/status");
  box.innerHTML = `<div class="panel" style="margin-top:14px">
    <h3>Conexión con Hik-Partner Pro</h3>
    ${st.configured ? `<p>API key cargada. Las alarmas llegan a <span class="mono">${esc(st.webhook_url || "(falta PUBLIC_URL)")}</span>.
        Última alarma recibida: <b>${fmtDate(st.last_alarm_at)}</b>. Cámaras en modo alarmas: <b>${st.cameras}</b>.</p>
      <div class="row">${isAdmin() ? `<button id="hpp-setup" class="primary">Conectar / actualizar suscripción</button>
        <label class="inline"><input type="checkbox" id="hpp-all"> Todos los equipos de la cuenta (no solo los cargados)</label>
        <button id="hpp-check">Probar conexión</button>` : ""}<button id="hpp-devices">Ver equipos de HPP</button></div>
      <div id="hpp-out" style="margin-top:10px"></div>`
    : !isAdmin() ? `<p class="muted">Todavía no está conectado. Lo configura un administrador.</p>` : `<p>Falta la API key. En el portal de Hik-Partner Pro: <b>Site &amp; Device → My Service → API Integration</b>; copiá AppKey y SecretKey
        a las variables de Vercel <span class="mono">HPP_APP_KEY</span> y <span class="mono">HPP_SECRET_KEY</span> y volvé a desplegar.</p>`}
  </div>`;
  if (!st.configured) return;
  const out = $("#hpp-out");
  if (isAdmin()) {
  $("#hpp-setup").addEventListener("click", async () => {
    try {
      const r = await api("POST", "/api/hpp/setup", { all_devices: $("#hpp-all").checked });
      toast("Listo: Hik-Partner Pro va a mandar las alarmas"); out.innerHTML = `<p class="muted">Suscriptos: ${r.serials ? r.serials.map(esc).join(", ") : "todos los equipos"}</p>`;
    } catch (e) { fail(e); }
  });
  $("#hpp-check").addEventListener("click", async () => {
    try {
      const r = await api("GET", "/api/hpp/status?check=true");
      out.innerHTML = r.error ? `<p class="error">${esc(r.error)}</p>` : `<p>Conexión OK. Webhook registrado en HPP: <span class="mono">${esc(JSON.stringify(r.remote || {}))}</span></p>`;
    } catch (e) { fail(e); }
  });
  }
  $("#hpp-devices").addEventListener("click", async () => {
    out.innerHTML = `<p class="muted">Consultando Hik-Partner Pro…</p>`;
    try {
      const devs = await api("GET", "/api/hpp/devices");
      out.innerHTML = `<div class="table-wrap"><table><thead><tr><th>Equipo</th><th>Serie</th><th>Canales</th></tr></thead><tbody>${devs.map((d) => `<tr>
        <td>${esc(d.name)} <span class="muted">${esc(d.model || "")}</span> ${d.online ? `<span class="chip ok">en línea</span>` : `<span class="chip bad">desconectado</span>`}</td>
        <td class="mono">${esc(d.serial)}</td>
        <td>${(d.channels || []).map((c) => c.error ? `<span class="muted">${esc(c.error)}</span>` : `<div>${c.channel}: ${esc(c.name || "")} ${c.camera_id ? `<span class="chip ok">cargada</span>` : `<a href="#/camaras?nueva=1&serie=${encodeURIComponent(d.serial)}&canal=${c.channel}&nombre=${encodeURIComponent(c.name || "")}">cargar</a>`}</div>`).join("") || "—"}</td>
      </tr>`).join("") || `<tr><td colspan="3" class="muted">La cuenta no tiene equipos.</td></tr>`}</tbody></table></div>`;
    } catch (e) { out.innerHTML = ""; fail(e); }
  });
}

// ------------------------------------------------------------------ editor de zonas
async function zonesView(v, id) {
  const cam = await api("GET", "/api/cameras/" + id);
  let zones = JSON.parse(JSON.stringify(cam.zones || []));
  const rules = cam.rules || {};
  let draft = [], selected = -1;
  // sin elegir a mano rige lo del tipo de cámara (el servidor lo manda en rules_default / gestures_default)
  const enabled = new Set(rules.enabled_rules || cam.rules_default || RULE_KEYS);
  const gestOn = new Set((cam.gestures && cam.gestures.enabled) || cam.gestures_default || []);
  const client = (await loadClients().catch(() => [])).find((c) => c.id === cam.client_id);
  const noTheft = client && !(client.plans || ["hurto"]).includes("hurto");
  v.innerHTML = `
    <div class="row"><a href="#/camara/${id}">← ${esc(cam.name)}</a></div>
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
          <label>Seg. en zona restringida<input id="r-restricted" type="number" step="0.5" min="0" value="${rules.restricted_min_seconds ?? 1}"></label>
          <label>Abre (opcional)<input id="r-open" type="time" value="${rules.opening_hours?.start || ""}"></label>
          <label>Cierra<input id="r-close" type="time" value="${rules.opening_hours?.end || ""}"></label>
        </div>
        <div style="margin-top:8px">${RULE_KEYS.map((k) => `<label class="inline" style="margin-right:12px"><input type="checkbox" class="r-en" value="${k}" ${enabled.has(k) ? "checked" : ""}> ${TYPE_NAMES[k]}</label>`).join("")}</div>
        <h3 style="margin-top:16px">Gestos</h3>
        <div class="gestures">${GESTURE_KEYS.map((k) => `<label class="inline"><input type="checkbox" class="g-en" value="${k}" ${gestOn.has(k) ? "checked" : ""}> ${TYPE_NAMES[k]}</label>`).join("")}</div>
        <p class="muted note">Avisan un <b>posible ocultamiento</b> (tomar algo y llevar la mano al bolsillo, dentro de la ropa o a la bolsa): es un indicio para revisar el clip, nunca una prueba. Hace falta ver a la persona de la cintura para arriba; mirar la postura usa un 30 % más de procesador.${noTheft ? " <b>Este cliente no tiene «Detección de hurtos»: no se usan.</b>" : cam.profile === "metricas" ? " <b>Esta cámara es «Solo métricas»: no se usan.</b>" : ""}</p>
        <div class="row" style="margin-top:14px"><button id="save" class="primary">Guardar</button></div>
      </div>
    </div>`;
  const cv = $("#cv"), ctx = cv.getContext("2d");
  const img = new Image();
  let hasImg = false;
  img.onload = () => { hasImg = true; cv.width = img.naturalWidth; cv.height = img.naturalHeight; draw(); };
  img.onerror = () => { cv.width = 1280; cv.height = 720; $("#bg-info").textContent = "Sin vista previa: el agente la envía al conectarse. También podés subir una captura de iVMS."; draw(); };
  if (cam.has_preview) { img.src = `/api/cameras/${id}/preview?t=${Date.now()}`; $("#bg-info").textContent = "Vista previa del " + fmtDate(cam.preview_at); }
  else {
    img.onerror();
    every(10000, async () => { // la PC central manda la primera imagen al conectarse: esperar sin recargar
      if (hasImg) return;
      const c = await api("GET", "/api/cameras/" + id).catch(() => null);
      if (c && c.has_preview) { img.src = `/api/cameras/${id}/preview?t=${Date.now()}`; $("#bg-info").textContent = "Vista previa del " + fmtDate(c.preview_at); }
    });
  }
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
    // lo que este formulario no muestra (por ejemplo cooldown_seconds, o lo que guardó la app) se conserva
    const r = { ...rules, shelf_dwell_seconds: +$("#r-shelf").value, checkout_min_seconds: +$("#r-checkout").value,
      restricted_min_seconds: +$("#r-restricted").value,
      enabled_rules: $$(".r-en").filter((x) => x.checked).map((x) => x.value) };
    delete r.loiter_seconds;  // el merodeo ya no existe
    if ($("#r-open").value && $("#r-close").value) r.opening_hours = { start: $("#r-open").value, end: $("#r-close").value };
    else delete r.opening_hours;
    // los gestos van aparte de las reglas. Si se dejaron como los del tipo de cámara (y no estaban elegidos a mano),
    // no se manda la lista: así siguen al tipo si después se cambia (p. ej. a «Caja», que no mira gestos)
    const marked = $$(".g-en").filter((x) => x.checked).map((x) => x.value);
    const pinned = cam.gestures && Array.isArray(cam.gestures.enabled);
    // lo que regiría por el tipo con las reglas que se están guardando (con todas apagadas, el tipo no prende gestos)
    const typeNow = r.enabled_rules.length ? (cam.profile_gestures || cam.gestures_default || []) : [];
    const sameAsType = [...marked].sort().join() === [...typeNow].sort().join();
    const body = { zones, rules: r };
    if (pinned || !sameAsType) body.gestures = { ...(cam.gestures || {}), enabled: marked };
    try {
      const saved = await api("PUT", `/api/cameras/${id}/analysis`, body);
      Object.assign(cam, saved);
      // las casillas, como quedó guardado (lo elegido a mano o lo del tipo)
      const on = new Set((cam.gestures && cam.gestures.enabled) || cam.gestures_default || []);
      $$(".g-en").forEach((x) => { x.checked = on.has(x.value); });
      toast("Guardado. El agente aplica los cambios en menos de un minuto.");
    } catch (e) { fail(e); }
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
        ${channels.length <= 1 ? `<br><span class="muted">Este cliente solo tiene el link/portal. Agregá Telegram, WhatsApp o email en Clientes para enviarle directo.</span>` : ""}
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

// ------------------------------------------------------------------ resumen
async function summaryView(v) {
  const [stats, pend, alarms] = await Promise.all([
    api("GET", "/api/stats?days=7"),
    api("GET", "/api/events?label=sin_etiqueta&limit=8"),
    api("GET", "/api/alarms?limit=1").catch(() => ({ counts_7d: {} })),
  ]);
  await Promise.all([loadCameras(), loadClients()]);
  const c7 = alarms.counts_7d || {};
  const alarmsTotal = Object.values(c7).reduce((a, b) => a + b, 0);
  const online = S.cameras.filter((c) => c.enabled !== false && c.status && c.status.agent_online && ["en_linea", "escuchando", "fuera_de_horario"].includes(c.status.state));
  const trouble = S.cameras.filter((c) => c.enabled !== false && !online.includes(c));
  const tile = (n, t, hint, href) => `<a class="panel stat" href="${href}" style="text-decoration:none;color:inherit"><span class="n">${n}</span><span>${t}</span>${hint ? `<span class="muted">${hint}</span>` : ""}</a>`;
  v.innerHTML = `
    <h1>Resumen</h1>
    <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(170px,1fr))">
      ${tile(pend.total, "Alertas sin revisar", "a etiquetar", "#/revisar")}
      ${tile(stats.total, "Alertas (7 días)", `${stats.by_label.hurto || 0} hurtos confirmados`, "#/alertas")}
      ${tile(alarmsTotal, "Alarmas del grabador (7 días)", alarmsTotal ? `${pct((c7.sin_persona || 0) / alarmsTotal)} descartadas solas` : "", "#/alarmas")}
      ${tile(`${online.length}/${S.cameras.filter((c) => c.enabled !== false).length}`, "Cámaras conectadas", trouble.length ? `${trouble.length} con problemas` : "", "#/camaras")}
      ${tile(S.clients.length, "Clientes", "", "#/clientes")}
    </div>
    ${trouble.length ? `<div class="panel" style="margin-top:14px"><b>Cámaras con problemas</b><div class="table-wrap"><table><tbody>${trouble.map((c) => `<tr><td>${esc(clientName(c.client_id))} · ${esc(c.name)}</td><td>${statusChip(c.status)}</td><td><a href="#/camara/${c.id}">ver</a></td></tr>`).join("")}</tbody></table></div></div>` : ""}
    <div class="row" style="margin-top:18px"><h2 style="margin:0">Últimas alertas sin revisar</h2><span class="spacer"></span><a class="btn primary" href="#/revisar">Revisar</a></div>
    <div class="grid" style="margin-top:10px">${pend.items.map(eventCard).join("") || `<div class="empty">No hay alertas pendientes. Cuando una cámara detecte algo, aparece acá.</div>`}</div>
    <p class="muted" style="margin-top:22px">Más: <a href="#/avisos">Avisos enviados a clientes</a> · <a href="#/videos">Videos para entrenar</a> · <a href="#/entrenamiento">Entrenamiento del modelo</a> · <a href="guia.html" target="_blank" rel="noopener">Guía de conexión</a></p>`;
  S.lastList = pend.items.map((e) => e.id);
  v.addEventListener("click", (e) => { const c = e.target.closest(".card[data-id]"); if (c) location.hash = "#/evento/" + c.dataset.id; });
}

// ------------------------------------------------------------------ métricas por cliente (planes comercio / estación)
const fmtDur = (s) => { s = Math.round(s || 0); return s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : s >= 90 ? `${Math.round(s / 60)} min` : `${s} s`; };
const avg = (sum, n) => (n ? sum / n : 0);

const fmtDay = (iso) => { const [y, m, d] = String(iso).split("-"); return d ? `${d}/${m}` : iso; };
const fmtFull = (iso) => { const [y, m, d] = String(iso).split("-"); return d ? `${d}/${m}/${y}` : iso; };

function barChart(items, key, label) {
  // gráfico de barras en SVG, sin librerías: items = [{hour|date, ...}]
  if (!items.length) return `<div class="empty">Sin datos en este período.</div>`;
  const vals = items.map((x) => +x[key] || 0), max = Math.max(1, ...vals);
  const W = 720, H = 180, T = 16, pad = 26, bw = Math.max(4, (W - pad * 2) / items.length - 4);  // T: lugar para el número de arriba
  return `<svg viewBox="0 0 ${W} ${H + 30 + T}" style="width:100%;max-width:${W}px;display:block">
    ${items.map((x, i) => { const v = vals[i], h = Math.round((H - 10) * v / max), xx = pad + i * ((W - pad * 2) / items.length);
      return `<rect x="${xx}" y="${T + H - h}" width="${bw}" height="${h}" rx="3" fill="var(--accent)"><title>${esc(x.hour !== undefined ? x.hour + ":00" : x.date)}: ${v} ${label}</title></rect>
        ${v ? `<text x="${xx + bw / 2}" y="${T + H - h - 4}" font-size="10" text-anchor="middle" fill="currentColor">${v}</text>` : ""}
        <text x="${xx + bw / 2}" y="${T + H + 14}" font-size="10" text-anchor="middle" fill="var(--muted)">${esc(x.hour !== undefined ? x.hour + "h" : fmtDay(x.date))}</text>`; }).join("")}
  </svg>`;
}

async function metricsView(v, clientId) {
  await loadClients();
  if (!clientId && isStaff() && S.clients.length) clientId = S.clients[0].id;  // el primero, y se cambia arriba
  const client = S.clients.find((c) => c.id === clientId);
  if (!client) { v.innerHTML = `<div class="empty">${isStaff() ? "Todavía no hay clientes." : "Tu usuario no tiene un local asociado."}</div>`; return; }
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const days = +(q.get("days") || 1);
  const [m, stats] = await Promise.all([
    api("GET", `/api/metrics?client_id=${clientId}&days=${days}`),
    api("GET", `/api/stats?client_id=${clientId}&days=${days}`).catch(() => null),
  ]);
  const t = m.total, plans = m.plans || [];
  const tile = (n, label, hint) => `<div class="panel stat"><span class="n">${n}</span><span>${label}</span>${hint ? `<span class="muted">${hint}</span>` : ""}</div>`;
  const period = (d) => `<a class="btn small ${days === d ? "primary" : ""}" href="#/${isStaff() ? "metricas?client_id=" + clientId + "&" : "resumen?"}days=${d}">${d === 1 ? "Hoy" : d + " días"}</a>`;
  const zones = Object.entries(t.zones || {}).sort((a, b) => b[1].visits - a[1].visits);
  const sinDatos = !t.frames;
  const pdf = `/api/metrics/report.pdf?days=${days}` + (isStaff() ? `&client_id=${clientId}` : "");
  v.innerHTML = `
    <div class="row"><h1>${isStaff() ? "Métricas" : `Resumen de ${esc(client.name)}`}</h1>
      ${isStaff() ? `<select id="m-client" class="inline-select">${options(S.clients.map((c) => [c.id, c.name]), clientId)}</select>` : ""}
      <span class="spacer"></span>${period(1)}${period(7)}${period(30)}
      <a class="btn small" href="${pdf}" download title="El reporte de este período en PDF, para guardarlo o mandarlo">⬇ Descargar PDF</a></div>
    <p class="muted">${days === 1 ? `Hoy, ${fmtFull(m.until)}` : `Del ${fmtFull(m.since)} al ${fmtFull(m.until)}`} · hora del local · planes: ${plans.map((p) => esc(PLAN_NAMES[p] || p)).join(", ") || "ninguno"}</p>
    ${sinDatos ? `<div class="empty">Todavía no hay métricas en este período. Se generan mientras la PC central analiza las cámaras en modo "En vivo" (se envían cada 5 minutos).</div>` : ""}
    ${plans.includes("comercio") || (!plans.includes("estacion") && t.people) ? `
      <h2>Gente</h2>
      <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(170px,1fr))">
        ${t.door ? tile(t.entries, "Entraron", `${t.exits} salieron${t.staff ? ` · ${t.staff} de personal descontados` : ""}`)
          : t.entries ? tile(t.entries, "Pasaron por la entrada", "entrando o saliendo: falta marcar el sentido de entrada")
          : tile("—", "Entraron", isStaff() ? "falta marcar la zona Entrada, con su sentido, en la cámara de la puerta" : "todavía no se cuenta la puerta")}
        ${tile(fmtDur(avg(t.people_dwell_s, t.people)), "Tiempo promedio en cámara", "")}
        ${tile(t.people_max, "Máximo al mismo tiempo", "")}
        ${zones.length ? tile(esc(zones[0][0]), "Zona más visitada", `${zones[0][1].visits} visitas`) : ""}
      </div>
      ${t.door || t.entries ? `<div class="panel" style="margin-top:12px"><b>${t.door ? "Entraron" : "Pasaron por la entrada"} ${days === 1 ? "por hora" : "por día"}</b>${barChart(days === 1 ? m.by_hour : m.by_day, "entries", t.door ? "entraron" : "pasaron")}</div>` : ""}
      ${zones.length ? `<div class="panel" style="margin-top:12px"><b>A dónde van</b><div class="table-wrap"><table><thead><tr><th>Zona</th><th>Tipo</th><th>Visitas</th><th>Tiempo promedio</th><th></th></tr></thead>
        <tbody>${zones.map(([name, z]) => `<tr><td>${esc(name)}</td><td>${esc(ZONE_NAMES[z.type] || z.type)}</td><td>${z.visits}</td><td>${fmtDur(avg(z.dwell_s, z.visits))}</td><td style="width:40%"><div class="bar"><i style="width:${Math.round(100 * z.visits / Math.max(1, zones[0][1].visits))}%"></i></div></td></tr>`).join("")}</tbody></table></div></div>` : ""}` : ""}
    ${plans.includes("estacion") ? `
      <h2>Estación de servicio</h2>
      <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(170px,1fr))">
        ${tile(t.vehicles, "Vehículos que llegaron", `${t.pump_vehicles} pasaron por un surtidor`)}
        ${tile(fmtDur(avg(t.vehicles_dwell_s, t.vehicles)), "Tiempo promedio en la estación", "")}
        ${tile(t.pump_wait_n ? fmtDur(avg(t.pump_wait_s, t.pump_wait_n)) : "—", "Espera hasta el pistero", t.pump_wait_n ? `${t.pump_wait_n} atendidos` : "necesita zona Surtidor")}
        ${tile(t.pump_unattended, "Se fueron sin atención", "vehículos en el surtidor sin que llegara nadie")}
        ${tile(t.shop_visits, "Bajaron a la tienda", "personas que entraron a la zona Tienda")}
        ${tile(t.vehicles_max, "Máximo de vehículos a la vez", "")}
      </div>
      <div class="panel" style="margin-top:12px"><b>${days === 1 ? "Vehículos por hora" : "Vehículos por día"}</b>${barChart(days === 1 ? m.by_hour : m.by_day, "vehicles", "vehículos")}</div>
      ${!plans.includes("comercio") ? `<div class="panel" style="margin-top:12px"><b>Personas</b> promedio ${fmtDur(avg(t.people_dwell_s, t.people))} en cámara · máximo ${t.people_max} al mismo tiempo</div>` : ""}` : ""}
    ${plans.includes("hurto") && stats ? `
      <h2>Alertas de hurto</h2>
      <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(170px,1fr))">
        ${tile(stats.total, "Alertas en el período", "")}
        ${tile(stats.by_label.hurto || 0, "Confirmadas como hurto", "")}
        ${tile(stats.by_label.sospechoso || 0, "Sospechosas", "")}
        ${tile(stats.by_label.sin_etiqueta || 0, "Sin revisar todavía", "")}
      </div>
      <p><a class="btn" href="#/alertas?client_id=${clientId}">Ver las alertas</a></p>` : ""}
    ${m.by_camera.length > 1 ? `<div class="panel" style="margin-top:12px"><b>Por cámara</b><div class="table-wrap"><table><thead><tr><th>Cámara</th><th>Tiempo promedio</th>${plans.includes("estacion") ? "<th>Vehículos</th>" : ""}<th>Máximo a la vez</th></tr></thead>
      <tbody>${m.by_camera.map((c) => `<tr><td>${esc(c.name)}</td><td>${fmtDur(avg(c.people_dwell_s, c.people))}</td>${plans.includes("estacion") ? `<td>${c.vehicles}</td>` : ""}<td>${c.people_max}</td></tr>`).join("")}</tbody></table></div></div>` : ""}
    <p class="muted small" style="margin-top:18px">Cómo se mide: "Entraron" cuenta a quien cruza la puerta hacia adentro en la cámara de la entrada (una sola cámara, así nadie se cuenta dos veces). No se muestra una suma de personas de todas las cámaras: la misma persona aparece en varias y, si sale del cuadro y vuelve, cuenta de nuevo. "A dónde van" cuenta visitas: quien vuelve a una zona suma otra visita. Personal: quien pisa la zona de empleados (detrás del mostrador, depósito) o aparece con el local cerrado no cuenta como cliente; los demás empleados sí. No reconoce caras. "Espera hasta el pistero" es el tiempo desde que el vehículo entra a la zona Surtidor hasta que una persona pisa esa misma zona.</p>`;
  const sel = $("#m-client", v);
  if (sel) sel.addEventListener("change", () => { location.hash = `#/metricas?client_id=${sel.value}&days=${days}`; });
}

// ------------------------------------------------------------------ clientes (administradores y empleados)
async function clientsView(v) {
  await loadClients(true);
  await loadCameras();
  const nCams = (id) => S.cameras.filter((c) => c.client_id === id).length;
  v.innerHTML = `
    <div class="row"><h1>Clientes</h1><span class="spacer"></span><button id="add" class="primary">Nuevo cliente</button></div>
    <p class="muted">Cada cliente es un comercio. Tocá el nombre para ver sus cámaras y agregar nuevas. El contacto se usa solo cuando alguien decide avisarle de un posible hurto.</p>
    <div id="client-form"></div>
    <div class="table-wrap panel" style="padding:0;margin-top:12px"><table><thead><tr><th>Nombre</th><th>Cámaras</th><th>Planes</th><th>Zona horaria</th><th>WhatsApp</th><th>Email</th><th>Telegram</th><th>Canales de aviso</th><th></th></tr></thead>
      <tbody>${S.clients.map((c) => `<tr><td><a href="#/cliente/${c.id}"><b>${esc(c.name)}</b></a></td><td><a href="#/cliente/${c.id}">${nCams(c.id)}</a></td><td>${(c.plans || []).map((p) => `<span class="chip">${esc(PLAN_NAMES[p] || p)}</span>`).join(" ")}</td><td>${esc(c.timezone)}</td><td class="mono">${esc(c.contact_whatsapp || "—")}</td><td class="mono">${esc(c.contact_email || "—")}</td><td class="mono">${esc(c.telegram_chat_id || "—")}</td><td>${(c.notify_channels || []).map((x) => esc(CHANNEL_NAMES[x] || x)).join(", ")}</td>
        <td class="row"><button class="small" data-edit="${c.id}">Editar</button><a class="btn small" href="#/metricas?client_id=${c.id}">Métricas</a><a class="btn small primary" href="#/cliente/${c.id}">Ver cámaras</a>${isAdmin() ? `<button class="small ghost" data-delclient="${c.id}" title="Borra el cliente con sus cámaras, alertas, métricas y usuarios">Eliminar</button>` : ""}</td></tr>`).join("") || `<tr><td colspan="9" class="muted">Todavía no hay clientes.</td></tr>`}</tbody></table></div>`;
  const form = (c = {}) => {
    $("#client-form").innerHTML = `<form class="form panel" id="cf" style="margin-top:12px">
      <label>Nombre<input name="name" required value="${esc(c.name || "")}"></label>
      <label>Zona horaria<input name="timezone" value="${esc(c.timezone || "America/Montevideo")}"></label>
      <label>WhatsApp del cliente (opcional)<input name="contact_whatsapp" value="${esc(c.contact_whatsapp || "")}" placeholder="+59891234567"></label>
      <label>Email del cliente (opcional)<input name="contact_email" type="email" value="${esc(c.contact_email || "")}" placeholder="dueno@local.com"></label>
      <label>Chat de Telegram (opcional)<input name="telegram_chat_id" value="${esc(c.telegram_chat_id || "")}" placeholder="-1001234567890"></label>
      <label>Webhook (opcional)<input name="webhook_url" value="${esc(c.webhook_url || "")}" placeholder="https://…"></label>
      <fieldset><legend>Planes contratados</legend>${Object.entries(PLAN_NAMES).map(([k, t]) => `<label class="inline"><input type="checkbox" name="plan" value="${k}" ${(c.plans || ["hurto"]).includes(k) ? "checked" : ""}> ${t}</label>`).join("")}
        <small class="muted" style="flex-basis:100%">Hurtos: alertas con clip. Comercio: cuánta gente entra, cuánto se queda y a dónde va (zonas Entrada y Góndola). Estación: vehículos, espera hasta el pistero y visitas a la tienda (zonas Surtidor y Tienda; el detector también marca vehículos).</small></fieldset>
      <div class="row"><button class="primary" type="submit">${c.id ? "Guardar" : "Crear cliente"}</button><button type="button" id="cancel">Cancelar</button></div>
    </form>`;
    $("#cancel").addEventListener("click", () => ($("#client-form").innerHTML = ""));
    $("#cf").addEventListener("submit", async (e) => {
      e.preventDefault(); const fd = new FormData(e.target); const f = Object.fromEntries(fd); delete f.plan; f.plans = fd.getAll("plan");
      for (const k of ["telegram_chat_id", "webhook_url", "contact_whatsapp", "contact_email"]) if (!f[k]) f[k] = null;
      try { await api(c.id ? "PUT" : "POST", c.id ? "/api/clients/" + c.id : "/api/clients", f); toast(c.id ? "Cliente guardado" : "Cliente creado"); S._clientsAt = 0; route(); } catch (err) { fail(err); }
    });
  };
  $("#add").addEventListener("click", () => form());
  $$("[data-edit]", v).forEach((b) => b.addEventListener("click", () => form(S.clients.find((c) => c.id === +b.dataset.edit))));
  $$("[data-delclient]", v).forEach((b) => b.addEventListener("click", async () => {
    const c = S.clients.find((x) => x.id === +b.dataset.delclient);
    const typed = prompt(`Esto borra "${c.name}" con TODAS sus cámaras, alertas y clips, alarmas, métricas y usuarios. No se puede deshacer.\n\nEscribí el nombre del cliente para confirmar:`);
    if (typed === null) return;
    if (typed.trim() !== c.name) return toast("El nombre no coincide: no se borró nada", true);
    try { await api("DELETE", "/api/clients/" + c.id); toast("Cliente eliminado"); S._clientsAt = 0; S._camsAt = 0; route(); } catch (e) { fail(e); }
  }));
  const editar = +(new URLSearchParams(location.hash.split("?")[1] || "").get("editar") || 0);
  if (editar) form(S.clients.find((c) => c.id === editar) || {});
}

// ------------------------------------------------------------------ administración (solo administradores)
async function adminView(v) {
  const [users, agents] = await Promise.all([api("GET", "/api/users"), api("GET", "/api/agents")]);
  await loadClients();
  const admins = users.filter((u) => u.role === "admin" && u.active).length;
  v.innerHTML = `
    <h1>Administración</h1>
    <h2>Usuarios</h2>
    <p class="muted"><b>Administradores</b> (${admins} activos): crean usuarios y agentes y conectan Hik-Partner Pro.
      <b>Empleados</b>: revisan y etiquetan los clips, crean clientes y configuran sus cámaras.
      <b>Clientes</b>: el dueño del comercio; solo ve lo suyo.</p>
    <div class="table-wrap panel" style="padding:0"><table><thead><tr><th>Usuario</th><th>Rol</th><th>Cliente</th><th>Activo</th><th></th></tr></thead>
      <tbody>${users.map((u) => `<tr><td>${esc(u.username)}${u.id === S.me.id ? ` <span class="chip">vos</span>` : ""}</td><td>${ROLE_NAMES[u.role]}</td><td>${u.client_id ? esc(clientName(u.client_id)) : "—"}</td><td>${u.active ? "Sí" : `<span class="chip bad">No</span>`}</td>
        <td><button class="small" data-user="${u.id}">Editar</button></td></tr>`).join("")}</tbody></table></div>
    <div id="user-form"></div>
    <div class="row" style="margin-top:10px"><button id="new-user" class="primary">Nuevo usuario</button></div>

    <h2>Agentes</h2>
    <p class="muted">El <b>agente</b> es el programa que corre en la PC central: llega a los grabadores de todos los clientes por internet (puertos abiertos en el router de cada uno), baja el tramo de cada alarma y lo analiza. Con una PC alcanza; si se suma otra, se reparten las cámaras.</p>
    <div class="panel" style="margin-bottom:12px"><b>Instalar el programa en una PC central (Windows)</b>
      <ol style="margin:6px 0 0 18px;padding:0">
        <li>Descargar <a href="/api/app/download">la app para Windows</a>. Después se actualiza sola con el botón «Buscar actualización» de la app.</li>
        <li>Descomprimirlo en una carpeta y abrir el programa (la primera vez se renombra solo a <span class="mono">Analítica Crítica.exe</span>). Si Windows lo frena: "Más información → Ejecutar de todas formas".</li>
        <li>Enter para el servidor y pegar el <b>token</b> del agente (botón "Nuevo token" acá abajo; se muestra una sola vez). Queda guardado en <span class="mono">agente.env</span> al lado del programa.</li>
        <li>Dejar la ventana abierta. Para que arranque con Windows: Programador de tareas → "Al iniciar sesión" → Analítica Crítica.exe.</li>
      </ol></div>
    <div class="table-wrap panel" style="padding:0"><table><thead><tr><th>Agente</th><th>Cliente</th><th>Estado</th><th>Última conexión</th><th>Pendientes de subir</th><th></th></tr></thead>
      <tbody>${agents.map((a) => `<tr><td>${esc(a.name)}</td><td>${a.central ? "Todos" : esc(clientName(a.client_id))}</td><td>${a.online ? `<span class="chip ok">En línea</span>` : `<span class="chip bad">Desconectado</span>`}</td>
        <td>${fmtDate(a.last_seen)}${a.status?.alarms_with_person != null ? `<div class="muted">Alarmas: ${a.status.alarms_with_person} con persona · ${a.status.alarms_discarded} descartadas · ${a.status.alarms_failed} con error</div>` : ""}</td><td>${a.status?.outbox_pending ?? "—"}</td><td class="row"><button class="small" data-rotate="${a.id}">Nuevo token</button><button class="small ghost" data-delagent="${a.id}">Eliminar</button></td></tr>`).join("") || `<tr><td colspan="6" class="muted">Sin agentes.</td></tr>`}</tbody></table></div>
    <form class="form panel" id="af" style="margin-top:10px">
      <label>Nombre de la PC<input name="name" required placeholder="PC central"></label>
      <button class="primary" type="submit">Crear agente</button>
    </form>
    <div id="token"></div>`;
  const userForm = (u = {}) => {
    $("#user-form").innerHTML = `<form class="form panel" id="uf" style="margin-top:10px">
      <label>Usuario<input name="username" required minlength="3" value="${esc(u.username || "")}"></label>
      <label>${u.id ? "Nueva clave (vacío = sin cambios)" : "Clave (mín. 8)"}<input name="password" type="password" ${u.id ? "" : "required"} minlength="8" autocomplete="new-password"></label>
      <label>Rol<select name="role">${options(Object.entries(ROLE_NAMES), u.role || "entrenador")}</select></label>
      <label>Cliente (solo rol cliente)<select name="client_id">${options(S.clients.map((c) => [c.id, c.name]), u.client_id, "—")}</select></label>
      ${u.id ? `<label class="inline"><input type="checkbox" name="active" ${u.active ? "checked" : ""}> Activo</label>` : ""}
      <div class="row"><button class="primary" type="submit">${u.id ? "Guardar" : "Crear usuario"}</button><button type="button" id="ucancel">Cancelar</button></div>
    </form>`;
    $("#ucancel").addEventListener("click", () => ($("#user-form").innerHTML = ""));
    $("#uf").addEventListener("submit", async (e) => {
      e.preventDefault(); const f = new FormData(e.target);
      const body = { username: f.get("username"), password: f.get("password") || null, role: f.get("role"),
        client_id: f.get("client_id") ? +f.get("client_id") : null, active: u.id ? !!f.get("active") : true };
      try { await api(u.id ? "PUT" : "POST", u.id ? "/api/users/" + u.id : "/api/users", body); toast(u.id ? "Usuario guardado" : "Usuario creado"); route(); } catch (err) { fail(err); }
    });
  };
  $("#new-user").addEventListener("click", () => userForm());
  $$("[data-user]", v).forEach((b) => b.addEventListener("click", () => userForm(users.find((u) => u.id === +b.dataset.user))));
  const showToken = (a) => {
    $("#token").innerHTML = `<div class="panel" style="margin-top:10px"><b>Token del agente "${esc(a.name)}"</b> — copialo ahora, no se vuelve a mostrar.
      <div class="token-box mono" style="margin:8px 0">${esc(a.token)}</div>
      <p class="muted">En la PC central:</p><div class="token-box mono">python -m agent --server ${esc(location.origin)} --token ${esc(a.token)}</div></div>`;
  };
  $("#af").addEventListener("submit", async (e) => {
    e.preventDefault(); const f = new FormData(e.target);
    try { const a = await api("POST", "/api/agents", { client_id: null, name: f.get("name") }); e.target.reset(); showToken(a); } catch (err) { fail(err); }
  });
  $$("[data-delagent]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("¿Eliminar este agente? Sus cámaras quedan sin PC central asignada y el programa que use ese token deja de funcionar.")) return;
    try { await api("DELETE", "/api/agents/" + b.dataset.delagent); toast("Agente eliminado"); S._camsAt = 0; route(); } catch (e) { fail(e); }
  }));
  $$("[data-rotate]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("El token anterior deja de funcionar. ¿Generar uno nuevo?")) return;
    try { showToken(await api("POST", `/api/agents/${b.dataset.rotate}/token`)); } catch (err) { fail(err); }
  }));
}

boot();
