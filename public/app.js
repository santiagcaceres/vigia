"use strict";
/* La web de Analítica Crítica. Desde el 9 nov (decisión del dueño: «en la web, solo info para los clientes; en la app
   todo lo de entrenamiento») la ven:
   - los clientes: su resumen (métricas y PDF), sus alertas, los incidentes que les mandamos y sus cámaras;
   - los administradores: solo las métricas de todos los clientes (con el PDF), por ejemplo desde el celular;
   - los operadores: solo la pantalla para bajar la app de escritorio.
   Todo lo del equipo (clientes, cámaras, zonas, revisar y etiquetar alertas, alarmas, avisos al cliente, videos,
   entrenamiento, usuarios) se hace en la app de escritorio (agent/oficina.py). */

const TYPE_NAMES = { salida_sin_pago: "Salida sin pago", merodeo: "Merodeo", zona_restringida: "Zona restringida", fuera_de_horario: "Fuera de horario", alarma_dvr: "Alarma con persona",
  mano_bolsillo: "Mano al bolsillo", mano_ropa: "Mano dentro de la ropa", mano_bolsa: "Mano en la bolsa", hurto_ia: "Posible hurto (IA de movimientos)" };
const ALARM_TYPES = { VMD: "Movimiento", fielddetection: "Intrusión", linedetection: "Cruce de línea", regionEntrance: "Entrada a zona", regionExiting: "Salida de zona" };
const MODE_NAMES = { alarmas_equipo: "Alarmas del grabador", alarmas: "Alarmas HPP", vivo: "En vivo" };
const alarmTypeName = (t) => String(t || "").split("+").map((x) => ALARM_TYPES[x] || x).join(" + ");
const LABEL_NAMES = { hurto: "Hurto", sospechoso: "Sospechoso", normal: "Normal", falso_positivo: "Falso positivo" };
const ZONE_NAMES = { shelf: "Góndola", aisle: "Pasillo", checkout: "Caja", exit: "Salida", restricted: "Restringida", entrance: "Entrada (conteo)", pump: "Surtidor", shop: "Tienda", staff: "Personal (empleados)", table: "Mesa" };
const PLAN_NAMES = { hurto: "Detección de hurtos", comercio: "Métricas de comercio", estacion: "Estación de servicio", empleados: "Seguimiento de empleados", gastronomia: "Gastronomía" };
const PLAN_KEYS = Object.keys(PLAN_NAMES);  // el orden de siempre (el mismo del servidor)
const PLAN_SHORT = { hurto: "Hurtos", comercio: "Comercio", estacion: "Estación", empleados: "Empleados", gastronomia: "Gastronomía" };
const STATE_NAMES = { escuchando: "Escuchando alarmas", en_linea: "En línea", sin_conexion: "Sin conexión", reconectando: "Reconectando", error: "Error", iniciando: "Iniciando", detenida: "Detenida", fuera_de_horario: "Local cerrado" };

const ROLE_NAMES = { admin: "Administrador", entrenador: "Operador", cliente: "Cliente" };
const S = { me: null, clients: [], cameras: [], lastList: [], timers: [] };

// ------------------------------------------------------------------ utilidades
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString("es-AR", { dateStyle: "short", timeStyle: "medium" }) : "—");
const fmtSec = (s) => { if (s == null) return "—"; s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60; return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(x).padStart(2, "0"); };
const pct = (v) => (v == null ? "—" : Math.round(v * 100) + "%");
const isAdmin = () => S.me && S.me.role === "admin";
const clientName = (id) => (S.clients.find((c) => c.id === id) || {}).name || "—";
/** Planes que contrató un cliente, en el orden de siempre (null si no está cargado). */
const clientPlans = (id) => { const cl = S.clients.find((c) => c.id === id); return cl ? PLAN_KEYS.filter((p) => (cl.plans || ["hurto"]).includes(p)) : null; };
/** Lo que mide en efecto una cámara: `measures` del servidor; si no viene (servidor viejo), lo elegido (`plans`) dentro
 * de lo del cliente, o todo lo del cliente. null si no se sabe. */
function cameraMeasures(cam, own = clientPlans(cam.client_id)) {
  if (Array.isArray(cam.measures)) return PLAN_KEYS.filter((p) => cam.measures.includes(p));
  if (!own) return null;
  const pick = Array.isArray(cam.plans) ? own.filter((p) => cam.plans.includes(p)) : [];
  return pick.length ? pick : own;
}

const NO_POSTS_TEXT = "Sin datos de los puestos: hace falta marcar (en la app) la zona «Personal (empleados)» en las cámaras (detrás del mostrador, la caja por dentro) y que la máquina que las analiza esté actualizada (se actualiza al reiniciarla). A los empleados se los reconoce por esa zona o porque llegan con el local cerrado.";

// la app de escritorio: ahí se hace todo lo del equipo (el servidor redirige al .zip de la última versión)
const APP_BUTTON = `<a class="btn primary" href="/api/app/download">Descargar la app para Windows</a>`;
/** El aviso arriba de las métricas de los administradores: lo demás va en la app. */
const appNote = () => `<div class="panel app-note"><span>Lo demás (clientes, cámaras, alertas, entrenamiento) se hace en la <b>app de escritorio</b>. Entrá con este mismo usuario.</span><span class="spacer"></span>${APP_BUTTON}</div>`;

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
const stopTimers = () => { S.timers.forEach(clearInterval); S.timers = []; };

const typeChip = (e) => `<span class="chip ${esc(e.severity)}">${esc(TYPE_NAMES[e.type] || e.type)}</span>`;
const labelChip = (l) => (l ? `<span class="chip label-${esc(l)}">${esc(LABEL_NAMES[l] || l)}</span>` : `<span class="chip">Sin revisar</span>`);
const options = (items, sel, empty) => (empty !== undefined ? `<option value="">${esc(empty)}</option>` : "") +
  items.map(([v, t]) => `<option value="${esc(v)}" ${String(v) === String(sel ?? "") ? "selected" : ""}>${esc(t)}</option>`).join("");

// ------------------------------------------------------------------ sesión y navegación
function showLogin() {
  S.me = null; stopTimers();  // (sin sesión no se late ni se refresca nada)
  $("#app").classList.add("hidden"); $("#login").classList.remove("hidden");
}

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
  S.clients = []; S.cameras = []; S._clientsAt = S._camsAt = 0; S.lastList = [];  // (otro usuario en la misma pestaña)
  if (S.me.role !== "entrenador") await loadClients(true);
  $("#login").classList.add("hidden");
  $("#app").classList.remove("hidden");
  $("#user-name").textContent = `${S.me.username} · ${ROLE_NAMES[S.me.role] || S.me.role}`;
  // Los clientes ven su local; los administradores, las métricas de todos (con el PDF); los operadores trabajan solo en
  // la app de escritorio.
  const links = S.me.role === "entrenador" ? []
    : isAdmin() ? [["#/metricas", "Métricas"]]
    : [["#/resumen", "Resumen"], ["#/alertas", "Alertas"], ["#/avisos", "Incidentes"], ["#/camaras", "Cámaras"]];
  $("#nav").innerHTML = links.map(([h, t]) => `<a href="${h}">${t}</a>`).join("");
  route();
}

window.addEventListener("hashchange", route);
async function route() {
  if (!S.me) return;
  stopTimers();
  const old = $("#view"), v = old.cloneNode(false); // nodo nuevo: descarta listeners de la vista anterior
  old.replaceWith(v);
  window.scrollTo(0, 0);
  if (S.me.role === "entrenador") return appOnlyView(v);
  const h = location.hash || (isAdmin() ? "#/metricas" : "#/resumen");
  const section = isAdmin() ? "#/metricas" : h.startsWith("#/evento") ? "#/alertas" : h.startsWith("#/camara/") ? "#/camaras" : h;
  $$("#nav a").forEach((a) => a.classList.toggle("active", section.startsWith(a.getAttribute("href"))));
  let m;
  try {
    // los administradores: solo métricas (cualquier otra dirección, p. ej. un link viejo, abre las métricas)
    if (isAdmin()) return await metricsView(v, h.startsWith("#/metricas") ? +(new URLSearchParams(h.split("?")[1] || "").get("client_id") || 0) : 0);
    if ((m = h.match(/^#\/evento\/(\d+)/))) return await eventView(v, +m[1]);
    if ((m = h.match(/^#\/camara\/(\d+)/))) return await cameraView(v, +m[1]);
    if (h.startsWith("#/camaras")) return await clientCamerasView(v, S.me.client_id);
    if (h.startsWith("#/avisos")) return await reportsView(v);
    if (h.startsWith("#/resumen")) return await metricsView(v, S.me.client_id);
    return await eventsView(v);
  } catch (e) { v.innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
}

/** Los operadores: todo su trabajo está en la app de escritorio. */
function appOnlyView(v) {
  $("#nav").innerHTML = "";
  v.innerHTML = `<div class="panel" style="max-width:640px;margin:40px auto">
      <h1>Usá la app de escritorio</h1>
      <p>Todo el trabajo del equipo (clientes, cámaras, zonas, revisar y etiquetar alertas, avisos a los clientes y el entrenamiento de la IA) se hace en la <b>app de escritorio de Analítica Crítica</b>. La web quedó para que los clientes vean su local.</p>
      <p>${APP_BUTTON}</p>
      <p class="muted">Descomprimila y abrí el programa (la primera vez se renombra solo a «Analítica Crítica.exe»). Entrá con este mismo usuario. Después se actualiza sola con «Buscar actualización».</p>
    </div>`;
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

// ------------------------------------------------------------------ alertas del cliente
async function eventsView(v) {
  await loadCameras();
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const stats = await api("GET", "/api/stats?days=30" + (q.get("client_id") ? "&client_id=" + q.get("client_id") : ""));
  v.innerHTML = `
    <div class="row"><h1>Alertas</h1></div>
    <p class="muted">Últimos 30 días: ${stats.total} alertas, ${stats.by_label.sin_etiqueta || 0} sin revisar. Cada alerta es un indicio para revisar, no una prueba.</p>
    <div class="toolbar">
      <label>Cámara<select id="f-camera">${options(S.cameras.map((c) => [c.id, c.name]), q.get("camera_id"), "Todas")}</select></label>
      <label>Tipo<select id="f-type">${options(Object.entries(TYPE_NAMES), q.get("type"), "Todos")}</select></label>
      <label>Revisión<select id="f-label">${options([["sin_etiqueta", "Sin revisar"], ...Object.entries(LABEL_NAMES)], q.get("label"), "Todas")}</select></label>
      <label>Origen<select id="f-source">${options([["alarma", "Alarmas del grabador"], ["vivo", "En vivo"], ["prueba", "Prueba (pantalla)"], ["importado", "Videos importados"]], q.get("source"), "Todos")}</select></label>
      <label>Alerta<select id="f-alert">${options([["true", "Solo alertas"], ["false", "Filtrados por el modelo"]], q.get("alert"), "Todos")}</select></label>
    </div>
    <div id="list" class="grid"></div>
    <div class="row" style="margin-top:14px"><span id="count" class="muted"></span><span class="spacer"></span><button id="more" class="hidden">Cargar más</button></div>`;
  const ids = ["camera", "type", "label", "source", "alert"];
  ids.forEach((k) => $("#f-" + k).addEventListener("change", () => {
    const p = new URLSearchParams();
    ids.forEach((x) => { const el = $("#f-" + x); if (el.value) p.set(x === "camera" ? "camera_id" : x, el.value); });
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
    ${e.snapshot_url ? `<img loading="lazy" src="${esc(e.snapshot_url)}" alt="">` : ""}
    <div class="body">
      <div class="row">${typeChip(e)}${labelChip(e.label)}${e.alert ? "" : `<span class="chip" title="El modelo lo consideró poco probable">filtrado</span>`}</div>
      <div class="title">${esc(e.camera_name)}</div>
      <div class="muted">${fmtDate(e.occurred_at)}${e.source === "importado" ? " · video importado" : ""}${e.source === "alarma" ? " · alarma " + esc(alarmTypeName(e.alarm?.dvr_event)) : ""}${e.source === "prueba" ? " · prueba desde la pantalla" : ""}${e.score != null ? ` · puntaje ${pct(e.score)}` : ""}</div>
    </div></div>`;
}

function mediaHtml(e) {
  if (e.clip_url) return `<div class="clip-box"><video src="${esc(e.clip_url)}" controls playsinline poster="${esc(e.snapshot_url || "")}"></video></div>`;
  if (e.snapshot_url) return `<div class="clip-box"><img src="${esc(e.snapshot_url)}" alt=""></div><p class="muted">El clip todavía no llegó (el agente lo sube unos segundos después).</p>`;
  return `<div class="empty">Sin imagen</div>`;
}

const originText = (e) => e.source === "importado" ? `Video importado (seg. ${fmtSec(e.video_ts)})`
  : e.source === "alarma" ? `Alarma del DVR: ${esc(alarmTypeName(e.alarm?.dvr_event))}${e.alarm?.triggers > 1 ? ` (${esc(e.alarm.triggers)} disparos)` : ""}`
  : e.source === "prueba" ? "Prueba desde la pantalla de HPP" : "En vivo";
const CLIP_NAMES = { nvr: "Grabación del grabador", buffer: "Buffer del agente", video: "Del video importado", prueba: "Grabado de la pantalla" };

async function eventView(v, id) {
  const e = await api("GET", "/api/events/" + id);
  const i = S.lastList.indexOf(id);
  const prev = i > 0 ? S.lastList[i - 1] : null, next = i >= 0 && i < S.lastList.length - 1 ? S.lastList[i + 1] : null;
  v.innerHTML = `
    <div class="row"><a href="#/alertas">← Alertas</a><span class="spacer"></span>
      ${prev ? `<a class="btn" href="#/evento/${prev}">← Anterior</a>` : ""}${next ? `<a class="btn" href="#/evento/${next}">Siguiente →</a>` : ""}</div>
    <div class="two" style="margin-top:12px">
      <div>${mediaHtml(e)}</div>
      <div class="panel">
        <div class="row">${typeChip(e)}${labelChip(e.label)}</div>
        <h1 style="margin-top:8px">${esc(e.camera_name)}</h1>
        <p>${esc(e.description)}</p>
        <dl class="kv">
          <dt>Fecha</dt><dd>${fmtDate(e.occurred_at)}</dd>
          <dt>Origen</dt><dd>${originText(e)}</dd>
          <dt>Clip</dt><dd>${e.clip_url ? (CLIP_NAMES[e.clip_source] || "Clip") + ` · <a href="${esc(e.clip_url)}" download>Descargar</a>` : "—"}</dd>
          ${e.score != null ? `<dt>Puntaje</dt><dd>${pct(e.score)} ${e.alert ? "" : "(no se envió alerta)"}</dd>` : ""}
        </dl>
        ${e.label ? `<p style="margin-top:14px">Revisión: ${labelChip(e.label)}</p>` : ""}
      </div>
    </div>`;
}

// ------------------------------------------------------------------ cámaras del cliente
function statusChip(st) {
  if (!st) return `<span class="chip">Sin datos del agente</span>`;
  if (!st.agent_online) return `<span class="chip bad">Agente desconectado</span>`;
  const ok = st.state === "en_linea" || st.state === "escuchando";
  if (st.state === "fuera_de_horario") return `<span class="chip" title="Fuera del horario del local no se mira (ahorra video y análisis)">Local cerrado</span>`;
  return `<span class="chip ${ok ? "ok" : "bad"}" title="${esc(st.error || "")}">${esc(STATE_NAMES[st.state] || st.state)}${ok && st.fps ? ` · ${esc(st.fps)} fps` : ""}</span>`;
}
/** Estado de una cámara en su tarjeta y su ficha. */
const camChip = (c) => (c.enabled ? (c.mode === "alarmas" ? `<span class="chip">${MODE_NAMES[c.mode]}</span>` : statusChip(c.status)) : `<span class="chip">deshabilitada</span>`);

const previewUrl = (c) => (c.has_preview ? `/api/cameras/${c.id}/preview?t=${Math.floor(Date.now() / 30000)}` : null);
const camThumb = (c) => (previewUrl(c) ? `<img loading="lazy" src="${previewUrl(c)}" alt="">` : `<div class="thumb-empty">Sin imagen todavía<br><small>la PC central la manda al conectarse</small></div>`);

/** «Cámaras de <comercio>»: miniaturas con la última imagen; un clic abre la ficha de la cámara. */
async function clientCamerasView(v, clientId) {
  await Promise.all([loadClients(), loadCameras()]);
  const client = S.clients.find((c) => c.id === clientId) || { id: clientId, name: S.me.client_name || "mi comercio" };
  const cams = S.cameras.filter((c) => c.client_id === clientId);
  v.innerHTML = `
    <div class="row"><h1>Cámaras de ${esc(client.name)}</h1></div>
    <div class="grid cams" style="margin-top:12px">${cams.map((c) => `<a class="card cam" href="#/camara/${c.id}" data-id="${c.id}">${camThumb(c)}
      <div class="body"><div class="row"><span class="title">${esc(c.name)}</span><span class="spacer"></span>${camChip(c)}</div>
      <span class="muted mono">${esc(MODE_NAMES[c.mode] || "")}</span></div></a>`).join("") || `<div class="empty">Este cliente todavía no tiene cámaras.</div>`}</div>`;
  // cada 30 s, la imagen nueva (la PC central manda una captura cada 5 min)
  every(30000, () => $$("a.cam img", v).forEach((i) => { const c = cams.find((x) => x.id === +i.closest("a").dataset.id); if (c) i.src = previewUrl(c); }));
}

/** Ficha de una cámara: imagen grande (en vivo mientras se mira), estado, qué mide y sus zonas. */
async function cameraView(v, id) {
  const cam = await api("GET", "/api/cameras/" + id);
  await loadClients();
  const measures = cameraMeasures(cam);  // qué planes del cliente mide (null si no se sabe)
  v.innerHTML = `
    <div class="row"><a href="#/camaras">← Cámaras de ${esc(clientName(cam.client_id))}</a></div>
    <div class="row"><h1>${esc(cam.name)}</h1>${camChip(cam)}<span class="spacer"></span>
      <a class="btn" href="#/alertas?camera_id=${cam.id}">Alertas</a></div>
    <div class="two" style="margin-top:12px">
      <div class="panel cam-big"><div class="live-box">${previewUrl(cam) ? `<img id="live-img" src="${previewUrl(cam)}" alt="">` : `<img id="live-img" alt="" class="hidden"><div class="thumb-empty" id="live-empty">Sin imagen todavía</div>`}<span class="live-tag" id="live-state">Conectando con la PC central…</span></div>
        <p class="muted" style="margin:8px 0 0" id="live-note">${cam.has_preview ? `Última imagen: ${fmtDate(cam.preview_at)}.` : ""} La PC central manda un cuadro por segundo con las personas marcadas mientras esta pantalla esté abierta.</p></div>
      <div class="panel"><h3 style="margin-top:0">Detalle</h3>
        <dl class="kv">
          <dt>Modo</dt><dd>${esc(MODE_NAMES[cam.mode] || cam.mode)}</dd>
          ${measures ? `<dt>Mide</dt><dd>${measures.length ? measures.map((p) => `<span class="chip">${esc(PLAN_NAMES[p] || p)}</span>`).join(" ") : `<span class="muted">nada: el cliente ya no tiene lo que se le eligió (elegilo en «Editar»)</span>`}</dd>` : ""}
          <dt>Zonas</dt><dd>${(cam.zones || []).length ? (cam.zones || []).map((z) => `${esc(z.name)} (${esc(ZONE_NAMES[z.type] || z.type)}${z.wait ? " · espera" : ""})`).join(", ") : `<span class="muted">ninguna: sin zonas no se disparan reglas</span>`}</dd>
          ${cam.status && cam.status.error ? `<dt>Error</dt><dd class="error">${esc(cam.status.error)}</dd>` : ""}
        </dl></div>
    </div>`;
  liveView(v, cam);
}

/** Vista en vivo liviana: latido cada 10 s ("alguien mira") y un cuadro por segundo con los recuadros. */
function liveView(v, cam) {
  const img = $("#live-img", v), tag = $("#live-state", v);
  let frameUrl = null, active = false;
  const setTag = (txt, cls) => { tag.textContent = txt; tag.className = "live-tag " + (cls || ""); };
  const beat = async () => {
    if (document.visibilityState === "hidden") return; // pestaña oculta: no gastar
    try {
      const r = await api("POST", `/api/cameras/${cam.id}/live`);
      frameUrl = r.frame_url; active = r.active;
      if (!r.agent_online) setTag("PC central desconectada", "bad");
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
    probe.onload = () => { img.src = url; img.classList.remove("hidden"); const e = $("#live-empty", v); if (e) e.remove(); };
    probe.src = url;
  });
}

// ------------------------------------------------------------------ incidentes (los avisos que le mandamos al cliente)
async function reportsView(v) {
  v.innerHTML = `
    <h1>Incidentes</h1>
    <p class="muted">Clips que te enviamos cuando detectamos una situación sospechosa en tus cámaras.</p>
    <div id="rp-body"></div>`;
  const data = await api("GET", "/api/reports?limit=100");
  if (!data.items.length) { $("#rp-body").innerHTML = `<div class="empty">Todavía no recibiste avisos.</div>`; return; }
  $("#rp-body").innerHTML = `<div class="grid">${data.items.map((r) => `<div class="card">
    <div class="body"><div class="row"><span class="chip bad">Posible hurto</span></div>
      <p>${esc(r.message)}</p>
      <div class="muted">${fmtDate(r.occurred_at || r.sent_at)}</div>
      <div class="row" style="margin-top:8px">${r.has_clip ? `<a class="btn small" href="${esc(r.link)}" target="_blank" rel="noopener">Ver el clip</a>` : `<span class="muted">Clip no disponible</span>`}</div>
    </div></div>`).join("")}</div>`;
}

// ------------------------------------------------------------------ métricas (el resumen del cliente y las de los administradores)
const fmtDur = (s) => { s = Math.round(s || 0); return s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : s >= 90 ? `${Math.round(s / 60)} min` : `${s} s`; };
const avg = (sum, n) => (n ? sum / n : 0);
// Redondeo como Python (round y format redondean al par en los empates exactos: 0,5 → 0, 2,5 → 2), así la web
// escribe los mismos números que la app y el PDF.
const roundEven = (x) => { const r = Math.round(x); return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r; };
function toFixedEven(x, digits) {
  const plain = x.toFixed(digits), long = Math.abs(x).toFixed(Math.min(100, digits + 30)), cut = long.length - 30;
  if (/^50*$/.test(long.slice(cut))) {  // empate exacto: si el último dígito ya es par, se trunca
    const low = long.slice(0, cut).replace(/\.$/, "");
    if (+low.slice(-1) % 2 === 0) return (x < 0 ? "-" : "") + low;
  }
  return plain;
}
/** Tiempos de espera y de los puestos: «1:20» (minutos:segundos) o, de una hora para arriba, «1 h 12 min» (como la app
 * y el PDF). Sin dato, «—». */
const fmtClock = (s) => {
  if (s == null) return "—";
  s = Math.max(0, roundEven(s));
  if (s >= 3600) { const mm = roundEven(s / 60); return `${Math.floor(mm / 60)} h ${mm % 60} min`; }
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};
/** Segundos como «mm:ss» también de una hora para arriba («90:00»): lo que `parseClock` vuelve a leer igual (con
 *  `fmtClock`, «1 h 30 min» se leía como 1 minuto al volver a guardar). */
const fmtMinSec = (s) => { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
/** «2:00» → 120 s; «2:30» → 150 s; un número suelto se toma como minutos («2» → 120 s). Vacío → 0. */
const parseClock = (s) => {
  s = String(s == null ? "" : s).trim().replace(",", ":");
  if (!s) return 0;
  if (s.includes(":")) { const p = s.split(":"); return (parseInt(p[0], 10) || 0) * 60 + (parseInt(p[1], 10) || 0); }
  return Math.round((parseFloat(s) || 0) * 60);
};
/** Un rato largo dicho en una frase: «12 min», «menos de 1 min» o «1 h 12 min». */
const fmtMinutes = (s) => {
  if (!s) return "0 min";
  if (s < 30) return "menos de 1 min";
  const mm = Math.floor(s / 60 + 0.5);  // (redondear «al par» daría «0 min» con 30 s)
  return mm >= 60 ? `${Math.floor(mm / 60)} h ${mm % 60} min` : `${mm} min`;
};
/** Número con coma decimal («1,3»). */
const fmtDec = (v, digits = 1) => (v == null ? "—" : toFixedEven(Number(v), digits).replace(".", ","));

// ---- tiempo de espera (plan «comercio») y puestos del personal (plan «empleados»). Lo mismo que core/metrics.py
// (wait_stats / post_stats, que usan la app y el PDF): si cambia allá, cambiar acá.
const WAIT_MIN_SECONDS = 5, WAIT_EDGES = [60, 180, 300, 600];
const num0 = (v) => (typeof v === "number" && Number.isFinite(v) ? v : 0);
/** Resumen de la espera en una zona: esperas, promedio y máximo (segundos), cuántos esperaron 1, 3, 5 y 10 min o más
 * (`over`), la fila promedio y máxima y, si la cámara tiene zona de personal, el rato con clientes esperando y nadie en
 * el puesto y cuántos esperaron solos (null sin zona de personal). */
function waitStats(w) {
  w = w || {};
  const hist = [0, 1, 2, 3, 4].map((i) => num0((w.hist || [])[i]));
  const n = num0(w.n), obs = num0(w.obs_s), staffed = Math.max(0, num0(w.staffed)) > 0;
  const over = {};
  WAIT_EDGES.forEach((edge, i) => { over[edge] = hist.slice(i + 1).reduce((a, b) => a + b, 0); });
  return { n, avg_s: n ? num0(w.sum_s) / n : null, max_s: n ? Math.max(0, num0(w.max_s)) : null, over,
    queue_avg: obs ? num0(w.queue_s) / obs : null, queue_max: Math.max(0, num0(w.queue_max)), busy_s: num0(w.busy_s),
    obs_s: obs, staffed, unattended_s: staffed ? num0(w.unattended_s) : null, left_alone: staffed ? num0(w.left_alone) : null };
}
/** Resumen de un puesto (zona de personal): qué parte del tiempo mirado hubo alguien (0-100), los segundos sin nadie,
 * el promedio y el máximo de gente en el puesto. */
function postStats(p) {
  p = p || {};
  const obs = num0(p.obs_s), covered = num0(p.covered_s);
  return { obs_s: obs, covered_s: covered, uncovered_s: Math.max(0, obs - covered), covered_pct: obs ? 100 * covered / obs : null,
    avg_people: obs ? num0(p.people_s) / obs : null, max: Math.max(0, num0(p.max)) };
}
/** Resumen de una mesa (plan «gastronomía»): lo mismo que core/metrics.py (table_stats): si cambia allá, cambiar acá. */
function tableStats(t) {
  t = t || {};
  const n = num0(t.n), obs = num0(t.obs_s), an = num0(t.attend_n), sn = num0(t.serve_n);
  return { n, avg_s: n ? num0(t.sum_s) / n : null, max_s: n ? num0(t.max_s) : null, obs_s: obs, occ_s: num0(t.occ_s),
    occupied_pct: obs ? 100 * Math.min(num0(t.occ_s), obs) / obs : null, party_avg: n ? num0(t.party) / n : null,
    visits: num0(t.visits), no_visit: num0(t.no_visit), attend_n: an, attend_avg_s: an ? num0(t.attend_s) / an : null,
    attend_max_s: an ? num0(t.attend_max_s) : null, serve_n: sn, serve_avg_s: sn ? num0(t.serve_s) / sn : null };
}
/** Todas las mesas de un cubo juntas. */
function tableTotal(b) {
  const keys = ["n", "sum_s", "obs_s", "occ_s", "party", "visits", "no_visit", "attend_n", "attend_s", "serve_n", "serve_s"];
  const agg = {};
  for (const t of Object.values((b && b.tables) || {})) {
    for (const k of keys) agg[k] = num0(agg[k]) + num0((t || {})[k]);
    agg.max_s = Math.max(num0(agg.max_s), num0((t || {}).max_s));
    agg.attend_max_s = Math.max(num0(agg.attend_max_s), num0((t || {}).attend_max_s));
  }
  return tableStats(agg);
}
const byName = (a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
/** Las zonas de espera con algo medido de un cubo (total, día, hora o cámara), la de más esperas primero. */
const waitZones = (b) => Object.entries((b && b.waits) || {}).map(([name, w]) => [name, waitStats(w)])
  .filter(([, w]) => w.n || w.obs_s).sort((a, b) => b[1].n - a[1].n || byName(a, b));
/** Todas las zonas de espera juntas (como el PDF): esperas y segundos sumados (→ promedio), el máximo el mayor. */
function waitTotal(zones) {
  const withN = zones.filter(([, w]) => w.n), n = withN.reduce((a, [, w]) => a + w.n, 0);
  return { n, avg_s: n ? withN.reduce((a, [, w]) => a + w.avg_s * w.n, 0) / n : null,
    max_s: withN.length ? Math.max(...withN.map(([, w]) => w.max_s)) : null,
    over5: zones.reduce((a, [, w]) => a + w.over[300], 0), queue_max: Math.max(0, ...zones.map(([, w]) => w.queue_max)) };
}
/** Los puestos (zonas de personal) mirados de un cubo, por nombre. */
const postZones = (b) => Object.entries((b && b.posts) || {}).map(([name, p]) => [name, postStats(p)]).filter(([, p]) => p.obs_s).sort(byName);

const fmtDay = (iso) => { const [y, m, d] = String(iso).split("-"); return d ? `${d}/${m}` : iso; };
const fmtFull = (iso) => { const [y, m, d] = String(iso).split("-"); return d ? `${d}/${m}/${y}` : iso; };

function barChart(items, key, label, fmt = String) {
  // gráfico de barras en SVG, sin librerías: items = [{hour|date, ...}]; `fmt` escribe el número de cada barra
  if (!items.length) return `<div class="empty">Sin datos en este período.</div>`;
  const vals = items.map((x) => +x[key] || 0), max = Math.max(1, ...vals);
  const W = 720, H = 180, T = 16, pad = 26, bw = Math.max(4, (W - pad * 2) / items.length - 4);  // T: lugar para el número de arriba
  return `<svg viewBox="0 0 ${W} ${H + 30 + T}" style="width:100%;max-width:${W}px;display:block">
    ${items.map((x, i) => { const v = vals[i], h = Math.round((H - 10) * v / max), xx = pad + i * ((W - pad * 2) / items.length);
      return `<rect x="${xx}" y="${T + H - h}" width="${bw}" height="${h}" rx="3" fill="var(--accent)"><title>${esc(x.hour !== undefined ? x.hour + ":00" : x.date)}: ${esc(fmt(v))} ${label}</title></rect>
        ${v ? `<text x="${xx + bw / 2}" y="${T + H - h - 4}" font-size="10" text-anchor="middle" fill="currentColor">${esc(fmt(v))}</text>` : ""}
        <text x="${xx + bw / 2}" y="${T + H + 14}" font-size="10" text-anchor="middle" fill="var(--muted)">${esc(x.hour !== undefined ? x.hour + "h" : fmtDay(x.date))}</text>`; }).join("")}
  </svg>`;
}

async function metricsView(v, clientId) {
  await loadClients();
  if (!clientId && isAdmin() && S.clients.length) clientId = S.clients[0].id;  // el primero, y se cambia arriba
  const client = S.clients.find((c) => c.id === clientId);
  if (!client) { v.innerHTML = isAdmin() ? appNote() + `<div class="empty">Todavía no hay clientes: se cargan en la app de escritorio.</div>` : `<div class="empty">Tu usuario no tiene un local asociado.</div>`; return; }
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const days = +(q.get("days") || 1);
  const [m, stats] = await Promise.all([
    api("GET", `/api/metrics?client_id=${clientId}&days=${days}`),
    api("GET", `/api/stats?client_id=${clientId}&days=${days}`).catch(() => null),
  ]);
  const t = m.total, plans = m.plans || [];
  const tile = (n, label, hint) => `<div class="panel stat"><span class="n">${n}</span><span>${label}</span>${hint ? `<span class="muted">${hint}</span>` : ""}</div>`;
  const period = (d) => `<a class="btn small ${days === d ? "primary" : ""}" href="#/${isAdmin() ? "metricas?client_id=" + clientId + "&" : "resumen?"}days=${d}">${d === 1 ? "Hoy" : d + " días"}</a>`;
  const zones = Object.entries(t.zones || {}).sort((a, b) => b[1].visits - a[1].visits);
  // «demora estándar» por sector (segundos). Si el tiempo promedio en la zona lo supera, se marca «demora».
  const limits = m.zone_limits || {};
  const zoneDelay = (name, z) => limits[name] > 0 && z.visits && avg(z.dwell_s, z.visits) >= limits[name];
  const delayed = zones.filter(([name, z]) => zoneDelay(name, z));
  const sinDatos = !t.frames;
  const pdf = `/api/metrics/report.pdf?days=${days}` + (isAdmin() ? `&client_id=${clientId}` : "");
  // cada cámara mide los planes del cliente que le eligieron (`measures`); lo que una cámara no mide sale «—»
  // (el servidor ya no lo suma). Sin `measures` (servidor viejo), como antes.
  const showMeasures = m.by_camera.some((c) => Array.isArray(c.measures));
  const measures = (c, plan) => !Array.isArray(c.measures) || !plans.includes(plan) || c.measures.includes(plan);
  const grid = `class="grid" style="grid-template-columns:repeat(auto-fill,minmax(170px,1fr))"`;
  // tiempo de espera (plan «comercio»): por zona marcada «Medir el tiempo de espera acá»
  const waits = plans.includes("comercio") ? waitZones(t) : [];
  let waitHtml = "";
  if (waits.length) {
    const tot = waitTotal(waits);
    const series = (days === 1 ? m.by_hour : m.by_day).map((x) => ({ ...x, wait_avg: waitTotal(waitZones(x)).avg_s || 0 }));
    const staffed = waits.filter(([, w]) => w.staffed);
    waitHtml = `
      <h2>Tiempo de espera</h2>
      <div ${grid}>
        ${tile(fmtClock(tot.avg_s), "Espera promedio", `${tot.n} espera${tot.n !== 1 ? "s" : ""} en ${waits.length} zona${waits.length !== 1 ? "s" : ""}`)}
        ${tile(fmtClock(tot.max_s), "Espera más larga", "el cliente que más esperó")}
        ${tile(tot.over5, "Más de 5 min", "clientes que esperaron 5 minutos o más")}
        ${tile(tot.queue_max, "Fila máxima", "personas esperando a la vez")}
      </div>
      <div class="panel" style="margin-top:12px"><b>Por zona</b><div class="table-wrap"><table class="nums"><thead><tr><th>Zona</th><th>Esperas</th><th>Promedio</th><th>Máximo</th><th>Más de 5 min</th><th>Más de 10 min</th><th>Fila prom.</th><th>Fila máx.</th></tr></thead>
        <tbody>${waits.map(([name, w]) => `<tr><td>${esc(name)}</td><td>${w.n}</td><td>${fmtClock(w.avg_s)}</td><td>${fmtClock(w.max_s)}</td><td>${w.over[300]}</td><td>${w.over[600]}</td><td>${fmtDec(w.queue_avg)}</td><td>${w.queue_max}</td></tr>`).join("")}</tbody></table></div>
        ${staffed.length ? `<ul class="plain">${staffed.map(([name, w]) => `<li><b>${esc(name)}</b>: con clientes esperando y nadie en su puesto (la zona de personal más cercana): ${fmtMinutes(w.unattended_s)} · se fueron sin que hubiera nadie: ${w.left_alone}</li>`).join("")}</ul>`
          : isAdmin() ? `<p class="muted note">Para saber si había alguien para atender, marcá en la app la zona «Personal (empleados)» (detrás del mostrador) en la misma cámara.</p>` : ""}</div>
      ${series.filter((x) => x.wait_avg).length > 1 ? `<div class="panel" style="margin-top:12px"><b>Espera promedio ${days === 1 ? "por hora" : "por día"}</b>${barChart(series, "wait_avg", "de espera promedio", fmtClock)}</div>` : ""}
      <p class="muted note">La espera es el tiempo de cada cliente (${WAIT_MIN_SECONDS} s o más) en la zona marcada: si la zona incluye donde se atiende, incluye la atención. En una fila apretada, quien queda tapado por otro no se ve.</p>`;
  } else if (plans.includes("comercio") && isAdmin() && !sinDatos) {
    waitHtml = `<p class="muted note">Sin datos de espera. Hace falta, en la app de escritorio, tildar «Medir el tiempo de espera acá» en la zona (caja, mostrador, fila) con «✎ Marcar zonas» de la cámara; que esa cámara mida «Métricas de comercio» (en «Editar»); y que la máquina que la analiza esté actualizada (se actualiza al reiniciarla).</p>`;
  }
  // empleados (plan «empleados»): puestos (zonas de personal), cuántos a la vista y dónde pasan el tiempo
  let staffHtml = "";
  if (plans.includes("empleados")) {
    const posts = postZones(t), watched = num0(t.staff_obs_s);  // segundos mirados, sumando cámaras (0: máquina sin actualizar)
    const dwell = Object.entries(t.staff_dwell || {}).filter(([, secs]) => num0(secs) > 0).sort((a, b) => b[1] - a[1]).slice(0, 6);
    staffHtml = `
      <h2>Empleados</h2>
      <div ${grid}>
        ${watched ? tile(num0(t.staff_max), "Máximo a la vez", "empleados a la vista en una cámara") : ""}
        ${watched ? tile(fmtDec(num0(t.staff_s) / watched), "Promedio a la vista", "empleados por cámara, en el tiempo mirado") : ""}
        ${posts.length === 1 ? tile(`${fmtDec(posts[0][1].covered_pct, 0)} %`, `${esc(posts[0][0])}: cubierto`, `sin nadie ${fmtClock(posts[0][1].uncovered_s)}`) : ""}
      </div>
      ${posts.length ? `<div class="panel" style="margin-top:12px"><b>Puestos</b><div class="table-wrap"><table class="nums"><thead><tr><th>Puesto</th><th>Cubierto</th><th>Sin nadie</th><th>Gente en el puesto (promedio)</th><th>Máximo</th><th></th></tr></thead>
        <tbody>${posts.map(([name, p]) => `<tr><td>${esc(name)}</td><td>${fmtDec(p.covered_pct, 0)} %</td><td>${fmtClock(p.uncovered_s)}</td><td>${fmtDec(p.avg_people)}</td><td>${p.max}</td><td style="width:30%"><div class="bar"><i style="width:${Math.round(p.covered_pct || 0)}%"></i></div></td></tr>`).join("")}</tbody></table></div></div>`
        : `<p class="warn-text note">${m.by_camera.length && m.by_camera.every((c) => Array.isArray(c.measures)) && !m.by_camera.some((c) => c.measures.includes("empleados")) ? (isAdmin() ? "Ninguna cámara mide «Seguimiento de empleados»: elegilo en la app, en «Qué mide esta cámara» (Editar de la cámara)." : "Todavía no hay datos de los puestos en este período.") : isAdmin() ? NO_POSTS_TEXT : "Todavía no hay datos de los puestos en este período."}</p>`}
      ${dwell.length ? `<div class="panel" style="margin-top:12px"><b>Dónde pasan el tiempo</b><div class="table-wrap"><table class="nums"><thead><tr><th>Zona</th><th>Tiempo de los empleados</th><th></th></tr></thead>
        <tbody>${dwell.map(([name, secs]) => `<tr><td>${esc(name)}</td><td>${fmtClock(secs)}</td><td style="width:40%"><div class="bar"><i style="width:${Math.round(100 * secs / dwell[0][1])}%"></i></div></td></tr>`).join("")}</tbody></table></div></div>` : ""}
      <p class="muted note">Reconoce a los empleados solo por su zona «Personal» (pisada 2 s o más) o por el horario (con el local cerrado); no sabe quién es quién ni los sigue de una cámara a otra; quien pisa el puesto cuenta como «alguien en el puesto», sea o no empleado.</p>`;
  }
  // gastronomía (plan «gastronomia»): las mesas
  let tablesHtml = "";
  if (plans.includes("gastronomia")) {
    const tables = Object.entries(t.tables || {}).map(([name, x]) => [name, tableStats(x)]).filter(([, x]) => x.n || x.obs_s).sort(byName);
    const tot = tableTotal(t);
    tablesHtml = `
      <h2>Gastronomía: mesas</h2>
      ${tables.length ? `<div ${grid}>
        ${tile(fmtClock(tot.avg_s), "Tiempo en la mesa", `promedio de ${tot.n} ocupaciones`)}
        ${tile(tot.occupied_pct == null ? "—" : `${fmtDec(tot.occupied_pct, 0)} %`, "Ocupación", `del tiempo mirado, ${tables.length} mesa${tables.length === 1 ? "" : "s"}`)}
        ${tile(fmtClock(tot.attend_avg_s), "Hasta la atención", "desde que se sientan hasta que se acerca el mozo")}
        ${tile(fmtClock(tot.serve_avg_s), "Hasta la entrega (estimada)", "hasta la segunda visita a la mesa")}
      </div>
      <div class="panel" style="margin-top:12px"><b>Por mesa</b><div class="table-wrap"><table class="nums"><thead><tr><th>Mesa</th><th>Veces</th><th>En la mesa</th><th>Ocupada</th><th>Comensales</th><th>Atención</th><th>Entrega (est.)</th><th>Sin atención</th></tr></thead>
        <tbody>${tables.map(([name, x]) => `<tr><td>${esc(name)}</td><td>${x.n}</td><td>${fmtClock(x.avg_s)}</td><td>${x.occupied_pct == null ? "—" : fmtDec(x.occupied_pct, 0) + " %"}</td><td>${fmtDec(x.party_avg)}</td><td>${fmtClock(x.attend_avg_s)}</td><td>${fmtClock(x.serve_avg_s)}</td><td>${x.no_visit}</td></tr>`).join("")}</tbody></table></div></div>`
        : `<p class="warn-text note">Sin datos de las mesas en este período.${isAdmin() ? " Hace falta marcar cada mesa con la zona «Mesa» (en la app, «✎ Marcar zonas») en una cámara que mida «Gastronomía»." : ""}</p>`}
      <p class="muted note">Una ocupación va desde que un cliente lleva 10 s en la mesa hasta que queda vacía un minuto. «Atención» es la primera vez que alguien se acerca a la mesa ya ocupada (el mozo); «entrega» es la segunda: no se ve qué lleva en la mano, es una estimación.</p>`;
  }
  // «Por cámara»: la espera y los puestos de cada una, si hay (con «—» donde la cámara no mide ese plan)
  const camWait = (c) => { const w = waitTotal(waitZones(c)); return measures(c, "comercio") && w.n ? fmtClock(w.avg_s) : "—"; };
  const camPosts = (c) => { const ps = postZones(c); return !measures(c, "empleados") || !ps.length ? "—" : ps.length === 1 ? `${fmtDec(ps[0][1].covered_pct, 0)} %` : ps.map(([n, p]) => `${esc(n)} ${fmtDec(p.covered_pct, 0)} %`).join(" · "); };
  const showWait = plans.includes("comercio") && m.by_camera.some((c) => measures(c, "comercio") && waitTotal(waitZones(c)).n);
  const showPosts = plans.includes("empleados") && m.by_camera.some((c) => measures(c, "empleados") && postZones(c).length);
  v.innerHTML = `
    ${isAdmin() ? appNote() : ""}
    <div class="row"><h1>${isAdmin() ? "Métricas" : `Resumen de ${esc(client.name)}`}</h1>
      ${isAdmin() ? `<select id="m-client" class="inline-select">${options(S.clients.map((c) => [c.id, c.name]), clientId)}</select>` : ""}
      <span class="spacer"></span>${period(1)}${period(7)}${period(30)}
      <a class="btn small" href="${pdf}" download title="El reporte de este período en PDF, para guardarlo o mandarlo">⬇ Descargar PDF</a></div>
    <p class="muted">${days === 1 ? `Hoy, ${fmtFull(m.until)}` : `Del ${fmtFull(m.since)} al ${fmtFull(m.until)}`} · hora del local · planes: ${plans.map((p) => esc(PLAN_NAMES[p] || p)).join(", ") || "ninguno"}</p>
    ${sinDatos ? `<div class="empty">Todavía no hay métricas en este período. Se generan mientras la PC central analiza las cámaras en modo "En vivo" (se envían cada 5 minutos).</div>` : ""}
    ${plans.includes("comercio") || (!plans.includes("estacion") && t.people) ? `
      <h2>Gente</h2>
      <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(170px,1fr))">
        ${t.door ? tile(t.entries, "Entraron", `${t.exits} salieron${t.staff ? ` · ${t.staff} de personal descontados` : ""}`)
          : t.entries ? tile(t.entries, "Pasaron por la entrada", "entrando o saliendo: falta marcar el sentido de entrada")
          : tile("—", "Entraron", isAdmin() ? "falta marcar la zona Entrada, con su sentido, en la cámara de la puerta" : "todavía no se cuenta la puerta")}
        ${tile(fmtDur(avg(t.people_dwell_s, t.people)), "Tiempo promedio en cámara", "")}
        ${tile(t.people_max, "Máximo al mismo tiempo", "")}
        ${zones.length ? tile(esc(zones[0][0]), "Zona más visitada", `${zones[0][1].visits} visitas`) : ""}
        ${delayed.length ? `<div class="panel stat"><span class="n" style="color:var(--warn)">${delayed.length}</span><span>${delayed.length === 1 ? "Sector con demora" : "Sectores con demora"}</span><span class="muted">${delayed.map(([n]) => esc(n)).join(", ")}</span></div>` : ""}
      </div>
      ${t.door || t.entries ? `<div class="panel" style="margin-top:12px"><b>${t.door ? "Entraron" : "Pasaron por la entrada"} ${days === 1 ? "por hora" : "por día"}</b>${barChart(days === 1 ? m.by_hour : m.by_day, "entries", t.door ? "entraron" : "pasaron")}</div>` : ""}
      ${zones.length ? `<div class="panel" style="margin-top:12px"><b>A dónde van</b><div class="table-wrap"><table><thead><tr><th>Zona</th><th>Tipo</th><th>Visitas</th><th>Tiempo promedio</th><th>Demora estándar</th><th></th></tr></thead>
        <tbody>${zones.map(([name, z]) => { const d = zoneDelay(name, z); return `<tr><td>${esc(name)}</td><td>${esc(ZONE_NAMES[z.type] || z.type)}</td><td>${z.visits}</td><td>${fmtDur(avg(z.dwell_s, z.visits))}${d ? ` <span class="warn-text">⚠ demora</span>` : ""}</td><td>${isAdmin() ? `<input class="zlimit" data-zone="${esc(name)}" value="${limits[name] > 0 ? fmtMinSec(limits[name]) : ""}" placeholder="mm:ss" style="width:70px">` : limits[name] > 0 ? fmtClock(limits[name]) : `<span class="muted">—</span>`}</td><td style="width:30%"><div class="bar"><i style="width:${Math.round(100 * z.visits / Math.max(1, zones[0][1].visits))}%"></i></div></td></tr>`; }).join("")}</tbody></table></div>
        ${isAdmin() ? `<div class="row" style="margin-top:8px"><button id="save-zlimits" class="btn small primary">Guardar demoras</button><span class="muted small" style="margin-left:10px">Poné cuánto es lo normal en cada sector (mm:ss). Si el promedio lo supera, el cliente ve «⚠ demora» acá y en su resumen.</span></div>` : ""}</div>` : ""}` : ""}
    ${waitHtml}
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
    ${staffHtml}
    ${tablesHtml}
    ${plans.includes("hurto") && stats ? `
      <h2>Alertas de hurto</h2>
      <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(170px,1fr))">
        ${tile(stats.total, "Alertas en el período", "")}
        ${tile(stats.by_label.hurto || 0, "Confirmadas como hurto", "")}
        ${tile(stats.by_label.sospechoso || 0, "Sospechosas", "")}
        ${tile(stats.by_label.sin_etiqueta || 0, "Sin revisar todavía", "")}
      </div>
      ${isAdmin() ? "" : `<p><a class="btn" href="#/alertas?client_id=${clientId}">Ver las alertas</a></p>`}` : ""}
    ${m.by_camera.length > 1 ? `<div class="panel" style="margin-top:12px"><b>Por cámara</b><div class="table-wrap"><table class="nums"><thead><tr><th>Cámara</th>${showMeasures ? "<th>Mide</th>" : ""}<th>Tiempo promedio</th>${plans.includes("estacion") ? "<th>Vehículos</th>" : ""}<th>Máximo a la vez</th>${showWait ? "<th>Espera prom.</th>" : ""}${showPosts ? "<th>Puesto cubierto</th>" : ""}</tr></thead>
      <tbody>${m.by_camera.map((c) => { const people = measures(c, "comercio"), cars = measures(c, "estacion"); return `<tr><td>${esc(c.name)}</td>${showMeasures ? `<td>${(c.measures || []).map((p) => esc(PLAN_SHORT[p] || p)).join(" · ") || "—"}</td>` : ""}<td>${people ? fmtDur(avg(c.people_dwell_s, c.people)) : "—"}</td>${plans.includes("estacion") ? `<td>${cars ? c.vehicles : "—"}</td>` : ""}<td>${people ? c.people_max : "—"}</td>${showWait ? `<td>${camWait(c)}</td>` : ""}${showPosts ? `<td>${camPosts(c)}</td>` : ""}</tr>`; }).join("")}</tbody></table></div></div>` : ""}
    <p class="muted small" style="margin-top:18px">Cómo se mide: "Entraron" cuenta a quien cruza la puerta hacia adentro en la cámara de la entrada (una sola cámara, así nadie se cuenta dos veces). No se muestra una suma de personas de todas las cámaras: la misma persona aparece en varias y, si sale del cuadro y vuelve, cuenta de nuevo. "A dónde van" cuenta visitas: quien vuelve a una zona suma otra visita. Personal: quien pisa la zona de empleados (detrás del mostrador, depósito) o aparece con el local cerrado no cuenta como cliente; los demás empleados sí. No reconoce caras. "Espera hasta el pistero" es el tiempo desde que el vehículo entra a la zona Surtidor hasta que una persona pisa esa misma zona. Cada cámara suma solo lo que mide (por ejemplo, la de afuera la estación y la de adentro la gente del comercio).</p>`;
  const sel = $("#m-client", v);
  if (sel) sel.addEventListener("change", () => { location.hash = `#/metricas?client_id=${sel.value}&days=${days}`; });
  const saveZ = $("#save-zlimits", v);  // «demora estándar» por sector (solo admin)
  if (saveZ) saveZ.addEventListener("click", async () => {
    const out = {};
    $$(".zlimit", v).forEach((i) => { const secs = parseClock(i.value); if (secs > 0) out[i.dataset.zone] = secs; });
    saveZ.disabled = true;
    try { await api("PUT", `/api/clients/${clientId}/zone-limits`, { limits: out }); toast("Demoras guardadas"); route(); }
    catch (e) { saveZ.disabled = false; fail(e); }
  });
}

boot();
