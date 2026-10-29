"use strict";
/* Pestaña Prueba: analizar en el navegador el video en vivo que muestra otra ventana
   (Hik-Partner Pro, iVMS-4200, SmartPSS...). Se comparte la pantalla, se marca el recuadro
   de la cámara y una zona (góndola), y el detector de personas corre acá mismo con
   ONNX Runtime Web. Si alguien se queda en la zona más de N segundos, se guarda una alerta
   con foto y un clip de 12 s en el panel, como cualquier otra.

   Es una prueba: la imagen viene comprimida por la nube y el detector corre en la CPU del
   navegador (1 a 3 cuadros por segundo). Sirve para probar con video real sin tocar el local. */

const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
const MODEL_URL = "models/personas.onnx";
const MODEL_SIZE = 640;
const CONF = 0.35, IOU = 0.5;
const CLIP_SECONDS = 12;
const ALERT_COOLDOWN_MS = 60000;

const P = { session: null, stream: null, timer: null, running: false, crop: null, zone: null, drawing: null,
  zoneSince: null, lastSeenInZone: 0, lastAlertAt: 0, people: 0, fps: 0, saving: false, alerts: [] };

function loadScript(src) {
  return new Promise((ok, bad) => {
    if (document.querySelector(`script[src="${src}"]`)) return ok();
    const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = () => bad(new Error("No se pudo cargar " + src));
    document.head.appendChild(s);
  });
}

async function loadModel(setStatus) {
  if (P.session) return P.session;
  setStatus("Cargando el detector de personas (10 MB, una sola vez)…");
  await loadScript(ORT_CDN + "ort.min.js");
  ort.env.wasm.wasmPaths = ORT_CDN;
  ort.env.wasm.numThreads = 1;
  const r = await fetch(MODEL_URL);
  if (!r.ok) throw new Error("No se encontró el modelo en " + MODEL_URL);
  const buf = await r.arrayBuffer();
  P.session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  return P.session;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

/* Devuelve [[x1,y1,x2,y2,score], ...] en píxeles del recorte. */
async function detectPeople(src, sx, sy, sw, sh) {
  const c = detectPeople._c || (detectPeople._c = document.createElement("canvas"));
  c.width = MODEL_SIZE; c.height = MODEL_SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const r = Math.min(MODEL_SIZE / sw, MODEL_SIZE / sh);
  const nw = Math.round(sw * r), nh = Math.round(sh * r), px = Math.floor((MODEL_SIZE - nw) / 2), py = Math.floor((MODEL_SIZE - nh) / 2);
  ctx.fillStyle = "rgb(114,114,114)"; ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  ctx.drawImage(src, sx, sy, sw, sh, px, py, nw, nh);
  const img = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const n = MODEL_SIZE * MODEL_SIZE;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { data[i] = img[i * 4] / 255; data[n + i] = img[i * 4 + 1] / 255; data[2 * n + i] = img[i * 4 + 2] / 255; }
  const input = new ort.Tensor("float32", data, [1, 3, MODEL_SIZE, MODEL_SIZE]);
  const out = await P.session.run({ [P.session.inputNames[0]]: input });
  const o = out[P.session.outputNames[0]];
  const [, rows, cols] = o.dims; // [1, 84, 8400]
  const d = o.data;
  const cand = [];
  for (let i = 0; i < cols; i++) {
    const score = d[4 * cols + i]; // clase 0 = persona
    if (score < CONF) continue;
    const cx = d[i], cy = d[cols + i], w = d[2 * cols + i], h = d[3 * cols + i];
    cand.push([(cx - w / 2 - px) / r, (cy - h / 2 - py) / r, (cx + w / 2 - px) / r, (cy + h / 2 - py) / r, score]);
  }
  cand.sort((a, b) => b[4] - a[4]);
  const keep = [];
  for (const b of cand) { if (keep.every((k) => iou(k, b) < IOU)) keep.push(b); }
  return keep.map((b) => [Math.max(0, b[0]), Math.max(0, b[1]), Math.min(sw, b[2]), Math.min(sh, b[3]), b[4]]);
}

async function pruebaView(v) {
  await loadCameras();
  const cams = S.cameras.filter((c) => c.enabled !== false);
  v.innerHTML = `
    <div class="row"><h1>Prueba: analizar la pantalla de HPP</h1></div>
    <p class="muted">Abrí Hik-Partner Pro (o iVMS-4200 / SmartPSS) en otra ventana con la cámara en vivo, lo más grande posible.
      Acá se comparte esa ventana y el detector de personas corre en este navegador. Si alguien se queda en la zona marcada más del tiempo
      elegido, se guarda una alerta con foto y un clip de ${CLIP_SECONDS} s en <a href="#/alertas">Alertas</a>. Es una prueba:
      la imagen llega comprimida y el detector corre en tu PC (1 a 3 cuadros por segundo).</p>
    <div class="panel">
      <div class="form">
        <label>Cámara a la que se asocian las alertas<select id="p-cam">${options(cams.map((c) => [c.id, `${clientName(c.client_id)} · ${c.name}`]), cams[0]?.id, cams.length ? undefined : "No hay cámaras cargadas")}</select></label>
        <label>Segundos en la zona para alertar<input id="p-secs" type="number" min="3" max="300" value="20"></label>
        <label>Nombre de la zona<input id="p-zone" value="gondola"></label>
        <div class="row">
          <button id="p-share" class="primary">Compartir pantalla</button>
          <button id="p-stop" class="hidden">Detener</button>
        </div>
      </div>
      <div class="row" style="margin-top:10px">
        <button id="p-crop" class="small" disabled>1. Marcar el recuadro de la cámara</button>
        <button id="p-mark" class="small" disabled>2. Marcar la zona (góndola)</button>
        <button id="p-save" class="small" disabled>Guardar alerta ahora</button>
        <span id="p-status" class="muted">Sin compartir.</span>
      </div>
      <div id="p-stage" style="position:relative;margin-top:12px;background:#000;border-radius:10px;overflow:hidden;max-height:70vh">
        <video id="p-video" autoplay muted playsinline style="display:block;width:100%;max-height:70vh;object-fit:contain"></video>
        <canvas id="p-canvas" style="position:absolute;inset:0;width:100%;height:100%;cursor:crosshair"></canvas>
      </div>
    </div>
    <h3 style="margin-top:16px">Alertas de esta prueba</h3>
    <div id="p-alerts" class="muted">Todavía ninguna.</div>`;
  const video = $("#p-video"), canvas = $("#p-canvas"), status = $("#p-status");
  const setStatus = (t) => (status.textContent = t);
  const stop = () => {
    P.running = false; clearTimeout(P.timer);
    if (P.stream) { P.stream.getTracks().forEach((t) => t.stop()); P.stream = null; }
    video.srcObject = null;
    $("#p-share").classList.remove("hidden"); $("#p-stop").classList.add("hidden");
    ["p-crop", "p-mark", "p-save"].forEach((id) => ($("#" + id).disabled = true));
    setStatus("Detenido.");
  };
  $("#p-stop").addEventListener("click", stop);
  S.timers.push(setInterval(() => { if (!document.body.contains(video)) stop(); }, 2000));

  // ---- dibujar recorte y zona con el mouse (coordenadas relativas 0..1 al video)
  const rel = (e) => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
  let mode = null;
  $("#p-crop").addEventListener("click", () => { mode = "crop"; setStatus("Arrastrá sobre la imagen para marcar el recuadro de la cámara."); });
  $("#p-mark").addEventListener("click", () => { mode = "zone"; setStatus("Arrastrá sobre la imagen para marcar la zona (sobre el piso frente a la góndola)."); });
  canvas.addEventListener("mousedown", (e) => { if (mode) P.drawing = { mode, a: rel(e), b: rel(e) }; });
  canvas.addEventListener("mousemove", (e) => { if (P.drawing) P.drawing.b = rel(e); });
  window.addEventListener("mouseup", (e) => {
    if (!P.drawing) return;
    const { a, b } = P.drawing;
    const rect = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
    if (rect[2] - rect[0] > 0.02 && rect[3] - rect[1] > 0.02) {
      if (P.drawing.mode === "crop") { P.crop = rect; P.zone = null; } else P.zone = rect;
    }
    P.drawing = null; mode = null; setStatus("Analizando.");
  });

  // ---- compartir pantalla y arrancar el análisis
  $("#p-share").addEventListener("click", async () => {
    try {
      await loadModel(setStatus);
      P.stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 10 }, audio: false });
      video.srcObject = P.stream;
      await video.play();
      P.stream.getVideoTracks()[0].addEventListener("ended", stop);
      P.crop = null; P.zone = null; P.zoneSince = null; P.running = true;
      $("#p-share").classList.add("hidden"); $("#p-stop").classList.remove("hidden");
      ["p-crop", "p-mark", "p-save"].forEach((id) => ($("#" + id).disabled = false));
      setStatus("Analizando. Marcá el recuadro de la cámara y la zona.");
      loop();
    } catch (e) { fail(e); }
  });

  const cropPx = () => {
    const w = video.videoWidth, h = video.videoHeight;
    const c = P.crop || [0, 0, 1, 1];
    return [Math.round(c[0] * w), Math.round(c[1] * h), Math.max(8, Math.round((c[2] - c[0]) * w)), Math.max(8, Math.round((c[3] - c[1]) * h))];
  };
  let lastBoxes = [];
  const draw = () => {
    const w = video.videoWidth, h = video.videoHeight;
    if (!w) return;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = Math.max(2, w / 500); ctx.font = `${Math.max(14, w / 60)}px system-ui`;
    const [sx, sy, sw, sh] = cropPx();
    if (P.crop) { ctx.setLineDash([8, 6]); ctx.strokeStyle = "#4c9ffe"; ctx.strokeRect(sx, sy, sw, sh); ctx.setLineDash([]); }
    if (P.zone) {
      const z = P.zone; ctx.fillStyle = "rgba(255,200,0,.18)"; ctx.strokeStyle = "#ffc800";
      ctx.fillRect(z[0] * w, z[1] * h, (z[2] - z[0]) * w, (z[3] - z[1]) * h); ctx.strokeRect(z[0] * w, z[1] * h, (z[2] - z[0]) * w, (z[3] - z[1]) * h);
      ctx.fillStyle = "#ffc800"; ctx.fillText($("#p-zone").value || "zona", z[0] * w + 6, z[1] * h + 18);
    }
    if (P.drawing) { const { a, b, mode: m } = P.drawing; ctx.strokeStyle = m === "crop" ? "#4c9ffe" : "#ffc800"; ctx.strokeRect(Math.min(a[0], b[0]) * w, Math.min(a[1], b[1]) * h, Math.abs(b[0] - a[0]) * w, Math.abs(b[1] - a[1]) * h); }
    for (const b of lastBoxes) {
      const inZone = b.inZone;
      ctx.strokeStyle = inZone ? "#e5484d" : "#2fb344";
      ctx.strokeRect(sx + b[0], sy + b[1], b[2] - b[0], b[3] - b[1]);
      ctx.fillStyle = ctx.strokeStyle; ctx.fillText(`persona ${Math.round(b[4] * 100)}%`, sx + b[0], Math.max(16, sy + b[1] - 4));
    }
    const dwell = P.zoneSince ? (performance.now() - P.zoneSince) / 1000 : 0;
    const txt = `${P.people} persona(s) · ${P.fps.toFixed(1)} cuadros/s` + (P.zone ? ` · en zona: ${dwell.toFixed(0)} s` : "");
    ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(0, h - 30, ctx.measureText(txt).width + 16, 30);
    ctx.fillStyle = "#fff"; ctx.fillText(txt, 8, h - 9);
  };
  const loop = async () => {
    if (!P.running) return;
    const t0 = performance.now();
    try {
      if (video.videoWidth) {
        const [sx, sy, sw, sh] = cropPx();
        const boxes = await detectPeople(video, sx, sy, sw, sh);
        const w = video.videoWidth, h = video.videoHeight;
        let anyInZone = false;
        for (const b of boxes) {
          const fx = (sx + (b[0] + b[2]) / 2) / w, fy = (sy + b[3]) / h; // los pies
          b.inZone = !!P.zone && fx >= P.zone[0] && fx <= P.zone[2] && fy >= P.zone[1] && fy <= P.zone[3];
          anyInZone = anyInZone || b.inZone;
        }
        lastBoxes = boxes; P.people = boxes.length;
        const now = performance.now();
        if (anyInZone) { if (!P.zoneSince) P.zoneSince = now; P.lastSeenInZone = now; }
        else if (P.zoneSince && now - P.lastSeenInZone > 1500) P.zoneSince = null;
        const secs = +$("#p-secs").value || 20;
        if (P.zoneSince && (now - P.zoneSince) / 1000 >= secs && now - P.lastAlertAt > ALERT_COOLDOWN_MS) {
          P.lastAlertAt = now;
          saveAlert(`Persona ${secs}+ s en «${$("#p-zone").value || "zona"}» (prueba desde la pantalla)`, "zona_restringida", (now - P.zoneSince) / 1000);
        }
      }
    } catch (e) { setStatus("Error del detector: " + e.message); }
    P.fps = 1000 / Math.max(1, performance.now() - t0);
    draw();
    P.timer = setTimeout(loop, 50);
  };
  S.timers.push(setInterval(draw, 100));

  // ---- guardar foto + clip de 12 s como alerta
  const saveAlert = async (description, type, dwell) => {
    if (P.saving) return;
    const camId = +$("#p-cam").value;
    if (!camId) return toast("Elegí una cámara para asociar la alerta", true);
    P.saving = true;
    const item = document.createElement("div"); item.className = "panel"; item.style.marginTop = "8px";
    item.textContent = `${new Date().toLocaleTimeString("es-AR")} · ${description} · grabando ${CLIP_SECONDS} s…`;
    const box = $("#p-alerts"); if (box.classList.contains("muted")) { box.classList.remove("muted"); box.innerHTML = ""; }
    box.prepend(item);
    try {
      const snapCanvas = document.createElement("canvas"); snapCanvas.width = video.videoWidth; snapCanvas.height = video.videoHeight;
      const sctx = snapCanvas.getContext("2d"); sctx.drawImage(video, 0, 0); sctx.drawImage(canvas, 0, 0);
      const snap = await new Promise((ok) => snapCanvas.toBlob(ok, "image/jpeg", 0.85));
      const fd = new FormData();
      const [sx, sy, sw, sh] = cropPx();
      fd.append("meta", JSON.stringify({ camera_id: camId, type, description, zone: $("#p-zone").value || null, ext: "webm",
        occurred_at: new Date().toISOString(), frame_size: [sw, sh],
        features: { shelf_total_s: dwell || 0, max_shelf_dwell_s: dwell || 0, track_age_s: dwell || 0, people_in_scene: P.people, shelf_visits: P.zone ? 1 : 0 } }));
      fd.append("snapshot", snap, "s.jpg");
      const ev = await api("POST", "/api/events/manual", fd);
      // clip: lo que muestra la pantalla + las marcas, durante CLIP_SECONDS
      const rec = document.createElement("canvas"); rec.width = video.videoWidth; rec.height = video.videoHeight;
      const rctx = rec.getContext("2d");
      const paint = setInterval(() => { rctx.drawImage(video, 0, 0); rctx.drawImage(canvas, 0, 0); }, 100);
      const chunks = [];
      const mr = new MediaRecorder(rec.captureStream(10), { mimeType: "video/webm" });
      mr.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      const done = new Promise((ok) => (mr.onstop = ok));
      mr.start(500);
      await new Promise((ok) => setTimeout(ok, CLIP_SECONDS * 1000));
      mr.stop(); await done; clearInterval(paint);
      const blob = new Blob(chunks, { type: "video/webm" });
      const put = await fetch(ev.upload_url, { method: "PUT", body: blob, headers: { "Content-Type": "video/webm", "x-upsert": "true" } });
      if (!put.ok) throw new Error("No se pudo subir el clip (" + put.status + ")");
      await api("POST", `/api/events/${ev.id}/clip-done`, { ext: "webm" });
      item.innerHTML = `${new Date().toLocaleTimeString("es-AR")} · ${esc(description)} · <a href="#/evento/${ev.id}">ver alerta</a>`;
      toast("Alerta guardada");
    } catch (e) { item.textContent += " · ERROR: " + e.message; fail(e); }
    finally { P.saving = false; }
  };
  $("#p-save").addEventListener("click", () => saveAlert("Guardada a mano desde la prueba", "alarma_dvr", P.zoneSince ? (performance.now() - P.zoneSince) / 1000 : 0));
}
