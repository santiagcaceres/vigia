// La página pública del aviso (aviso.html): trae el clip y la foto con el token del enlace. Va en un archivo aparte
// (no dentro del HTML) para que la política de contenido (CSP) no tenga que permitir scripts en línea.
"use strict";
var esc = function (v) { return String(v == null ? "" : v).replace(/[&<>"']/g, function (c) {
  return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); };
// el token va en el fragmento (#t=…, los links nuevos: no viaja al servidor) o en la consulta (?t=…, los de antes)
var token = new URLSearchParams(location.hash.replace(/^#/, "")).get("t") || new URLSearchParams(location.search).get("t");
var box = document.getElementById("content");
function state(title, detail) {
  box.innerHTML = '<div class="state"><h1>' + esc(title) + '</h1>' +
    (detail ? '<p class="muted">' + esc(detail) + '</p>' : '') + '</div>';
}
function fmt(iso) { try { return new Date(iso).toLocaleString("es-UY", { dateStyle: "long", timeStyle: "short" }); } catch (e) { return iso || ""; } }

if (!token) {
  state("Enlace inválido", "Faltan datos en el enlace. Pedí que te lo reenvíen.");
} else {
  fetch("/api/r/" + encodeURIComponent(token)).then(function (r) {
    if (r.status === 410) { state("El enlace venció", "Por seguridad, los enlaces caducan. Pedí uno nuevo al monitoreo."); return null; }
    if (r.status === 404) { state("El enlace no existe", "Puede haber sido dado de baja. Pedí que te lo reenvíen."); return null; }
    if (!r.ok) { state("No se pudo cargar", "Probá de nuevo en un rato."); return null; }
    return r.json();
  }).then(function (d) {
    if (!d) return;
    var clip = "/api/r/" + encodeURIComponent(token) + "/clip";
    var snap = "/api/r/" + encodeURIComponent(token) + "/snapshot";
    var media = d.has_clip
      ? '<video controls playsinline autoplay muted poster="' + (d.has_snapshot ? snap : "") + '"><source src="' + clip + '" type="video/mp4"></video>'
      : (d.has_snapshot ? '<img class="poster" src="' + snap + '" alt="">' : "");
    box.innerHTML = '<div class="card">' + media + '<div class="pad">' +
      '<span class="tag">Posible hurto</span>' +
      '<h1>' + esc(d.client_name) + (d.camera_name ? ' · ' + esc(d.camera_name) : '') + '</h1>' +
      (d.occurred_at ? '<div class="muted">' + esc(fmt(d.occurred_at)) + '</div>' : '') +
      '<p class="msg">' + esc(d.message) + '</p>' +
      (d.has_clip ? '<a class="btn" href="' + clip + '" download>Descargar el clip</a>' : '') +
      '</div></div>';
  }).catch(function () { state("No se pudo cargar", "Revisá tu conexión e intentá de nuevo."); });
}
