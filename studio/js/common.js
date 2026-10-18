// Small helpers shared by the Canvas, the form tabs and Settings.

export const $ = (s, root = document) => root.querySelector(s);
export const $$ = (s, root = document) => [...root.querySelectorAll(s)];

export async function api(path, body = undefined, method = body === undefined ? "GET" : "POST") {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || j.message || j.error || JSON.stringify(j);
      if (typeof detail !== "string") detail = JSON.stringify(detail);
    } catch {}
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

export function fillSelect(sel, items, value, { empty = null } = {}) {
  sel.innerHTML = (empty !== null ? `<option value="">${escapeHtml(empty)}</option>` : "") + items.map((i) => `<option value="${escapeHtml(i.value)}">${escapeHtml(i.label)}</option>`).join("");
  if (value !== undefined && items.some((i) => i.value === value)) sel.value = value;
}

export const randomSeed = () => Math.floor(Math.random() * 2 ** 31);


export function formatBytes(n) {
  if (!n) return "";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i > 1 ? 1 : 0)} ${u[i]}`;
}

export function withQualityPrefix(prompt, enabled, info) {
  prompt = prompt.trim();
  const prefix = info?.prompt_defaults?.positive_prefix;
  if (!enabled || !prefix) return prompt;
  return prompt.startsWith(prefix.trim().replace(/,$/, "")) ? prompt : prefix + prompt;
}

export const ADETAILER_DEFAULT = {
  enabled: true,
  units: [
    { model: "face_yolov8n.pt", prompt: "", negative: "", confidence: 0.3, denoise: 0.4, maskBlur: 4, dilate: 4, padding: 32 },
    { model: "None", prompt: "", negative: "", confidence: 0.3, denoise: 0.4, maskBlur: 4, dilate: 4, padding: 32 },
  ],
};

let toastTimer = 0;
export function toast(msg, kind = "error") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = `toast on ${kind === "info" ? "info" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast"), kind === "info" ? 2500 : 6000);
}

export function loadStore(key, fallback) {
  try { return { ...fallback, ...(JSON.parse(localStorage.getItem(key)) || {}) }; } catch { return { ...fallback }; }
}
const saveTimers = {};
export function saveStore(key, value) {
  clearTimeout(saveTimers[key]);
  saveTimers[key] = setTimeout(() => { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} }, 300);
}

export function downloadDataURL(src, name) {
  const a = document.createElement("a");
  a.href = src;
  a.download = name;
  a.click();
}

/** Full-window image viewer (comparison grids on the Canvas, which can't be applied as a layer). */
export function showLightbox(src, name = "venno-grid.png") {
  let box = document.getElementById("lightbox");
  if (!box) {
    box = document.createElement("div");
    box.id = "lightbox";
    box.innerHTML = `<img alt=""><div class="row"><button class="lb-download">Download</button><button class="lb-close primary">Close</button></div>`;
    document.body.appendChild(box);
    box.onclick = (e) => { if (e.target === box || e.target.classList.contains("lb-close")) box.hidden = true; };
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !box.hidden) { e.stopPropagation(); box.hidden = true; } }, true);
  }
  box.querySelector("img").src = src;
  box.querySelector(".lb-download").onclick = () => downloadDataURL(src, name);
  box.hidden = false;
}

// Recommended preset for a checkpoint (by file name)
export function presetForCheckpoint(name = "") {
  if (/turbo/i.test(name)) return "turbo";
  if (/3[._-]?8b/i.test(name)) return "anima38";
  if (/2[._-]?9b/i.test(name)) return "anima29";
  return "standard";
}
