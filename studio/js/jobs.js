// Client for the Venno job API (/venno/v1/jobs + /venno/v1/events).
// One shared WebSocket delivers job updates and live previews; polling takes over if it is down.

import { api } from "./common.js";

const V = "/venno/v1";
const watchers = new Map(); // job id -> {onUpdate, onPreview, resolve, reject}
let ws = null;
let wsReady = false;
let retry = 0;

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}${V}/events`);
  ws.onopen = () => { wsReady = true; retry = 0; };
  ws.onclose = () => {
    wsReady = false;
    ws = null;
    if (watchers.size) setTimeout(connect, Math.min(5000, 500 * 2 ** retry++));
  };
  ws.onerror = () => ws && ws.close();
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    const w = watchers.get(msg.id);
    if (!w) return;
    if (msg.type === "preview") w.onPreview?.(msg.image);
    else handleUpdate(msg);
  };
}

function handleUpdate(job) {
  const w = watchers.get(job.id);
  if (!w) return;
  w.onUpdate?.(job);
  if (job.status === "done") finish(job.id, () => w.resolve(job));
  else if (job.status === "cancelled") finish(job.id, () => w.resolve(job));
  else if (job.status === "error") finish(job.id, () => w.reject(new Error(job.error || "Generation failed")));
}

function finish(id, fn) {
  const w = watchers.get(id);
  watchers.delete(id);
  clearTimeout(w?.pollTimer);
  fn();
}

// Fast polling while the WebSocket is down; a slow safety poll otherwise
// (also catches events that arrived before the job was registered here)
function poll(id) {
  const w = watchers.get(id);
  if (!w) return;
  w.pollTimer = setTimeout(async () => {
    if (!watchers.has(id)) return;
    try { handleUpdate(await api(`${V}/jobs/${id}`)); } catch {}
    poll(id);
  }, wsReady ? 3000 : 800);
}

/**
 * Submit a job and wait for it to finish.
 * Resolves with the final job summary (status "done" or "cancelled"), rejects on error.
 */
export async function runJob(job, { onSubmit, onUpdate, onPreview } = {}) {
  connect();
  const submitted = await api(`${V}/jobs`, job);
  onSubmit?.(submitted);
  return new Promise((resolve, reject) => {
    watchers.set(submitted.id, { onUpdate, onPreview, resolve, reject });
    handleUpdate(submitted);
    poll(submitted.id);
  });
}

export function cancelJob(id) {
  return api(`${V}/jobs/${id}/cancel`, {}).catch(() => {});
}

// Job summary → progress bar text
export function jobProgress(j) {
  if (j.status === "queued") return { pct: 0, text: j.queue_position > 1 ? `Queued (#${j.queue_position})` : "Queued…" };
  const pct = Math.round((j.progress || 0) * 100);
  const part = j.job_count > 1 ? ` · pass ${j.job_no + 1}/${j.job_count}` : "";
  const step = j.steps ? ` · step ${j.step}/${j.steps}` : "";
  const eta = j.eta && pct >= 5 ? ` · ~${Math.round(j.eta)}s left` : "";
  return { pct, text: `${pct}%${part}${step}${eta}` };
}

// Shared model selection → job.model
export function jobModel(model) {
  return { checkpoint: model.checkpoint || "", text_encoder: model.textEncoder || "", vae: model.vae || "" };
}
