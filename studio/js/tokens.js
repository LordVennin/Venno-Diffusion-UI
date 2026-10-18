// Token counters on the prompt boxes: "N / 512", the length of the context Anima sees (venno/engine/tokens.py).
// Can be switched off in Settings → Studio (prefs.tokenCounter).

import { prefs } from "./common.js";

const counters = new Set();
let pending = new Set();
let timer = 0;
let limit = 512;

/** Show a counter in the bottom-right corner of a prompt box. opts.text: the text actually sent (default: the box's value). */
export function attachTokenCounter(ta, opts = {}) {
  if (ta.dataset.tc) return;
  ta.dataset.tc = "1";
  const wrap = document.createElement("div");
  wrap.className = "tc-wrap";
  ta.replaceWith(wrap);
  wrap.appendChild(ta);
  const badge = document.createElement("span");
  badge.className = "tc";
  badge.title = "Prompt length in tokens (Anima is trained on prompts up to 512)";
  wrap.appendChild(badge);
  const c = { ta, badge, text: opts.text || (() => ta.value) };
  counters.add(c);
  ta.addEventListener("input", () => schedule(c));
  schedule(c, 0);
}

/** Recount (e.g. after a prompt was set from code, or the quality prefix changed). */
export function refreshTokenCounters() {
  for (const c of counters) if (c.ta.isConnected) schedule(c, 0);
}

function schedule(c, delay = 250) {
  pending.add(c);
  clearTimeout(timer);
  timer = setTimeout(flush, delay);
}

async function flush() {
  const batch = [...pending].filter((c) => c.ta.isConnected);
  pending = new Set();
  for (const c of counters) if (!c.ta.isConnected) counters.delete(c); // region rows are re-rendered
  for (const c of batch) c.badge.hidden = !prefs.tokenCounter;
  const todo = batch.filter(() => prefs.tokenCounter);
  if (!todo.length) return;
  let res;
  try {
    const r = await fetch("/venno/v1/tokens", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ texts: todo.map((c) => c.text()) }) });
    if (!r.ok) return;
    res = await r.json();
  } catch { return; }
  limit = res.limit || limit;
  todo.forEach((c, i) => {
    const n = res.counts[i] ?? 0;
    c.badge.textContent = `${n} / ${limit}`;
    c.badge.classList.toggle("over", n > limit);
    c.badge.classList.toggle("empty", n === 0);
  });
}
