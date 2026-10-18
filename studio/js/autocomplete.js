// Tag autocomplete for every prompt box (delegated, so layer / detailer prompts created later get it too).
// Danbooru tags come from the server already in Anima's format (spaces, @artist, escaped parentheses).
// Typing "@" lists artists only; "__" lists wildcard files. ↑/↓ to choose, Enter or Tab to insert, Esc to close.
// Can be switched off in Settings → Studio (prefs.autocomplete).

import { escapeHtml, prefs } from "./common.js";

const SEPARATORS = /[,\n]/;
let wildcardNames = [];
export const setWildcards = (names) => (wildcardNames = names || []);

const fmtCount = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : `${n}`);

/** Caret position in page coordinates (mirror-div technique). */
function caretXY(ta) {
  const div = document.createElement("div");
  const cs = getComputedStyle(ta);
  for (const p of ["boxSizing", "width", "fontFamily", "fontSize", "fontWeight", "lineHeight", "letterSpacing", "paddingTop", "paddingRight", "paddingBottom", "paddingLeft", "borderTopWidth", "borderLeftWidth", "whiteSpace", "wordWrap", "tabSize"]) div.style[p] = cs[p];
  Object.assign(div.style, { position: "absolute", visibility: "hidden", whiteSpace: "pre-wrap", wordWrap: "break-word", overflow: "hidden", top: "0", left: "-9999px" });
  div.textContent = ta.value.slice(0, ta.selectionStart);
  const mark = document.createElement("span");
  mark.textContent = "​";
  div.appendChild(mark);
  document.body.appendChild(div);
  const r = ta.getBoundingClientRect();
  const x = r.left + mark.offsetLeft - ta.scrollLeft;
  const y = r.top + mark.offsetTop - ta.scrollTop + parseFloat(cs.lineHeight || "16");
  div.remove();
  return { x: Math.min(x, r.right - 40), y: Math.min(y, r.bottom + 4) };
}

/** The tag being typed: from the last separator to the caret. */
function currentToken(ta) {
  const caret = ta.selectionStart;
  if (caret !== ta.selectionEnd) return null;
  const before = ta.value.slice(0, caret);
  let start = before.length;
  while (start > 0 && !SEPARATORS.test(before[start - 1])) start--;
  const raw = before.slice(start);
  const lead = raw.length - raw.trimStart().length;
  const token = raw.trim();
  if (/<lora:[^>]*$/i.test(before) || /^\[?\d/.test(token)) return null; // LoRA tag or a number
  const open = token.match(/^[([{]+/); // typing inside "(tag" or "[a" → complete what follows the bracket
  const skip = open ? open[0].length : 0;
  return { start: start + lead + skip, end: caret, text: token.slice(skip) };
}

class Popup {
  constructor() {
    this.el = document.createElement("ul");
    this.el.className = "ac-popup";
    this.el.hidden = true;
    document.body.appendChild(this.el);
    this.items = [];
    this.active = 0;
    this.ta = null;
    this.token = null;
    this.el.addEventListener("mousedown", (e) => {
      const li = e.target.closest("li");
      if (!li) return;
      e.preventDefault(); // keep focus in the textarea
      this.accept(Number(li.dataset.i));
    });
  }

  show(ta, token, items) {
    this.ta = ta;
    this.token = token;
    this.items = items;
    this.active = 0;
    if (!items.length) return this.hide();
    this.el.innerHTML = items.map((it, i) => `<li data-i="${i}" class="${i === 0 ? "on" : ""}"><span class="cat ${it.category}"></span><span class="t">${escapeHtml(it.text)}</span>${it.alias ? `<span class="alias">← ${escapeHtml(it.alias)}</span>` : ""}<span class="n">${it.count != null ? fmtCount(it.count) : escapeHtml(it.note || "")}</span></li>`).join("");
    const { x, y } = caretXY(ta);
    this.el.hidden = false;
    const w = this.el.offsetWidth, h = this.el.offsetHeight;
    this.el.style.left = `${Math.max(8, Math.min(x, innerWidth - w - 8))}px`;
    this.el.style.top = `${y + h > innerHeight - 8 ? Math.max(8, y - h - 24) : y}px`;
  }

  hide() { this.el.hidden = true; this.items = []; }
  get open() { return !this.el.hidden; }

  move(d) {
    this.active = (this.active + d + this.items.length) % this.items.length;
    [...this.el.children].forEach((li, i) => li.classList.toggle("on", i === this.active));
    this.el.children[this.active]?.scrollIntoView({ block: "nearest" });
  }

  accept(i = this.active) {
    const it = this.items[i];
    const ta = this.ta, tok = this.token;
    if (!it || !ta) return;
    const after = ta.value.slice(tok.end);
    const sep = /^\s*[,)\]]/.test(after) || it.noComma ? "" : ", ";
    ta.value = ta.value.slice(0, tok.start) + it.text + sep + after.replace(/^[^\s,()[\]{}|]*/, ""); // also replace the rest of a half-typed word
    const pos = tok.start + it.text.length + sep.length;
    ta.setSelectionRange(pos, pos);
    this.hide();
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }
}

let popup = null;
let timer = 0;
let seq = 0;

async function suggest(ta) {
  const tok = currentToken(ta);
  if (!tok || tok.text.length < 2) return popup.hide();
  const mySeq = ++seq;
  if (tok.text.startsWith("__")) {
    const q = tok.text.slice(2).replace(/_+$/, "").toLowerCase();
    const items = wildcardNames.filter((n) => n.toLowerCase().includes(q)).slice(0, 12).map((n) => ({ text: `__${n}__`, category: "wildcard", note: "wildcard" }));
    return popup.show(ta, tok, items);
  }
  let items = [];
  try {
    const res = await fetch(`/venno/v1/tags?limit=12&q=${encodeURIComponent(tok.text)}`);
    items = await res.json();
  } catch { return popup.hide(); }
  if (mySeq !== seq || document.activeElement !== ta) return;
  // the exact tag already typed: nothing to suggest
  if (items.length === 1 && items[0].text === tok.text) return popup.hide();
  popup.show(ta, tok, items);
}

/** Have the server load its tag list now (it is read on first use), so the first suggestion isn't slow. */
export function warmTags() {
  if (prefs.autocomplete) fetch("/venno/v1/tags?limit=1&q=").catch(() => {});
}

export function enableAutocomplete(root = document) {
  popup = popup || new Popup();
  warmTags();
  root.addEventListener("input", (e) => {
    const ta = e.target;
    if (!prefs.autocomplete || !(ta instanceof HTMLTextAreaElement) || ta.dataset.noac !== undefined) return;
    clearTimeout(timer);
    timer = setTimeout(() => suggest(ta), 90);
  });
  root.addEventListener("keydown", (e) => {
    if (!popup.open || e.target !== popup.ta) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); e.stopPropagation(); popup.move(e.key === "ArrowDown" ? 1 : -1); }
    else if ((e.key === "Enter" || e.key === "Tab") && !e.ctrlKey && !e.metaKey && !e.shiftKey) { e.preventDefault(); e.stopPropagation(); popup.accept(); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); popup.hide(); }
  }, true);
  root.addEventListener("focusout", (e) => { if (e.target === popup.ta) setTimeout(() => popup.hide(), 120); });
  root.addEventListener("scroll", () => popup.open && popup.hide(), true);
}
