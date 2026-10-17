// "Image Info" tab (like Forge's PNG Info): drop an image, read its generation parameters, send them on.

import { escapeHtml } from "./common.js";
import { readImageSettings } from "./params.js";

export class InfoPage {
  /** ctx: { send(target, src, settings) } — target: txt2img | img2img | inpaint | canvas */
  constructor(root, ctx) {
    this.root = root;
    this.ctx = ctx;
    this.src = null;
    this.settings = null;
    root.innerHTML = `
      <div class="info-drop">
        <div class="info-empty">Drop, paste or <button class="i-open">open</button> an image<br><small>PNG from Venno, Forge or A1111, or an A1111 JPEG / WebP</small></div>
        <img class="info-img" alt="" hidden>
      </div>
      <div class="info-side">
        <div class="row info-send" hidden>
          <span class="lbl">Send to</span>
          <button data-to="txt2img">Text to Image</button>
          <button data-to="img2img">Image to Image</button>
          <button data-to="inpaint">Inpaint</button>
          <button data-to="canvas">Canvas</button>
        </div>
        <pre class="info-text">Generation parameters appear here.</pre>
        <table class="info-table"></table>
      </div>`;
    const file = document.createElement("input");
    file.type = "file";
    file.accept = "image/*";
    file.onchange = () => { if (file.files[0]) this.load(file.files[0]); file.value = ""; };
    root.querySelector(".i-open").onclick = () => file.click();
    const drop = root.querySelector(".info-drop");
    drop.onclick = (e) => { if (e.target === drop || e.target.classList.contains("info-img")) file.click(); };
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragging"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("dragging");
      const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
      if (f) this.load(f);
    });
    root.querySelectorAll(".info-send button").forEach((b) => (b.onclick = () => this.ctx.send(b.dataset.to, this.src, this.settings)));
  }

  async load(fileOrBlob) {
    this.src = URL.createObjectURL(fileOrBlob);
    const img = this.root.querySelector(".info-img");
    img.src = this.src;
    img.hidden = false;
    this.root.querySelector(".info-empty").hidden = true;
    const res = await readImageSettings(fileOrBlob);
    this.settings = res.found ? res.settings : null;
    this.root.querySelector(".info-text").textContent = res.found ? res.text : res.note || "No generation parameters in this image.";
    this.root.querySelector(".info-send").hidden = false;
    this.root.querySelectorAll('.info-send button:not([data-to="canvas"]):not([data-to="img2img"]):not([data-to="inpaint"])').forEach((b) => (b.disabled = !res.found));
    const raw = res.found ? res.settings.raw || {} : {};
    this.root.querySelector(".info-table").innerHTML = Object.entries(raw).map(([k, v]) => `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(String(v))}</td></tr>`).join("");
  }
}
