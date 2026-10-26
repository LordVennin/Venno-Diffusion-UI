// Viewport: rendering, navigation and painting / selection tools.

import { LAYER_CONTROL, LAYER_PAINT, LAYER_REGION, ctx2d, makeCanvas } from "./doc.js";

export const TOOLS = {
  brush: { label: "Brush", key: "b", paints: true },
  eraser: { label: "Eraser", key: "e", paints: true },
  fill: { label: "Fill", key: "g" },
  "shape-rect": { label: "Rectangle (fills a region on region layers; Ctrl = erase)", key: "u" },
  picker: { label: "Color picker", key: "i" },
  "select-rect": { label: "Rectangle select", key: "m" },
  "select-lasso": { label: "Lasso select", key: "l" },
  "select-brush": { label: "Selection brush", key: "s", paints: true },
  move: { label: "Move layer", key: "v" },
  pan: { label: "Pan", key: "h" },
};

export class View {
  constructor(doc, container, settings) {
    this.doc = doc;
    this.container = container;
    this.settings = settings; // shared brush / tool settings object
    this.canvas = document.createElement("canvas");
    this.canvas.className = "viewport-canvas";
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.tool = "brush";
    this.showRegions = true;
    this.pointer = null; // last pointer pos in screen px
    this.spaceDown = false;
    this.overlay = null; // {canvas, rect} result / progress preview
    this.genRect = null; // generation area outline
    this.strokeBuf = null;
    this.scratch = null;
    this.stroke = null;
    this.selTint = null;
    this.selBounds = null;
    this.onStrokeEnd = () => {};
    this.onColorPicked = () => {};
    this.onToast = () => {};
    this.onZoom = () => {};
    this._raf = 0;

    new ResizeObserver(() => this.resize()).observe(container);
    this.bindEvents();
    doc.on((what) => {
      if (what === "selection" || what === "reset") this.updateSelectionTint();
      if (what === "reset") { this.strokeBuf = null; this.scratch = null; this.fit(); }
      this.render();
    });
    this.resize();
    this.fit();
    this.updateSelectionTint();
  }

  // ---------- geometry ----------
  resize() {
    const r = this.container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.cssW = r.width;
    this.cssH = r.height;
    this.canvas.width = Math.max(1, Math.round(r.width * dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    this.canvas.style.width = `${r.width}px`;
    this.canvas.style.height = `${r.height}px`;
    this.render();
  }

  fit() {
    if (!this.cssW) return;
    const pad = 40;
    this.zoom = Math.min((this.cssW - pad * 2) / this.doc.width, (this.cssH - pad * 2) / this.doc.height, 4);
    this.panX = (this.cssW - this.doc.width * this.zoom) / 2;
    this.panY = (this.cssH - this.doc.height * this.zoom) / 2;
    this.onZoom(this.zoom);
    this.render();
  }

  setZoom(z, cx = this.cssW / 2, cy = this.cssH / 2) {
    z = Math.min(32, Math.max(0.05, z));
    const dx = (cx - this.panX) / this.zoom;
    const dy = (cy - this.panY) / this.zoom;
    this.zoom = z;
    this.panX = cx - dx * z;
    this.panY = cy - dy * z;
    this.onZoom(z);
    this.render();
  }

  toDoc(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    return { x: (clientX - r.left - this.panX) / this.zoom, y: (clientY - r.top - this.panY) / this.zoom };
  }

  // ---------- overlays ----------
  setOverlay(canvas, rect) {
    this.overlay = canvas ? { canvas, rect } : null;
    this.render();
  }
  setRegionPreview(canvas) {
    this.regionPreview = canvas;
    this.render();
  }
  setGenRect(rect) {
    this.genRect = rect;
    this.render();
  }

  updateSelectionTint() {
    const d = this.doc;
    this.selBounds = d.selectionBounds();
    if (!this.selBounds) { this.selTint = null; return; }
    if (!this.selTint || this.selTint.width !== d.width || this.selTint.height !== d.height) this.selTint = makeCanvas(d.width, d.height);
    const c = ctx2d(this.selTint);
    c.clearRect(0, 0, d.width, d.height);
    c.drawImage(d.selection, 0, 0);
    c.globalCompositeOperation = "source-in";
    c.fillStyle = "rgba(64, 196, 255, 1)";
    c.fillRect(0, 0, d.width, d.height);
    c.globalCompositeOperation = "source-over";
  }

  // ---------- rendering ----------
  render() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      this.draw();
    });
  }

  draw() {
    const { ctx, doc } = this;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.setTransform(dpr * this.zoom, 0, 0, dpr * this.zoom, dpr * this.panX, dpr * this.panY);
    ctx.imageSmoothingEnabled = this.zoom < 2;

    // checkerboard under the document (visible if the background layer is hidden/erased)
    ctx.fillStyle = this._checker || (this._checker = makeChecker(ctx));
    ctx.fillRect(0, 0, doc.width, doc.height);

    for (const l of doc.layers) {
      if (!l.visible) continue;
      if (l.type === LAYER_REGION) continue; // drawn after paint layers
      ctx.globalAlpha = l.opacity;
      ctx.drawImage(this.layerWithStroke(l), 0, 0);
    }
    ctx.globalAlpha = 1;

    if (this.showRegions && !this.overlay) {
      // draw top-first underneath what is already there, so a higher region hides lower ones, as generated
      const regions = doc.layers.filter((l) => l.visible && l.type === LAYER_REGION);
      {
        const tint = this.tintCanvas && this.tintCanvas.width === doc.width && this.tintCanvas.height === doc.height ? this.tintCanvas : (this.tintCanvas = makeCanvas(doc.width, doc.height));
        const tc = ctx2d(tint);
        tc.clearRect(0, 0, tint.width, tint.height);
        tc.globalCompositeOperation = "destination-over";
        for (const l of [...regions].reverse()) {
          tc.globalAlpha = l.id === doc.activeId ? 1 : 0.65;
          tc.drawImage(this.layerWithStroke(l), 0, 0);
        }
        tc.globalCompositeOperation = "source-over";
        tc.globalAlpha = 1;
        ctx.globalAlpha = 0.5;
        ctx.drawImage(tint, 0, 0);
      }
      ctx.globalAlpha = 1;
    }

    if (this.regionPreview && !this.overlay) {
      ctx.globalAlpha = 0.4;
      ctx.drawImage(this.regionPreview, 0, 0);
      ctx.globalAlpha = 1;
    }

    // result / progress preview (region tints are hidden so the result is seen as-is)
    if (this.overlay) {
      const { canvas, rect } = this.overlay;
      ctx.drawImage(canvas, rect.x, rect.y, rect.w, rect.h);
    }

    // selection
    if (this.selTint || (this.stroke && this.stroke.target === doc.selection)) {
      ctx.globalAlpha = 0.28;
      if (this.stroke && this.stroke.target === doc.selection) ctx.drawImage(this.selectionWithStroke(), 0, 0);
      else ctx.drawImage(this.selTint, 0, 0);
      ctx.globalAlpha = 1;
    }
    const lw = 1 / this.zoom;
    if (this.selBounds) {
      const b = this.selBounds;
      ctx.lineWidth = lw;
      ctx.setLineDash([5 * lw, 4 * lw]);
      ctx.strokeStyle = "#000";
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.lineDashOffset = 5 * lw;
      ctx.strokeStyle = "#fff";
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.setLineDash([]);
      ctx.lineDashOffset = 0;
    }

    // shape previews
    const s = this.stroke;
    if (s && s.kind === "rect") {
      const r = normRect(s.start, s.last);
      if (s.shape && !s.erase) {
        ctx.globalAlpha = 0.35;
        ctx.fillStyle = s.layer.type === LAYER_REGION ? s.layer.color : this.settings.color;
        ctx.fillRect(r.x, r.y, r.w, r.h);
        ctx.globalAlpha = 1;
      }
      ctx.lineWidth = lw * 1.5;
      ctx.strokeStyle = s.erase ? "#ff5c5c" : "#40c4ff";
      ctx.strokeRect(r.x, r.y, r.w, r.h);
    } else if (s && s.kind === "lasso" && s.points.length > 1) {
      ctx.lineWidth = lw * 1.5;
      ctx.strokeStyle = "#40c4ff";
      ctx.beginPath();
      s.points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.closePath();
      ctx.stroke();
    }

    if (this.genRect) {
      const r = this.genRect;
      ctx.lineWidth = lw * 2;
      ctx.setLineDash([10 * lw, 6 * lw]);
      ctx.strokeStyle = "#ffb000";
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.setLineDash([]);
    }

    // document border
    ctx.lineWidth = lw;
    ctx.strokeStyle = "rgba(0,0,0,.5)";
    ctx.strokeRect(0, 0, doc.width, doc.height);

    // brush cursor
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.pointer && TOOLS[this.tool]?.paints && !this.spaceDown) {
      const r = (this.settings.size / 2) * this.zoom;
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(0,0,0,.8)";
      ctx.beginPath();
      ctx.arc(this.pointer.x, this.pointer.y, Math.max(1, r), 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = "rgba(255,255,255,.8)";
      ctx.beginPath();
      ctx.arc(this.pointer.x, this.pointer.y, Math.max(1, r - 1), 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // layer canvas with the in-progress stroke applied (for live display)
  layerWithStroke(layer) {
    const s = this.stroke;
    if (!s || s.kind !== "paint" || s.target !== layer.canvas) {
      if (s && s.kind === "move" && s.layer === layer) return this.movedLayer(layer, s);
      return layer.canvas;
    }
    const sc = this.getScratch();
    const c = ctx2d(sc);
    c.clearRect(0, 0, sc.width, sc.height);
    c.drawImage(layer.canvas, 0, 0);
    this.applyStroke(c, s);
    return sc;
  }

  selectionWithStroke() {
    const sc = this.getScratch();
    const c = ctx2d(sc);
    c.clearRect(0, 0, sc.width, sc.height);
    c.drawImage(this.doc.selection, 0, 0);
    this.applyStroke(c, this.stroke);
    c.globalCompositeOperation = "source-in";
    c.fillStyle = "rgba(64, 196, 255, 1)";
    c.fillRect(0, 0, sc.width, sc.height);
    c.globalCompositeOperation = "source-over";
    return sc;
  }

  movedLayer(layer, s) {
    const sc = this.getScratch();
    const c = ctx2d(sc);
    c.clearRect(0, 0, sc.width, sc.height);
    c.drawImage(layer.canvas, Math.round(s.dx), Math.round(s.dy));
    return sc;
  }

  // composite the stroke buffer onto ctx using the stroke's mode / opacity / selection clip
  applyStroke(c, s) {
    let buf = this.strokeBuf;
    if (s.clip && this.doc.hasSelection) {
      const clip = this.getClipBuf();
      const cc = ctx2d(clip);
      cc.clearRect(0, 0, clip.width, clip.height);
      cc.drawImage(buf, 0, 0);
      cc.globalCompositeOperation = "destination-in";
      cc.drawImage(this.doc.selection, 0, 0);
      cc.globalCompositeOperation = "source-over";
      buf = clip;
    }
    c.globalAlpha = s.opacity;
    c.globalCompositeOperation = s.erase ? "destination-out" : "source-over";
    c.drawImage(buf, 0, 0);
    c.globalAlpha = 1;
    c.globalCompositeOperation = "source-over";
  }

  getScratch() {
    const d = this.doc;
    if (!this.scratch || this.scratch.width !== d.width || this.scratch.height !== d.height) this.scratch = makeCanvas(d.width, d.height);
    return this.scratch;
  }
  getClipBuf() {
    const d = this.doc;
    if (!this._clip || this._clip.width !== d.width || this._clip.height !== d.height) this._clip = makeCanvas(d.width, d.height);
    return this._clip;
  }
  getStrokeBuf() {
    const d = this.doc;
    if (!this.strokeBuf || this.strokeBuf.width !== d.width || this.strokeBuf.height !== d.height) this.strokeBuf = makeCanvas(d.width, d.height);
    ctx2d(this.strokeBuf).clearRect(0, 0, d.width, d.height);
    return this.strokeBuf;
  }

  // ---------- input ----------
  bindEvents() {
    const cv = this.canvas;
    cv.addEventListener("pointerdown", (e) => this.pointerDown(e));
    cv.addEventListener("pointermove", (e) => this.pointerMove(e));
    cv.addEventListener("pointerup", (e) => this.pointerUp(e));
    cv.addEventListener("pointercancel", (e) => this.pointerUp(e));
    cv.addEventListener("pointerleave", () => { this.pointer = null; this.render(); });
    cv.addEventListener("contextmenu", (e) => e.preventDefault());
    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      if (e.ctrlKey || !e.shiftKey) {
        const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
        this.setZoom(this.zoom * factor, e.clientX - r.left, e.clientY - r.top);
      } else {
        this.panX -= e.deltaY;
        this.render();
      }
    }, { passive: false });
  }

  activeTool(e) {
    if (this.spaceDown || e.button === 1) return "pan";
    if (e.altKey && TOOLS[this.tool]?.paints && this.tool !== "select-brush") return "picker";
    return this.tool;
  }

  pointerDown(e) {
    if (e.button === 2) return;
    this.canvas.setPointerCapture(e.pointerId);
    const p = this.toDoc(e.clientX, e.clientY);
    const tool = this.activeTool(e);
    const doc = this.doc;
    const layer = doc.active;

    if (tool === "pan") {
      this.stroke = { kind: "pan", sx: e.clientX, sy: e.clientY, px: this.panX, py: this.panY };
      return;
    }
    if (tool === "picker") {
      this.pickColor(p);
      this.stroke = { kind: "picker" };
      return;
    }
    if (tool === "select-rect") {
      this.stroke = { kind: "rect", start: p, last: p, mode: selectionMode(e, this.settings.selectMode) };
      return;
    }
    if (tool === "select-lasso") {
      this.stroke = { kind: "lasso", points: [p], mode: selectionMode(e, this.settings.selectMode) };
      return;
    }
    if (tool === "shape-rect") {
      if (!this.checkLayer(layer)) return;
      this.stroke = { kind: "rect", shape: true, layer, start: p, last: p, erase: e.ctrlKey || e.metaKey || e.button === 2 };
      return;
    }
    if (tool === "fill") {
      if (!this.checkLayer(layer)) return;
      this.floodFill(layer, p);
      return;
    }
    if (tool === "move") {
      if (!this.checkLayer(layer)) return;
      this.stroke = { kind: "move", layer, start: p, dx: 0, dy: 0 };
      return;
    }

    // painting tools
    let target, color, erase;
    if (tool === "select-brush") {
      target = doc.selection;
      erase = e.button === 0 && (e.ctrlKey || e.metaKey || this.settings.selectMode === "subtract");
      color = "#ffffff";
    } else {
      if (!this.checkLayer(layer)) return;
      target = layer.canvas;
      erase = tool === "eraser";
      color = layer.type === LAYER_REGION ? layer.color : this.settings.color;
    }
    this.getStrokeBuf();
    this.stroke = {
      kind: "paint",
      target,
      erase,
      color,
      opacity: tool === "select-brush" || layer.type === LAYER_REGION ? 1 : this.settings.opacity,
      clip: target !== doc.selection,
      last: null,
      layerType: layer.type,
    };
    this.dab(p, e);
    this.render();
  }

  pointerMove(e) {
    const r = this.canvas.getBoundingClientRect();
    this.pointer = { x: e.clientX - r.left, y: e.clientY - r.top };
    const s = this.stroke;
    if (!s) { this.render(); return; }
    if (s.kind === "pan") {
      this.panX = s.px + (e.clientX - s.sx);
      this.panY = s.py + (e.clientY - s.sy);
    } else if (s.kind === "picker") {
      this.pickColor(this.toDoc(e.clientX, e.clientY));
    } else if (s.kind === "rect") {
      s.last = this.toDoc(e.clientX, e.clientY);
    } else if (s.kind === "lasso") {
      s.points.push(this.toDoc(e.clientX, e.clientY));
    } else if (s.kind === "move") {
      const p = this.toDoc(e.clientX, e.clientY);
      s.dx = p.x - s.start.x;
      s.dy = p.y - s.start.y;
    } else if (s.kind === "paint") {
      const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
      for (const ev of events.length ? events : [e]) this.dab(this.toDoc(ev.clientX, ev.clientY), ev);
    }
    this.render();
  }

  pointerUp(e) {
    const s = this.stroke;
    if (!s) return;
    this.stroke = null;
    const doc = this.doc;
    if (s.kind === "paint") {
      doc.editPixels(s.target, (c) => this.applyStroke(c, s), s.target === doc.selection ? "selection" : "pixels");
      this.onStrokeEnd(s.target === doc.selection ? "selection" : "pixels");
    } else if (s.kind === "rect" && s.shape) {
      const r = normRect(s.start, s.last);
      if (r.w >= 2 && r.h >= 2) {
        const l = s.layer;
        doc.editPixels(l.canvas, (c) => {
          if (s.erase) { c.clearRect(r.x, r.y, r.w, r.h); return; }
          if (l.type === LAYER_CONTROL) {  // guides: an outline, like a sketched box
            c.strokeStyle = this.settings.color;
            c.lineWidth = Math.max(1, this.settings.size / 3);
            c.strokeRect(r.x, r.y, r.w, r.h);
            return;
          }
          c.globalAlpha = l.type === LAYER_REGION ? 1 : this.settings.opacity;
          c.fillStyle = l.type === LAYER_REGION ? l.color : this.settings.color;
          c.fillRect(r.x, r.y, r.w, r.h);
          c.globalAlpha = 1;
        });
        this.onStrokeEnd("pixels");
      }
    } else if (s.kind === "rect") {
      const r = normRect(s.start, s.last);
      if (r.w < 2 || r.h < 2) { if (s.mode === "replace") doc.deselect(); }
      else this.applySelectionShape(s.mode, (c) => c.fillRect(r.x, r.y, r.w, r.h));
      this.onStrokeEnd("selection");
    } else if (s.kind === "lasso") {
      if (s.points.length > 2) {
        this.applySelectionShape(s.mode, (c) => {
          c.beginPath();
          s.points.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)));
          c.closePath();
          c.fill();
        });
      }
      this.onStrokeEnd("selection");
    } else if (s.kind === "move") {
      const dx = Math.round(s.dx), dy = Math.round(s.dy);
      if (dx || dy) {
        doc.editPixels(s.layer.canvas, (c) => {
          const tmp = makeCanvas(doc.width, doc.height);
          ctx2d(tmp).drawImage(s.layer.canvas, 0, 0);
          c.clearRect(0, 0, doc.width, doc.height);
          c.drawImage(tmp, dx, dy);
        });
        this.onStrokeEnd("pixels");
      }
    }
    this.render();
  }

  applySelectionShape(mode, drawShape) {
    const doc = this.doc;
    doc.editPixels(doc.selection, (c) => {
      if (mode === "replace") c.clearRect(0, 0, doc.width, doc.height);
      c.globalCompositeOperation = mode === "subtract" ? "destination-out" : "source-over";
      c.fillStyle = "#fff";
      drawShape(c);
      c.globalCompositeOperation = "source-over";
    }, "selection");
  }

  checkLayer(layer) {
    if (!layer.visible) {
      this.onToast("The active layer is hidden");
      return false;
    }
    return true;
  }

  // ---------- brush engine ----------
  dab(p, e) {
    const s = this.stroke;
    const st = this.settings;
    const pressure = e.pointerType === "pen" ? Math.max(0.05, e.pressure || 0.5) : 1;
    const size = Math.max(1, st.size * (st.pressureSize ? pressure : 1));
    const flow = st.pressureOpacity && e.pointerType === "pen" ? pressure : 1;
    const c = ctx2d(this.strokeBuf);
    const stamp = this.getStamp(size, st.hardness, s.erase ? "#000" : s.color);
    const spacing = Math.max(0.5, size * 0.08);
    const draw = (x, y) => {
      c.globalAlpha = flow;
      c.drawImage(stamp, x - stamp.width / 2, y - stamp.height / 2);
    };
    if (!s.last) {
      draw(p.x, p.y);
      s.last = { ...p, carry: 0 };
      return;
    }
    const dx = p.x - s.last.x, dy = p.y - s.last.y;
    const dist = Math.hypot(dx, dy);
    let t = spacing - s.last.carry;
    while (t <= dist) {
      draw(s.last.x + (dx * t) / dist, s.last.y + (dy * t) / dist);
      t += spacing;
    }
    s.last = { x: p.x, y: p.y, carry: dist - (t - spacing) };
    c.globalAlpha = 1;
  }

  getStamp(size, hardness, color) {
    const key = `${Math.round(size * 2)}|${hardness}|${color}`;
    if (this._stampKey === key) return this._stamp;
    const r = size / 2;
    const dim = Math.ceil(size) + 2;
    const c = makeCanvas(dim, dim);
    const x = ctx2d(c);
    const cx = dim / 2;
    if (hardness >= 0.99) {
      x.fillStyle = color;
      x.beginPath();
      x.arc(cx, cx, r, 0, Math.PI * 2);
      x.fill();
    } else {
      const g = x.createRadialGradient(cx, cx, r * hardness, cx, cx, r);
      g.addColorStop(0, color);
      g.addColorStop(1, hexToRgba(color, 0));
      x.fillStyle = g;
      x.fillRect(0, 0, dim, dim);
    }
    this._stampKey = key;
    this._stamp = c;
    return c;
  }

  // ---------- picker & fill ----------
  pickColor(p) {
    const x = Math.floor(p.x), y = Math.floor(p.y);
    if (x < 0 || y < 0 || x >= this.doc.width || y >= this.doc.height) return;
    const comp = this.doc.composite({ x, y, w: 1, h: 1 }, 1, 1);
    const d = ctx2d(comp).getImageData(0, 0, 1, 1).data;
    this.onColorPicked(rgbToHex(d[0], d[1], d[2]));
  }

  floodFill(layer, p) {
    const doc = this.doc;
    const W = doc.width, H = doc.height;
    const sx = Math.floor(p.x), sy = Math.floor(p.y);
    if (sx < 0 || sy < 0 || sx >= W || sy >= H) return;
    // sample from the composite of paint layers (what the user sees)
    const src = ctx2d(doc.composite()).getImageData(0, 0, W, H).data;
    const tol = this.settings.fillTolerance;
    const i0 = (sy * W + sx) * 4;
    const r0 = src[i0], g0 = src[i0 + 1], b0 = src[i0 + 2];
    const filled = new Uint8Array(W * H);
    const stack = [sx, sy];
    const match = (i) => Math.abs(src[i] - r0) <= tol && Math.abs(src[i + 1] - g0) <= tol && Math.abs(src[i + 2] - b0) <= tol;
    while (stack.length) {
      const y = stack.pop(), x0 = stack.pop();
      let x = x0;
      while (x >= 0 && !filled[y * W + x] && match((y * W + x) * 4)) x--;
      x++;
      let up = false, down = false;
      while (x < W && !filled[y * W + x] && match((y * W + x) * 4)) {
        filled[y * W + x] = 1;
        if (y > 0) {
          const m = !filled[(y - 1) * W + x] && match(((y - 1) * W + x) * 4);
          if (m && !up) { stack.push(x, y - 1); up = true; } else if (!m) up = false;
        }
        if (y < H - 1) {
          const m = !filled[(y + 1) * W + x] && match(((y + 1) * W + x) * 4);
          if (m && !down) { stack.push(x, y + 1); down = true; } else if (!m) down = false;
        }
        x++;
      }
    }
    const mask = makeCanvas(W, H);
    const mc = ctx2d(mask);
    const img = mc.createImageData(W, H);
    const color = hexToRgb(layer.type === LAYER_REGION ? layer.color : this.settings.color);
    // grow by 1px so fills tuck under anti-aliased line art
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const k = y * W + x;
        if (filled[k] || (x > 0 && filled[k - 1]) || (x < W - 1 && filled[k + 1]) || (y > 0 && filled[k - W]) || (y < H - 1 && filled[k + W])) {
          const o = k * 4;
          img.data[o] = color[0];
          img.data[o + 1] = color[1];
          img.data[o + 2] = color[2];
          img.data[o + 3] = 255;
        }
      }
    }
    mc.putImageData(img, 0, 0);
    if (doc.hasSelection) {
      mc.globalCompositeOperation = "destination-in";
      mc.drawImage(doc.selection, 0, 0);
    }
    doc.editPixels(layer.canvas, (c) => {
      c.globalAlpha = layer.type === LAYER_REGION ? 1 : this.settings.opacity;
      c.drawImage(mask, 0, 0);
      c.globalAlpha = 1;
    });
    this.onStrokeEnd("pixels");
  }
}

// ---------- utils ----------
function selectionMode(e, fallback) {
  if (e.shiftKey) return "add";
  if (e.altKey || e.ctrlKey || e.metaKey) return "subtract";
  return fallback || "replace";
}

function normRect(a, b) {
  const x = Math.round(Math.min(a.x, b.x)), y = Math.round(Math.min(a.y, b.y));
  return { x, y, w: Math.round(Math.abs(a.x - b.x)), h: Math.round(Math.abs(a.y - b.y)) };
}

function makeChecker(ctx) {
  const c = makeCanvas(16, 16);
  const x = ctx2d(c);
  x.fillStyle = "#cfcfcf";
  x.fillRect(0, 0, 16, 16);
  x.fillStyle = "#f2f2f2";
  x.fillRect(0, 0, 8, 8);
  x.fillRect(8, 8, 8, 8);
  return ctx.createPattern(c, "repeat");
}

export function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function hexToRgba(hex, a) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

export function rgbToHex(r, g, b) {
  return "#" + [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("");
}

export { LAYER_CONTROL, LAYER_PAINT };
