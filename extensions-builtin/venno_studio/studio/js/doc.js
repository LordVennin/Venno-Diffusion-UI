// Document model: layers, selection, history and compositing helpers.

export const LAYER_PAINT = "paint";
export const LAYER_REGION = "region";
export const LAYER_CONTROL = "control";

const REGION_COLORS = ["#ff5c5c", "#4fa3ff", "#4fd17a", "#ffb84f", "#c67cff", "#4fe0e0", "#ff7cc8", "#d6d64f"];

let nextId = 1;

export function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w));
  c.height = Math.max(1, Math.round(h));
  return c;
}

export function ctx2d(c) {
  return c.getContext("2d", { willReadFrequently: true });
}

export class Layer {
  constructor(doc, { type = LAYER_PAINT, name = null, canvas = null } = {}) {
    this.id = nextId++;
    this.type = type;
    this.canvas = canvas || makeCanvas(doc.width, doc.height);
    this.visible = true;
    this.opacity = 1;
    this.name = name || defaultName(type, this.id);
    // region layers
    this.prompt = "";
    this.color = REGION_COLORS[(doc.layers.filter((l) => l.type === LAYER_REGION).length) % REGION_COLORS.length];
    this.regionStrength = 1;
    // control layers
    this.controlModel = "";
    this.controlWeight = 0.8;
    this.controlStart = 0;
    this.controlEnd = 0.8;
  }
}

function defaultName(type, id) {
  if (type === LAYER_REGION) return `Region ${id}`;
  if (type === LAYER_CONTROL) return `Sketch guide ${id}`;
  return `Layer ${id}`;
}

export class History {
  constructor(limit = 60) {
    this.undoStack = [];
    this.redoStack = [];
    this.limit = limit;
    this.onChange = () => {};
  }
  push(entry) {
    this.undoStack.push(entry);
    if (this.undoStack.length > this.limit) this.undoStack.shift();
    this.redoStack.length = 0;
    this.onChange();
  }
  undo() {
    const e = this.undoStack.pop();
    if (!e) return;
    e.undo();
    this.redoStack.push(e);
    this.onChange();
  }
  redo() {
    const e = this.redoStack.pop();
    if (!e) return;
    e.redo();
    this.undoStack.push(e);
    this.onChange();
  }
  clear() {
    this.undoStack.length = 0;
    this.redoStack.length = 0;
    this.onChange();
  }
}

export class Doc {
  constructor(width = 1024, height = 1024) {
    this.listeners = new Set();
    this.history = new History();
    this.reset(width, height);
  }

  reset(width, height) {
    this.width = width;
    this.height = height;
    this.layers = [];
    this.selection = makeCanvas(width, height); // alpha = selected
    this.hasSelection = false;
    const bg = new Layer(this, { name: "Background" });
    const c = ctx2d(bg.canvas);
    c.fillStyle = "#ffffff";
    c.fillRect(0, 0, width, height);
    this.layers.push(bg);
    const sketch = new Layer(this, { name: "Sketch" });
    this.layers.push(sketch);
    this.activeId = sketch.id;
    this.history.clear();
    this.emit("reset");
  }

  on(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
  emit(what) {
    for (const fn of this.listeners) fn(what);
  }

  get active() {
    return this.layers.find((l) => l.id === this.activeId) || this.layers[this.layers.length - 1];
  }
  layerById(id) {
    return this.layers.find((l) => l.id === id);
  }

  // ---- layer structure (undoable) ----
  addLayer(type = LAYER_PAINT, { name = null, canvas = null, above = null } = {}) {
    const layer = new Layer(this, { type, name, canvas });
    const ref = above ? this.layers.indexOf(above) : this.layers.indexOf(this.active);
    const idx = ref < 0 ? this.layers.length : ref + 1;
    this._structural(() => {
      this.layers.splice(idx, 0, layer);
      this.activeId = layer.id;
    });
    return layer;
  }

  removeLayer(layer) {
    if (this.layers.length <= 1) return;
    this._structural(() => {
      const idx = this.layers.indexOf(layer);
      this.layers.splice(idx, 1);
      if (this.activeId === layer.id) this.activeId = this.layers[Math.max(0, idx - 1)].id;
    });
  }

  moveLayer(layer, delta) {
    const idx = this.layers.indexOf(layer);
    const to = idx + delta;
    if (to < 0 || to >= this.layers.length) return;
    this._structural(() => {
      this.layers.splice(idx, 1);
      this.layers.splice(to, 0, layer);
    });
  }

  duplicateLayer(layer) {
    const c = makeCanvas(this.width, this.height);
    ctx2d(c).drawImage(layer.canvas, 0, 0);
    const copy = this.addLayer(layer.type, { name: `${layer.name} copy`, canvas: c, above: layer });
    Object.assign(copy, { prompt: layer.prompt, opacity: layer.opacity, controlModel: layer.controlModel, controlWeight: layer.controlWeight, controlStart: layer.controlStart, controlEnd: layer.controlEnd, regionStrength: layer.regionStrength });
    return copy;
  }

  mergeDown(layer) {
    const idx = this.layers.indexOf(layer);
    if (idx <= 0) return;
    const below = this.layers[idx - 1];
    if (below.type !== layer.type) return;
    const before = snapshot(below.canvas);
    const layersBefore = [...this.layers];
    const activeBefore = this.activeId;
    const c = ctx2d(below.canvas);
    c.globalAlpha = layer.opacity;
    c.drawImage(layer.canvas, 0, 0);
    c.globalAlpha = 1;
    const after = snapshot(below.canvas);
    this.layers.splice(idx, 1);
    this.activeId = below.id;
    const layersAfter = [...this.layers];
    this.history.push({
      undo: () => { restore(below.canvas, before); this.layers = [...layersBefore]; this.activeId = activeBefore; this.emit("layers"); },
      redo: () => { restore(below.canvas, after); this.layers = [...layersAfter]; this.activeId = below.id; this.emit("layers"); },
    });
    this.emit("layers");
  }

  _structural(fn) {
    const before = [...this.layers];
    const activeBefore = this.activeId;
    fn();
    const after = [...this.layers];
    const activeAfter = this.activeId;
    this.history.push({
      undo: () => { this.layers = [...before]; this.activeId = activeBefore; this.emit("layers"); },
      redo: () => { this.layers = [...after]; this.activeId = activeAfter; this.emit("layers"); },
    });
    this.emit("layers");
  }

  setLayerProp(layer, key, value) {
    layer[key] = value;
    this.emit(key === "prompt" ? "region-prompt" : "layer-props");
  }

  // ---- pixel edits (undoable) ----
  // Wrap a pixel edit to `target` (a layer canvas or the selection canvas).
  editPixels(target, fn, what = "pixels") {
    const before = snapshot(target);
    fn(ctx2d(target));
    const after = snapshot(target);
    const isSel = target === this.selection;
    this.history.push({
      undo: () => { restore(target, before); if (isSel) this.refreshSelectionFlag(); this.emit(what); },
      redo: () => { restore(target, after); if (isSel) this.refreshSelectionFlag(); this.emit(what); },
    });
    if (isSel) this.refreshSelectionFlag();
    this.emit(what);
  }

  // ---- selection ----
  refreshSelectionFlag() {
    this.hasSelection = !!alphaBounds(this.selection);
  }
  selectAll() {
    this.editPixels(this.selection, (c) => { c.fillStyle = "#fff"; c.fillRect(0, 0, this.width, this.height); }, "selection");
  }
  deselect() {
    if (!this.hasSelection) return;
    this.editPixels(this.selection, (c) => c.clearRect(0, 0, this.width, this.height), "selection");
  }
  invertSelection() {
    this.editPixels(this.selection, (c) => {
      const tmp = makeCanvas(this.width, this.height);
      const t = ctx2d(tmp);
      t.fillStyle = "#fff";
      t.fillRect(0, 0, this.width, this.height);
      t.globalCompositeOperation = "destination-out";
      t.drawImage(this.selection, 0, 0);
      c.clearRect(0, 0, this.width, this.height);
      c.drawImage(tmp, 0, 0);
    }, "selection");
  }
  selectionBounds() {
    return this.hasSelection ? alphaBounds(this.selection) : null;
  }

  // ---- resize / new ----
  resizeCanvas(width, height) {
    const olds = this.layers.map((l) => l.canvas);
    const oldSel = this.selection;
    const ow = this.width, oh = this.height;
    const apply = (w, h, canvases, sel) => {
      this.width = w; this.height = h;
      this.layers.forEach((l, i) => (l.canvas = canvases[i]));
      this.selection = sel;
      this.refreshSelectionFlag();
      this.emit("reset");
    };
    const news = this.layers.map((l, i) => {
      const c = makeCanvas(width, height);
      const x = ctx2d(c);
      if (i === 0 && l.type === LAYER_PAINT) { x.fillStyle = "#fff"; x.fillRect(0, 0, width, height); }
      x.drawImage(olds[i], 0, 0);
      return c;
    });
    const newSel = makeCanvas(width, height);
    ctx2d(newSel).drawImage(oldSel, 0, 0);
    apply(width, height, news, newSel);
    this.history.push({ undo: () => apply(ow, oh, olds, oldSel), redo: () => apply(width, height, news, newSel) });
  }

  // ---- compositing ----
  // Composite visible paint layers inside `rect` into a canvas of size outW x outH.
  composite(rect = null, outW = null, outH = null, { background = "#ffffff" } = {}) {
    rect = rect || { x: 0, y: 0, w: this.width, h: this.height };
    outW = outW || rect.w;
    outH = outH || rect.h;
    const out = makeCanvas(outW, outH);
    const c = ctx2d(out);
    if (background) { c.fillStyle = background; c.fillRect(0, 0, outW, outH); }
    c.imageSmoothingQuality = "high";
    for (const l of this.layers) {
      if (!l.visible || l.type !== LAYER_PAINT) continue;
      c.globalAlpha = l.opacity;
      c.drawImage(l.canvas, rect.x, rect.y, rect.w, rect.h, 0, 0, outW, outH);
    }
    c.globalAlpha = 1;
    return out;
  }

  // A single canvas cropped and scaled (used for masks, regions and control images).
  crop(source, rect, outW, outH, { background = null, smoothing = true } = {}) {
    const out = makeCanvas(outW, outH);
    const c = ctx2d(out);
    if (background) { c.fillStyle = background; c.fillRect(0, 0, outW, outH); }
    c.imageSmoothingEnabled = smoothing;
    c.imageSmoothingQuality = "high";
    c.drawImage(source, rect.x, rect.y, rect.w, rect.h, 0, 0, outW, outH);
    return out;
  }

  contentBounds() {
    let b = null;
    for (const l of this.layers) {
      if (!l.visible || l.type !== LAYER_PAINT) continue;
      b = unionRect(b, alphaBounds(l.canvas));
    }
    return b;
  }
}

// ---------- helpers ----------

export function snapshot(canvas) {
  return ctx2d(canvas).getImageData(0, 0, canvas.width, canvas.height);
}

export function restore(canvas, data) {
  if (canvas.width !== data.width || canvas.height !== data.height) {
    canvas.width = data.width;
    canvas.height = data.height;
  }
  ctx2d(canvas).putImageData(data, 0, 0);
}

// Bounding box of pixels with alpha > threshold (null if empty). Downsampled scan for speed.
export function alphaBounds(canvas, threshold = 8) {
  const w = canvas.width, h = canvas.height;
  const data = ctx2d(canvas).getImageData(0, 0, w, h).data;
  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++) {
    const row = y * w * 4;
    for (let x = 0; x < w; x++) {
      if (data[row + x * 4 + 3] > threshold) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

export function unionRect(a, b) {
  if (!a) return b;
  if (!b) return a;
  const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
  return { x, y, w: Math.max(a.x + a.w, b.x + b.w) - x, h: Math.max(a.y + a.h, b.y + b.h) - y };
}

export function hasContentIn(canvas, rect, threshold = 8) {
  const d = ctx2d(canvas).getImageData(rect.x, rect.y, rect.w, rect.h).data;
  for (let i = 3; i < d.length; i += 4) if (d[i] > threshold) return true;
  return false;
}

// Convert an alpha mask canvas into a white-on-black opaque image (for inpaint masks).
export function alphaToMask(canvas, { blur = 0 } = {}) {
  const out = makeCanvas(canvas.width, canvas.height);
  const c = ctx2d(out);
  c.fillStyle = "#000";
  c.fillRect(0, 0, out.width, out.height);
  const tmp = makeCanvas(canvas.width, canvas.height);
  const t = ctx2d(tmp);
  t.drawImage(canvas, 0, 0);
  t.globalCompositeOperation = "source-in";
  t.fillStyle = "#fff";
  t.fillRect(0, 0, tmp.width, tmp.height);
  if (blur > 0) c.filter = `blur(${blur}px)`;
  c.drawImage(tmp, 0, 0);
  c.filter = "none";
  return out;
}

export function canvasToDataURL(canvas, type = "image/png") {
  return canvas.toDataURL(type);
}

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
