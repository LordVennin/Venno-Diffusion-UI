// Turns the canvas state into Venno jobs and manages results / live mode.

import { LAYER_CONTROL, LAYER_PAINT, LAYER_REGION, alphaToMask, ctx2d, hasContentIn, loadImage, makeCanvas, regionMasks, strokeRegionMask } from "./doc.js";
import { ADETAILER_DEFAULT, withQualityPrefix } from "./common.js";
import { cancelJob, jobModel, runJob } from "./jobs.js";
import { jobLoras } from "./loras.js";

// Choose a model resolution for a canvas area: ~targetPixels, multiples of 64, Anima-friendly bounds.
export function fitResolution(w, h, targetPixels) {
  const scale = Math.sqrt(targetPixels / (w * h));
  const snap = (v) => Math.min(2048, Math.max(256, Math.round((v * scale) / 64) * 64));
  return [snap(w), snap(h)];
}

function expandRect(r, pad, W, H) {
  const x = Math.max(0, Math.floor(r.x - pad));
  const y = Math.max(0, Math.floor(r.y - pad));
  const x2 = Math.min(W, Math.ceil(r.x + r.w + pad));
  const y2 = Math.min(H, Math.ceil(r.y + r.h + pad));
  return { x, y, w: x2 - x, h: y2 - y };
}

const randomSeed = () => Math.floor(Math.random() * 2 ** 31);

export class Generator {
  constructor(doc, view, state) {
    this.doc = doc;
    this.view = view;
    this.state = state; // shared app state: gen params, model selection, info
    this.busy = false;
    this.results = []; // {id, images: [canvas], rect, mask, seeds, info}
    this.selected = null; // {result, index}
    this.live = { enabled: false, pending: false, timer: 0, seed: randomSeed(), result: null };
    this.listeners = new Set();
    this.progress = null;
    this._resultId = 1;
  }

  on(fn) { this.listeners.add(fn); }
  emit(what, data) { for (const fn of this.listeners) fn(what, data); }

  // ---------- request building ----------
  plan({ live = false } = {}) {
    const doc = this.doc;
    const g = this.state.gen;
    const sel = doc.selectionBounds();
    const W = doc.width, H = doc.height;

    let rect;
    if (sel) {
      // surrounding context helps the model match style / lighting; scale with both the selection and the canvas
      const pad = Math.max(Math.max(sel.w, sel.h) * g.context, Math.max(W, H) * g.context * 0.25, 16);
      rect = expandRect(sel, pad, W, H);
    }
    else rect = { x: 0, y: 0, w: W, h: H };

    const [gw, gh] = fitResolution(rect.w, rect.h, g.resolution * g.resolution);
    const content = doc.contentBounds();
    const canvasHasContent = !!content && hasAnyNonWhite(doc, rect);
    const strength = g.strength;
    const txt2img = !sel && (strength >= 0.999 || !canvasHasContent);

    return { rect, gw, gh, sel, strength, txt2img, live };
  }

  buildJob(plan) {
    const doc = this.doc;
    const st = this.state;
    const g = st.gen;
    const { rect, gw, gh, sel, txt2img } = plan;
    const scale = gw / rect.w;

    const seed = plan.live ? this.live.seed : g.seedLocked ? g.seed : randomSeed();
    if (!plan.live) g.seed = seed;

    const job = {
      mode: txt2img ? "txt2img" : "img2img",
      model: jobModel(st.model),
      prompt: withQualityPrefix(g.prompt, g.qualityPrefix, st.info),
      negative: g.negative,
      loras: jobLoras(g.loras),
      switch_prompt: g.switchPrompt.trim() ? withQualityPrefix(g.switchPrompt, g.qualityPrefix, st.info) : "",
      switch_at: g.switchAt,
      width: gw,
      height: gh,
      batch: plan.live ? 1 : g.batch,
      count: 1,
      seed,
      sampler: g.sampler,
      scheduler: g.scheduler,
      steps: g.steps,
      cfg: g.cfg,
      shift: g.shift,
      save: !plan.live && g.saveImages,
      tag: plan.live ? "live" : "canvas",
    };

    if (!txt2img) {
      job.init_image = doc.composite(rect, gw, gh).toDataURL("image/png");
      job.strength = Math.min(1, Math.max(0.01, plan.strength));
      if (sel) {
        job.mask = alphaToMask(doc.crop(doc.selection, rect, gw, gh)).toDataURL("image/png");
        job.mask_blur = Math.round(g.feather * scale);
        job.fill = "original";
        job.only_masked = false;
        job.soft_inpaint = g.softInpaint;
      }
    }

    // regional prompts: region layers, plus paint / guide layers that carry a prompt (shaped by their strokes)
    job.regions = [];
    for (const { layer: l, mask: source } of regionMasks(doc, { mode: g.regionOverlap || "top" })) {
      if (!hasContentIn(source, rect)) continue;
      let rp = l.prompt.trim();
      if (g.regionAppendRoot && g.prompt.trim()) rp = `${rp}, ${g.prompt.trim()}`;
      let rn = (l.negative || "").trim();
      if (rn && g.regionAppendRoot && g.negative.trim()) rn = `${rn}, ${g.negative.trim()}`;
      job.regions.push({ prompt: rp, negative: rn, mask: doc.crop(source, rect, gw, gh).toDataURL("image/png"), strength: l.regionStrength });
    }
    job.region_base = g.regionBase;
    job.region_end = g.regionEnd;

    // sketch guides (ControlNet / ControlLLLite)
    job.guides = [];
    for (const l of doc.layers) {
      if (l.type !== LAYER_CONTROL || !l.visible || !l.controlModel) continue;
      if (!hasContentIn(l.canvas, rect)) continue;
      const img = doc.crop(l.canvas, rect, gw, gh, { background: "#ffffff" });
      // the user always draws dark lines on white; scribble-type models expect white lines on black
      const invert = l.controlInvert === "yes" || (l.controlInvert === "auto" && /scribble|hed|pidi/i.test(l.controlModel));
      if (invert) {
        const c = ctx2d(img);
        c.globalCompositeOperation = "difference";
        c.fillStyle = "#ffffff";
        c.fillRect(0, 0, gw, gh);
        c.globalCompositeOperation = "source-over";
      }
      job.guides.push({ model: l.controlModel, image: img.toDataURL("image/png"), weight: l.controlWeight, start: l.controlStart, end: l.controlEnd });
    }

    // detailer: re-render detected faces (skipped in live mode to keep it fast)
    const adModels = st.info?.adetailer_models || [];
    job.detailer = [];
    if (g.adetailer && !plan.live && adModels.length) {
      const face = adModels.includes("face_yolov8n.pt") ? "face_yolov8n.pt" : adModels.find((m) => /face/i.test(m));
      const u = ADETAILER_DEFAULT.units[0];
      if (face) job.detailer.push({ model: face, confidence: u.confidence, denoise: u.denoise, mask_blur: u.maskBlur, dilate: u.dilate, padding: u.padding });
    }

    return job;
  }

  // Feathered selection mask cropped to rect (alpha canvas) used to blend results back.
  blendMask(rect) {
    const doc = this.doc;
    if (!doc.hasSelection) return null;
    const feather = this.state.gen.feather;
    const m = makeCanvas(rect.w, rect.h);
    const c = ctx2d(m);
    if (feather > 0) c.filter = `blur(${feather / 2}px)`;
    c.drawImage(doc.selection, rect.x, rect.y, rect.w, rect.h, 0, 0, rect.w, rect.h);
    c.filter = "none";
    return m;
  }

  // ---------- running ----------
  async run({ live = false } = {}) {
    if (this.busy) {
      if (live) this.live.pending = true;
      return;
    }
    if (!this.state.model.checkpoint) {
      this.emit("error", "Select an Anima checkpoint first (top bar).");
      return;
    }
    const plan = this.plan({ live });
    let job;
    try {
      job = this.buildJob(plan);
    } catch (e) {
      this.emit("error", `Could not prepare request: ${e.message}`);
      return;
    }
    const mask = this.blendMask(plan.rect);

    this.busy = true;
    this.currentJob = null;
    this.view.setGenRect(plan.rect);
    this.emit("busy", { live, plan });
    try {
      const done = await runJob(job, {
        onSubmit: (j) => (this.currentJob = j.id),
        onUpdate: (j) => this.emit("progress", j),
        onPreview: async (src) => {
          if (!this.state.showProgressPreview) return;
          const img = await loadImage(src).catch(() => null);
          if (img && this.busy) this.view.setOverlay(img, plan.rect);
        },
      });
      if (done.status === "cancelled") throw new Error("Cancelled"); // ignore half-denoised images from an interrupted run
      const images = await Promise.all(done.images.map((url) => loadImage(url)));
      if (!images.length) throw new Error("No image returned");
      const result = { id: this._resultId++, images, rect: plan.rect, mask, seeds: done.seeds, infotexts: done.infotexts, prompt: job.prompt, txt2img: plan.txt2img };
      if (live) {
        this.live.result = result;
        this.emit("live-result", result);
        if (this.state.liveOnCanvas) this.preview(result, 0);
      } else {
        this.results.unshift(result);
        if (this.results.length > 40) this.results.pop();
        this.emit("results");
        this.preview(result, 0);
      }
    } catch (e) {
      if (e.message !== "Cancelled") this.emit("error", e.message);
      if (!this.selected) this.view.setOverlay(null);
      else this.preview(this.selected.result, this.selected.index);
    } finally {
      this.busy = false;
      this.currentJob = null;
      this.view.setGenRect(null);
      this.emit("idle");
      if (this.live.enabled && this.live.pending) {
        this.live.pending = false;
        this.scheduleLive(50);
      }
    }
  }

  async interrupt() {
    this.live.pending = false;
    if (this.currentJob) await cancelJob(this.currentJob);
  }

  // ---------- results ----------
  compose(result, index) {
    const { rect, mask } = result;
    const out = makeCanvas(rect.w, rect.h);
    const c = ctx2d(out);
    c.imageSmoothingQuality = "high";
    c.drawImage(result.images[index], 0, 0, rect.w, rect.h);
    if (mask) {
      c.globalCompositeOperation = "destination-in";
      c.drawImage(mask, 0, 0);
      c.globalCompositeOperation = "source-over";
    }
    return out;
  }

  preview(result, index) {
    this.selected = result ? { result, index } : null;
    if (!result) { this.view.setOverlay(null); this.emit("selected"); return; }
    this.view.setOverlay(this.compose(result, index), result.rect);
    this.emit("selected");
  }

  discardPreview() {
    this.selected = null;
    this.view.setOverlay(null);
    this.emit("selected");
  }

  apply(result = this.selected?.result, index = this.selected?.index ?? 0) {
    if (!result) return;
    const doc = this.doc;
    const layerCanvas = makeCanvas(doc.width, doc.height);
    ctx2d(layerCanvas).drawImage(this.compose(result, index), result.rect.x, result.rect.y);
    // place above the active paint layer (or the top-most paint layer)
    let above = doc.active;
    if (above.type !== LAYER_PAINT) above = [...doc.layers].reverse().find((l) => l.type === LAYER_PAINT) || above;
    const seed = result.seeds?.[index];
    doc.addLayer(LAYER_PAINT, { name: `Generated${seed !== undefined ? ` ${seed}` : ""}`, canvas: layerCanvas, above });
    this.discardPreview();
  }

  // ---------- live mode ----------
  setLive(enabled) {
    this.live.enabled = enabled;
    if (enabled) this.scheduleLive(10);
    else {
      this.live.pending = false;
      clearTimeout(this.live.timer);
      if (this.selected?.result === this.live.result) this.discardPreview();
    }
    this.emit("live");
  }

  scheduleLive(delay = 700) {
    if (!this.live.enabled) return;
    clearTimeout(this.live.timer);
    this.live.timer = setTimeout(() => this.run({ live: true }), delay);
  }

  newLiveSeed() {
    this.live.seed = randomSeed();
    this.scheduleLive(10);
  }
}

function hasAnyNonWhite(doc, rect) {
  // small thumbnail check: is there anything other than a blank white canvas?
  const t = doc.composite(rect, 64, 64);
  const d = ctx2d(t).getImageData(0, 0, 64, 64).data;
  for (let i = 0; i < d.length; i += 4) if (d[i] < 250 || d[i + 1] < 250 || d[i + 2] < 250) return true;
  return false;
}
