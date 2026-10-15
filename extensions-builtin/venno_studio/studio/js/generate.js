// Turns the canvas state into Forge API requests and manages results / live mode.

import { LAYER_CONTROL, LAYER_PAINT, LAYER_REGION, alphaToMask, ctx2d, hasContentIn, loadImage, makeCanvas } from "./doc.js";

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

  buildPayload(plan) {
    const doc = this.doc;
    const st = this.state;
    const g = st.gen;
    const { rect, gw, gh, sel, txt2img } = plan;
    const scale = gw / rect.w;

    let prompt = g.prompt.trim();
    if (g.qualityPrefix && st.info?.prompt_defaults?.positive_prefix) {
      const prefix = st.info.prompt_defaults.positive_prefix;
      if (!prompt.startsWith(prefix.trim().replace(/,$/, ""))) prompt = prefix + prompt;
    }

    const seed = plan.live ? this.live.seed : g.seedLocked ? g.seed : randomSeed();
    if (!plan.live) g.seed = seed;

    const modules = [st.model.textEncoder, st.model.vae].filter(Boolean);
    const payload = {
      prompt,
      negative_prompt: g.negative,
      steps: g.steps,
      cfg_scale: g.cfg,
      distilled_cfg_scale: g.shift,
      sampler_name: g.sampler,
      scheduler: g.scheduler,
      seed,
      width: gw,
      height: gh,
      batch_size: plan.live ? 1 : g.batch,
      n_iter: 1,
      send_images: true,
      save_images: !plan.live && g.saveImages,
      do_not_save_grid: true,
      override_settings: {},
      override_settings_restore_afterwards: false,
      alwayson_scripts: {},
    };
    if (st.model.checkpoint) payload.override_settings.sd_model_checkpoint = st.model.checkpoint;
    if (modules.length) payload.override_settings.forge_additional_modules = modules;

    if (!txt2img) {
      payload.init_images = [doc.composite(rect, gw, gh).toDataURL("image/png")];
      payload.denoising_strength = Math.min(1, Math.max(0.01, plan.strength));
      payload.resize_mode = 0;
      if (sel) {
        const m = alphaToMask(doc.crop(doc.selection, rect, gw, gh));
        payload.mask = m.toDataURL("image/png");
        payload.mask_blur = Math.round(g.feather * scale);
        payload.inpainting_fill = 1; // original
        payload.inpaint_full_res = false;
        payload.inpainting_mask_invert = 0;
        if (g.softInpaint) {
          // Forge "soft inpainting": per-step latent blending for seamless edges
          payload.alwayson_scripts["soft inpainting"] = { args: [true, 1, 0.5, 4, 0, 0.5, 2] };
        }
      }
    }

    // regional prompts
    const regions = [];
    for (const l of doc.layers) {
      if (l.type !== LAYER_REGION || !l.visible || !l.prompt.trim()) continue;
      if (!hasContentIn(l.canvas, rect)) continue;
      const mask = doc.crop(l.canvas, rect, gw, gh);
      let rp = l.prompt.trim();
      if (g.regionAppendRoot && g.prompt.trim()) rp = `${rp}, ${g.prompt.trim()}`;
      regions.push({ prompt: rp, mask: mask.toDataURL("image/png"), strength: l.regionStrength });
    }
    if (regions.length) {
      payload.alwayson_scripts["venno regional prompting"] = {
        args: [true, JSON.stringify({ base_strength: g.regionBase, end_at: g.regionEnd, regions })],
      };
    }

    // sketch guides (ControlNet / ControlLLLite units)
    const units = [];
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
      units.push({
        enabled: true,
        module: "None",
        model: l.controlModel,
        weight: l.controlWeight,
        image: img.toDataURL("image/png"),
        resize_mode: "Just Resize",
        guidance_start: l.controlStart,
        guidance_end: l.controlEnd,
        pixel_perfect: false,
        control_mode: "Balanced",
        save_detected_map: false,
      });
    }
    if (units.length) payload.alwayson_scripts.controlnet = { args: units };

    return payload;
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
    let payload;
    try {
      payload = this.buildPayload(plan);
    } catch (e) {
      this.emit("error", `Could not prepare request: ${e.message}`);
      return;
    }
    const mask = this.blendMask(plan.rect);
    const endpoint = plan.txt2img ? "/sdapi/v1/txt2img" : "/sdapi/v1/img2img";

    this.busy = true;
    this.view.setGenRect(plan.rect);
    this.emit("busy", { live, plan });
    const poll = this.pollProgress(plan);
    try {
      const res = await api(endpoint, payload);
      const info = safeJSON(res.info);
      const count = payload.batch_size;
      const images = [];
      for (const b64 of (res.images || []).slice(0, count)) {
        images.push(await loadImage(b64.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`));
      }
      if (!images.length) throw new Error("No image returned (interrupted?)");
      const seeds = info?.all_seeds || [payload.seed];
      const result = { id: this._resultId++, images, rect: plan.rect, mask, seeds, info, prompt: payload.prompt, txt2img: plan.txt2img };
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
      this.emit("error", e.message);
      if (!this.selected) this.view.setOverlay(null);
    } finally {
      this.busy = false;
      poll.stop();
      this.progress = null;
      this.view.setGenRect(null);
      this.emit("idle");
      if (this.live.enabled && this.live.pending) {
        this.live.pending = false;
        this.scheduleLive(50);
      }
    }
  }

  pollProgress(plan) {
    let stopped = false;
    const tick = async () => {
      if (stopped) return;
      try {
        const p = await api("/sdapi/v1/progress?skip_current_image=false");
        if (stopped) return;
        this.progress = p;
        this.emit("progress", p);
        if (p.current_image && this.state.showProgressPreview) {
          const img = await loadImage(`data:image/jpeg;base64,${p.current_image}`).catch(() => null);
          if (img && !stopped) this.view.setOverlay(img, plan.rect);
        }
      } catch {}
      if (!stopped) setTimeout(tick, 600);
    };
    setTimeout(tick, 400);
    return { stop: () => (stopped = true) };
  }

  async interrupt() {
    this.live.pending = false;
    try { await api("/sdapi/v1/interrupt", {}); } catch {}
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

function safeJSON(s) {
  try { return typeof s === "string" ? JSON.parse(s) : s; } catch { return null; }
}
