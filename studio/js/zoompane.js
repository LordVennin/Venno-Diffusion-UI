// Zoom and pan for the input image of the Image to Image / Inpaint tabs (Venno and Venno MOSS).
// The wheel zooms at the pointer (Shift + wheel is handed to `onBrush`, for the brush size); when zoomed in, the
// pane scrolls: drag with the middle button, or hold Space and drag, to move around.

export class ZoomPane {
  /**
   * @param {HTMLElement} drop   the pane around the image (becomes the scroll container while zoomed)
   * @param {HTMLElement} stage  the element holding the image and its canvases; sized here
   * @param {{ max?: number, onChange?: (zoom: number) => void, onBrush?: (direction: 1 | -1) => void }} [opts]
   */
  constructor(drop, stage, { max = 8, onChange = () => {}, onBrush = null } = {}) {
    Object.assign(this, { drop, stage, max, onChange, zoom: 1, base: null });
    drop.addEventListener("wheel", (e) => {
      if (stage.hidden || !this.base) return;
      e.preventDefault();
      const d = e.deltaY || e.deltaX; // Shift turns the wheel sideways in some browsers
      if (!d) return;
      if (e.shiftKey && onBrush) return onBrush(d < 0 ? 1 : -1);
      this.zoomAt(e.clientX, e.clientY, d < 0 ? 1.2 : 1 / 1.2);
    }, { passive: false });

    let pan = null, space = false;
    const typing = () => /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
    addEventListener("keydown", (e) => {
      if (e.code !== "Space" || !drop.matches(":hover") || typing()) return;
      e.preventDefault(); // no page scroll, no click on a focused button
      space = true;
      drop.classList.add("pan-ready");
    });
    addEventListener("keyup", (e) => {
      if (e.code !== "Space" || !space) return;
      e.preventDefault();
      space = false;
      drop.classList.remove("pan-ready");
    });
    drop.addEventListener("mousedown", (e) => { if (e.button === 1) e.preventDefault(); }); // no browser auto-scroll
    drop.addEventListener("pointerdown", (e) => {
      if (!(e.button === 1 || (e.button === 0 && space)) || this.zoom <= 1) return;
      e.preventDefault();
      e.stopPropagation(); // before the stage starts a brush stroke
      pan = { x: e.clientX, y: e.clientY, left: drop.scrollLeft, top: drop.scrollTop };
      drop.setPointerCapture(e.pointerId);
      drop.classList.add("panning");
    }, true);
    drop.addEventListener("pointermove", (e) => {
      if (!pan) return;
      drop.scrollLeft = pan.left - (e.clientX - pan.x);
      drop.scrollTop = pan.top - (e.clientY - pan.y);
    });
    const end = () => { pan = null; drop.classList.remove("panning"); };
    drop.addEventListener("pointerup", end);
    drop.addEventListener("pointercancel", end);
  }

  /** Size the stage: `width` × `height` is its size at zoom 1 (fitted to the pane). */
  apply(width, height) {
    this.base = [width, height];
    this.stage.style.width = `${Math.round(width * this.zoom)}px`;
    this.stage.style.height = `${Math.round(height * this.zoom)}px`;
    this.drop.classList.toggle("zoomed", this.zoom > 1);
  }

  /** Zoom by `factor`, keeping the image point under (clientX, clientY) where it is. */
  zoomAt(clientX, clientY, factor) {
    let z = Math.min(this.max, Math.max(1, this.zoom * factor));
    if (z < 1.03) z = 1;
    if (z === this.zoom || !this.base) return;
    const before = this.stage.getBoundingClientRect();
    const fx = (clientX - before.left) / before.width, fy = (clientY - before.top) / before.height;
    this.zoom = z;
    this.apply(...this.base);
    const after = this.stage.getBoundingClientRect();
    this.drop.scrollLeft += after.left + fx * after.width - clientX;
    this.drop.scrollTop += after.top + fy * after.height - clientY;
    this.onChange(z);
  }

  reset() {
    if (this.zoom === 1) return;
    this.zoom = 1;
    if (this.base) this.apply(...this.base);
    this.onChange(1);
  }
}
