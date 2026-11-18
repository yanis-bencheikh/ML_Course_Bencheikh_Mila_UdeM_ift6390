// Small canvas plotting helper shared by every demo. No dependency.
(function () {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function Plot(canvas, opt) {
    this.c = canvas;
    this.o = Object.assign({ pad: { l: 48, r: 14, t: 14, b: 42 }, xticks: 6, yticks: 6 }, opt);
    this.resize();
  }

  Plot.prototype.resize = function () {
    const r = this.c.getBoundingClientRect();
    const d = window.devicePixelRatio || 1;
    this.w = r.width; this.h = r.height;
    this.c.width = Math.round(r.width * d); this.c.height = Math.round(r.height * d);
    this.g = this.c.getContext('2d');
    this.g.setTransform(d, 0, 0, d, 0, 0);
  };
  Plot.prototype.sx = function (x) { const p = this.o.pad; return p.l + (x - this.o.xmin) / (this.o.xmax - this.o.xmin) * (this.w - p.l - p.r); };
  Plot.prototype.sy = function (y) { const p = this.o.pad; return this.h - p.b - (y - this.o.ymin) / (this.o.ymax - this.o.ymin) * (this.h - p.t - p.b); };
  Plot.prototype.ix = function (px) { const p = this.o.pad; return this.o.xmin + (px - p.l) / (this.w - p.l - p.r) * (this.o.xmax - this.o.xmin); };
  Plot.prototype.iy = function (py) { const p = this.o.pad; return this.o.ymin + (this.h - p.b - py) / (this.h - p.t - p.b) * (this.o.ymax - this.o.ymin); };

  Plot.prototype.clear = function () {
    this.resize();
    this.g.clearRect(0, 0, this.w, this.h);
  };

  Plot.prototype.clip = function (fn) {
    const g = this.g, p = this.o.pad;
    g.save(); g.beginPath(); g.rect(p.l, p.t, this.w - p.l - p.r, this.h - p.t - p.b); g.clip();
    fn(g); g.restore();
  };

  Plot.prototype.axes = function () {
    const g = this.g, o = this.o, p = o.pad;
    g.font = '12px system-ui, sans-serif';
    g.lineWidth = 1;
    const ticks = (a, b, n) => {
      const raw = (b - a) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
      const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw);
      const out = [];
      for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(Math.abs(v) < 1e-9 ? 0 : v);
      return out;
    };
    const fmt = (v) => (Math.abs(v) >= 100 || Number.isInteger(v) ? String(Math.round(v * 100) / 100) : v.toFixed(Math.abs(v) < 1 ? 2 : 1).replace(/0$/, '').replace(/\.$/, ''));
    g.strokeStyle = css('--grid'); g.fillStyle = css('--ink-3');
    g.textAlign = 'center'; g.textBaseline = 'top';
    ticks(o.xmin, o.xmax, o.xticks).forEach((v) => {
      const x = this.sx(v);
      g.beginPath(); g.moveTo(x, p.t); g.lineTo(x, this.h - p.b); g.stroke();
      g.fillText(fmt(v), x, this.h - p.b + 6);
    });
    g.textAlign = 'right'; g.textBaseline = 'middle';
    ticks(o.ymin, o.ymax, o.yticks).forEach((v) => {
      const y = this.sy(v);
      g.beginPath(); g.moveTo(p.l, y); g.lineTo(this.w - p.r, y); g.stroke();
      g.fillText(fmt(v), p.l - 7, y);
    });
    g.fillStyle = css('--ink-2'); g.font = '13px system-ui, sans-serif';
    if (o.xlabel) { g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillText(o.xlabel, (p.l + this.w - p.r) / 2, this.h - 4); }
    if (o.ylabel) {
      g.save(); g.translate(13, (p.t + this.h - p.b) / 2); g.rotate(-Math.PI / 2);
      g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(o.ylabel, 0, 0); g.restore();
    }
  };

  Plot.prototype.line = function (pts, colour, width, dash) {
    this.clip((g) => {
      g.strokeStyle = colour; g.lineWidth = width || 2; g.setLineDash(dash || []); g.lineJoin = 'round';
      g.beginPath();
      pts.forEach((q, i) => { const x = this.sx(q[0]), y = this.sy(q[1]); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
      g.stroke(); g.setLineDash([]);
    });
  };

  Plot.prototype.dot = function (x, y, colour, radius, ring) {
    const g = this.g;
    g.beginPath(); g.arc(this.sx(x), this.sy(y), radius || 4, 0, 2 * Math.PI);
    g.fillStyle = colour; g.fill();
    if (ring) { g.lineWidth = 2; g.strokeStyle = ring; g.stroke(); }
  };

  Plot.prototype.label = function (text, x, y, colour, align) {
    const g = this.g;
    g.font = '600 13px system-ui, sans-serif'; g.textAlign = align || 'left'; g.textBaseline = 'middle';
    g.lineWidth = 4; g.strokeStyle = css('--panel'); g.strokeText(text, this.sx(x), this.sy(y));
    g.fillStyle = colour || css('--ink'); g.fillText(text, this.sx(x), this.sy(y));
  };

  // Fills the plot area from a function value(x, y) -> [r, g, b, a] sampled on an n x n grid.
  Plot.prototype.field = function (n, value) {
    const off = document.createElement('canvas'); off.width = n; off.height = n;
    const og = off.getContext('2d'), img = og.createImageData(n, n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = this.o.xmin + (i + 0.5) / n * (this.o.xmax - this.o.xmin);
      const y = this.o.ymax - (j + 0.5) / n * (this.o.ymax - this.o.ymin);
      const c = value(x, y), k = 4 * (j * n + i);
      img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = c[3];
    }
    og.putImageData(img, 0, 0);
    const p = this.o.pad;
    this.g.imageSmoothingEnabled = true;
    this.g.drawImage(off, p.l, p.t, this.w - p.l - p.r, this.h - p.t - p.b);
  };

  // Pointer helper: calls move(x, y, px, py) in data coordinates while the pointer is over the plot, leave() when it exits.
  Plot.prototype.pointer = function (move, leave, drag) {
    const c = this.c; let down = false;
    const at = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    c.addEventListener('pointerdown', (e) => { down = true; c.setPointerCapture(e.pointerId); const [px, py] = at(e); if (drag) drag(this.ix(px), this.iy(py)); });
    c.addEventListener('pointerup', () => { down = false; });
    c.addEventListener('pointermove', (e) => { const [px, py] = at(e); if (down && drag) drag(this.ix(px), this.iy(py)); if (move) move(this.ix(px), this.iy(py), px, py); });
    c.addEventListener('pointerleave', () => { if (leave) leave(); });
  };

  function hex(h) { h = h.replace('#', ''); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }

  function bindRange(id, onChange, format) {
    const el = document.getElementById(id), out = document.querySelector('output[for="' + id + '"]');
    const show = () => { if (out) out.textContent = format ? format(+el.value) : el.value; };
    el.addEventListener('input', () => { show(); onChange(); });
    show();
    return el;
  }

  function tip(el, html, px, py) {
    if (html === null) { el.style.display = 'none'; return; }
    el.innerHTML = html; el.style.display = 'block';
    const w = el.offsetWidth, parent = el.parentElement.clientWidth;
    el.style.left = Math.max(0, Math.min(parent - w, px + 12)) + 'px';
    el.style.top = Math.max(0, py - 38) + 'px';
  }

  window.Viz = { Plot, css, hex, bindRange, tip };
})();
