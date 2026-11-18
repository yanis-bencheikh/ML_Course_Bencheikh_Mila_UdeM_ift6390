// The mathematics behind the demos, kept free of any drawing code so it can be tested on its own (see scripts/test_ml.js).
(function (root) {
  // ---- Lab 4: four surrogate losses, written as functions of the margin m = y * z
  const LOSSES = {
    linear: { name: 'Linear regression', formula: '½ (m − 1)²', colour: '--violet', loss: (m) => 0.5 * (m - 1) * (m - 1), slope: (m) => m - 1 },
    perceptron: { name: 'Perceptron', formula: 'max(0, −m)', colour: '--orange', loss: (m) => Math.max(0, -m), slope: (m) => (m < 0 ? -1 : 0) },
    svm: { name: 'SVM (hinge)', formula: 'max(0, 1 − m)', colour: '--magenta', loss: (m) => Math.max(0, 1 - m), slope: (m) => (m < 1 ? -1 : 0) },
    logistic: { name: 'Logistic regression', formula: 'log(1 + e^(−m))', colour: '--teal', loss: (m) => Math.log1p(Math.exp(-m)), slope: (m) => -1 / (1 + Math.exp(m)) },
  };
  const zeroOne = (m) => (m < 0 ? 1 : 0);

  // Objective L(w) = mean of loss(m_i) + lambda/2 ||w||^2, for points x (rows of length d) and labels y in {-1, +1}.
  function objective(key, w, X, y, lambda) {
    let s = 0;
    for (let i = 0; i < X.length; i++) { let z = 0; for (let j = 0; j < w.length; j++) z += w[j] * X[i][j]; s += LOSSES[key].loss(y[i] * z); }
    let n2 = 0; for (const v of w) n2 += v * v;
    return s / X.length + 0.5 * lambda * n2;
  }

  // Its gradient: mean of slope(m_i) * y_i * x_i, plus lambda * w. The same line for the four losses.
  function gradient(key, w, X, y, lambda) {
    const g = w.map((v) => lambda * v);
    for (let i = 0; i < X.length; i++) {
      let z = 0; for (let j = 0; j < w.length; j++) z += w[j] * X[i][j];
      const c = LOSSES[key].slope(y[i] * z) * y[i] / X.length;
      for (let j = 0; j < w.length; j++) g[j] += c * X[i][j];
    }
    return g;
  }

  function descend(key, w0, X, y, lambda, eta, steps) {
    const path = [w0.slice()]; let w = w0.slice();
    for (let t = 0; t < steps; t++) {
      const g = gradient(key, w, X, y, lambda);
      w = w.map((v, j) => v - eta * g[j]);
      path.push(w);
    }
    return path;
  }

  function errors(w, X, y) {
    let e = 0;
    for (let i = 0; i < X.length; i++) { let z = 0; for (let j = 0; j < w.length; j++) z += w[j] * X[i][j]; if (y[i] * z < 0) e++; }
    return e;
  }

  // ---- Lab 8: the smallest network of the lab. 1 input, 2 hidden ReLU neurons, 1 output, squared error.
  function tinyNet(p) {
    const z = [p.w[0] * p.x, p.w[1] * p.x];
    const h = z.map((v) => Math.max(0, v));
    const out = p.v[0] * h[0] + p.v[1] * h[1];
    const L = (out - p.y) * (out - p.y);
    const dOut = 2 * (out - p.y);                    // dL/d(out)
    const dV = h.map((v) => dOut * v);               // dL/dv_i = dL/d(out) * h_i
    const dH = p.v.map((v) => dOut * v);             // dL/dh_i = dL/d(out) * v_i
    const dZ = dH.map((v, i) => (z[i] > 0 ? v : 0)); // through ReLU: local slope 1 if z > 0, else 0
    const dW = dZ.map((v) => v * p.x);               // dL/dw_i = dL/dz_i * x
    return { z, h, out, L, dOut, dV, dH, dZ, dW };
  }

  // ---- Lab 8: a 2-H-1 network trained by Adam on the mean squared error (XOR demo).
  function rng(seed) { // mulberry32, so that every visitor sees the same run
    return function () { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function normal(r) { return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }

  function xorData(n, seed) {
    const r = rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) {
      const a = r() < 0.5 ? 0 : 1, b = r() < 0.5 ? 0 : 1;
      X.push([a + 0.13 * normal(r), b + 0.13 * normal(r)]); y.push(a ^ b);
    }
    return { X, y };
  }

  function Mlp(hidden, relu, seed) {
    const r = rng(seed);
    this.H = hidden; this.relu = relu; this.t = 0;
    // parameters, flattened: W (2 x H), b (H), v (H), c (1)
    this.p = new Float64Array(4 * hidden + 1);
    for (let k = 0; k < 2 * hidden; k++) this.p[k] = 0.7 * normal(r);
    for (let k = 3 * hidden; k < 4 * hidden; k++) this.p[k] = 0.7 * normal(r);
    this.m = new Float64Array(this.p.length); this.s = new Float64Array(this.p.length);
  }
  Mlp.prototype.forward = function (x0, x1) {
    const H = this.H, p = this.p; let out = p[4 * H];
    for (let j = 0; j < H; j++) {
      let z = p[j] * x0 + p[H + j] * x1 + p[2 * H + j];
      if (this.relu && z < 0) z = 0;
      out += p[3 * H + j] * z;
    }
    return out;
  };
  Mlp.prototype.evaluate = function (X, y) {
    let loss = 0, ok = 0;
    for (let i = 0; i < X.length; i++) { const o = this.forward(X[i][0], X[i][1]); loss += (o - y[i]) * (o - y[i]); if ((o > 0.5) === (y[i] > 0.5)) ok++; }
    return { loss: loss / X.length, accuracy: ok / X.length };
  };
  Mlp.prototype.step = function (X, y, lr) { // one epoch: forward, backward by hand, Adam update
    const H = this.H, p = this.p, g = new Float64Array(p.length), n = X.length;
    for (let i = 0; i < n; i++) {
      const x0 = X[i][0], x1 = X[i][1], z = new Float64Array(H); let out = p[4 * H];
      for (let j = 0; j < H; j++) { z[j] = p[j] * x0 + p[H + j] * x1 + p[2 * H + j]; out += p[3 * H + j] * (this.relu ? Math.max(0, z[j]) : z[j]); }
      const dOut = 2 * (out - y[i]) / n;
      g[4 * H] += dOut;
      for (let j = 0; j < H; j++) {
        const h = this.relu ? Math.max(0, z[j]) : z[j];
        g[3 * H + j] += dOut * h;
        const dZ = this.relu && z[j] <= 0 ? 0 : dOut * p[3 * H + j];
        g[j] += dZ * x0; g[H + j] += dZ * x1; g[2 * H + j] += dZ;
      }
    }
    this.t++;
    for (let k = 0; k < p.length; k++) {
      this.m[k] = 0.9 * this.m[k] + 0.1 * g[k]; this.s[k] = 0.999 * this.s[k] + 0.001 * g[k] * g[k];
      p[k] -= lr * (this.m[k] / (1 - Math.pow(0.9, this.t))) / (Math.sqrt(this.s[k] / (1 - Math.pow(0.999, this.t))) + 1e-8);
    }
  };

  const api = { LOSSES, zeroOne, objective, gradient, descend, errors, tinyNet, xorData, Mlp };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.ML = api;
})(typeof window !== 'undefined' ? window : globalThis);
