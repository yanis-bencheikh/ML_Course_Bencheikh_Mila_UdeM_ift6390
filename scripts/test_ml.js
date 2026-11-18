// Checks the mathematics of the demos against the numbers worked out by hand in the labs.
// Usage: node scripts/test_ml.js
const assert = require('assert');
const path = require('path');
const ML = require(path.join(__dirname, '..', 'docs', 'assets', 'ml.js'));
global.window = {};
require(path.join(__dirname, '..', 'docs', 'assets', 'iris.js'));
const IRIS = global.window.IRIS;
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= (tol || 1e-9), `${msg || ''} expected ${b}, got ${a}`);

// Lab 8, section 1: x = 2, target 3, w = (1, 0.5), v = (2, 1).
const net = ML.tinyNet({ x: 2, y: 3, w: [1, 0.5], v: [2, 1] });
assert.deepStrictEqual(net.z, [2, 1]);
close(net.out, 5); close(net.L, 4); close(net.dOut, 4);
assert.deepStrictEqual(net.dV, [8, 4]);
assert.deepStrictEqual(net.dH, [8, 4]);
assert.deepStrictEqual(net.dW, [16, 8]);
// one step of gradient descent with a step of 0.01: output 4.03, loss 1.07
const after = ML.tinyNet({ x: 2, y: 3, w: [1 - 0.16, 0.5 - 0.08], v: [2 - 0.08, 1 - 0.04] });
close(after.out, 4.03, 0.005); close(after.L, 1.07, 0.005);
// a hidden neuron with a negative pre-activation passes no gradient
assert.deepStrictEqual(ML.tinyNet({ x: 2, y: 3, w: [-0.5, 0.5], v: [2, 1] }).dW[0], 0);
// finite-difference check of dL/dw1
const eps = 1e-6;
close((ML.tinyNet({ x: 2, y: 3, w: [1 + eps, 0.5], v: [2, 1] }).L - net.L) / eps, 16, 1e-3);

// Lab 4: flowers A, B, C with w = (1, 1, -0.4): scores 2.40, -2.60, -0.70.
const w = [1, 1, -0.4];
const flowers = [[1.6, 1.2, 1, 1], [-1.3, -0.9, 1, -1], [-1.6, 1.3, 1, 1]];
const scores = flowers.map((f) => f[0] * w[0] + f[1] * w[1] + f[2] * w[2]);
[2.4, -2.6, -0.7].forEach((s, i) => close(scores[i], s, 1e-9));
const mC = flowers[2][3] * scores[2]; // margin of C: -0.70
close(ML.LOSSES.linear.loss(mC), 1.44, 0.01); close(ML.LOSSES.perceptron.loss(mC), 0.70, 0.01);
close(ML.LOSSES.svm.loss(mC), 1.70, 0.01); close(ML.LOSSES.logistic.loss(mC), 1.10, 0.01);
// every analytic slope agrees with a finite difference (away from the kinks)
for (const key of Object.keys(ML.LOSSES)) for (const m of [-2.3, -0.4, 0.6, 1.7]) {
  close((ML.LOSSES[key].loss(m + eps) - ML.LOSSES[key].loss(m)) / eps, ML.LOSSES[key].slope(m), 1e-4, key);
}
// gradient of the full objective agrees with finite differences on the iris data
const X = IRIS.x.map((r) => [r[0], r[1], 1]);
for (const key of Object.keys(ML.LOSSES)) {
  const w0 = [-0.3, 0.8, 0.1], g = ML.gradient(key, w0, X, IRIS.y, 0.1);
  for (let j = 0; j < 3; j++) {
    const wp = w0.slice(); wp[j] += eps;
    close((ML.objective(key, wp, X, IRIS.y, 0.1) - ML.objective(key, w0, X, IRIS.y, 0.1)) / eps, g[j], 1e-4, key);
  }
}
// with the settings of the lab, each method separates setosa from versicolor after 100 iterations
const settings = { linear: [0.1, 0.2], perceptron: [0, 1], svm: [0.1, 1], logistic: [0.1, 0.2] };
for (const key of Object.keys(settings)) {
  const p = ML.descend(key, [-3, 3, 0.1], X, IRIS.y, settings[key][0], settings[key][1], 100);
  assert.strictEqual(ML.errors(p[100], X, IRIS.y), 0, key);
}

// XOR: without an activation the network cannot do better than a line; with ReLU it separates the four clusters.
const data = ML.xorData(240, 7);
const results = {};
for (const relu of [false, true]) {
  const mlp = new ML.Mlp(8, relu, 3);
  for (let t = 0; t < 600; t++) mlp.step(data.X, data.y, 0.05);
  results[relu] = mlp.evaluate(data.X, data.y);
}
assert.ok(results[false].accuracy < 0.8, 'linear network should fail on XOR');
assert.ok(results[true].accuracy > 0.97, 'ReLU network should solve XOR');
console.log('all checks pass', results);
