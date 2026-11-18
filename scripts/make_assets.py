"""Builds the data file used by the interactive demos and the animated previews shown in the README.

Usage: python scripts/make_assets.py
Outputs: docs/assets/iris.js, assets/gifs/linear_classifiers.gif, assets/gifs/xor_relu.gif
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import load_iris

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, GREEN, RED, INK, GREY = "#2563eb", "#16a34a", "#dc2626", "#111827", "#6b7280"
LOSS_COLOURS = {"Linear regression": "#9333ea", "Perceptron": "#ea580c", "SVM (hinge)": "#c026d3", "Logistic regression": "#0891b2"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------- data: setosa (-1) vs versicolor (+1), petals
def iris_two_classes():
    data = load_iris()
    keep = data.target < 2
    X = data.data[keep][:, [2, 3]]                     # petal length, petal width (cm)
    y = np.where(data.target[keep] == 0, -1.0, 1.0)    # setosa -> -1, versicolor -> +1
    X = (X - X.mean(axis=0)) / X.std(axis=0)           # standardised on these 100 flowers
    return X, y


# ---------------------------------------------------------------- the four losses: slope l'(m) as a function of the margin
SLOPES = {
    "Linear regression": lambda m: m - 1,
    "Perceptron": lambda m: -(m < 0).astype(float),
    "SVM (hinge)": lambda m: -(m < 1).astype(float),
    "Logistic regression": lambda m: -1 / (1 + np.exp(m)),
}
SETTINGS = {"Linear regression": (0.1, 0.2), "Perceptron": (0.0, 1.0), "SVM (hinge)": (0.1, 1.0), "Logistic regression": (0.1, 0.2)}  # (lambda, eta), as in the lab


def descend(X1, y, name, w0=(-3.0, 3.0, 0.1), steps=100):
    """Gradient descent on L(w) = mean l(m_i) + lambda/2 ||w||^2. Every loss shares the same gradient line."""
    lam, eta = SETTINGS[name]
    w = np.array(w0, dtype=float)
    path = [w.copy()]
    for _ in range(steps):
        m = y * (X1 @ w)
        grad = ((SLOPES[name](m) * y)[:, None] * X1).mean(axis=0) + lam * w
        w = w - eta * grad
        path.append(w.copy())
    return np.array(path)


def gif_linear_classifiers(X, y):
    X1 = np.column_stack([X, np.ones(len(X))])
    paths = {name: descend(X1, y, name) for name in SLOPES}
    frames = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 35, 40, 50, 60, 70, 80, 90, 100] + [100] * 8
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.9), dpi=80)
    g = np.linspace(-2.6, 2.6, 160)
    G0, G1 = np.meshgrid(g, g)

    def draw(k):
        t = frames[k]
        for ax, name in zip(axes, SLOPES):
            ax.clear()
            w = paths[name][t]
            z = w[0] * G0 + w[1] * G1 + w[2]
            ax.contourf(G0, G1, z, levels=[-1e9, 0, 1e9], colors=[GREEN, BLUE], alpha=0.10)
            ax.contour(G0, G1, z, levels=[0], colors=[RED], linewidths=2.2)
            ax.scatter(*X[y > 0].T, c=BLUE, s=16, alpha=0.8, linewidths=0)
            ax.scatter(*X[y < 0].T, c=GREEN, s=16, alpha=0.8, linewidths=0)
            errors = int(np.sum(y * (X1 @ w) < 0))
            ax.set_title(name, color=LOSS_COLOURS[name], fontweight="bold")
            ax.set_xlabel(f"iteration {t}   ·   {errors} misclassified")
            ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Same flowers, same starting point, four losses: gradient descent finds four different boundaries", y=0.99, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    FuncAnimation(fig, draw, frames=len(frames)).save(os.path.join(ROOT, "assets/gifs/linear_classifiers.gif"), writer=PillowWriter(fps=5))
    plt.close(fig)


# ---------------------------------------------------------------- XOR: one hidden layer of 8 neurons, with and without ReLU
def xor_data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    corners = rng.integers(0, 2, size=(n, 2))
    X = corners + rng.normal(0, 0.13, size=(n, 2))
    y = (corners[:, 0] ^ corners[:, 1]).astype(float)
    return X, y


def train_xor(X, y, relu, epochs=600, lr=0.05, hidden=8, seed=1, every=20):
    """Tiny 2-8-1 network trained by Adam on the mean squared error; returns snapshots of the parameters."""
    rng = np.random.default_rng(seed)
    P = [rng.normal(0, 0.7, (2, hidden)), np.zeros(hidden), rng.normal(0, 0.7, hidden), np.zeros(1)]
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    snaps = []
    for t in range(epochs + 1):
        W, b, v, c = P
        z = X @ W + b
        h = np.maximum(z, 0) if relu else z
        out = h @ v + c
        if t % every == 0:
            snaps.append((t, [p.copy() for p in P], float(np.mean((out - y) ** 2)), float(np.mean((out > 0.5) == (y > 0.5)))))
        d_out = 2 * (out - y) / len(y)
        d_h = np.outer(d_out, v)
        d_z = d_h * (z > 0) if relu else d_h
        grads = [X.T @ d_z, d_z.sum(axis=0), h.T @ d_out, np.array([d_out.sum()])]
        for i, gr in enumerate(grads):
            M[i] = 0.9 * M[i] + 0.1 * gr; V[i] = 0.999 * V[i] + 0.001 * gr ** 2
            P[i] = P[i] - lr * (M[i] / (1 - 0.9 ** (t + 1))) / (np.sqrt(V[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return snaps


def gif_xor():
    X, y = xor_data()
    runs = {"Without activation: the network stays linear": train_xor(X, y, relu=False), "With ReLU: the boundary can bend": train_xor(X, y, relu=True)}
    g = np.linspace(-0.5, 1.5, 140)
    G0, G1 = np.meshgrid(g, g)
    grid = np.column_stack([G0.ravel(), G1.ravel()])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9), dpi=80)
    n = len(next(iter(runs.values())))

    def draw(k):
        k = min(k, n - 1)
        for ax, (title, snaps) in zip(axes, runs.items()):
            ax.clear()
            t, (W, b, v, c), loss, acc = snaps[k]
            z = grid @ W + b
            h = np.maximum(z, 0) if "ReLU" in title else z
            out = (h @ v + c).reshape(G0.shape)
            ax.contourf(G0, G1, out, levels=[-1e9, 0.5, 1e9], colors=[RED, BLUE], alpha=0.13)
            ax.contour(G0, G1, out, levels=[0.5], colors=[INK], linewidths=2)
            ax.scatter(*X[y > 0.5].T, c=BLUE, s=14, linewidths=0)
            ax.scatter(*X[y < 0.5].T, c=RED, s=14, linewidths=0)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(f"epoch {t}   ·   accuracy {acc * 100:.0f} %")
            ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("XOR, 2 inputs, 8 hidden neurons, 1 output: the only difference is the activation", fontsize=12.5)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    FuncAnimation(fig, draw, frames=n + 8).save(os.path.join(ROOT, "assets/gifs/xor_relu.gif"), writer=PillowWriter(fps=6))
    plt.close(fig)
    return {title: (snaps[-1][2], snaps[-1][3]) for title, snaps in runs.items()}


if __name__ == "__main__":
    X, y = iris_two_classes()
    with open(os.path.join(ROOT, "docs/assets/iris.js"), "w") as f:
        f.write("// Iris, setosa (y = -1) against versicolor (y = +1): petal length and width, standardised. Fisher (1936).\n")
        f.write("window.IRIS = " + json.dumps({"x": np.round(X, 4).tolist(), "y": y.astype(int).tolist()}) + ";\n")
    gif_linear_classifiers(X, y)
    print(gif_xor())
