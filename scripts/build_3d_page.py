"""Builds docs/lab4_3d.html from the 3D demos saved in the outputs of the linear classifiers notebook.

The three demos (the score plane, the ball on the surface of the objective, the perceptron one correction
at a time) are self-contained: their data and their JavaScript are stored in the notebook. This script lifts
them out unchanged, so the page shows exactly the tool used in class.

Usage: python scripts/build_3d_page.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, "notebooks/lab4_linear_classifiers/IFT3395_Lab4_Classifieurs_Lineaires.ipynb")
CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
LOCAL = "assets/vendor/plotly-2.35.2.min.js"

DEMOS = [  # (marker in the cell source, anchor, kicker, title, intro, prediction, answer)
    ("demo_S()", "ball", "3D tool 1", "The ball on the surface of the objective",
     "Gradient descent seen as a hiker in the fog. The surface is the objective L(w₀, w₁) with the bias held at its final value: an elongated bowl with a single bottom, the gold star w*. The gold ball is the current weight vector, the grey square its starting point, and the grey circles on the floor are the level lines of the same surface. On the right, the decision boundary that this weight vector draws on the flowers; below, the altitude L at each iteration.",
     "Before moving the iteration slider: will the first moves of the ball be longer or shorter than the last ones? The step η does not change.",
     "Longer. Each move has length η times the norm of the gradient: η is fixed, the slope is steep at the start and almost flat at the bottom. With η = 0.5 the ball moves by 0.73 at the first iteration and by 0.001 at the hundredth. Then raise η: the path overshoots the valley, zigzags, and past a threshold the descent diverges."),
    ("demo_P()", "plane", "3D tool 2", "The score is a height: the tilted plane",
     "A linear classifier computes the score z = w₀x₀ + w₁x₁ + b. Drawn above the plane of the inputs, the score is a tilted plane; the decision boundary is the line where that plane crosses the floor z = 0. Ten toy points, the angle of w and the bias b are adjustable, in 3D and seen from above.",
     "If you only change the bias b, does the plane tilt differently, or does it do something else?",
     "It does not tilt: the plane moves up or down as a whole, so its intersection with the floor, the boundary, slides parallel to itself. Changing the angle of w is what rotates the slope, and the boundary with it."),
    ("demo_R()", "perceptron", "3D tool 3", "The perceptron, one correction at a time",
     "Rosenblatt's rule on the same ten points, three steps per correction: a misclassified point is found under the floor, the vector η·y·x is added to w (seen from above as a sum of vectors), and the plane pivots. Starting point w = (−1.5, 0.5, −0.5), step η = 0.5.",
     "The start misclassifies 7 points out of 10. How many corrections do you expect before no point is misclassified: 7, more, or fewer?",
     "Fewer: 4 corrections are enough. Each correction rotates the whole plane, so fixing one point usually fixes several others at once."),
]

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gradient descent in 3D: the visualisation tool used in class</title>
<meta name="description" content="The 3D visualisation tool used in the linear classifiers lab of IFT3395/IFT6390 (Université de Montréal): gradient descent as a ball on the surface of the objective.">
<link rel="stylesheet" href="assets/style.css">
<style>
  :root {{ color-scheme: light; --surface: #fcfcfb; --panel: #ffffff; --ink: #111827; --ink-2: #4b5563; --ink-3: #6b7280; --rule: #e5e7eb; --accent: #2563eb; --gold: #d99a00; }}
  .tool {{ margin-top: 16px; overflow-x: auto; }}
  .tool .lab4-demo {{ min-width: 860px; }}
  .note {{ font-size: 14px; color: var(--ink-3); }}
</style>
</head>
<body>
<header class="top"><nav>
  <a class="brand" href="index.html">IFT3395 / IFT6390 · Teaching portfolio</a>
  <a href="lab4_3d.html" aria-current="page">Gradient descent in 3D</a>
  <a href="lab4.html">Linear classifiers</a>
  <a href="lab8.html">PyTorch and backpropagation</a>
  <a href="https://github.com/yanis-bencheikh/ML_Course_Bencheikh_Mila_UdeM_ift6390">GitHub</a>
</nav></header>
<main>
  <div class="kicker">Lab 4 · 27 October 2026</div>
  <h1>Gradient descent in 3D</h1>
  <p class="lede">The 3D visualisation tool we used in class to understand gradient descent, exactly as it runs in the lab notebook. Drag a scene to rotate it, scroll to zoom, and use the sliders above each figure.</p>
  <p class="note">The labels inside the figures are in French, the language of the section. The scenes need WebGL and are easier to use on a computer than on a phone.</p>
{sections}
</main>
<footer>Yanis Bencheikh · Mila and Université de Montréal · IFT3395 / IFT6390, Fall 2026. Figures drawn with Plotly.js (MIT licence).</footer>
</body>
</html>
"""

SECTION = """
  <section class="demo" id="{anchor}">
    <div class="kicker">{kicker}</div>
    <h2>{title}</h2>
    <p>{intro}</p>
    <div class="predict"><b>Predict first.</b> {prediction}</div>
    <div class="tool">{html}</div>
    <details class="answer"><summary>Compare with the answer</summary><p>{answer}</p></details>
  </section>"""


def saved_html(cells, marker):
    for cell in cells:
        if cell["cell_type"] == "code" and marker in "".join(cell["source"]):
            for output in cell.get("outputs", []):
                if "text/html" in output.get("data", {}):
                    return "".join(output["data"]["text/html"])
    raise SystemExit(f"no saved HTML output for {marker}")


if __name__ == "__main__":
    cells = json.load(open(NOTEBOOK, encoding="utf-8"))["cells"]
    sections = []
    for marker, anchor, kicker, title, intro, prediction, answer in DEMOS:
        html = saved_html(cells, marker)
        assert CDN in html
        sections.append(SECTION.format(anchor=anchor, kicker=kicker, title=title, intro=intro, prediction=prediction, answer=answer, html=html.replace(CDN, LOCAL)))
    with open(os.path.join(ROOT, "docs/lab4_3d.html"), "w", encoding="utf-8") as f:
        f.write(PAGE.format(sections="".join(sections)))
    print("wrote docs/lab4_3d.html")
