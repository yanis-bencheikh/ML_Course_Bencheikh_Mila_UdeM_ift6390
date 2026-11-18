"""Records the animations of the 3D tool shown in the README by driving the real page in a headless browser.

Usage: python scripts/capture_3d_gifs.py   (needs playwright and a local Chrome; set CHROME to its path)
"""
import asyncio
import io
import os

from PIL import Image
from playwright.async_api import async_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = os.environ.get("CHROME", "/usr/bin/google-chrome")
SET = "([el, v]) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})); }"


async def record(page, anchor, moves, name, width=860, hold=6, ms=140, trim=24):
    """moves: list of (index of the control inside the demo, value). One frame is captured after each move."""
    demo = page.locator(f"#{anchor} .lab4-demo")
    await demo.scroll_into_view_if_needed()
    controls = demo.locator("input[type=range]")
    frames = []
    for index, value in moves:
        await page.evaluate(SET, [await controls.nth(index).element_handle(), str(value)])
        await page.wait_for_timeout(350)
        shot = Image.open(io.BytesIO(await demo.screenshot())).convert("RGB")
        frames.append(shot)
    height = min(f.height for f in frames) - trim  # same size for every frame; drops the printed summary lines under the figure
    frames = [f.crop((0, 0, f.width, height)) for f in frames]
    frames = [f.resize((width, round(height * width / f.width)), Image.LANCZOS) for f in frames]
    frames += [frames[-1]] * hold
    quantised = [f.quantize(colors=160, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames]  # one palette per frame keeps the gold ball gold
    path = os.path.join(ROOT, "assets/gifs", name)
    quantised[0].save(path, save_all=True, append_images=quantised[1:], duration=ms, loop=0, optimize=True)
    print(name, len(frames), "frames", os.path.getsize(path) // 1024, "kB")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROME, args=["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        page = await browser.new_page(viewport={"width": 1180, "height": 1100})
        await page.goto("file://" + os.path.join(ROOT, "docs/lab4_3d.html"))
        await page.wait_for_timeout(5000)
        steps = list(range(0, 16)) + list(range(16, 41, 2)) + [50, 60, 80, 100]
        # the ball going down the bowl with the default step, then with a step large enough to zigzag
        await record(page, "ball", [(0, 3)] + [(1, t) for t in steps], "descent_3d_ball.gif")
        await record(page, "ball", [(1, 0), (0, 4)] + [(1, t) for t in range(0, 41)], "descent_3d_large_step.gif", ms=200)
        # the perceptron: the plane pivots at each correction
        await record(page, "perceptron", [(0, k) for k in range(14)], "perceptron_3d.gif", ms=900, hold=2, trim=24)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
