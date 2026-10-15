"""Simple text CAPTCHA server (assets-driven).

Standalone Flask app that loads challenge config from
`assets/text_captcha_4/{MODE}.json` (generative mode) or
`assets/text_captcha_4/{MODE}/{TEST_ID:03d}/{image.png,meta.json}` (static mode),
then renders a distorted text CAPTCHA and verifies user input.

Designed to run inside a Docker container for cua-lite gym, but can also be
tested locally:

    pip install flask pillow
    python lite/gym/envs/captcha/servers/text_captcha_4.py
    # Open http://localhost:5000 in a browser

Environment variables:
    CAPTCHA_MODE        : asset mode name (default "train_eval")
    CAPTCHA_TEST_ID     : index for static-mode challenges (default 0)
    CAPTCHA_SEED        : seed `random` for deterministic challenge
    CAPTCHA_RESULT_FILE : where to dump the submission result
    PORT                : Flask port (default 5000)
"""

from __future__ import annotations

import io
import json
import math
import os
import random
from pathlib import Path

from flask import Flask, request, send_file
from PIL import Image, ImageDraw, ImageFont

# Deterministic mode: if CAPTCHA_SEED is set, seed random before any generation
# so repeated runs produce the same challenge (used by eval task variants).
_seed = os.environ.get("CAPTCHA_SEED")
if _seed is not None:
    random.seed(int(_seed))

ASSETS = Path(__file__).parent.parent / ".cache" / "assets" / "text_captcha_4"
MODE = os.environ.get("CAPTCHA_MODE", "train_eval")
TEST_ID = int(os.environ.get("CAPTCHA_TEST_ID", "0"))


# ---------------------------------------------------------------------------
# CAPTCHA generation (pure renderer, all params from config)
# ---------------------------------------------------------------------------

def generate_text(content_cfg: dict) -> str:
    """Sample a random CAPTCHA string from the configured char set."""
    return "".join(random.choices(content_cfg["char_set"], k=content_cfg["text_length"]))


def render_distorted_text(text: str, render_cfg: dict) -> Image.Image:
    """Render `text` to a distorted PIL image using params from `render_cfg`."""
    width, height = render_cfg["canvas_size"]
    bg = tuple(render_cfg["bg_color"])
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    font_size = random.randint(*render_cfg["font_size_range"])
    font = None
    for font_path in render_cfg["fonts"]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    color_r = render_cfg["char_color_range"]["r"]
    color_g = render_cfg["char_color_range"]["g"]
    color_b = render_cfg["char_color_range"]["b"]
    rot_lo, rot_hi = render_cfg["rotation_range_deg"]
    sp_lo, sp_hi = render_cfg["char_spacing_range"]

    char_width = width // (len(text) + 1)
    for i, ch in enumerate(text):
        x = char_width * (i + 0.5) + random.randint(sp_lo, sp_hi)
        y = height // 2 - font_size // 2 + random.randint(-10, 10)

        ch_img = Image.new("RGBA", (font_size + 20, font_size + 20), (255, 255, 255, 0))
        ch_draw = ImageDraw.Draw(ch_img)
        ch_draw.text(
            (10, 10), ch, font=font,
            fill=(random.randint(*color_r), random.randint(*color_g), random.randint(*color_b)),
        )
        ch_img = ch_img.rotate(random.randint(rot_lo, rot_hi), expand=True, fillcolor=(255, 255, 255, 0))
        img.paste(ch_img, (int(x), int(y)), ch_img)

    # Interference lines
    for _ in range(random.randint(4, 7)):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 2))

    # Noise dots
    for _ in range(random.randint(200, 400)):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        draw.point((x, y), fill=color)

    # Wave distortion
    img2 = Image.new("RGB", (width, height), bg)
    amplitude = random.uniform(3, 6)
    period = random.uniform(0.05, 0.1)
    phase = random.uniform(0, 2 * math.pi)
    for y_pos in range(height):
        shift = int(amplitude * math.sin(period * y_pos + phase))
        row = img.crop((0, y_pos, width, y_pos + 1))
        img2.paste(row, (shift, y_pos))
    return img2


def load_challenge() -> tuple[str, Image.Image]:
    """Mode dispatch: generative (.json) vs static (NNN/image.png + meta.json)."""
    mode_file = ASSETS / f"{MODE}.json"
    if mode_file.exists():
        cfg = json.loads(mode_file.read_text())
        text = generate_text(cfg["content"])
        return text, render_distorted_text(text, cfg["render"])
    chal_dir = ASSETS / MODE / f"{TEST_ID:03d}"
    img = Image.open(chal_dir / "image.png").convert("RGB")
    return json.loads((chal_dir / "meta.json").read_text())["answer"], img


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

RESULT_FILE = os.environ.get("CAPTCHA_RESULT_FILE", "/tmp/captcha_result.json")

app = Flask(__name__)

# Load challenge at startup
ANSWER, IMAGE = load_challenge()
_buf = io.BytesIO()
IMAGE.save(_buf, "PNG")
CAPTCHA_IMAGE = _buf.getvalue()



@app.route("/")
def index():
    """Main page: CAPTCHA image + input form."""
    return f"""<!DOCTYPE html>
<html>
<head><title>CAPTCHA Challenge</title></head>
<body style="display:flex; justify-content:center; align-items:center;
             min-height:100vh; margin:0; background:#f5f5f5;
             font-family:Arial,sans-serif;">
  <div style="text-align:center; background:white; padding:40px;
              border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.1);">
    <h1>CAPTCHA Challenge</h1>
    <p>Type the text you see in the image below:</p>
    <img src="/captcha.png" style="border:1px solid #ccc; border-radius:4px;" /><br><br>
    <form action="/submit" method="post">
      <input name="answer" type="text" placeholder="Enter CAPTCHA"
             style="font-size:20px; padding:8px 16px; width:200px;
                    text-align:center; border:2px solid #ccc; border-radius:4px;" />
      <br><br>
      <button type="submit"
              style="font-size:18px; padding:10px 32px; background:#4CAF50;
                     color:white; border:none; border-radius:4px; cursor:pointer;">
        Submit
      </button>
    </form>
  </div>
</body>
</html>"""


@app.route("/captcha.png")
def captcha_image():
    """Serve the CAPTCHA image."""
    return send_file(io.BytesIO(CAPTCHA_IMAGE), mimetype="image/png")


def check_answer(user_answer: str, expected: str) -> bool:
    """Grade a text-CAPTCHA answer: case-insensitive, whitespace-stripped on
    BOTH sides. Previously only the user input was upper-cased while the
    expected answer was compared verbatim, so a correctly-read CAPTCHA whose
    char_set/answer contained any lowercase character could never match.
    """
    return user_answer.strip().upper() == expected.strip().upper()


@app.route("/submit", methods=["POST"])
def submit():
    """Verify the user's answer and write result to file."""
    user_answer = request.form.get("answer", "").strip()
    correct = check_answer(user_answer, ANSWER)

    result = {"submitted": True, "correct": correct, "user_answer": user_answer, "expected": ANSWER}
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f)

    if correct:
        return """<!DOCTYPE html>
<html><body style="display:flex; justify-content:center; align-items:center;
                    min-height:100vh; margin:0; background:#f5f5f5;
                    font-family:Arial,sans-serif;">
  <div style="text-align:center;">
    <h1 style="color:#4CAF50; font-size:48px;">Correct!</h1>
    <p style="font-size:20px;">You entered the CAPTCHA correctly.</p>
  </div>
</body></html>"""
    else:
        return f"""<!DOCTYPE html>
<html><body style="display:flex; justify-content:center; align-items:center;
                    min-height:100vh; margin:0; background:#f5f5f5;
                    font-family:Arial,sans-serif;">
  <div style="text-align:center;">
    <h1 style="color:#f44336; font-size:48px;">Wrong!</h1>
    <p style="font-size:20px;">You entered: <b>{user_answer}</b></p>
    <p><a href="/">Try again</a></p>
  </div>
</body></html>"""


@app.route("/result")
def result():
    """API endpoint for evaluate_final_fn to read the result."""
    try:
        with open(RESULT_FILE) as f:
            return f.read()
    except FileNotFoundError:
        return json.dumps({"submitted": False, "correct": False})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
