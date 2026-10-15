"""Rotation CAPTCHA server.

Standalone Flask app that generates a rotation CAPTCHA: a circular image
is randomly rotated, and the user must drag a slider to rotate it back
to the correct orientation. Core approach is standard across open-source
implementations (rotate-captcha-crack, RotateCaptchaBreak, etc.).

All challenge constants (canvas size, rotation range, tolerance, etc.) are
loaded from a JSON config under `assets/rotation/`. The mode is selected
via the `CAPTCHA_MODE` env var (default `train_eval`). For static challenges
(pre-rendered image + meta), `CAPTCHA_TEST_ID` selects a numbered subdir.

    pip install flask pillow
    python lite/gym/envs/captcha/servers/rotation.py
    # Open http://localhost:5000 in a browser
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import urllib.request
from pathlib import Path

from flask import Flask, request, send_file

from PIL import Image, ImageDraw

# Deterministic mode: if CAPTCHA_SEED is set, seed random before any generation
# so repeated runs produce the same challenge (used by eval task variants).
_seed = os.environ.get("CAPTCHA_SEED")
if _seed is not None:
    random.seed(int(_seed))

ASSETS = Path(__file__).parent.parent / ".cache" / "assets" / "rotation"
MODE = os.environ.get("CAPTCHA_MODE", "train_eval")
TEST_ID = int(os.environ.get("CAPTCHA_TEST_ID", "0"))
RESULT_FILE = os.environ.get("CAPTCHA_RESULT_FILE", "/tmp/captcha_result.json")


# ---------------------------------------------------------------------------
# Image generation (config-driven)
# ---------------------------------------------------------------------------

def _download_image(url_template: str, size: int) -> Image.Image | None:
    """Download a random square photograph using the configured URL template.

    The template uses `{seed}` and `{size}` placeholders. When CAPTCHA_SEED
    is unset, a random seed is used so each run gets a fresh image.
    """
    try:
        seed = _seed if _seed is not None else str(random.randint(0, 10**9))
        url = url_template.format(seed=seed, size=size)
        req = urllib.request.Request(url, headers={"User-Agent": "RotationCaptcha/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception as e:
        print(f"[ROTATION CAPTCHA] Failed to download image: {e}")
        return None


def _load_local_image(img_dir: str, size: int) -> Image.Image | None:
    """Pick a random jpg from `img_dir` (relative to assets/rotation/) and
    open it as RGBA, resizing to (size, size) if needed.
    """
    try:
        root = ASSETS / img_dir
        jpgs = sorted(root.glob("*.jpg"))
        if not jpgs:
            return None
        chosen = random.choice(jpgs)
        img = Image.open(chosen).convert("RGBA")
        if img.size != (size, size):
            img = img.resize((size, size), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"[ROTATION CAPTCHA] Failed to load local image: {e}")
        return None


def _generate_fallback_image(size: int, shape_count: int, triangle_size: int) -> Image.Image:
    """Generate a colorful fallback image with enough visual features."""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)

    # Gradient background
    base_hue = random.randint(0, 360)
    for y in range(size):
        for x in range(size):
            r = int(128 + 80 * math.sin(0.03 * x + base_hue))
            g = int(128 + 80 * math.sin(0.03 * y + base_hue + 2))
            b = int(128 + 60 * math.sin(0.02 * (x + y) + base_hue + 4))
            img.putpixel((x, y), (r, g, b))

    # Random shapes for visual orientation cues
    for _ in range(shape_count):
        x1 = random.randint(0, size - 60)
        y1 = random.randint(0, size - 60)
        x2 = x1 + random.randint(20, 60)
        y2 = y1 + random.randint(20, 60)
        color = (random.randint(50, 220), random.randint(50, 220), random.randint(50, 220))
        draw.rectangle([x1, y1, x2, y2], fill=color)

    # A triangle for clear orientation
    cx, cy = size // 2, size // 4
    draw.polygon([
        (cx, cy - triangle_size),
        (cx - triangle_size, cy + triangle_size),
        (cx + triangle_size, cy + triangle_size),
    ], fill=(255, 80, 80))

    return img.convert("RGBA")


def _apply_circular_mask(img: Image.Image, aa_scale: int) -> Image.Image:
    """Crop image to a circle with antialiased edges (supersampled by aa_scale)."""
    size = img.size[0]
    mask_hi = Image.new("L", (size * aa_scale, size * aa_scale), 0)
    draw = ImageDraw.Draw(mask_hi)
    draw.ellipse([0, 0, size * aa_scale - 1, size * aa_scale - 1], fill=255)
    mask = mask_hi.resize((size, size), Image.LANCZOS)

    result = img.copy()
    result.putalpha(mask)
    return result


def generate_rotation_challenge(content: dict, render: dict) -> dict:
    """Generate a rotation CAPTCHA from a content+render config.

    Returns a dict with: image (PNG bytes of the rotated circle),
    original_image (PNG bytes of the upright circle, for debugging),
    target_angle_deg, canvas_size, tolerance_deg.
    """
    size = render["canvas_size"][0]
    aa_scale = render["aa_supersample"]
    angle_lo, angle_hi = render["initial_rotation_range_deg"]
    tolerance = render["tolerance_deg"]

    img = None
    src = content["image_source"]
    if src == "local":
        img = _load_local_image(content["images_dir"], size)
    elif src == "picsum":
        img = _download_image(content["image_fallback_url"], size)
    if img is None:
        img = _generate_fallback_image(
            size,
            shape_count=render["fallback_shape_count"],
            triangle_size=render["fallback_triangle_size"],
        )
    img = img.resize((size, size), Image.LANCZOS)

    circle_img = _apply_circular_mask(img, aa_scale)

    # Random rotation angle — avoid trivially small angles (within tolerance of 0/360)
    angle = random.randint(angle_lo, angle_hi)
    while angle < tolerance or angle > 360 - tolerance:
        angle = random.randint(angle_lo, angle_hi)

    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    rotated_circle = _apply_circular_mask(rotated, aa_scale)

    orig_buf = io.BytesIO()
    circle_img.save(orig_buf, "PNG")
    rot_buf = io.BytesIO()
    rotated_circle.save(rot_buf, "PNG")

    return {
        "image": rot_buf.getvalue(),
        "original_image": orig_buf.getvalue(),
        "target_angle_deg": angle,
        "canvas_size": render["canvas_size"],
        "tolerance_deg": tolerance,
    }


def _load_static_challenge() -> dict:
    """Load a pre-rendered challenge from assets/rotation/<MODE>/<TEST_ID>/."""
    chal_dir = ASSETS / MODE / f"{TEST_ID:03d}"
    meta = json.loads((chal_dir / "meta.json").read_text())
    rotated_png = (chal_dir / "image.png").read_bytes()
    original_png = (chal_dir / "original.png").read_bytes() if (chal_dir / "original.png").exists() else rotated_png
    return {
        "image": rotated_png,
        "original_image": original_png,
        "target_angle_deg": meta["target_angle_deg"],
        "canvas_size": meta["canvas_size"],
        "tolerance_deg": meta["tolerance_deg"],
    }


def load_challenge() -> dict:
    """Load a challenge for the current MODE.

    If `assets/rotation/<MODE>.json` exists, generate a fresh challenge from
    that config. Otherwise treat MODE as a directory containing pre-rendered
    static challenges keyed by TEST_ID.
    """
    mode_file = ASSETS / f"{MODE}.json"
    if mode_file.exists():
        cfg = json.loads(mode_file.read_text())
        return generate_rotation_challenge(cfg["content"], cfg["render"])
    return _load_static_challenge()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Generate at startup
CHALLENGE = load_challenge()
ORIG_PNG = CHALLENGE["original_image"]
ROTATED_PNG = CHALLENGE["image"]
ROTATION_ANGLE = CHALLENGE["target_angle_deg"]
IMG_SIZE = CHALLENGE["canvas_size"][0]
TOLERANCE = CHALLENGE["tolerance_deg"]


@app.route("/")
def index():
    """Main page: rotated circular image + rotation slider."""
    return f"""<!DOCTYPE html>
<html>
<head><title>Rotation CAPTCHA</title></head>
<body style="display:flex; justify-content:center; align-items:center;
             min-height:100vh; margin:0; background:#f5f5f5;
             font-family:Arial,sans-serif; user-select:none;">
  <div style="text-align:center; background:white; padding:30px;
              border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.1);">
    <h2>Rotation CAPTCHA</h2>
    <p>Drag the slider to rotate the image to its correct orientation</p>

    <div style="position:relative; width:{IMG_SIZE}px; height:{IMG_SIZE}px;
                margin:0 auto; border-radius:50%; overflow:hidden;
                border:3px solid #ccc; background:#eee;">
      <img id="rotated-img" src="/rotated.png"
           style="width:{IMG_SIZE}px; height:{IMG_SIZE}px;
                  transform:rotate(0deg); transition:none;"
           draggable="false"/>
    </div>

    <div id="slider-track" style="position:relative; width:{IMG_SIZE}px; height:44px;
                margin:10px auto 0; background:#e8e8e8; border-radius:22px;
                border:1px solid #ccc;">
      <div id="slider-fill" style="position:absolute; top:0; left:0; height:100%;
                  background:#76c442; border-radius:22px; width:0;"></div>
      <div id="slider-handle" style="position:absolute; top:2px; left:0;
                  width:40px; height:40px; box-sizing:border-box;
                  background:#fff; border-radius:50%;
                  border:2px solid #ccc; cursor:grab; display:flex;
                  align-items:center; justify-content:center; font-size:18px;
                  box-shadow:0 1px 3px rgba(0,0,0,0.2);">&#10095;</div>
      <span id="slider-text" style="position:absolute; top:0; left:0; width:100%;
                  text-align:center; line-height:44px; color:#999; pointer-events:none;">
        Drag to rotate image</span>
    </div>

    <button id="submit-btn" onclick="submitAnswer()"
            style="margin-top:16px; font-size:18px; padding:10px 32px;
                   background:#4CAF50; color:white; border:none;
                   border-radius:4px; cursor:pointer;">
      Submit
    </button>

    <div id="result-msg" style="margin-top:12px; font-size:18px; min-height:28px;"></div>
  </div>

<script>
(function() {{
  var handle = document.getElementById('slider-handle');
  var fill = document.getElementById('slider-fill');
  var img = document.getElementById('rotated-img');
  var track = document.getElementById('slider-track');
  var text = document.getElementById('slider-text');

  var trackWidth = {IMG_SIZE};
  var handleWidth = 40;
  var maxMove = trackWidth - handleWidth;
  var isDragging = false;
  var originX = 0;
  var currentAngle = 0;

  window._userAngle = 0;

  function onStart(e) {{
    isDragging = true;
    originX = (e.clientX || e.touches[0].clientX);
    handle.style.cursor = 'grabbing';
    text.style.display = 'none';
    e.preventDefault();
  }}

  function onMove(e) {{
    if (!isDragging) return;
    var clientX = (e.clientX || e.touches[0].clientX);
    var moveX = clientX - originX;
    if (moveX < 0) moveX = 0;
    if (moveX > maxMove) moveX = maxMove;

    handle.style.left = moveX + 'px';
    fill.style.width = (moveX + handleWidth / 2) + 'px';

    // Map slider position to 0-360 degrees
    currentAngle = Math.round((moveX / maxMove) * 360);
    window._userAngle = currentAngle;
    img.style.transform = 'rotate(' + currentAngle + 'deg)';
  }}

  function onEnd(e) {{
    if (!isDragging) return;
    isDragging = false;
    handle.style.cursor = 'grab';
  }}

  handle.addEventListener('mousedown', onStart);
  handle.addEventListener('touchstart', onStart);
  document.addEventListener('mousemove', onMove);
  document.addEventListener('touchmove', onMove);
  document.addEventListener('mouseup', onEnd);
  document.addEventListener('touchend', onEnd);
}})();

function submitAnswer() {{
  var msg = document.getElementById('result-msg');
  var fill = document.getElementById('slider-fill');
  var handle = document.getElementById('slider-handle');
  var img = document.getElementById('rotated-img');
  var text = document.getElementById('slider-text');
  var xhr = new XMLHttpRequest();
  xhr.open('POST', '/verify', true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.onload = function() {{
    var resp = JSON.parse(xhr.responseText);
    if (resp.correct) {{
      msg.innerHTML = '<span style="color:#4CAF50; font-weight:bold;">&#10003; Correct!</span>';
      fill.style.background = '#4CAF50';
    }} else {{
      msg.innerHTML = '<span style="color:#f44336; font-weight:bold;">&#10007; Wrong! Try again</span>';
      fill.style.background = '#f44336';
      setTimeout(function() {{
        handle.style.left = '0px';
        fill.style.width = '0px';
        fill.style.background = '#76c442';
        img.style.transform = 'rotate(0deg)';
        window._userAngle = 0;
        msg.innerHTML = '';
        text.style.display = '';
      }}, 1000);
    }}
  }};
  xhr.send(JSON.stringify({{angle: window._userAngle}}));
}}
</script>
</body>
</html>"""


@app.route("/rotated.png")
def rotated_image():
    """Serve the rotated circular image."""
    return send_file(io.BytesIO(ROTATED_PNG), mimetype="image/png")


@app.route("/original.png")
def original_image():
    """Serve the original (unrotated) circular image (for debugging)."""
    return send_file(io.BytesIO(ORIG_PNG), mimetype="image/png")


# Once a correct submit happens, lock the result so a subsequent wrong
# submit doesn't overwrite it — the page stays interactive after a correct
# answer, and the env reads /result only once at the terminal step.
# Mirrors slider.py's _LOCKED_CORRECT.
_LOCKED_CORRECT = False


@app.route("/verify", methods=["POST"])
def verify():
    """Verify rotation angle. Writes result to file for evaluate_final_fn.

    The image was rotated by ROTATION_ANGLE degrees (PIL counter-clockwise).
    To restore it, the user must apply a CSS rotation (clockwise) of
    ROTATION_ANGLE degrees. We compare the user's CSS rotation to the target.
    """
    global _LOCKED_CORRECT
    data = request.get_json(force=True)
    user_angle = data.get("angle", -9999)

    target = ROTATION_ANGLE

    # Angular distance (handle wraparound)
    diff = abs(user_angle - target) % 360
    distance = min(diff, 360 - diff)
    correct = distance <= TOLERANCE
    if correct:
        _LOCKED_CORRECT = True

    result = {
        "submitted": True,
        "correct": _LOCKED_CORRECT,
        "user_angle": user_angle,
        "target_angle": target,
        "distance": distance,
        "tolerance": TOLERANCE,
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f)

    return json.dumps(result), 200, {"Content-Type": "application/json"}


@app.route("/result")
def result():
    """API endpoint for evaluate_final_fn to read the result."""
    try:
        with open(RESULT_FILE) as f:
            return f.read()
    except FileNotFoundError:
        return json.dumps({"submitted": False, "correct": False})


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
