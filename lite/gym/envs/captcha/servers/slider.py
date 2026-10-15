"""Slider puzzle CAPTCHA server.

Standalone Flask app that generates a slider puzzle CAPTCHA: a background
image with a puzzle-piece gap and a separate puzzle piece that the user
must drag to the correct position. Core algorithm referenced from
ArgoZhang/SliderCaptcha (304 stars).

Render hyperparameters (puzzle size, piece size range, tolerance, etc.) and
the picsum background-fallback URL live in `assets/slider/<MODE>.json`.
The server is a pure renderer that dispatches on `CAPTCHA_MODE`:

  CAPTCHA_MODE          : asset stem under assets/slider/  (default: train_eval)
  CAPTCHA_TEST_ID       : index for static-asset modes     (default: 0)
  CAPTCHA_SEED          : seeds random; also used by picsum's seeded endpoint
  CAPTCHA_RESULT_FILE   : where to write the verify result JSON

    pip install flask pillow
    python lite/gym/envs/captcha/servers/slider.py
    # http://localhost:5000
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

# ---------------------------------------------------------------------------
# Env / asset dispatch
# ---------------------------------------------------------------------------

ASSETS = Path(__file__).parent.parent / ".cache" / "assets" / "slider"
MODE = os.environ.get("CAPTCHA_MODE", "train_eval")
TEST_ID = int(os.environ.get("CAPTCHA_TEST_ID", "0"))

_seed = os.environ.get("CAPTCHA_SEED")
if _seed is not None:
    random.seed(int(_seed))

RESULT_FILE = os.environ.get("CAPTCHA_RESULT_FILE", "/tmp/captcha_result.json")


# ---------------------------------------------------------------------------
# Background sourcing
# ---------------------------------------------------------------------------

def _download_background(url_template: str, width: int, height: int) -> Image.Image | None:
    """Download a random photograph from the configured background source.

    Substitutes {seed}, {w}, {h} into `url_template`. When CAPTCHA_SEED is
    unset, replaces {seed} with "random" — picsum treats unknown seeds as
    new random images.
    """
    try:
        seed = _seed if _seed is not None else "random"
        url = url_template.format(seed=seed, w=width, h=height)
        req = urllib.request.Request(url, headers={"User-Agent": "SliderCaptcha/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"[SLIDER CAPTCHA] Failed to download background: {e}")
        return None


def _load_local_background(bg_dir: str, width: int, height: int) -> Image.Image | None:
    """Pick a random jpg from `bg_dir` (relative to assets/slider/) and open
    it. Resizes to (width, height) if needed. Returns None on failure so
    callers can fall back to the synthetic generator.
    """
    try:
        root = ASSETS / bg_dir
        jpgs = sorted(root.glob("*.jpg"))
        if not jpgs:
            return None
        chosen = random.choice(jpgs)
        img = Image.open(chosen).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"[SLIDER CAPTCHA] Failed to load local background: {e}")
        return None


def _generate_background(width: int, height: int) -> Image.Image:
    """Generate a colorful background image using random rectangles and gradients."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Base gradient
    base_hue = random.randint(0, 360)
    for x in range(width):
        r = int(128 + 80 * math.sin(0.02 * x + base_hue))
        g = int(128 + 80 * math.sin(0.02 * x + base_hue + 2))
        b = int(128 + 80 * math.sin(0.02 * x + base_hue + 4))
        draw.line([(x, 0), (x, height)], fill=(r, g, b))

    # Random rectangles for texture (so the gap is visually distinguishable)
    for _ in range(15):
        x1 = random.randint(0, width - 40)
        y1 = random.randint(0, height - 40)
        x2 = x1 + random.randint(20, 80)
        y2 = y1 + random.randint(20, 60)
        color = (random.randint(50, 220), random.randint(50, 220), random.randint(50, 220))
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255))

    return img


# ---------------------------------------------------------------------------
# Puzzle geometry
# ---------------------------------------------------------------------------

def _draw_puzzle_path(x: int, y: int, size: int, knob_r: int) -> list[tuple[float, float]]:
    """Build a puzzle-piece polygon (square + top knob + right knob).

    Mirrors ArgoZhang/SliderCaptcha: arc-based knobs on the top and right
    edges of a square. Returned as a polygon-ready point list.
    """
    points: list[tuple[float, float]] = []

    # Bottom-left to top-left
    points.append((x, y + size))
    points.append((x, y))

    # Top edge with knob: left part -> knob arc -> right part
    knob_cx = x + size // 2
    knob_cy = y
    points.append((knob_cx - knob_r, y))
    for angle_deg in range(180, -1, -10):
        angle = math.radians(angle_deg)
        px = knob_cx + knob_r * math.cos(angle)
        py = knob_cy - knob_r * math.sin(angle)
        points.append((px, py))
    points.append((knob_cx + knob_r, y))
    points.append((x + size, y))

    # Right edge with knob
    knob_cx2 = x + size
    knob_cy2 = y + size // 2
    points.append((x + size, knob_cy2 - knob_r))
    for angle_deg in range(90, -91, -10):
        angle = math.radians(angle_deg)
        px = knob_cx2 + knob_r * math.cos(angle)
        py = knob_cy2 - knob_r * math.sin(angle)
        points.append((px, py))
    points.append((x + size, knob_cy2 + knob_r))

    # Right edge bottom -> bottom edge
    points.append((x + size, y + size))
    points.append((x, y + size))

    return points


# ---------------------------------------------------------------------------
# Renderer (pure: cfg in, challenge dict out)
# ---------------------------------------------------------------------------

def generate_slider_challenge(content_cfg: dict, render_cfg: dict) -> dict:
    """Render a slider puzzle CAPTCHA from config.

    Returns a dict with keys: bg_image (PIL.Image, RGB), piece_image (PIL.Image,
    RGBA), target_x (int), target_y (int), tolerance_px (int), and the full
    render_cfg echoed back for downstream HTML layout.
    """
    width, height = render_cfg["puzzle_size"]
    piece_lo, piece_hi = render_cfg["piece_size_range"]
    piece_size = random.randint(piece_lo, piece_hi)
    knob_r = render_cfg["knob_radius"]
    tolerance_px = render_cfg["tolerance_px"]
    edge_margin = render_cfg["edge_margin"]
    left_start_offset = render_cfg["left_start_offset"]
    aa = render_cfg["aa_scale"]

    bg = None
    src = content_cfg["background_source"]
    if src == "local":
        bg = _load_local_background(content_cfg["backgrounds_dir"], width, height)
    elif src == "picsum":
        bg = _download_background(content_cfg["background_fallback_url"], width, height)
    if bg is None:
        bg = _generate_background(width, height)

    # Random position for the puzzle piece (ensure it's not too close to edges)
    pad = piece_size + knob_r + edge_margin
    target_x = random.randint(pad + left_start_offset, width - pad)
    target_y = random.randint(knob_r + edge_margin, height - piece_size - edge_margin)

    # --- Cut puzzle piece from background ---
    total_w = piece_size + knob_r  # extra width for right knob
    total_h = piece_size + knob_r  # extra height (knob extends above)
    del total_h  # noqa: F841 — kept to document layout intent

    piece_strip = bg.crop((target_x - 2, 0, target_x - 2 + total_w + 4, height))
    piece_img = piece_strip.convert("RGBA")

    # Antialiased puzzle-shape mask via supersampling
    mask_hi = Image.new("L", ((total_w + 4) * aa, height * aa), 0)
    mask_hi_draw = ImageDraw.Draw(mask_hi)
    path = _draw_puzzle_path(2 * aa, (target_y - knob_r) * aa, piece_size * aa, knob_r * aa)
    mask_hi_draw.polygon(path, fill=255)
    mask = mask_hi.resize((total_w + 4, height), Image.LANCZOS)
    piece_img.putalpha(mask)

    # --- Draw gap on background ---
    overlay_hi = Image.new("RGBA", (width * aa, height * aa), (0, 0, 0, 0))
    overlay_hi_draw = ImageDraw.Draw(overlay_hi)
    gap_path_hi = _draw_puzzle_path(target_x * aa, (target_y - knob_r) * aa, piece_size * aa, knob_r * aa)
    overlay_hi_draw.polygon(gap_path_hi, fill=(0, 0, 0, 100))
    overlay = overlay_hi.resize((width, height), Image.LANCZOS)
    bg_with_gap = Image.alpha_composite(bg.convert("RGBA"), overlay)
    bg_final = bg_with_gap.convert("RGB")

    # Antialiased white border around gap
    border_hi = Image.new("RGBA", (width * aa, height * aa), (0, 0, 0, 0))
    border_hi_draw = ImageDraw.Draw(border_hi)
    border_hi_draw.polygon(gap_path_hi, outline=(255, 255, 255), width=aa)
    border = border_hi.resize((width, height), Image.LANCZOS)
    bg_final = Image.alpha_composite(bg_final.convert("RGBA"), border).convert("RGB")

    return {
        "bg_image": bg_final,
        "piece_image": piece_img,
        "target_x": target_x,
        "target_y": target_y,
        "tolerance_px": tolerance_px,
        "render_cfg": render_cfg,
    }


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

def load_challenge() -> dict:
    """Load a challenge for the current MODE / TEST_ID.

    Two layouts are supported:
      * `assets/slider/<MODE>.json` — config-driven generative challenge
      * `assets/slider/<MODE>/<TEST_ID:03d>/` — static pre-rendered challenge
        with `bg.png`, `piece.png`, `meta.json` (future Halligan-style import)
    """
    mode_file = ASSETS / f"{MODE}.json"
    if mode_file.exists():
        cfg = json.loads(mode_file.read_text())
        return generate_slider_challenge(cfg["content"], cfg["render"])

    # Static asset path
    chal_dir = ASSETS / MODE / f"{TEST_ID:03d}"
    meta = json.loads((chal_dir / "meta.json").read_text())
    bg_image = Image.open(chal_dir / "bg.png")
    piece_image = Image.open(chal_dir / "piece.png")
    return {
        "bg_image": bg_image,
        "piece_image": piece_image,
        "target_x": meta["target_x"],
        "target_y": meta.get("target_y", 0),
        "tolerance_px": meta.get("tolerance_px", 10),
        "render_cfg": meta.get("render_cfg", {
            "puzzle_size": list(bg_image.size),
            "track_height": meta.get("track_height", 44),
        }),
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

_challenge = load_challenge()
_BG_IMAGE: Image.Image = _challenge["bg_image"]
_PIECE_IMAGE: Image.Image = _challenge["piece_image"]
TARGET_X: int = _challenge["target_x"]
TOLERANCE: int = _challenge["tolerance_px"]
_RENDER_CFG: dict = _challenge["render_cfg"]
BG_WIDTH, BG_HEIGHT = _RENDER_CFG["puzzle_size"]
TRACK_HEIGHT = _RENDER_CFG.get("track_height", 44)

# Pre-encode to PNG so the serve handlers stay cheap
def _encode(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


BG_PNG: bytes = _encode(_BG_IMAGE)
PIECE_PNG: bytes = _encode(_PIECE_IMAGE)


@app.route("/")
def index():
    """Main page: background image with gap + draggable puzzle piece."""
    return f"""<!DOCTYPE html>
<html>
<head><title>Slider CAPTCHA</title></head>
<body style="display:flex; justify-content:center; align-items:center;
             min-height:100vh; margin:0; background:#f5f5f5;
             font-family:Arial,sans-serif; user-select:none;">
  <div style="text-align:center; background:white; padding:30px;
              border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.1);">
    <h2>Slider CAPTCHA</h2>
    <p>Drag the puzzle piece to fill the gap</p>

    <div id="captcha-container" style="position:relative; width:{BG_WIDTH}px;
                height:{BG_HEIGHT}px; margin:0 auto; overflow:hidden;
                border:1px solid #ccc; border-radius:4px;">
      <img id="bg" src="/bg.png" style="position:absolute; top:0; left:0;
                  width:{BG_WIDTH}px; height:{BG_HEIGHT}px;" draggable="false"/>
      <img id="piece" src="/piece.png" style="position:absolute; top:0; left:0;
                  height:{BG_HEIGHT}px; cursor:grab;" draggable="false"/>
    </div>

    <div id="slider-track" style="position:relative; width:{BG_WIDTH}px; height:{TRACK_HEIGHT}px;
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
                  text-align:center; line-height:{TRACK_HEIGHT}px; color:#999; pointer-events:none;">
        Drag to complete puzzle</span>
    </div>

    <div id="result-msg" style="margin-top:12px; font-size:18px; min-height:28px;"></div>
  </div>

<script>
(function() {{
  var handle = document.getElementById('slider-handle');
  var fill = document.getElementById('slider-fill');
  var piece = document.getElementById('piece');
  var track = document.getElementById('slider-track');
  var msg = document.getElementById('result-msg');
  var text = document.getElementById('slider-text');

  var trackWidth = {BG_WIDTH};
  var handleWidth = 40;
  var maxMove = trackWidth - handleWidth;
  var originX = 0;
  var isDragging = false;
  var trail = [];

  function onStart(e) {{
    isDragging = true;
    originX = (e.clientX || e.touches[0].clientX);
    trail = [];
    handle.style.cursor = 'grabbing';
    text.style.display = 'none';
  }}

  function onMove(e) {{
    if (!isDragging) return;
    var clientX = (e.clientX || e.touches[0].clientX);
    var clientY = (e.clientY || e.touches[0].clientY);
    var moveX = clientX - originX;
    if (moveX < 0) moveX = 0;
    if (moveX > maxMove) moveX = maxMove;

    handle.style.left = moveX + 'px';
    fill.style.width = (moveX + handleWidth/2) + 'px';
    piece.style.left = moveX + 'px';
    trail.push(Math.round(clientY));
  }}

  function onEnd(e) {{
    if (!isDragging) return;
    isDragging = false;
    handle.style.cursor = 'grab';

    var left = parseInt(handle.style.left) || 0;

    // POST to server for verification
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
          piece.style.left = '0px';
          msg.innerHTML = '';
          text.style.display = '';
        }}, 1000);
      }}
    }};
    xhr.send(JSON.stringify({{x: left, trail: trail}}));
  }}

  handle.addEventListener('mousedown', onStart);
  handle.addEventListener('touchstart', onStart);
  document.addEventListener('mousemove', onMove);
  document.addEventListener('touchmove', onMove);
  document.addEventListener('mouseup', onEnd);
  document.addEventListener('touchend', onEnd);
}})();
</script>
</body>
</html>"""


@app.route("/bg.png")
def bg_image():
    """Serve background image with gap."""
    return send_file(io.BytesIO(BG_PNG), mimetype="image/png")


@app.route("/piece.png")
def piece_image():
    """Serve the puzzle piece image."""
    return send_file(io.BytesIO(PIECE_PNG), mimetype="image/png")


# Once a correct drag happens, lock the result so subsequent wrong drags
# don't overwrite it. Mirrors the UX of text/math/icon_click/rotation which
# redirect to a success page after correct submission (so the agent can't
# accidentally "un-win" by re-submitting). Without this, slider agents often
# land within tolerance on an early turn but lose the reward on a later wrong
# drag, since /tmp/captcha_result.json gets overwritten each /verify.
_LOCKED_CORRECT = False


@app.route("/verify", methods=["POST"])
def verify():
    """Verify slider position. Writes result to file for evaluate_final_fn."""
    global _LOCKED_CORRECT
    data = request.get_json(force=True)
    user_x = data.get("x", -9999)
    this_correct = abs(user_x - TARGET_X) <= TOLERANCE
    if this_correct:
        _LOCKED_CORRECT = True

    result = {
        "submitted": True,
        "correct": _LOCKED_CORRECT,
        "user_x": user_x,
        "target_x": TARGET_X,
        "distance": abs(user_x - TARGET_X),
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
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
