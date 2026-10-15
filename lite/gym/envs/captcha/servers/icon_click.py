"""Icon-click CAPTCHA server (assets-driven).

Standalone Flask app that loads challenge config from
`assets/icon_click/{MODE}.json` (generative mode) or
`assets/icon_click/{MODE}/{TEST_ID:03d}/{image.png,meta.json}` (static mode),
renders a background with various icons (animals, objects) scattered on it,
and prompts the user to click all icons of a specific category
(e.g., "Click all the cats"). Verification checks that the user clicked all
target icons and no distractors, within a pixel tolerance.

Icons are sourced from Twemoji (Twitter's open-source emoji set, CC-BY 4.0).

    pip install flask pillow
    python lite/gym/envs/captcha/servers/icon_click.py
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
import urllib.request
from pathlib import Path

from flask import Flask, request, send_file

from PIL import Image, ImageDraw

# Deterministic mode: if CAPTCHA_SEED is set, seed random before any generation
# so repeated runs produce the same challenge (used by eval task variants).
_seed = os.environ.get("CAPTCHA_SEED")
if _seed is not None:
    random.seed(int(_seed))

ASSETS = Path(__file__).parent.parent / ".cache" / "assets" / "icon_click"
MODE = os.environ.get("CAPTCHA_MODE", "train_eval")
TEST_ID = int(os.environ.get("CAPTCHA_TEST_ID", "0"))
RESULT_FILE = os.environ.get("CAPTCHA_RESULT_FILE", "/tmp/captcha_result.json")

# Cache for downloaded emoji images: (codepoint, size) → PIL Image (RGBA)
_emoji_cache: dict[tuple[str, int], Image.Image] = {}


# ---------------------------------------------------------------------------
# Twemoji icon loading
# ---------------------------------------------------------------------------

def _download_emoji(codepoint: str, size: int, url_template: str) -> Image.Image | None:
    """Download a single Twemoji PNG and return as RGBA PIL Image."""
    key = (codepoint, size)
    if key in _emoji_cache:
        return _emoji_cache[key]
    try:
        url = url_template.format(codepoint=codepoint)
        req = urllib.request.Request(url, headers={"User-Agent": "IconClickCaptcha/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        img = img.resize((size, size), Image.LANCZOS)
        _emoji_cache[key] = img
        return img
    except Exception as e:
        print(f"[ICON CLICK CAPTCHA] Failed to download emoji {codepoint}: {e}")
        return None


def _preload_all_icons(
    categories: dict[str, tuple[str, list[str]]], size: int, url_template: str,
) -> None:
    """Download all emoji icons at startup."""
    for _, codepoints in categories.values():
        for cp in codepoints:
            _download_emoji(cp, size, url_template)


# ---------------------------------------------------------------------------
# Fallback icon drawing (used when emoji download fails)
# ---------------------------------------------------------------------------

def _draw_fallback_icon(
    draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, category: str,
) -> None:
    """Draw a simple geometric icon as fallback."""
    r = s // 2 - 2
    colors = {
        "cat": (255, 165, 0), "dog": (180, 130, 70), "fish": (70, 150, 220),
        "bird": (220, 60, 60), "star": (255, 210, 0), "heart": (220, 40, 60),
        "tree": (50, 160, 50), "flower": (220, 100, 220),
    }
    color = colors.get(category, (150, 150, 150))

    if category == "star":
        # 5-pointed star
        points = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            rad = r if i % 2 == 0 else r * 0.4
            points.append((cx + rad * math.cos(angle), cy + rad * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=1)
    elif category == "heart":
        draw.ellipse([cx - r, cy - r // 2, cx, cy + r // 2], fill=color)
        draw.ellipse([cx, cy - r // 2, cx + r, cy + r // 2], fill=color)
        draw.polygon([(cx - r, cy), (cx + r, cy), (cx, cy + r + 2)], fill=color)
    else:
        # Generic labeled circle
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(0, 0, 0), width=2)
        # Draw first letter
        draw.text((cx - 5, cy - 6), category[0].upper(), fill=(255, 255, 255))


# ---------------------------------------------------------------------------
# Background generation
# ---------------------------------------------------------------------------

def _download_background(width: int, height: int, fallback_url: str) -> Image.Image | None:
    """Download a random photograph from picsum.photos.

    When CAPTCHA_SEED is set, use picsum's seeded endpoint for reproducibility.
    """
    try:
        seed = _seed if _seed is not None else random.randint(0, 2**31 - 1)
        url = fallback_url.format(seed=seed, w=width, h=height)
        req = urllib.request.Request(url, headers={"User-Agent": "IconClickCaptcha/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        # Lighten the background so icons stand out
        overlay = Image.new("RGBA", (width, height), (255, 255, 255, 100))
        return Image.alpha_composite(img, overlay)
    except Exception as e:
        print(f"[ICON CLICK CAPTCHA] Failed to download background: {e}")
        return None


def _load_local_background(bg_dir: str, width: int, height: int) -> Image.Image | None:
    """Pick a random jpg from `bg_dir` (relative to assets/icon_click/) and
    open it. Resizes to (width, height) if needed and applies the same white
    overlay as the picsum path so icons stand out. Returns None on failure
    so callers can fall back to the synthetic generator.
    """
    try:
        root = ASSETS / bg_dir
        jpgs = sorted(root.glob("*.jpg"))
        if not jpgs:
            return None
        chosen = random.choice(jpgs)
        img = Image.open(chosen).convert("RGBA")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        overlay = Image.new("RGBA", (width, height), (255, 255, 255, 100))
        return Image.alpha_composite(img, overlay)
    except Exception as e:
        print(f"[ICON CLICK CAPTCHA] Failed to load local background: {e}")
        return None


def _generate_background(width: int, height: int) -> Image.Image:
    """Generate a light patterned background."""
    img = Image.new("RGBA", (width, height), (240, 240, 240, 255))
    draw = ImageDraw.Draw(img)

    # Subtle grid pattern
    for x in range(0, width, 20):
        draw.line([(x, 0), (x, height)], fill=(220, 220, 220, 255), width=1)
    for y in range(0, height, 20):
        draw.line([(0, y), (width, y)], fill=(220, 220, 220, 255), width=1)

    # Random soft colored rectangles
    for _ in range(6):
        x1 = random.randint(0, width - 60)
        y1 = random.randint(0, height - 60)
        x2 = x1 + random.randint(30, 80)
        y2 = y1 + random.randint(30, 60)
        color = (random.randint(200, 240), random.randint(200, 240),
                 random.randint(200, 240), 180)
        draw.rectangle([x1, y1, x2, y2], fill=color)

    return img


# ---------------------------------------------------------------------------
# CAPTCHA generation (pure renderer, all params from config)
# ---------------------------------------------------------------------------

def _grid_positions(
    width: int, height: int, cols: int, rows: int, margin: int, count: int, jitter: int,
) -> list[tuple[int, int]]:
    """Pick `count` random grid cell centers from a cols x rows grid."""
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    cell_w = usable_w // cols
    cell_h = usable_h // rows

    cells = [(c, r) for r in range(rows) for c in range(cols)]
    chosen = random.sample(cells, min(count, len(cells)))

    positions = []
    for c, r in chosen:
        cx = margin + c * cell_w + cell_w // 2 + random.randint(-jitter, jitter)
        cy = margin + r * cell_h + cell_h // 2 + random.randint(-jitter, jitter)
        positions.append((cx, cy))

    return positions


def _paste_icon(
    bg: Image.Image, category: str, codepoints: list[str],
    cx: int, cy: int, icon_size: int, url_template: str,
) -> None:
    """Paste an emoji icon onto the background at (cx, cy). Falls back to drawing."""
    cp = random.choice(codepoints)
    icon = _download_emoji(cp, icon_size, url_template)
    if icon is not None:
        x = cx - icon_size // 2
        y = cy - icon_size // 2
        bg.paste(icon, (x, y), icon)
    else:
        draw = ImageDraw.Draw(bg)
        _draw_fallback_icon(draw, cx, cy, icon_size, category)


def generate_icon_click_challenge(
    categories: dict[str, tuple[str, list[str]]],
    content_cfg: dict,
    render_cfg: dict,
) -> dict:
    """Generate an icon-click CAPTCHA.

    Returns a dict with keys: image (PIL Image), target_plural (str),
    target_positions (list[tuple[int, int]]), tolerance_px (int),
    canvas_size (tuple[int, int]).
    """
    width, height = render_cfg["canvas_size"]
    icon_size = render_cfg["icon_size"]
    tolerance_px = render_cfg["tolerance_px"]
    grid_cols = render_cfg["grid_cols"]
    grid_rows = render_cfg["grid_rows"]
    margin = render_cfg["margin"]
    jitter = render_cfg["jitter_px"]
    icon_lo, icon_hi = render_cfg["icon_count_range"]
    tgt_lo, tgt_hi = render_cfg["target_count_range"]
    fallback_url = content_cfg["background_fallback_url"]
    url_template = content_cfg["icon_url_template"]
    bg_source = content_cfg.get("background_source", "picsum")

    total_icons = random.randint(icon_lo, icon_hi)
    num_targets = random.randint(tgt_lo, tgt_hi)
    num_targets = min(num_targets, total_icons)
    num_distractors = total_icons - num_targets

    bg = None
    if bg_source == "local":
        bg = _load_local_background(content_cfg["backgrounds_dir"], width, height)
    elif bg_source == "picsum":
        bg = _download_background(width, height, fallback_url)
    if bg is None:
        bg = _generate_background(width, height)

    # Pick target category
    target_key = random.choice(list(categories.keys()))
    target_plural, target_codepoints = categories[target_key]

    # Pick distractor categories (exclude target)
    distractor_keys = [k for k in categories if k != target_key]

    positions = _grid_positions(width, height, grid_cols, grid_rows, margin, total_icons, jitter)
    random.shuffle(positions)

    target_positions = positions[:num_targets]
    distractor_positions = positions[num_targets:]

    # Paste distractor icons first
    for cx, cy in distractor_positions:
        cat = random.choice(distractor_keys)
        _paste_icon(bg, cat, categories[cat][1], cx, cy, icon_size, url_template)

    # Paste target icons
    for cx, cy in target_positions:
        _paste_icon(bg, target_key, target_codepoints, cx, cy, icon_size, url_template)

    return {
        "image": bg.convert("RGB"),
        "target_plural": target_plural,
        "target_positions": target_positions,
        "tolerance_px": tolerance_px,
        "canvas_size": (width, height),
    }


def load_challenge() -> dict:
    """Mode dispatch: generative (.json) vs static (NNN/image.png + meta.json)."""
    mode_file = ASSETS / f"{MODE}.json"
    if mode_file.exists():
        cfg = json.loads(mode_file.read_text())
        # JSON stores categories as 2-element lists; reconstruct as tuples.
        categories = {k: (v[0], v[1]) for k, v in cfg["content"]["categories"].items()}
        # Preload all icons at startup so render is fast.
        print("[ICON CLICK CAPTCHA] Downloading emoji icons...")
        _preload_all_icons(categories, cfg["render"]["icon_size"], cfg["content"]["icon_url_template"])
        print("[ICON CLICK CAPTCHA] Icons ready.")
        return generate_icon_click_challenge(categories, cfg["content"], cfg["render"])

    # Static mode: load pre-rendered image + meta
    chal_dir = ASSETS / MODE / f"{TEST_ID:03d}"
    img = Image.open(chal_dir / "image.png").convert("RGB")
    meta = json.loads((chal_dir / "meta.json").read_text())
    return {
        "image": img,
        "target_plural": meta["target_plural"],
        "target_positions": [tuple(p) for p in meta["target_positions"]],
        "tolerance_px": meta["tolerance_px"],
        "canvas_size": tuple(meta["canvas_size"]),
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Load challenge at startup
_CHALLENGE = load_challenge()
TARGET_PLURAL: str = _CHALLENGE["target_plural"]
TARGET_POSITIONS: list[tuple[int, int]] = _CHALLENGE["target_positions"]
TOLERANCE: int = _CHALLENGE["tolerance_px"]
BG_WIDTH, BG_HEIGHT = _CHALLENGE["canvas_size"]

_buf = io.BytesIO()
_CHALLENGE["image"].save(_buf, "PNG")
CAPTCHA_PNG = _buf.getvalue()



@app.route("/")
def index():
    """Main page: image with icons + click interface."""
    return f"""<!DOCTYPE html>
<html>
<head><title>Icon Click CAPTCHA</title></head>
<body style="display:flex; justify-content:center; align-items:center;
             min-height:100vh; margin:0; background:#f5f5f5;
             font-family:Arial,sans-serif; user-select:none;">
  <div style="text-align:center; background:white; padding:30px;
              border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.1);">
    <h2>Icon Click CAPTCHA</h2>
    <p>Click all the <b>{TARGET_PLURAL}</b> in the image below</p>

    <div id="captcha-container" style="position:relative; width:{BG_WIDTH}px;
                height:{BG_HEIGHT}px; margin:0 auto; overflow:hidden;
                border:1px solid #ccc; border-radius:4px; cursor:crosshair;">
      <img id="captcha-img" src="/captcha.png"
           style="width:{BG_WIDTH}px; height:{BG_HEIGHT}px; display:block;"
           draggable="false"/>
    </div>

    <div style="margin-top:10px;">
      <button id="undo-btn" onclick="undoClick()"
              style="font-size:16px; padding:8px 20px; background:#ff9800;
                     color:white; border:none; border-radius:4px;
                     cursor:pointer; margin-right:10px;">
        Undo
      </button>
      <button id="submit-btn" onclick="submitAnswer()"
              style="font-size:18px; padding:10px 32px; background:#4CAF50;
                     color:white; border:none; border-radius:4px;
                     cursor:pointer;">
        Submit
      </button>
    </div>

    <div id="result-msg" style="margin-top:12px; font-size:18px; min-height:28px;"></div>
  </div>

<script>
(function() {{
  var container = document.getElementById('captcha-container');
  var clicks = [];
  var markers = [];

  container.addEventListener('click', function(e) {{
    var rect = container.getBoundingClientRect();
    var x = Math.round(e.clientX - rect.left);
    var y = Math.round(e.clientY - rect.top);
    clicks.push({{x: x, y: y}});

    // Draw numbered marker
    var marker = document.createElement('div');
    marker.style.cssText = 'position:absolute; width:24px; height:24px; ' +
      'border-radius:50%; background:rgba(76,175,80,0.8); color:white; ' +
      'font-size:14px; font-weight:bold; text-align:center; line-height:24px; ' +
      'pointer-events:none; border:2px solid white; box-shadow:0 1px 3px rgba(0,0,0,0.3);';
    marker.style.left = (x - 12) + 'px';
    marker.style.top = (y - 12) + 'px';
    marker.textContent = clicks.length;
    container.appendChild(marker);
    markers.push(marker);
  }});

  window._getClicks = function() {{ return clicks; }};

  window.undoClick = function() {{
    if (markers.length > 0) {{
      var m = markers.pop();
      container.removeChild(m);
      clicks.pop();
    }}
  }};

  window.submitAnswer = function() {{
    var msg = document.getElementById('result-msg');
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/verify', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {{
      var resp = JSON.parse(xhr.responseText);
      if (resp.correct) {{
        msg.innerHTML = '<span style="color:#4CAF50; font-weight:bold;">&#10003; Correct!</span>';
      }} else {{
        msg.innerHTML = '<span style="color:#f44336; font-weight:bold;">&#10007; Wrong! Try again</span>';
        // Reset after 1 second
        setTimeout(function() {{
          while (markers.length > 0) {{
            container.removeChild(markers.pop());
          }}
          clicks.length = 0;
          msg.innerHTML = '';
        }}, 1000);
      }}
    }};
    xhr.send(JSON.stringify({{clicks: clicks}}));
  }};
}})();
</script>
</body>
</html>"""


@app.route("/captcha.png")
def captcha_image():
    """Serve the CAPTCHA image with icons."""
    return send_file(io.BytesIO(CAPTCHA_PNG), mimetype="image/png")


# Once a correct submit happens, lock the result so a subsequent wrong
# submit (different click set) doesn't overwrite it — the env reads
# /result only once at the terminal step. Mirrors slider.py.
_LOCKED_CORRECT = False


@app.route("/verify", methods=["POST"])
def verify():
    """Verify click positions against target icon positions.

    A click matches a target if it's within TOLERANCE pixels of the target center.
    The user must match ALL targets with no extra (unmatched) clicks.
    Click order does not matter.
    """
    global _LOCKED_CORRECT
    data = request.get_json(force=True)
    user_clicks = data.get("clicks", [])

    # Match each click to the nearest target
    matched_targets = set()
    unmatched_clicks = 0

    for click in user_clicks:
        cx, cy = click.get("x", -999), click.get("y", -999)
        best_dist = float("inf")
        best_idx = -1
        for i, (tx, ty) in enumerate(TARGET_POSITIONS):
            dist = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_dist <= TOLERANCE and best_idx not in matched_targets:
            matched_targets.add(best_idx)
        else:
            unmatched_clicks += 1

    correct = len(matched_targets) == len(TARGET_POSITIONS) and unmatched_clicks == 0
    if correct:
        _LOCKED_CORRECT = True

    result = {
        "submitted": True,
        "correct": _LOCKED_CORRECT,
        "matched": len(matched_targets),
        "total_targets": len(TARGET_POSITIONS),
        "unmatched_clicks": unmatched_clicks,
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
