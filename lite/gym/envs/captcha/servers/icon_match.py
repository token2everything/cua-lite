"""Icon-match CAPTCHA server.

Standalone Flask app that generates an icon-match CAPTCHA: a canvas with
PIL-rendered icons scattered on it. Two of the icons are visually
identical (same shape + same color); the rest are distractors. The user
must drag one of the matching icons onto the other. Verification checks
that the source is part of the pair AND the drop position falls within
the configured pixel tolerance of the OTHER pair member's center.

All vocabulary (shapes, colors) and render parameters (canvas, icon size,
icon count, drop tolerance, grid) are loaded from
`assets/icon_match/{MODE}.json`.

    pip install flask pillow
    python lite/gym/envs/captcha/servers/icon_match.py
    # Open http://localhost:5000 in a browser

Environment variables:
  CAPTCHA_MODE         : asset stem under assets/icon_match/   (default: train_eval)
                         e.g. "train_eval", "test/crowded"
  CAPTCHA_TEST_ID      : integer challenge id when MODE points to a directory
                         of pre-rendered images (default: 0)
  CAPTCHA_SEED         : seed Python's `random` for reproducible challenges
  CAPTCHA_RESULT_FILE  : where /submit writes the JSON verdict
                         (default: /tmp/captcha_result.json)
  HOST, PORT           : Flask bind address                     (default: 127.0.0.1:5000)
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import random
from pathlib import Path

from flask import Flask, request, send_file

from PIL import Image, ImageDraw

# Deterministic mode: if CAPTCHA_SEED is set, seed random before any generation
# so repeated runs produce the same challenge (used by eval task variants).
_seed = os.environ.get("CAPTCHA_SEED")
if _seed is not None:
    random.seed(int(_seed))

# ---------------------------------------------------------------------------
# Asset / mode dispatch
# ---------------------------------------------------------------------------

ASSETS = Path(__file__).parent.parent / ".cache" / "assets" / "icon_match"
MODE = os.environ.get("CAPTCHA_MODE", "train_eval")
TEST_ID = int(os.environ.get("CAPTCHA_TEST_ID", "0"))
RESULT_FILE = os.environ.get("CAPTCHA_RESULT_FILE", "/tmp/captcha_result.json")


# ---------------------------------------------------------------------------
# Icon rendering
# ---------------------------------------------------------------------------


def _shape_points(shape: str, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """Return polygon vertex list for a given shape centered at (cx, cy)."""
    if shape == "square":
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if shape == "triangle":
        # Equilateral-ish triangle pointing up
        return [
            (cx, cy - r),
            (cx + r * math.cos(math.radians(30)), cy + r * math.sin(math.radians(30))),
            (cx - r * math.cos(math.radians(30)), cy + r * math.sin(math.radians(30))),
        ]
    if shape == "diamond":
        return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    if shape == "star":
        pts = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            rad = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rad * math.cos(angle), cy + rad * math.sin(angle)))
        return pts
    # circle handled separately by caller
    return []


def render_icon(
    shape: str,
    color: tuple[int, int, int],
    size: int,
    aa_scale: int = 4,
) -> Image.Image:
    """Render a single shape+color icon with antialiased edges. Returns RGBA Image."""
    s = aa_scale
    hi = Image.new("RGBA", (size * s, size * s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hi)

    cx = size * s / 2
    cy = size * s / 2
    r = (size * s) / 2 - 4 * s  # leave room for outline
    outline = (40, 40, 40, 255)
    outline_w = max(1, s)

    if shape == "circle":
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=color + (255,),
            outline=outline,
            width=outline_w,
        )
    else:
        pts = _shape_points(shape, cx, cy, r)
        draw.polygon(pts, fill=color + (255,), outline=outline)
        # Pillow's polygon outline width is 1 — re-stroke for thickness
        draw.line(pts + [pts[0]], fill=outline, width=outline_w)

    return hi.resize((size, size), Image.LANCZOS)


def _icon_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Challenge generation (pure: no globals, no magic numbers)
# ---------------------------------------------------------------------------


def _grid_positions(
    count: int,
    canvas_w: int,
    canvas_h: int,
    icon_size: int,
    grid_cols: int,
    grid_rows: int,
    margin: int,
) -> list[tuple[int, int]]:
    """Pick `count` jittered grid cell centers, non-overlapping by construction."""
    if count > grid_cols * grid_rows:
        raise RuntimeError(
            f"n_icons={count} exceeds grid capacity {grid_cols}x{grid_rows}"
            f"={grid_cols * grid_rows}"
        )

    usable_w = canvas_w - 2 * margin
    usable_h = canvas_h - 2 * margin
    cell_w = usable_w // grid_cols
    cell_h = usable_h // grid_rows

    cells = [(c, r) for r in range(grid_rows) for c in range(grid_cols)]
    chosen = random.sample(cells, count)

    # Jitter must keep icon fully inside its cell so icons don't overlap
    half = icon_size // 2
    jitter_x = max(0, cell_w // 2 - half - 4)
    jitter_y = max(0, cell_h // 2 - half - 4)

    positions = []
    for c, r in chosen:
        cx = margin + c * cell_w + cell_w // 2 + random.randint(-jitter_x, jitter_x)
        cy = margin + r * cell_h + cell_h // 2 + random.randint(-jitter_y, jitter_y)
        positions.append((cx, cy))
    return positions


def generate_icon_match_challenge(
    shapes: list[str],
    colors: dict[str, tuple[int, int, int]],
    render_cfg: dict,
) -> dict:
    """Build a full icon-match challenge from a vocabulary + render config.

    Args:
        shapes: list of shape names (must each be supported by `render_icon`).
        colors: {color_name: (r, g, b)} — RGB tuples.
        render_cfg: dict with keys
            canvas_size: [w, h]
            icon_size: int
            n_icons_range: [min, max] inclusive
            tolerance_px: int
            grid_cols, grid_rows, margin_px: int
            aa_scale: int (optional, default 4)

    Returns:
        Challenge dict with keys:
            icons:       list of {id, shape, color_name, color, x, y, png_b64, is_pair}
            pair_ids:    (int, int)
            canvas:      (w, h)
            icon_size:   int
            tolerance:   int
    """
    canvas_w, canvas_h = render_cfg["canvas_size"]
    icon_size = render_cfg["icon_size"]
    n_min, n_max = render_cfg["n_icons_range"]
    tolerance = render_cfg["tolerance_px"]
    grid_cols = render_cfg["grid_cols"]
    grid_rows = render_cfg["grid_rows"]
    margin = render_cfg["margin_px"]
    aa_scale = render_cfg.get("aa_scale", 4)

    color_items = list(colors.items())  # [(name, (r,g,b)), ...]

    num_icons = random.randint(n_min, n_max)
    positions = _grid_positions(
        num_icons, canvas_w, canvas_h, icon_size, grid_cols, grid_rows, margin,
    )
    random.shuffle(positions)

    # Pick the matching pair's (shape, color)
    pair_shape = random.choice(shapes)
    pair_color_name, pair_color = random.choice(color_items)

    # Distractor (shape, color_name, color) combos — must differ from the pair
    all_combos = [
        (sh, cn, co)
        for sh in shapes
        for cn, co in color_items
        if (sh, cn) != (pair_shape, pair_color_name)
    ]
    distractor_combos = random.sample(all_combos, num_icons - 2)

    combos: list[tuple[str, str, tuple[int, int, int]]] = [
        (pair_shape, pair_color_name, pair_color),
        (pair_shape, pair_color_name, pair_color),
    ] + distractor_combos
    random.shuffle(combos)

    icons: list[dict] = []
    pair_ids: list[int] = []
    for i, ((shape, color_name, color), (x, y)) in enumerate(zip(combos, positions)):
        img = render_icon(shape, color, icon_size, aa_scale=aa_scale)
        png_b64 = base64.b64encode(_icon_to_png_bytes(img)).decode("ascii")
        is_pair = (shape, color_name) == (pair_shape, pair_color_name)
        icons.append({
            "id": i,
            "shape": shape,
            "color_name": color_name,
            "color": list(color),
            "x": x,
            "y": y,
            "png_b64": png_b64,
            "is_pair": is_pair,
        })
        if is_pair:
            pair_ids.append(i)

    assert len(pair_ids) == 2, f"Expected exactly 2 pair members, got {len(pair_ids)}"
    return {
        "icons": icons,
        "pair_ids": (pair_ids[0], pair_ids[1]),
        "canvas": (canvas_w, canvas_h),
        "icon_size": icon_size,
        "tolerance": tolerance,
    }


def load_challenge() -> dict:
    """Resolve CAPTCHA_MODE/TEST_ID into a challenge dict.

    Two layouts are supported:
      * `assets/icon_match/<MODE>.json` — generative config (this is the
        only layout used today; pre-rendered tests are a future addition).
      * `assets/icon_match/<MODE>/<TEST_ID:03d>/image.png` — pre-rendered
        challenge directory (placeholder branch for parity with other
        captcha servers).
    """
    mode_file = ASSETS / f"{MODE}.json"
    if mode_file.exists():
        cfg = json.loads(mode_file.read_text())
        # JSON colors are lists of ints; convert to tuples for PIL
        colors = {name: tuple(rgb) for name, rgb in cfg["content"]["colors"].items()}
        shapes = cfg["content"]["shapes"]
        return generate_icon_match_challenge(shapes, colors, cfg["render"])

    chal_dir = ASSETS / MODE / f"{TEST_ID:03d}"
    raise NotImplementedError(
        f"Pre-rendered challenge layout not implemented yet (looked for {chal_dir})."
    )


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

CHALLENGE = load_challenge()
ICONS: list[dict] = CHALLENGE["icons"]
PAIR_IDS: tuple[int, int] = CHALLENGE["pair_ids"]
CANVAS_WIDTH, CANVAS_HEIGHT = CHALLENGE["canvas"]
ICON_SIZE = CHALLENGE["icon_size"]
TOLERANCE = CHALLENGE["tolerance"]

# Lock the result on first correct submission so the agent can't "un-win" with
# a subsequent stray drop. Mirrors slider.py's _LOCKED_CORRECT pattern.
_LOCKED_CORRECT = False


@app.route("/")
def index():
    """Main page: canvas with icons + drag-to-match interaction."""
    # Build the icon <img> elements. Each icon is absolutely positioned at its
    # center (x, y), with the icon image translated by -ICON_SIZE/2 via offset.
    half = ICON_SIZE // 2
    icon_html_parts = []
    for ic in ICONS:
        left = ic["x"] - half
        top = ic["y"] - half
        icon_html_parts.append(
            f'<img class="icon" data-id="{ic["id"]}" '
            f'src="data:image/png;base64,{ic["png_b64"]}" '
            f'style="position:absolute; left:{left}px; top:{top}px; '
            f'width:{ICON_SIZE}px; height:{ICON_SIZE}px; cursor:grab;" '
            f'draggable="false" />'
        )
    icons_html = "\n      ".join(icon_html_parts)

    return f"""<!DOCTYPE html>
<html>
<head><title>Icon Match CAPTCHA</title></head>
<body style="display:flex; justify-content:center; align-items:center;
             min-height:100vh; margin:0; background:#f5f5f5;
             font-family:Arial,sans-serif; user-select:none;">
  <div style="text-align:center; background:white; padding:30px;
              border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.1);">
    <h2>Icon Match CAPTCHA</h2>
    <p>Find the two identical icons and <b>drag one onto the other</b></p>

    <div id="captcha-container" style="position:relative; width:{CANVAS_WIDTH}px;
                height:{CANVAS_HEIGHT}px; margin:0 auto; overflow:hidden;
                border:1px solid #ccc; border-radius:4px; background:#fafafa;">
      {icons_html}
    </div>

    <div id="result-msg" style="margin-top:12px; font-size:18px; min-height:28px;"></div>
  </div>

<script>
(function() {{
  var container = document.getElementById('captcha-container');
  var msg = document.getElementById('result-msg');
  var icons = container.querySelectorAll('.icon');

  var dragging = null;        // the icon img being dragged
  var dragId = null;          // its data-id (int)
  var ghost = null;           // floating clone shown at cursor
  var submitted = false;

  function onMouseDown(e) {{
    if (submitted) return;
    e.preventDefault();
    dragging = e.currentTarget;
    dragId = parseInt(dragging.getAttribute('data-id'));

    // Create a floating ghost clone at cursor
    ghost = dragging.cloneNode(true);
    ghost.style.position = 'fixed';
    ghost.style.pointerEvents = 'none';
    ghost.style.opacity = '0.85';
    ghost.style.zIndex = '9999';
    moveGhost(e.clientX, e.clientY);
    document.body.appendChild(ghost);

    dragging.style.opacity = '0.3';
    dragging.style.cursor = 'grabbing';
  }}

  function moveGhost(clientX, clientY) {{
    if (!ghost) return;
    ghost.style.left = (clientX - {half}) + 'px';
    ghost.style.top = (clientY - {half}) + 'px';
  }}

  function onMouseMove(e) {{
    if (!dragging) return;
    moveGhost(e.clientX, e.clientY);
  }}

  function onMouseUp(e) {{
    if (!dragging || submitted) return;
    submitted = true;

    // Compute drop position relative to the canvas container
    var rect = container.getBoundingClientRect();
    var dropX = Math.round(e.clientX - rect.left);
    var dropY = Math.round(e.clientY - rect.top);
    var sourceId = dragId;

    // Cleanup ghost + restore source styling
    if (ghost) {{ document.body.removeChild(ghost); ghost = null; }}
    dragging.style.opacity = '1.0';
    dragging.style.cursor = 'grab';
    dragging = null;
    dragId = null;

    // Disable further interaction
    icons.forEach(function(ic) {{
      ic.style.cursor = 'default';
      ic.style.pointerEvents = 'none';
    }});

    // POST to server
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/submit', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {{
      var resp = JSON.parse(xhr.responseText);
      if (resp.correct) {{
        msg.innerHTML = '<span style="color:#4CAF50; font-weight:bold;">' +
          '&#10003; Verified!</span>';
      }} else {{
        msg.innerHTML = '<span style="color:#f44336; font-weight:bold;">' +
          '&#10007; Incorrect, please reload</span>';
      }}
    }};
    xhr.send(JSON.stringify({{
      source_id: sourceId, drop_x: dropX, drop_y: dropY,
    }}));
  }}

  icons.forEach(function(ic) {{
    ic.addEventListener('mousedown', onMouseDown);
  }});
  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('mouseup', onMouseUp);
}})();
</script>
</body>
</html>"""


@app.route("/icon/<int:icon_id>")
def icon_image(icon_id: int):
    """Serve an individual icon PNG (alternate path; HTML uses inline base64)."""
    ic = ICONS[icon_id]
    png = base64.b64decode(ic["png_b64"])
    return send_file(io.BytesIO(png), mimetype="image/png")


@app.route("/submit", methods=["POST"])
def submit():
    """Verify drag drop. Source must be a pair member; drop must land within
    TOLERANCE px of the OTHER pair member's center."""
    global _LOCKED_CORRECT
    data = request.get_json(force=True)
    source_id = data.get("source_id", -1)
    drop_x = data.get("drop_x", -9999)
    drop_y = data.get("drop_y", -9999)

    source_in_pair = source_id in PAIR_IDS
    target_id = PAIR_IDS[1] if source_id == PAIR_IDS[0] else PAIR_IDS[0]
    target = ICONS[target_id]
    dist = math.sqrt((drop_x - target["x"]) ** 2 + (drop_y - target["y"]) ** 2)
    this_correct = source_in_pair and dist <= TOLERANCE
    if this_correct:
        _LOCKED_CORRECT = True

    result = {
        "submitted": True,
        "correct": _LOCKED_CORRECT,
        "source_id": source_id,
        "source_in_pair": source_in_pair,
        "drop_x": drop_x,
        "drop_y": drop_y,
        "target_x": target["x"],
        "target_y": target["y"],
        "distance": round(dist, 2),
        "tolerance": TOLERANCE,
        "pair_ids": list(PAIR_IDS),
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
