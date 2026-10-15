"""Paged carousel CAPTCHA server.

Standalone Flask app that displays a carousel of N cards (count and content
controlled by an assets JSON), each showing a single PIL-rendered icon
(geometric shape + color). One card at a time is visible; navigation dots
and left/right arrows below the card let the user switch between cards.
A prompt at the top describes the unique target card (e.g. "Show the card
with a red triangle"); the user navigates to that card and clicks Submit.

Asset modes (selected via CAPTCHA_MODE):
  train_eval        -> assets/paged/train_eval.json   (default; 4-8 cards)
  test/many_cards   -> assets/paged/test/many_cards.json (OOD; 8-12 cards)

    pip install flask pillow
    python lite/gym/envs/captcha/servers/paged.py
    # Open http://localhost:5000 in a browser
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import random
from pathlib import Path

from flask import Flask, request

from PIL import Image, ImageDraw

# Deterministic mode: if CAPTCHA_SEED is set, seed random before any generation
# so repeated runs produce the same challenge (used by eval task variants).
_seed = os.environ.get("CAPTCHA_SEED")
if _seed is not None:
    random.seed(int(_seed))

# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------

ASSETS = Path(__file__).parent.parent / ".cache" / "assets" / "paged"
MODE = os.environ.get("CAPTCHA_MODE", "train_eval")
RESULT_FILE = os.environ.get("CAPTCHA_RESULT_FILE", "/tmp/captcha_result.json")


# ---------------------------------------------------------------------------
# Icon rendering
# ---------------------------------------------------------------------------

def _render_icon(
    shape: str,
    color_name: str,
    colors: dict[str, tuple[int, int, int]],
    size: int,
) -> bytes:
    """Render a single shape-color icon centered on a white card. Returns PNG bytes."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Subtle border so the card edge is visible against the page background.
    draw.rectangle([0, 0, size - 1, size - 1], outline=(220, 220, 220), width=1)

    color = colors[color_name]
    outline = (40, 40, 40)
    cx, cy = size // 2, size // 2
    r = int(size * 0.32)  # icon "radius" — leaves a comfortable margin

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=outline, width=3)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color, outline=outline, width=3)
    elif shape == "triangle":
        # Equilateral-ish triangle, point up.
        points = [
            (cx, cy - r),
            (cx - int(r * math.sin(math.radians(60))), cy + r // 2),
            (cx + int(r * math.sin(math.radians(60))), cy + r // 2),
        ]
        draw.polygon(points, fill=color, outline=outline)
    elif shape == "diamond":
        points = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw.polygon(points, fill=color, outline=outline)
    elif shape == "star":
        # 5-pointed star with alternating outer/inner radii.
        points = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            rad = r if i % 2 == 0 else r * 0.45
            points.append((cx + rad * math.cos(angle), cy + rad * math.sin(angle)))
        draw.polygon(points, fill=color, outline=outline)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Challenge generation
# ---------------------------------------------------------------------------

def generate_paged_challenge(
    shapes: list[str],
    colors: dict[str, tuple[int, int, int]],
    render_cfg: dict,
) -> dict:
    """Generate a paged carousel challenge from asset config.

    Args:
        shapes: shape vocabulary (e.g. ["circle", "square", ...]).
        colors: name -> RGB tuple mapping (e.g. {"red": (220, 60, 60), ...}).
        render_cfg: render section from the asset JSON. Required keys:
            card_size (int), card_count_range ([min, max]).

    Returns:
        Dict with keys: combos, card_pngs, target_index, card_size.
    """
    card_size = render_cfg["card_size"]
    min_cards, max_cards = render_cfg["card_count_range"]

    n_cards = random.randint(min_cards, max_cards)
    all_combos = [(s, c) for s in shapes for c in colors]
    combos = random.sample(all_combos, n_cards)
    card_pngs = [_render_icon(s, c, colors, card_size) for s, c in combos]
    target_index = random.randrange(n_cards)
    return {
        "combos": combos,
        "card_pngs": card_pngs,
        "target_index": target_index,
        "card_size": card_size,
    }


def load_challenge() -> dict:
    """Load the asset config for MODE and generate a challenge."""
    mode_file = ASSETS / f"{MODE}.json"
    cfg = json.loads(mode_file.read_text())
    colors = {name: tuple(rgb) for name, rgb in cfg["content"]["colors"].items()}
    shapes = cfg["content"]["shapes"]
    return generate_paged_challenge(shapes, colors, cfg["render"])


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Generate at startup
_CHALLENGE = load_challenge()
COMBOS = _CHALLENGE["combos"]
CARD_PNGS = _CHALLENGE["card_pngs"]
TARGET_INDEX = _CHALLENGE["target_index"]
CARD_SIZE = _CHALLENGE["card_size"]
TARGET_SHAPE, TARGET_COLOR = COMBOS[TARGET_INDEX]
N_CARDS = len(COMBOS)

# Base64-embed card PNGs so the page is fully self-contained.
CARD_DATA_URIS = [
    "data:image/png;base64," + base64.b64encode(p).decode("ascii") for p in CARD_PNGS
]



@app.route("/")
def index():
    """Main page: card carousel + navigation dots + Submit button."""
    # Build the <img> elements (all rendered, only target card visible at a time).
    card_imgs_html = "".join(
        f'<img class="card-img" data-idx="{i}" src="{uri}" '
        f'style="width:{CARD_SIZE}px; height:{CARD_SIZE}px; '
        f'display:{"block" if i == 0 else "none"};" draggable="false"/>'
        for i, uri in enumerate(CARD_DATA_URIS)
    )
    # Build the dots row.
    dots_html = "".join(
        f'<span class="dot" data-idx="{i}" '
        f'style="display:inline-block; width:14px; height:14px; margin:0 5px; '
        f'border-radius:50%; cursor:pointer; '
        f'background:{"#4CAF50" if i == 0 else "#ccc"};"></span>'
        for i in range(N_CARDS)
    )

    return f"""<!DOCTYPE html>
<html>
<head><title>Paged CAPTCHA</title></head>
<body style="display:flex; justify-content:center; align-items:center;
             min-height:100vh; margin:0; background:#f5f5f5;
             font-family:Arial,sans-serif; user-select:none;">
  <div style="text-align:center; background:white; padding:30px;
              border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.1);">
    <h2>Paged CAPTCHA</h2>
    <p>Show the card with a <b>{TARGET_COLOR} {TARGET_SHAPE}</b>, then click Submit.</p>

    <div style="display:flex; justify-content:center; align-items:center; gap:14px;">
      <button id="prev-btn" aria-label="Previous card"
              style="width:44px; height:44px; border-radius:50%; border:1px solid #ccc;
                     background:white; font-size:24px; line-height:1; cursor:pointer;
                     display:flex; align-items:center; justify-content:center;
                     padding:0; color:#444;">&#8249;</button>

      <div id="card-container" style="position:relative; width:{CARD_SIZE}px;
                  height:{CARD_SIZE}px;
                  border:1px solid #ccc; border-radius:6px; overflow:hidden;">
        {card_imgs_html}
      </div>

      <button id="next-btn" aria-label="Next card"
              style="width:44px; height:44px; border-radius:50%; border:1px solid #ccc;
                     background:white; font-size:24px; line-height:1; cursor:pointer;
                     display:flex; align-items:center; justify-content:center;
                     padding:0; color:#444;">&#8250;</button>
    </div>

    <div id="dots" style="margin-top:14px;">
      {dots_html}
    </div>

    <div style="margin-top:18px;">
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
  var nCards = {N_CARDS};
  var activeIndex = 0;
  var cards = document.querySelectorAll('.card-img');
  var dots = document.querySelectorAll('.dot');
  var submitBtn = document.getElementById('submit-btn');
  var msg = document.getElementById('result-msg');
  var locked = false;

  function showCard(idx) {{
    if (locked) return;
    if (idx < 0 || idx >= nCards) return;
    activeIndex = idx;
    for (var i = 0; i < nCards; i++) {{
      cards[i].style.display = (i === idx) ? 'block' : 'none';
      dots[i].style.background = (i === idx) ? '#4CAF50' : '#ccc';
    }}
  }}

  for (var i = 0; i < nCards; i++) {{
    (function(idx) {{
      dots[idx].addEventListener('click', function() {{ showCard(idx); }});
    }})(i);
  }}

  // Left/right arrows wrap around the carousel.
  document.getElementById('prev-btn').addEventListener('click', function() {{
    showCard((activeIndex - 1 + nCards) % nCards);
  }});
  document.getElementById('next-btn').addEventListener('click', function() {{
    showCard((activeIndex + 1) % nCards);
  }});

  window.submitAnswer = function() {{
    if (locked) return;
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/submit', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {{
      var resp = JSON.parse(xhr.responseText);
      locked = true;
      submitBtn.disabled = true;
      submitBtn.style.background = '#9e9e9e';
      submitBtn.style.cursor = 'not-allowed';
      for (var i = 0; i < nCards; i++) {{
        dots[i].style.cursor = 'not-allowed';
      }}
      ['prev-btn', 'next-btn'].forEach(function(id) {{
        var b = document.getElementById(id);
        b.disabled = true;
        b.style.cursor = 'not-allowed';
        b.style.opacity = '0.5';
      }});
      if (resp.correct) {{
        msg.innerHTML = '<span style="color:#4CAF50; font-weight:bold;">&#10003; Verified!</span>';
      }} else {{
        msg.innerHTML = '<span style="color:#f44336; font-weight:bold;">&#10007; Wrong card.</span>';
      }}
    }};
    xhr.send(JSON.stringify({{active_index: activeIndex}}));
  }};
}})();
</script>
</body>
</html>"""


@app.route("/submit", methods=["POST"])
def submit():
    """Verify which card was visible at submission time."""
    data = request.get_json(force=True)
    active_index = int(data.get("active_index", -1))
    correct = active_index == TARGET_INDEX

    result = {
        "submitted": True,
        "correct": correct,
        "active_index": active_index,
        "target_index": TARGET_INDEX,
        "target_shape": TARGET_SHAPE,
        "target_color": TARGET_COLOR,
        "n_cards": N_CARDS,
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
    )
