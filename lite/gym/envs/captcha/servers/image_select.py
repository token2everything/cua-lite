"""Image-select CAPTCHA server (unified: crop / full / halligan modes).

3x3 grid where the user selects tiles that contain a target. Three data
source modes are switched via `CAPTCHA_MODE`:

  crop     — one source photo (600x600) is sliced into 9 tiles on the
             server. Each source photo has a sidecar JSON listing which
             tile indices contain the target.
             Data:  assets/image_select/crop/<category>/{NNN.jpg, NNN.json}

  full     — each of the 9 tiles is an independent ~200x200 photo drawn
             from labeled positive/negative pools at startup. N positives
             (2..4) + (9-N) negatives are shuffled into the grid.
             Data:  assets/image_select/full/<category>/{positive,negative}/*.jpg

  halligan — one of 72 fixed real-reCAPTCHA-v2 challenges from the
             Halligan benchmark (Teoh et al., USENIX Security '25).
             Selected by `CAPTCHA_HALLIGAN_ID` (0..71). Used for eval.
             Data:  assets/image_select/test/halligan/NNN/{tile_0..8.jpg, meta.json}

Scoring:
  crop, full  → strict set equality
  halligan    → Jaccard index >= 0.75 (matches the original benchmark)

Environment variables:
  CAPTCHA_MODE           : crop | full | halligan        (default: full)
  CAPTCHA_CATEGORY       : crop/full only                (default: crosswalk)
  CAPTCHA_HALLIGAN_ID    : halligan only, int 0..71
  CAPTCHA_SEED           : seeds random (crop picks which source; full
                           samples pools); ignored by halligan
  CAPTCHA_RESULT_FILE    : where to write the verify result JSON

    pip install flask pillow
    python lite/gym/envs/captcha/servers/image_select.py
    # http://localhost:5000
"""

from __future__ import annotations

import io
import json
import os
import random
from pathlib import Path

from flask import Flask, abort, request, send_file
from PIL import Image

# ---------------------------------------------------------------------------
# Constants / env
# ---------------------------------------------------------------------------

MODE = os.environ.get("CAPTCHA_MODE", "full").lower()
if MODE not in {"crop", "full", "halligan"}:
    raise RuntimeError(f"Invalid CAPTCHA_MODE={MODE!r}; expected crop|full|halligan")

_seed_raw = os.environ.get("CAPTCHA_SEED")
if _seed_raw is not None and MODE != "halligan":
    random.seed(int(_seed_raw))

RESULT_FILE = os.environ.get("CAPTCHA_RESULT_FILE", "/tmp/captcha_result.json")

# Debug/test mode: when a wrong answer is submitted, leak the ground-truth
# tile indices + their pixel centers in the verify response, and have the
# browser briefly highlight the correct tiles in green. Useful for manual
# testing; do NOT enable when benchmarking an agent.
DEBUG_MODE = os.environ.get("CAPTCHA_DEBUG", "").lower() in ("1", "true", "yes", "on")

_ASSETS_DIR = Path(__file__).parent.parent / ".cache" / "assets" / "image_select"


def _load_mode_config(mode: str) -> dict:
    """Read `assets/image_select/<mode>/train_eval.json`."""
    cfg_path = _ASSETS_DIR / mode / "train_eval.json"
    return json.loads(cfg_path.read_text())


# ---------------------------------------------------------------------------
# Mode: crop — one source photo sliced into 9 tiles on startup
# ---------------------------------------------------------------------------

def _load_crop(content_cfg: dict) -> tuple[list[bytes], set[int], str, dict]:
    data_dir = _ASSETS_DIR / content_cfg["data_dir"]
    if not data_dir.is_dir():
        raise RuntimeError(f"Dataset directory missing: {data_dir}")

    entries = []
    for jp in sorted(data_dir.glob("*.json")):
        data = json.loads(jp.read_text())
        if not data.get("labeled"):
            continue
        tiles = data.get("positive_tiles") or []
        if not tiles:
            continue
        img_path = data_dir / f"{jp.stem}.jpg"
        if not img_path.is_file():
            continue
        entries.append({"stem": jp.stem, "img_path": img_path,
                        "positive_tiles": sorted(int(t) for t in tiles)})

    if not entries:
        raise RuntimeError(f"No labeled crop images in {data_dir}")

    chosen = random.choice(entries)
    img = Image.open(chosen["img_path"]).convert("RGB")
    w, h = img.size
    tw, th = w // 3, h // 3
    tile_bytes = []
    for idx in range(9):
        r, c = divmod(idx, 3)
        tile = img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
        buf = io.BytesIO()
        tile.save(buf, "JPEG", quality=90)
        tile_bytes.append(buf.getvalue())

    meta = {"source": f"{content_cfg['data_dir']}/{chosen['stem']}.jpg"}
    return tile_bytes, set(chosen["positive_tiles"]), \
        content_cfg["target_label"], meta


# ---------------------------------------------------------------------------
# Mode: full — sample N positives + (9-N) negatives from labeled pools
# ---------------------------------------------------------------------------

def _load_full(content_cfg: dict) -> tuple[list[bytes], set[int], str, dict]:
    pos_dir = _ASSETS_DIR / content_cfg["positive_dir"]
    neg_dir = _ASSETS_DIR / content_cfg["negative_dir"]
    pos = sorted(pos_dir.glob("*.jpg"))
    neg = sorted(neg_dir.glob("*.jpg"))
    if not pos or not neg:
        raise RuntimeError(f"Missing positive/negative pool in {pos_dir.parent}")

    pos_min, pos_max = content_cfg["pos_count_range"]
    n_pos = random.randint(pos_min, min(pos_max, len(pos), 8))
    n_neg = 9 - n_pos
    poses = random.sample(pos, n_pos)
    negs = (random.sample(neg, n_neg) if n_neg <= len(neg)
            else [random.choice(neg) for _ in range(n_neg)])

    tagged = [("+", p) for p in poses] + [("-", p) for p in negs]
    random.shuffle(tagged)

    paths = [p for _, p in tagged]
    pos_idx = {i for i, (t, _) in enumerate(tagged) if t == "+"}
    tile_bytes = [p.read_bytes() for p in paths]

    meta = {"tile_sources": [f"{p.parent.name}/{p.name}" for p in paths]}
    return tile_bytes, pos_idx, content_cfg["target_label"], meta


# ---------------------------------------------------------------------------
# Mode: halligan — fixed real-reCAPTCHA challenge by id
# ---------------------------------------------------------------------------

def _load_halligan(challenge_id: int) -> tuple[list[bytes], set[int], str, dict]:
    root = _ASSETS_DIR / "test" / "halligan" / f"{int(challenge_id):03d}"
    if not root.is_dir():
        raise RuntimeError(
            f"Halligan challenge {challenge_id} missing at {root}."
        )
    meta = json.loads((root / "meta.json").read_text())
    tile_bytes = [(root / f"tile_{i}.jpg").read_bytes() for i in range(9)]
    pos_idx = set(meta["labels"])
    return tile_bytes, pos_idx, meta["instruction"], \
        {"halligan_id": int(challenge_id), "source_index": meta.get("index")}


# ---------------------------------------------------------------------------
# Dispatch — pick loader + scoring based on mode
# ---------------------------------------------------------------------------

if MODE == "crop":
    _cfg = _load_mode_config("crop")
    _TILE_BYTES, _POSITIVE_TILES, _TARGET_LABEL, _META = _load_crop(_cfg["content"])
    TILE_DISPLAY = _cfg["render"]["tile_display_px"]
    GRID_GAP = _cfg["render"]["grid_gap_px"]
    _SCORING = "strict"
elif MODE == "full":
    _cfg = _load_mode_config("full")
    _TILE_BYTES, _POSITIVE_TILES, _TARGET_LABEL, _META = _load_full(_cfg["content"])
    TILE_DISPLAY = _cfg["render"]["tile_display_px"]
    GRID_GAP = _cfg["render"]["grid_gap_px"]
    _SCORING = "strict"
elif MODE == "halligan":
    _hid = int(os.environ.get("CAPTCHA_HALLIGAN_ID", "0"))
    _TILE_BYTES, _POSITIVE_TILES, _TARGET_LABEL, _META = _load_halligan(_hid)
    # Halligan keeps hardcoded render defaults — static test data, no config file.
    TILE_DISPLAY = 150
    GRID_GAP = 4
    _SCORING = "jaccard_075"
IMG_DISPLAY = TILE_DISPLAY * 3 + GRID_GAP * 2


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Image Select CAPTCHA ({MODE})</title>
</head>
<body style="display:flex; justify-content:center; align-items:center;
             min-height:100vh; margin:0; background:#f5f5f5;
             font-family:Arial,sans-serif; user-select:none;">
  <div style="background:white; border-radius:8px; overflow:hidden;
              box-shadow:0 2px 10px rgba(0,0,0,0.12);
              width:{IMG_DISPLAY + 40}px;">
    <div style="background:#4a90e2; color:white; padding:14px 20px; position:relative;">
      <div style="font-size:14px; opacity:0.9;">Select all images containing</div>
      <div style="font-size:22px; font-weight:bold; margin-top:4px;">
        {_TARGET_LABEL}
      </div>
      {_render_debug_badge()}
    </div>

    <div style="padding:20px; text-align:center;">
      <div id="grid" style="width:{IMG_DISPLAY}px;
                  height:{IMG_DISPLAY}px; margin:0 auto;
                  display:grid;
                  grid-template-columns: repeat(3, {TILE_DISPLAY}px);
                  grid-template-rows:    repeat(3, {TILE_DISPLAY}px);
                  gap:{GRID_GAP}px;">
        {_render_tiles()}
      </div>

      <div style="margin-top:16px; display:flex; justify-content:space-between;
                  align-items:center;">
        <button onclick="location.reload()" title="New challenge"
                style="background:none; border:none; font-size:22px;
                       cursor:pointer; color:#666; padding:6px 10px;">
          ↻
        </button>
        <button onclick="submitAnswer()"
                style="font-size:16px; padding:10px 28px; background:#4a90e2;
                       color:white; border:none; border-radius:4px;
                       cursor:pointer; font-weight:bold;">
          Verify
        </button>
      </div>

      <div id="result-msg" style="margin-top:12px; font-size:16px;
                  min-height:24px;"></div>
    </div>
  </div>

<script>
(function() {{
  var selected = new Set();
  var tiles = document.querySelectorAll('.tile');
  tiles.forEach(function(t) {{
    t.addEventListener('click', function() {{
      var idx = parseInt(t.dataset.idx, 10);
      if (selected.has(idx)) {{
        selected.delete(idx);
        t.classList.remove('selected');
      }} else {{
        selected.add(idx);
        t.classList.add('selected');
      }}
    }});
  }});

  window.submitAnswer = function() {{
    var msg = document.getElementById('result-msg');
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/verify', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {{
      var resp = JSON.parse(xhr.responseText);

      // Debug mode — if server leaked the ground truth, print coords to
      // the console, highlight correct tiles, and show them in the msg.
      if (resp.expected_tiles) {{
        console.log('[DEBUG] expected_tiles:', resp.expected_tiles);
        console.log('[DEBUG] expected_coords (grid-local px):', resp.expected_coords);
        resp.expected_tiles.forEach(function(i) {{
          tiles[i].classList.add('reveal');
        }});
      }}

      if (resp.correct) {{
        msg.innerHTML = '<span style="color:#4CAF50; font-weight:bold;">' +
                        '&#10003; Correct</span>';
      }} else {{
        var extra = '';
        if (resp.expected_tiles) {{
          extra = '<br/><span style="font-size:13px; color:#666;">' +
                  'Correct tiles: [' + resp.expected_tiles.join(', ') + ']</span>';
        }}
        msg.innerHTML = '<span style="color:#f44336; font-weight:bold;">' +
                        '&#10007; Try again</span>' + extra;
        setTimeout(function() {{
          selected.clear();
          tiles.forEach(function(t) {{
            t.classList.remove('selected');
            t.classList.remove('reveal');
          }});
          msg.innerHTML = '';
        }}, 3000);
      }}
    }};
    xhr.send(JSON.stringify({{selected: Array.from(selected)}}));
  }};
}})();
</script>
<style>
  .tile {{
    background-size: cover;
    background-position: center;
    cursor: pointer;
    position: relative;
    transition: transform 0.1s;
  }}
  .tile.selected {{
    outline: 4px solid #4a90e2;
    outline-offset: -4px;
  }}
  .tile.selected::after {{
    content: '';
    position: absolute; top:0; left:0; right:0; bottom:0;
    background: rgba(74, 144, 226, 0.25);
    pointer-events: none;
  }}
  /* Debug mode — server-revealed correct answer (green). */
  .tile.reveal {{
    outline: 4px solid #4CAF50;
    outline-offset: -4px;
  }}
  .tile.reveal::before {{
    content: '\u2713';
    position: absolute; top: 4px; right: 4px;
    width: 22px; height: 22px; line-height: 22px; text-align: center;
    border-radius: 50%;
    background: #4CAF50; color: white;
    font-weight: bold; font-size: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    pointer-events: none; z-index: 3;
  }}
  .tile:hover {{ transform: scale(1.02); z-index: 2; }}
</style>
</body>
</html>"""


def _render_debug_badge() -> str:
    if not DEBUG_MODE:
        return ""
    return (
        '<span style="position:absolute; top:10px; right:12px; '
        'background:#ff9800; color:white; font-size:11px; font-weight:bold; '
        'padding:2px 8px; border-radius:10px; letter-spacing:1px;">DEBUG</span>'
    )


def _render_tiles() -> str:
    return "\n        ".join(
        f'<div class="tile" data-idx="{i}" '
        f'style="background-image:url(\'/tile/{i}.jpg\');"></div>'
        for i in range(9)
    )


@app.route("/tile/<int:idx>.jpg")
def tile(idx: int):
    if not 0 <= idx < 9:
        abort(404)
    return send_file(io.BytesIO(_TILE_BYTES[idx]), mimetype="image/jpeg")


def _tile_center(idx: int) -> tuple[int, int]:
    """Center pixel of tile `idx` in GRID-LOCAL coordinates.

    (0, 0) is the top-left of the 3x3 grid container (not the page).
    Each cell is TILE_DISPLAY x TILE_DISPLAY with GRID_GAP between cells.
    """
    row, col = divmod(idx, 3)
    stride = TILE_DISPLAY + GRID_GAP
    return (col * stride + TILE_DISPLAY // 2,
            row * stride + TILE_DISPLAY // 2)


def _is_correct(selected: set[int]) -> bool:
    if _SCORING == "strict":
        return selected == _POSITIVE_TILES
    if _SCORING == "jaccard_075":
        union = selected | _POSITIVE_TILES
        if not union:
            return True  # both empty
        return len(selected & _POSITIVE_TILES) / len(union) >= 0.75
    raise RuntimeError(f"unknown scoring: {_SCORING}")


# Once a correct submit happens, lock the result so a subsequent wrong
# submit (tiles re-toggled + Verify again) doesn't overwrite it — the env
# reads /result only once at the terminal step. Mirrors slider.py.
_LOCKED_CORRECT = False


@app.route("/verify", methods=["POST"])
def verify():
    global _LOCKED_CORRECT
    data = request.get_json(force=True)
    try:
        selected = {int(i) for i in data.get("selected", []) if 0 <= int(i) < 9}
    except (TypeError, ValueError):
        selected = set()

    correct = _is_correct(selected)
    if correct:
        _LOCKED_CORRECT = True
    correct = _LOCKED_CORRECT

    with open(RESULT_FILE, "w") as f:
        json.dump({
            "submitted": True,
            "correct": correct,
            "mode": MODE,
            "scoring": _SCORING,
            "selected": sorted(selected),
            "expected": sorted(_POSITIVE_TILES),
            "target": _TARGET_LABEL,
            **_META,
        }, f)

    resp = {"submitted": True, "correct": correct}
    # In debug mode, leak the answer on every verify (not only wrong ones),
    # so the JS can visualize where the correct tiles are.
    if DEBUG_MODE:
        expected_sorted = sorted(_POSITIVE_TILES)
        resp["expected_tiles"] = expected_sorted
        resp["expected_coords"] = [
            {"tile": i, "x": x, "y": y}
            for i, (x, y) in ((i, _tile_center(i)) for i in expected_sorted)
        ]
        resp["target_label"] = _TARGET_LABEL
    return json.dumps(resp), 200, {"Content-Type": "application/json"}


@app.route("/result")
def result():
    try:
        with open(RESULT_FILE) as f:
            return f.read()
    except FileNotFoundError:
        return json.dumps({"submitted": False, "correct": False})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
