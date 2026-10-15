"""Single source of truth for the lite.osworld synth-asset bundle root.

The asset bundle (pdf / html / photos / gutenberg / code / csv referenced by
synth tasks via ``host_push``) is hosted on HuggingFace
(``cua-lite/lite.osworld-assets``) and downloaded by
``scripts/utils/assets.sh pull`` into
``<env>/.cache/assets/pulled/synth/``. It is NOT committed to git (it is
bulky and vendored; see ``data/assets/synth/README.md``).

``asset_root()`` is the ONE place that resolves where the bundle lives, used
by both the runtime staging path (``src/utils/dispatch.py``) and the codegen
staging path (``src/gen/train/synth/_utils.py`` + ``libreoffice_writer.py``).

Resolution order:
  1. ``$CUA_LITE_OSWORLD_ASSETS``      — explicit override (any layout)
  2. ``<env>/.cache/assets/pulled/synth`` — HF download, gated on the
     ``.complete`` + ``.asset_identity`` stamps

``snapshot_download`` leaves its own siblings in the cache dir
(``.gitattributes``, ``.cache/huggingface/``, plus our stamps);
``host_push`` resolves only specific rel paths (e.g. ``photos/x.jpg``), so
these extras are inert.

The committed ``data/assets/synth/{MANIFEST.csv,README.md}`` stay in git as
the inventory. ``scripts/utils/assets.sh build`` rebuilds the bundle from
source URLs, after which ``scripts/utils/assets.sh push`` can re-publish it to
HF.
"""

from __future__ import annotations

import os
from pathlib import Path

from lite.gym.utils.config.manifest import load_asset_lock

# src/utils/assets.py → parents[2] = the env root (lite/gym/envs/lite/osworld).
_ENV_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = _ENV_ROOT / ".cache" / "assets" / "pulled" / "synth"

#: Completion markers written by ``scripts/utils/assets.sh pull`` only after a
#: full ``snapshot_download`` succeeds. Gate on both so stale/partial downloads
#: are not silently trusted.
_CACHE_COMPLETE = _CACHE_ROOT / ".complete"
_CACHE_IDENTITY = _CACHE_ROOT / ".asset_identity"

#: Env var to point the asset root at an arbitrary location.
ENV_OVERRIDE = "CUA_LITE_OSWORLD_ASSETS"


def _expected_identity() -> str:
    return load_asset_lock(_ENV_ROOT).component_identity("synth")


def _fresh_pulled_cache() -> bool:
    if not (_CACHE_ROOT.is_dir() and _CACHE_COMPLETE.is_file() and _CACHE_IDENTITY.is_file()):
        return False
    return _CACHE_IDENTITY.read_text().strip() == _expected_identity()


def asset_root() -> Path:
    """Return the directory holding the synth asset bundle (see module docstring).

    Never raises (it is called at import time): if nothing resolves, it returns
    the expected cache path, so a genuinely missing asset fails loudly and
    locally at stage time (``_stage_asset`` → FileNotFoundError) rather than here.
    """
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        return Path(override)
    if _fresh_pulled_cache():            # a COMPLETED install stamped the cache
        return _CACHE_ROOT
    return _CACHE_ROOT                   # expected location; stage-time error if unpopulated
