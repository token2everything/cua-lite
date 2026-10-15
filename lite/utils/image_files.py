"""Atomic persistence of encoded images, addressed by their byte content."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def write_content_addressed_image(
    directory: Path,
    image_bytes: bytes,
    *,
    extension: str = "png",
) -> Path:
    """Return a SHA256-named image path, publishing each complete file atomically.

    Existing content is reused. Concurrent writers of the same image can only
    replace the destination with identical complete bytes; readers never see a
    partial image. Failed publications remove their temporary file.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = directory / f"{hashlib.sha256(image_bytes).hexdigest()}.{extension}"
    if destination.exists():
        return destination
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".image-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(image_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
