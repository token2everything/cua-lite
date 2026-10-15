"""Credential-safe API request logs with content-addressed image files."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from lite.utils.image_files import write_content_addressed_image


def redact(value: Any, secrets: tuple[str, ...]) -> Any:
    """Remove credential fields and configured secret values from log payloads."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            sensitive = normalized in {"authorization", "cookie", "setcookie"} or (
                normalized.endswith(
                    ("apikey", "accesskey", "secretkey", "password", "token", "secret")
                )
            )
            result[key] = "[redacted]" if sensitive else redact(item, secrets)
        return result
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
    return value


def write_log(
    directory: Path,
    name: str,
    value: Any,
    secrets: tuple[str, ...],
    *,
    image_directory: Path | None = None,
) -> None:
    """Atomically save a redacted JSON body, externalizing data-URL images.

    Only the persisted copy is transformed. Encrypted reasoning and provider
    output items remain intact in the in-memory stream replayed to the gateway.
    Image references in the JSON are relative to ``directory``. Images default
    to ``directory/images``; an explicit ``image_directory`` can share the
    trajectory's image files without changing the in-memory API request.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def archive_images(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: archive_images(child) for key, child in item.items()}
        if isinstance(item, list):
            return [archive_images(child) for child in item]
        if isinstance(item, str) and item.startswith("data:image/") and ";base64," in item:
            header, encoded = item.split(";base64,", 1)
            try:
                image = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                return item
            mime = header.removeprefix("data:image/")
            extension = {"png": "png", "jpeg": "jpg", "webp": "webp", "gif": "gif"}.get(mime, "bin")
            destination = write_content_addressed_image(
                image_directory if image_directory is not None else directory / "images",
                image,
                extension=extension,
            )
            return Path(os.path.relpath(destination, directory)).as_posix()
        return item

    payload = (
        json.dumps(
            redact(archive_images(value), secrets), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n"
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, directory / name)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
