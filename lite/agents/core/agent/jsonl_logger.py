"""Publish one viewer-compatible JSONL row per whole model trajectory, atomically."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from lite.agents.core.agent.hooks import SampleHook
from lite.core import LiteRLSample, LiteSample
from lite.core.messages.content import INLINE_REASONING_PART, MODEL_VISIBLE_PART_TYPES
from lite.utils.image import encode_png
from lite.utils.image_files import write_content_addressed_image


def project_viewer_messages(messages: list[dict]) -> list[dict]:
    """Keep viewer content to text/images and retain Lite annotations separately.

    SFT Data Explorer validates content part types even for ``base_messages``.
    Lite metadata, descriptions, and summaries belong outside that content list.
    Part positions are retained so this projection does not discard information.
    Supplied inline reasoning fills the display field when no provider reasoning
    field exists; an existing provider field remains unchanged.
    This operates on a copy and never changes the history sent to the model.
    """
    projected = copy.deepcopy(messages)
    for message in projected:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        visible, annotations, reasoning = [], [], []
        for index, part in enumerate(content):
            if part["type"] in MODEL_VISIBLE_PART_TYPES:
                visible.append(part)
            else:
                annotations.append({"index": index, "content": part})
                if part["type"] == INLINE_REASONING_PART:
                    reasoning.append(part["text"])
        message["content"] = visible
        if annotations:
            message["content_annotations"] = annotations
        if reasoning and "reasoning_content" not in message:
            message["reasoning_content"] = "\n\n".join(reasoning)
    return projected


class JsonlTrajectoryLogger(SampleHook):
    """Export model messages and screenshots for a live trace viewer.

    Attach alongside TrajectoryLogger to retain native parquet/debug artifacts.
    Live updates require a loop that calls on_trajectory. No reasoning is
    reconstructed: the row includes only content present in the model sample.
    This viewer export is distinct from the canonical training row schema.
    Use absolute image paths when the viewer shares the output filesystem;
    relative paths require configuring the viewer's media base directory.
    Images use SHA256 filenames, so identical PNG bytes share a single file
    while their separate trajectory positions remain intact.
    Supply the Agent's tool builder as ``get_tools`` to retain its tool
    declarations in ``metadata.tools`` on every live update.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        model_id: str,
        secrets=(),
        absolute_image_paths: bool = False,
        get_tools: Callable[[], list[dict]] | None = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.secrets = tuple(secrets)
        self.absolute_image_paths = absolute_image_paths
        self.get_tools = get_tools
        self._images: list[dict] = []
        self._row = None

    def _publish(self) -> None:
        text = json.dumps(self._row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for secret in self.secrets:
            if secret:
                text = text.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[redacted]")
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.directory, prefix=".trajectory-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(text + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                temporary.replace(self.directory / "agent_trajectory.jsonl")
            finally:
                temporary.unlink(missing_ok=True)

    def on_trajectory(self, trajectory: LiteSample) -> None:
        images_dir = self.directory / "images"
        images_dir.mkdir(exist_ok=True)
        for index in range(len(self._images), len(trajectory.images)):
            image = trajectory.images[index]
            destination = write_content_addressed_image(images_dir, encode_png(image))
            url = (
                str(destination.resolve())
                if self.absolute_image_paths
                else destination.relative_to(self.directory).as_posix()
            )
            self._images.append({"url": url, "image_wh": list(image.size)})
        messages = project_viewer_messages(trajectory.messages)
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        part["image"] = self._images[part.pop("index")]
        self._row = {
            "messages": messages,
            "for_agent": True,
            "source": ["cua-lite"],
            "metadata": {
                "trace_scope": "model_messages",
                "model_id": self.model_id,
                "status": "running",
                "environment": trajectory.metadata.to_dict(),
            },
        }
        if self.get_tools is not None:
            self._row["metadata"]["tools"] = copy.deepcopy(self.get_tools())
        self._publish()

    def on_complete(self, lite_rl_sample: LiteRLSample | None) -> None:
        if lite_rl_sample is not None:
            self.on_trajectory(lite_rl_sample.lite_sample)
            self._row["metadata"].update(
                status="completed",
                episode_return=lite_rl_sample.episode_return,
                terminated=lite_rl_sample.terminated,
                truncated=lite_rl_sample.truncated,
            )
        elif self._row is not None:
            self._row["metadata"]["status"] = "failed"
        if self._row is not None:
            self._publish()
