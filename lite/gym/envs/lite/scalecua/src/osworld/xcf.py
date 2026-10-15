"""Small XCF readers used by ScaleCUA generated GIMP judges."""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

PROP_END = 0
PROP_GROUP_ITEM = 37


def parse_xcf_layers(path: str | os.PathLike[str]) -> list[str]:
    """Return layer names from a GIMP XCF file in stored stack order."""

    return [layer["name"] for layer in parse_xcf_layer_info(path)]


def parse_xcf_layer_groups(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Return layer names plus the official PROP_GROUP_ITEM flag."""

    return [
        {"name": layer["name"], "is_group": layer["is_group"]}
        for layer in parse_xcf_layer_info(path)
    ]


def parse_xcf_file_directly(env, xcf_path: str) -> list[str]:
    """Compatibility helper expected by several generated ScaleCUA getters."""

    try:
        data = env.controller.get_file(xcf_path)
    except Exception:
        data = None
    if not data:
        return []
    return parse_xcf_layers_from_bytes(bytes(data))


def parse_xcf_layer_info(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return parse_xcf_layer_info_from_bytes(Path(path).read_bytes())


def parse_xcf_layers_from_bytes(data: bytes) -> list[str]:
    return [layer["name"] for layer in parse_xcf_layer_info_from_bytes(data)]


def parse_xcf_layer_info_from_bytes(data: bytes) -> list[dict[str, Any]]:
    if not data.startswith(b"gimp xcf "):
        return []
    try:
        null_pos = data.index(b"\x00", 9)
    except ValueError:
        return []
    version = data[9:null_pos].decode("ascii", errors="replace")
    offset_size = _offset_size(version)
    pos = null_pos + 1
    if pos + 12 > len(data):
        return []
    pos += 12
    pos = _skip_properties(data, pos)[0]
    layer_offsets: list[int] = []
    while pos + offset_size <= len(data):
        offset = _read_offset(data, pos, offset_size)
        pos += offset_size
        if offset == 0:
            break
        if 0 <= offset < len(data):
            layer_offsets.append(offset)
    layers: list[dict[str, Any]] = []
    for offset in layer_offsets:
        layer = _parse_layer(data, offset)
        if layer is not None:
            layers.append(layer)
    return layers


def _offset_size(version: str) -> int:
    try:
        version_number = int(version.lstrip("v") or "0")
    except ValueError:
        version_number = 0
    return 8 if version_number >= 11 else 4


def _read_u32(data: bytes, offset: int) -> int:
    return struct.unpack(">I", data[offset : offset + 4])[0]


def _read_offset(data: bytes, offset: int, size: int) -> int:
    fmt = ">Q" if size == 8 else ">I"
    return struct.unpack(fmt, data[offset : offset + size])[0]


def _parse_layer(data: bytes, offset: int) -> dict[str, Any] | None:
    if offset + 16 > len(data):
        return None
    width = _read_u32(data, offset)
    height = _read_u32(data, offset + 4)
    layer_type = _read_u32(data, offset + 8)
    name_len = _read_u32(data, offset + 12)
    name_start = offset + 16
    name_end = name_start + name_len
    if name_len == 0 or name_end > len(data):
        return None
    name = data[name_start:name_end].rstrip(b"\x00").decode("utf-8", errors="replace")
    _props_end, props = _skip_properties(data, name_end)
    group_value = props.get(PROP_GROUP_ITEM, b"")
    is_group = len(group_value) >= 4 and _read_u32(group_value, 0) != 0
    return {
        "name": name,
        "is_group": is_group,
        "width": width,
        "height": height,
        "type": layer_type,
    }


def _skip_properties(data: bytes, pos: int) -> tuple[int, dict[int, bytes]]:
    props: dict[int, bytes] = {}
    while pos + 8 <= len(data):
        prop_type = _read_u32(data, pos)
        prop_size = _read_u32(data, pos + 4)
        pos += 8
        prop_end = pos + prop_size
        if prop_end > len(data):
            return len(data), props
        props[prop_type] = data[pos:prop_end]
        pos = prop_end
        if prop_type == PROP_END:
            break
    return pos, props
