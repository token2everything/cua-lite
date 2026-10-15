"""Provider-free Lite action-space contract helpers."""

from lite.core.tools.action_space.base import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    LITE_ACTION_SET_TOOL_NAMES,
    LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER,
    LITE_DESKTOP_KEY_ACTION_NAMES,
    LITE_MOBILE_ACTION_NAMES_IN_SCHEMA_ORDER,
    LITE_VALID_ACTION_NAMES,
    LiteBBoxActionSet,
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
    is_lite_action_name_or_action_batch_tool_name,
    lite_action_batch_tool_name_for_platform,
    lite_action_names_by_action_batch_tool,
    lite_action_names_for_action_batch_tool,
    lite_action_set_tool_names_for_metadata,
    lite_builtin_tool_names_for_metadata,
)
from lite.core.tools.action_space.batches import (
    LITE_COMPUTER_ACTION_BATCH_TOOL_NAME,
    LITE_MOBILE_ACTION_BATCH_TOOL_NAME,
    lite_action_batch_child_name_errors,
    make_lite_action_batch_call,
    make_lite_action_batch_schema,
    merge_adjacent_lite_action_batches,
    unpack_action_batch_call,
    validate_lite_action_batch_structure,
)
from lite.core.tools.action_space.geometry import (
    COORDINATE_ARGUMENT_NAMES,
    MAX_NORM,
    action_coordinate_arguments_out_of_range,
    clamp_norm,
    norm_coord_out_of_range,
    pixel_to_norm,
)
from lite.core.tools.action_space.keys import normalize_keys

__all__ = [
    # Action classes.
    "LiteDesktopActionSet",
    "LiteMobileActionSet",
    "LitePointActionSet",
    "LiteBBoxActionSet",

    # Sets derived from the action classes.
    "LITE_VALID_ACTION_NAMES",
    "LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER",
    "LITE_DESKTOP_KEY_ACTION_NAMES",
    "LITE_MOBILE_ACTION_NAMES_IN_SCHEMA_ORDER",

    # Tool names derived from action sets.
    "LITE_ACTION_SET_TOOL_NAMES",
    "LITE_ACTION_BATCH_TOOL_NAMES",
    "lite_action_set_tool_names_for_metadata",
    "lite_builtin_tool_names_for_metadata",

    # Action-batch helpers.
    "LITE_COMPUTER_ACTION_BATCH_TOOL_NAME",
    "LITE_MOBILE_ACTION_BATCH_TOOL_NAME",
    "make_lite_action_batch_call",
    "make_lite_action_batch_schema",
    "merge_adjacent_lite_action_batches",
    "unpack_action_batch_call",
    "lite_action_batch_child_name_errors",
    "validate_lite_action_batch_structure",

    # Action-batch selectors.
    "lite_action_names_by_action_batch_tool",
    "lite_action_names_for_action_batch_tool",
    "lite_action_batch_tool_name_for_platform",

    # Cross-layer predicate.
    "is_lite_action_name_or_action_batch_tool_name",

    # Geometry and key helpers.
    "COORDINATE_ARGUMENT_NAMES",
    "MAX_NORM",
    "action_coordinate_arguments_out_of_range",
    "clamp_norm",
    "norm_coord_out_of_range",
    "normalize_keys",
    "pixel_to_norm",
]
