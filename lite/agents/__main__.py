"""Command-line listing for ``python -m lite.agents``."""

from __future__ import annotations

from typing import Any

from lite.agents.bootstrap import register_all


def _registered_name(registry: Any, key: str) -> str:
    item = registry.get_class(key)
    return getattr(item, "__name__", item.__class__.__name__)


def main() -> int:
    register_all()

    from lite.agents.core.action_space.base import ActionSpaceRegistry
    from lite.agents.core.adapter import AgentAdapterRegistry
    from lite.agents.core.protocol import ProtocolRegistry

    print("CUA-Lite Agents Module")
    print("=" * 40)

    print("\nRegistered action spaces:")
    for key in ActionSpaceRegistry.list():
        print(f"  - {key}: {_registered_name(ActionSpaceRegistry, key)}")

    if ActionSpaceRegistry.list_patterns():
        print("\nAction space patterns:")
        for pattern in ActionSpaceRegistry.list_patterns():
            print(f"  - {pattern}")

    print("\nRegistered agent adapters:")
    for name in AgentAdapterRegistry.list():
        print(f"  - {name}: {_registered_name(AgentAdapterRegistry, name)}")

    if AgentAdapterRegistry.list_patterns():
        print("\nAgent adapter patterns:")
        for pattern in AgentAdapterRegistry.list_patterns():
            print(f"  - {pattern}")

    print("\nRegistered protocols:")
    for key in ProtocolRegistry.list():
        print(f"  - {key}: {_registered_name(ProtocolRegistry, key)}")

    print("\nExample usage:")
    print("  action_space = ActionSpaceRegistry.get('lite@desktop')")
    print("  adapter = AgentAdapterRegistry.get('qwen3_vl@desktop@use')")
    print("  protocol = ProtocolRegistry.get('ui_tars.history', full_history_size=5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
