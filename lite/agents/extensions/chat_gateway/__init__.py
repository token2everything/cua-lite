"""Explicit Chat Completions gateway agents, without model-name registration."""

from lite.agents.extensions.chat_gateway.agent import (
    ChatGatewayAgent,
    ChatGatewayError,
    ChatGatewayProtocolError,
)

__all__ = ["ChatGatewayAgent", "ChatGatewayError", "ChatGatewayProtocolError"]
