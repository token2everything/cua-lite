"""Explicit Chat Completions gateway agents, without model-name registration."""

from lite.agents.extensions.chat_gateway.agent import ChatGatewayAgent, ChatGatewayError

__all__ = ["ChatGatewayAgent", "ChatGatewayError"]
