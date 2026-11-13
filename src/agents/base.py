"""Lightweight agent primitives used by the in-class loop workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


@dataclass
class AgentEvent:
    agent: str
    status: str
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))


class Agent:
    """Single-responsibility worker that logs to a shared event bus."""

    def __init__(self, name: str, role: str, fn: Callable[..., Any]):
        self.name = name
        self.role = role
        self.fn = fn

    def run(self, events: list[AgentEvent], **kwargs: Any) -> Any:
        events.append(AgentEvent(self.name, "running", f"{self.role} started"))
        result = self.fn(**kwargs)
        events.append(
            AgentEvent(
                self.name,
                "success",
                f"{self.role} finished",
                payload=_safe(result),
            )
        )
        return result


def _safe(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return {
            k: v
            for k, v in result.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
    return {"result_type": type(result).__name__}
