"""Common base class every agent in the brain implements.

Keeping a shared, tiny interface (`name`, `act`) is what lets
`brain/brain_service.py` treat the DQN agent, the coordinator, the
emergency preemption agent, and the route allocation agent uniformly when
logging decisions and computing decision-latency metrics -- important once
"industry grade" observability is layered on top.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def act(self, *args: Any, **kwargs: Any) -> Any:
        """Produce a decision. Signature varies per agent; see each subclass."""
