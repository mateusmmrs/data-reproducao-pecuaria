"""Abstract base class shared by all agents."""

from abc import ABC, abstractmethod
from typing import Any

from utils.helpers import log_step, get_logger


class BaseAgent(ABC):
    """
    Every agent receives a shared `context` dict and returns it enriched.
    The context dict is the single source of truth passed along the pipeline.
    """

    name: str = "BaseAgent"

    def __init__(self):
        self.logger = get_logger(self.name)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        log_step(self.name, self._description())
        result = self.execute(context)
        self._summarize(result)
        return result

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Core logic — must be implemented by each agent."""
        ...

    def _description(self) -> str:
        return "Running..."

    def _summarize(self, context: dict[str, Any]) -> None:
        pass
