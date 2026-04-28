from abc import ABC, abstractmethod
from typing import Tuple


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(self, prompt: str):
        pass