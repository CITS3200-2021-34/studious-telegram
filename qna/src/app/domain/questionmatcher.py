from abc import ABC, abstractmethod
from typing import List, Tuple


class AbstractQuestionMatcher(ABC):
    """Abstract question matcher interface.

    Concrete question matcher implementations must be a subclass and implement
    all abstract methods."""

    @abstractmethod
    def getSuggestions(self, question: str) -> List[Tuple[str, float]]:
        pass
