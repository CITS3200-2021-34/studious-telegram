from abc import ABC, abstractmethod
from ..domain import AbstractQuestionMatcher

class AbstractUserInterface(ABC):
    """Abstract user interface.
    
    Concrete user interface implementations must be a subclass and implement
    all abstract methods."""

    @abstractmethod
    def start(self) -> None:
        """Start the main user interface loop."""
        pass