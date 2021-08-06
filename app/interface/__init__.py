from abc import ABC, abstractmethod
import domain

class AbstractUserInterface(ABC):
    """Abstract user interface.
    
    Concrete user interface implementations must be a subclass and implement
    all abstract methods."""

    @abstractmethod
    def setQuestionMatcher(self, matcher: domain.AbstractQuestionMatcher) -> None:
        """Set the question matcher used by the interface."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the main user interface loop."""
        pass
