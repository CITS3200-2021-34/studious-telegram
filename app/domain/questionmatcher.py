from abc import ABC, abstractmethod
from typing import List

class AbstractQuestionMatcher(ABC):
    """Abstract question matcher interface.
    
    Concrete question matcher implementations must be a subclass and implement
    all abstract methods."""
    
    @abstractmethod
    def getSuggestions(self, question: str) -> List[str]:
        pass