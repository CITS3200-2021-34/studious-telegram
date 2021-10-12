from abc import ABC, abstractmethod
from typing import List, Tuple

from .question import Question


class AbstractQuestionMatcher(ABC):
    """Abstract question matcher interface.

    Concrete question matcher implementations must be a subclass and implement
    all abstract methods."""

    @abstractmethod
    def getSuggestions(self, question: str,
                       text_vec: bool) -> List[Tuple[str, float]]:
        pass

    @abstractmethod
    def addQuestions(self, questions: List[Question]) -> None:
        """Adds the given question to the Question Matcher so it can be returned as a suggestion."""
        pass
