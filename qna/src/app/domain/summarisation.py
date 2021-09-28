from abc import ABC, abstractmethod
from typing import List


class AbstractSummarisation(ABC):

    @abstractmethod
    def getSummarisations(self, question: str) -> List[str]:
        pass
