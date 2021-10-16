from abc import ABC, abstractmethod
from typing import List


class AbstractSummarisation(ABC):
    '''
    Abstract summarisation interface.

    Concrete summarisation implementations must be a subclass and implement
    all abstract methods.
    '''

    @abstractmethod
    def getSummarisation(self, question: str) -> List[str]:
        pass
