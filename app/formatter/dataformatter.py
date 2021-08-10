from abc import ABC, abstractmethod

from typing import List

class AbstractDataFormater(ABC):

    @abstractmethod
    def parse_text(self) -> List[str]:
        pass
