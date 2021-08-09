from abc import ABC, abstractmethod


class AbstractDataFormater(ABC):

    @abstractmethod
    def parse_text(self) -> list[str]:
        pass
