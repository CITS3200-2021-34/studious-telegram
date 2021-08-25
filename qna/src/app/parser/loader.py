from abc import ABC, abstractmethod


class AbstractLoader(ABC):

    @abstractmethod
    def read_data(self):
        pass
