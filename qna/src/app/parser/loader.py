from abc import ABC, abstractmethod


class AbstractLoader(ABC):
    '''
    Abstract data loader.

    Concrete data loader implementations must be a subclass and implement
    all abstract methods.
    '''
    @abstractmethod
    def read_data(self):
        pass
