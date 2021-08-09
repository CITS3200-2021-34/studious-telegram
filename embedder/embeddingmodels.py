from abc import ABC, abstractmethod


class AbstractEmbeddingModel(ABC):

    @abstractmethod
    def doc2vec(self):
        pass
