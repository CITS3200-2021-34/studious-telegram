from abc import ABC, abstractmethod


class AbstractEmbeddingModel(ABC):

    @abstractmethod
    def doc2vec(self):
        pass

    @abstractmethod
    def sent_BERT(self):
        pass

    @abstractmethod
    def universal_encoder(self):
        pass
