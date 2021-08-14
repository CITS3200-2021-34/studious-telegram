from .questionmatcher import AbstractQuestionMatcher

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import operator

from typing import List

class SentBERT(AbstractQuestionMatcher):
    def __init__(self, questions):
        # Load the model and pass in questions as a list to get embeddings
        self.__model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.__question_list = list(questions.keys())
        self.__sentence_embeddings = self.__model.encode(self.__question_list)

    def getSuggestions(self, question: str) -> List[str]:
        # pass the asked question into model to get embedding
        query_embedding = self.__model.encode([question])[0]
        query_embedding = query_embedding.reshape(-1, 1)

        # Loop through all the sentence embeddings and find cosine between
        # the asked question embedding
        similarity_dict = {}
        for i, sentence_embedding in enumerate(self.__sentence_embeddings):
            sentence_embedding = sentence_embedding.reshape(-1, 1)
            similarity_dict[self.__question_list[i]] = 1 - cosine(sentence_embedding, query_embedding)

        # Order dicitonary to a list, such that higher cosine are first
        similarity_dict = sorted(similarity_dict.items(),
                          key=operator.itemgetter(1), reverse=True)

        return [k[0] for k in similarity_dict]
