from .questionmatcher import AbstractQuestionMatcher

import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial.distance import cosine
import operator

from typing import List

class UniversalEncoder(AbstractQuestionMatcher):
    MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

    def __init__(self, questions):
        self.__model = hub.load(self.MODULE_URL)
        self.__questions_list = list(questions.keys())
        self.__sentence_embeddings = self.model(self.__question_list)

    def getSuggestions(self, question: str) -> List[str]:
        query_embedding = self.__model([question])[0]
        query_embedding = tf.reshape(query_embedding, (-1, 1))

        similarity_dict = {}
        for i, sentence_embedding in enumerate(self.__sentence_embeddings):
            sentence_embedding = tf.reshape(sentence_embedding, (-1, 1))
            similarity_dict[self.__questions_list[i]] = 1 - cosine(sentence_embedding, query_embedding)

        similarity_dict = sorted(similarity_dict.items(),
                          key=operator.itemgetter(1), reverse=True)

        return [k[0] for k in similarity_dict]
