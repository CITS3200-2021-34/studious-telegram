from .questionmatcher import AbstractQuestionMatcher

import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial.distance import cosine
import operator

from typing import List

class UniversalEncoder(AbstractQuestionMatcher):
    def __init__(self, questions, answers):
        self.__questions = questions
        self.__answers = answers

    def getSuggestions(self, question: str) -> List[str]:
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)
        question_list = [k for k in self.__questions.keys()]
        sentence_embeddings = model(question_list)
        query_vec = model([question])[0]
        query_vec = tf.reshape(query_vec, (-1, 1))
        sim_dict = {}

        temp = self.__questions.items()
        li = list(temp)

        for i, sent in enumerate(sentence_embeddings):
            sent = tf.reshape(sent, (-1, 1))
            sim_dict[li[i][0]] = 1 - cosine(sent, query_vec)

        sim_dict = sorted(sim_dict.items(),
                          key=operator.itemgetter(1), reverse=True)

        return [k[0] for k in sim_dict]
