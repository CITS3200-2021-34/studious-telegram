from .questionmatcher import AbstractQuestionMatcher

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import operator
import numpy as np

from typing import List


class SentBERT(AbstractQuestionMatcher):
    def __init__(self, questions, answers):
        self.__questions = questions
        self.__answers = answers

    def getSuggestions(self, question: str) -> List[str]:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        question_list = [k for k in self.__questions.keys()]
        sentence_embeddings = model.encode(question_list)

        query = model.encode([input])[0]
        query = query.reshape(-1, 1)

        temp = self.questions.items()
        li = list(temp)

        sim_dict = {}
        for i, sent in enumerate(sentence_embeddings):
            sent = sent.reshape(-1, 1)
            sim_dict[li[i][0]] = 1 - cosine(sent, query)

        sim_dict = sorted(sim_dict.items(),
                          key=operator.itemgetter(1), reverse=True)

        return [k[0] for k in sim_dict]
