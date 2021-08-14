from .questionmatcher import AbstractQuestionMatcher

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import operator
import numpy as np

from typing import List


class SentBERT(AbstractQuestionMatcher):
    def __init__(self, questions):
        self.__questions = questions

    def getSuggestions(self, question: str) -> List[str]:

        # Load the model and pass in questions as a list to get embeddings
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        question_list = [k for k in self.__questions.keys()]
        sentence_embeddings = model.encode(question_list)

        # pass the asked question into model to get embedding
        query = model.encode([question])[0]
        query = query.reshape(-1, 1)

        # preprocessing for dicitonary key indexing
        temp = self.__questions.items()
        li = list(temp)

        # Loop through all the sentence embeddings and find cosine between
        # the asked question embedding
        sim_dict = {}
        for i, sent in enumerate(sentence_embeddings):
            sent = sent.reshape(-1, 1)
            sim_dict[li[i][0]] = 1 - cosine(sent, query)

        # Order dicitonary to a list, such that higher cosine are first
        sim_dict = sorted(sim_dict.items(),
                          key=operator.itemgetter(1), reverse=True)

        return [k[0] for k in sim_dict]
