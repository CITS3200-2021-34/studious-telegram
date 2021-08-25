from .questionmatcher import AbstractQuestionMatcher

from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
import operator
import numpy as np

from typing import List


class Doc2Vec(AbstractQuestionMatcher):

    def __init__(self, questions):
        self.__questions = questions

    def getSuggestions(self, question: str) -> List[str]:

        # Load model
        model = Doc2Vec.load('app/pretrained/d2v.model')

        # Loop through all questions and get embeddings
        vect_questions = {}
        for k in self.questions.keys():
            token_input = word_tokenize(k.lower())
            vect = model.infer_vector(token_input)
            vect_questions[k] = vect

         # Tokenize the question into a list of words
        token_input = word_tokenize(question.lower())

        # pass token list into model to get embedding
        question_vect = model.infer_vector(token_input)

        # reshape vector for cosine similarity
        question_vect = question_vect.reshape(1, -1)

        # loop through questions and find cosine between asked question
        sim_dict = {}
        for k, v in vect_questions.items():
            vec = v.reshape(1, -1)
            sim_dict[k] = cosine(question_vect, vec)

        # sort by highest cosine value
        sim_dict = sorted(sim_dict.items(),
                          key=operator.itemgetter(1), reverse=True)

        return [k[0] for k in sim_dict]
