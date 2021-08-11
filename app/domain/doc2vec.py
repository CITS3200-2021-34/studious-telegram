from .questionmatcher import AbstractQuestionMatcher

from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
import operator

from typing import List


class Doc2Vec(AbstractQuestionMatcher):
    def __init__(self, questions, answers):
        self.__questions = questions
        self.__answers = answers

    def getSuggestions(self, question: str) -> List[str]:
        token_input = word_tokenize(question.lower())
        model = Doc2Vec.load('app/pretrained/d2v.model')
        vect = model.infer_vector(token_input)

        vect_questions = {}
        for k in self.questions.keys():
            token_input = word_tokenize(k.lower())
            vect = model.infer_vector(token_input)
            vect_questions[k] = vect

        token_input = word_tokenize(input.lower())
        question_vect = model.infer_vector(token_input)
        question_vect = question_vect.reshape(1, -1)

        sim_dict = {}
        for k, v in vect_questions.items():
            vec = v.reshape(1, -1)
            sim_dict[k] = cosine(question_vect, vec)

        sim_dict = sorted(sim_dict.items(),
                          key=operator.itemgetter(1), reverse=True)

        return [k[0] for k in sim_dict]
