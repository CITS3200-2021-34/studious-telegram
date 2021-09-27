from .questionmatcher import AbstractQuestionMatcher

import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial.distance import cosine
import operator

from typing import List, Tuple


class UniversalEncoder(AbstractQuestionMatcher):
    '''
    This is a class for sentence embedding of question subjects using Universal
    Sentence Encoder.
    '''
    MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

    def __init__(self, questions):
        '''
        Constructor for the UniversalEncoder class.

        :param self: Instance of the UniversalEncoder object
        :param questions: Dictionary of question threads
        '''
        # Load the model and pass in questions as a list to get embeddings
        self.__model = hub.load(self.MODULE_URL)
        self.__questions = questions
        self.__question_list = list(questions.keys())
        #self.__sentence_embeddings = self.__model(self.__question_list)


    def getSuggestions(self, question: str) -> List[Tuple[str, float]]:
        '''
        Determines question suggestions for a given question, based on the 
        similarity of their subject-line.

        :param self: Instance of the UniversalEncoder object
        :param question: An element of the question dictionary
        :return [k[0] for k in similarity_dict]: List of all questions from 
            question dictionary ordered from most similar to least
        '''
        # Pass the asked question into model to get embedding
        query_embedding = self.__model([question])[0]
        query_embedding = tf.reshape(query_embedding, (-1, 1))

        # Loop through the sentence embedding of each question, finding the cosine 
        # between this and the embedding of the asked question
        similarity_dict = {}
        for i, subject in enumerate(self.__questions.keys()):
            sentence_embedding = tf.reshape(
                self.__questions[subject]['Subject_vec'], (-1, 1))
            similarity_dict[self.__question_list[i]] = 1 - \
                cosine(sentence_embedding, query_embedding)

        # Order dictionary to a list, such that higher cosines are first
        similarity_dict = sorted(similarity_dict.items(),
                                 key=operator.itemgetter(1), reverse=True)

        return similarity_dict
