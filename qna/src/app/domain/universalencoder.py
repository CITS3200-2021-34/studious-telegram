from .questionmatcher import AbstractQuestionMatcher
from ..parser.parser import preprocess
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
        # self.__sentence_embeddings = self.__model(self.__question_list)

    def getSuggestions(self, question: str, body: str) -> List[Tuple[str, float]]:
        '''
        Determines question suggestions for a given question, based on the
        similarity of their subject-line.

        :param self: Instance of the UniversalEncoder object
        :param question: An element of the question dictionary
        :return [k[0] for k in similarity_dict]: List of all questions from
            question dictionary ordered from most similar to least
        '''

        # If we pass the model an empty question and body then return None
        if(question == "" and body == ""):
            return None

        Embedding_type = ""  # This will be either 'Subject_vec' or 'Text_vec' depending on stage

        # If we pass only the question, then find similarity of question only
        if(question != "" and body == ""):
            Embedding_type = 'Subject_vec'
            question = preprocess(question)

        # If we pass question and body, then combine them and find similarity
        if(body != ""):
            Embedding_type = 'Text_vec'
            question = preprocess(question) + body

        query_embedding = self.__model([question])[0]
        query_embedding = tf.reshape(query_embedding, (-1, 1))

        # Loop through the sentence embedding of each question, finding the cosine
        # between this and the embedding of the asked question
        similarity_dict = {}
        for i, subject in enumerate(self.__questions.keys()):
            sentence_embedding = tf.reshape(
                self.__questions[subject][Embedding_type], (-1, 1))
            similarity_dict[self.__question_list[i]] = 1 - \
                cosine(sentence_embedding, query_embedding)

        # Order dictionary to a list, such that higher cosines are first
        similarity_dict = sorted(similarity_dict.items(),
                                 key=operator.itemgetter(1), reverse=True)

        return similarity_dict, query_embedding
