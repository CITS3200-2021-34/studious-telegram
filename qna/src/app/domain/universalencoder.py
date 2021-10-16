from typing import List, Tuple
from .t5 import T5
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial.distance import cosine
import operator
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import string

from .question import Question
from .questionmatcher import AbstractQuestionMatcher


class UniversalEncoder(AbstractQuestionMatcher):
    '''
    This is a class for sentence embedding of question subjects using Universal
    Sentence Encoder.
    '''
    MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

    def __init__(self):
        '''
        Constructor for the UniversalEncoder class.

        :param self: Instance of the UniversalEncoder object
        '''

        self.__model = hub.load(self.MODULE_URL)
        self.__questions: List[Question] = []
        self.__question_embeddings = []
        self.__body_embeddings = []
        self.__summariser = T5(50, 5, 't5-small')

    def addQuestions(self, questions: List[Question]) -> None:
        self.__questions += questions

        subject_embeddings = self.__model([question.subject for question in questions])
        self.__question_embeddings += [tf.reshape(embedding, (-1, 1))
                                       for embedding in subject_embeddings]

        summarisations = []

        for question in questions:
            summarisations.append(question.subject + " " + self.__summariser.getSummarisation(question.body))
            # summarisations.append(f"{question.subject} {question.subject} {question.subject} {question.body}")

        body_embeddings = self.__model(summarisations)
        self.__body_embeddings += [tf.reshape(embedding, (-1, 1))
                                   for embedding in body_embeddings]

    def getSuggestions(self, question: str, body: str) -> List[Tuple[str, float, List[str]]]:
        '''
        Determines question suggestions for a given question, based on the
        similarity of their subject-line.

        :param self: Instance of the UniversalEncoder object
        :param question: An element of the question dictionary
        :returns:
            - suggestions - A list of suggestion tuples containing the suggestion's subject string,
                    similarity value, and list of authors who have answered it
            - query_embedding - The embedding value for the asked question
        '''

        # If we pass the model an empty question and body then return empty list
        if(question == "" and body == ""):
            return []

        embedding_type = ""  # This will be either 'Subject_vec' or 'Text_vec' depending on stage

        # If we pass only the question, then find similarity of question only
        if(question != "" and body == ""):
            embedding_type = 'Subject_vec'
            question = preprocess(question)

        # If we pass question and body, then combine them and find similarity
        if(body != ""):
            embedding_type = 'Text_vec'
            question = preprocess(question) + self.__summariser.getSummarisation(body)

        query_embedding = self.__model([question])[0]
        query_embedding = tf.reshape(query_embedding, (-1, 1))

        # Loop through the sentence embedding of each question, finding the cosine
        # between this and the embedding of the asked question
        suggestions = []
        for i, oldQuestion in enumerate(self.__questions):
            question_embedding = self.__question_embeddings[i]

            if embedding_type == "Text_vec":
                question_embedding = self.__body_embeddings[i]

            # Determine level of answer approval
            highest_author = []

            for author in oldQuestion.answer_authors:
                # Note if the question has been answered by the lecturer
                if author == "chris.mcdonald@uwa.edu.au" or author == "lecturer@uwa.edu.au":
                    highest_author.append("lecturer")
                # Note if the question has been answered by a tutor
                if author == "poster023@student.uwa.edu.au" or author == "tutor@uwa.edu.au":
                    highest_author.append("tutor")
                # Else, not answered by lecturer or tutor, therefore no need to
                # record author

            suggestions.append(
                (oldQuestion.subject,
                 1 -
                 cosine(
                     question_embedding,
                     query_embedding), highest_author))

        # Order dictionary to a list, such that higher cosines are first
        suggestions.sort(key=operator.itemgetter(1), reverse=True)

        return suggestions, query_embedding


def preprocess(data):

    # Tokenize question and remove punctuation and lower strings
    data = word_tokenize(data)
    data = [i for i in data if i not in string.punctuation]
    data = [i.lower() for i in data]

    # Convert words to stem form
    # e.g. 'playing' is converted to 'play'
    lemmatizer = WordNetLemmatizer()
    data = [lemmatizer.lemmatize(i) for i in data]

    # Remove stopwords as they don't add value to the sentence meaning
    # and select only the top 10 stop words.
    # e.g. 'the' is not a valuable word
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords = stopwords[0:10]
    data = [i for i in data if i not in stopwords]

    return " ".join(data)
