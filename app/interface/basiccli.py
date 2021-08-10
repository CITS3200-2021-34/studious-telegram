from app.embedder.sentence_embeddings import SentEmbeddings
from ..domain import AbstractQuestionMatcher
from ..preprocessor.preprocess import PreProcessor
from ..formatter.dataparser import DataParser
from ..embedder import SentEmbeddings
from .userinterface import AbstractUserInterface
from gensim.models.doc2vec import Doc2Vec
import os


class BasicCLI(AbstractUserInterface):
    '''
    Class for constructing a basic command line interface for the user 
    to interact with when asking questions and viewing suggestions.
    '''
    __matcher: AbstractQuestionMatcher = None

    def __init__(self, matcher: AbstractQuestionMatcher) -> None:
        '''
        Constructor for the BasicCLI class.

        :param self: Instance of the BasicCLI object
        :param matcher: Abstract question matcher interface
        '''
        super().__init__()
        self.setQuestionMatcher(matcher)
        print(os.getcwd())
        self.parser = DataParser('app/testfiles/help2002-2017.txt')
        self.processor = PreProcessor()
        self.questions = None
        self.answers = None
        self.getQuestionAnswer()
        self.embedder = SentEmbeddings(self.questions, self.answer)

    def setQuestionMatcher(self, matcher: AbstractQuestionMatcher):
        self.__matcher = matcher

    def getQuestionAnswer(self):
        '''
        Stores data as individual posts using get_posts() function and 
        then uses create_data_structures() function to place them into 
        either a question or an answer dictionary.

        :param self: Instance of the BasicCLI object
        '''
        posts = self.processor.get_posts(self.parser.parse_text())
        self.questions, self.answer = self.processor.create_data_structures(
            posts)

    def start(self):
        '''
        Gathers and prints top 3 suggestions, based on user's command line 
        entry, allowing user to select one of the suggestions to which the 
        answer will be printed.

        :param self: Instance of the BasicCLI object
        '''
        if self.__matcher == None:
            raise RuntimeError("Matcher has not been set.")

        threads = self.parser.parse_text()
        posts = self.processor.get_posts(threads)
        questions, answers = self.processor.create_data_structures(posts)

        # use dictionary to get sentence embeddings per subject
        e = SentEmbeddings(questions, answers)
        # model, dict = e.doc2vec()

        # load model and find similarities
        model = Doc2Vec.load("app/embedder/pretrained/d2v.model")
        vect_q = e.vectorised_data(model)
        while True:
            question = input("Please enter your question >> ")
            print(f'QUESTIONS: {question}')
            print()
            print("TOP SUGGESTED:")
            dict = self.embedder.get_similarity(model, question, vect_q)
            top_suggestions = e.get_top(dict, 3)
            print()
            temp = input(
                "Please Enter the suggested number (0 if no suggestion helps)>> ")
            suggestion = top_suggestions[int(temp)-1]
            print()
            e.get_question_answers(suggestion)
            print()
