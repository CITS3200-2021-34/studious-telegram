from app.embedder.sentence_embeddings import SentEmbeddings
from ..domain import AbstractQuestionMatcher
from ..preprocessor.preprocess import PreProcessor
from ..formatter.dataparser import DataParser
from ..embedder import SentEmbeddings
from .userinterface import AbstractUserInterface
from gensim.models.doc2vec import Doc2Vec
import os


class BasicCLI(AbstractUserInterface):
    __matcher: AbstractQuestionMatcher = None

    def __init__(self, matcher: AbstractQuestionMatcher) -> None:
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
        posts = self.processor.get_posts(self.parser.parse_text())
        self.questions, self.answer = self.processor.create_data_structures(
            posts)

    def start(self):
        if self.__matcher == None:
            raise RuntimeError("Matcher has not been set.")

       #threads = self.parser.parse_text()
       # posts = self.processor.get_posts(threads)
       # questions, answers = self.processor.create_data_structures(posts)

        # use dicitonary to get sentence embeddings per subject
        #e = SentEmbeddings(questions, answers)
        #model, dict = e.doc2vec()

        # Load model and find similarities
        #model = Doc2Vec.load("app/embedder/pretrained/d2v.model")
        #vect_q = e.vectorised_data(model)
        while True:
            question = input("Please enter your question >> ")
            print(f'QUESTIONS: {question}')
            print()
            print("TOP SUGGESTED:")
            dict = self.embedder.universal_encoder(question)
            top_suggestions = self.embedder.get_top(dict, 10)
            print()
            temp = input(
                "Please Enter the suggested number (0 if no suggestion helps)>> ")
            suggestion = top_suggestions[int(temp)-1]
            print()
            self.embedder.get_question_answers(suggestion)
            print()
