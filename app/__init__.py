from app.interface import BasicCLI
from app.domain import UniversalEncoder, SentBERT, Doc2Vec 
from app.parser import parseQuestionsAnswersFromFile

class App():
    def __init__(self, targetModel = "UniversalEncoder") -> None:


        questions, answers = parseQuestionsAnswersFromFile('app/testfiles/help2002-2017.txt')

        if targetModel == "UniversalEncoder":
            questionMatcher = UniversalEncoder(questions, answers)
        elif targetModel == "BERT":
            questionMatcher = SentBERT(questions)
        elif targetModel == "doc2vec":
            questionMatcher = Doc2Vec(questions)
        else: 
            raise ValueError(f"targetModel ({targetModel}) is not valid")

        self.__cli = BasicCLI(questionMatcher)

    def start(self) -> None:
        self.__cli.start()
