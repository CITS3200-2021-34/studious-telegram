from .interface import BasicCLI, TornadoWebInterface
from .domain import UniversalEncoder, SentBERT, Doc2Vec, T5
from .parser import parseQuestionsAnswersFromFile
from .parser import old_parseQuestionsAnswersFromFile


class App():
    def __init__(self, target_model: str, target_interface: str):
        questions = parseQuestionsAnswersFromFile(
            'app/testfiles/help2002-2017.txt')

        if target_model == "UniversalEncoder":
            questionMatcher = UniversalEncoder()
        elif target_model == "BERT":
            questionMatcher = SentBERT()
        elif target_model == "doc2vec":
            questionMatcher = Doc2Vec()
        else:
            raise ValueError(f"targetModel ({target_model}) is not valid")

        questionMatcher.addQuestions(questions)

        if target_interface == "cli":
            self.__interface = BasicCLI(questionMatcher, questions)
        elif target_interface == "web":
            self.__interface = TornadoWebInterface(8080, questionMatcher, questions)
        else:
            raise ValueError(
                f"target_interface ({target_interface}) is not valid")

    def start(self) -> None:
        self.__interface.start()
