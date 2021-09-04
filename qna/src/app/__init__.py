from app.interface import BasicCLI, WebInterface
from app.domain import UniversalEncoder, SentBERT, Doc2Vec
from app.parser import parseQuestionsAnswersFromFile
from app.parser import JsonLoader


class App():

    def __init__(self, target_model="UniversalEncoder", target_interface="cli") -> None:

        # Uncomment this if you want to create a new json file
        '''
        questions, answers = parseQuestionsAnswersFromFile(
            'app/testfiles/help2002-2017.txt')
        '''
        json = JsonLoader('app/storage/questions2017_UE.json')
        questions = json.read_data()

        if target_model == "UniversalEncoder":
            questionMatcher = UniversalEncoder(questions)
        elif target_model == "BERT":
            questionMatcher = SentBERT(questions)
        elif target_model == "doc2vec":
            questionMatcher = Doc2Vec(questions)
        else:
            raise ValueError(f"targetModel ({target_model}) is not valid")

        if target_interface == "cli":
            self.__interface = BasicCLI(questionMatcher, questions)
        elif target_interface == "web":
            self.__interface = WebInterface(questionMatcher)
        else:
            raise ValueError(f"target_interface ({target_interface}) is not valid")

    def start(self) -> None:
        self.__interface.start()
