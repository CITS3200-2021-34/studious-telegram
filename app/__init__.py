from app.interface import BasicCLI
from app.domain import UniversalEncoder, SentBERT, Doc2Vec
from app.parser import parseQuestionsAnswersFromFile
from app.parser import JsonLoader


class App():

    def __init__(self, targetModel="UniversalEncoder") -> None:

        # Uncomment this if you want to create a new json file
        '''
        questions, answers = parseQuestionsAnswersFromFile(
            'app/testfiles/help2002-2017.txt')
        '''
        json = JsonLoader('app/storage/questions2017_UE.json')
        questions = json.read_data()

        if targetModel == "UniversalEncoder":
            questionMatcher = UniversalEncoder(questions)
        elif targetModel == "BERT":
            questionMatcher = SentBERT(questions)
        elif targetModel == "doc2vec":

            questionMatcher = Doc2Vec(questions)
        else:
            raise ValueError(f"targetModel ({targetModel}) is not valid")

        self.__cli = BasicCLI(questionMatcher, questions)

    def start(self) -> None:
        self.__cli.start()
