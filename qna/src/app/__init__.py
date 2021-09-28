from .interface import BasicCLI, WebInterface
from .domain import UniversalEncoder, SentBERT, Doc2Vec, T5
from .parser import parseQuestionsAnswersFromFile
from .parser import JsonLoader
import os.path
import os


class App():

    def __init__(self, target_model, target_interface):

        # If the diretory does not exists create one
        if not os.path.exists('app/storage'):
            os.makedirs('app/storage')

        # If we don't have any files already in the directory create
        # based off the target_model
        if len(os.listdir("app/storage")) == 0:
            questions = parseQuestionsAnswersFromFile(
                'app/testfiles/help2002-2017.txt', target_model)

        # Find the first file that contains the target model
        else:
            flag = False
            for file in os.listdir('app/storage'):
                if target_model in file:
                    file = f'app/storage/{file}'
                    json = JsonLoader(file)
                    questions = json.read_data()
                    flag = True
                    break
            if not flag:
                questions = parseQuestionsAnswersFromFile(
                    'app/testfiles/help2002-2017.txt', target_model)

        if target_model == "UniversalEncoder":
            questionMatcher = UniversalEncoder(questions)
        elif target_model == "BERT":
            questionMatcher = SentBERT(questions)
        elif target_model == "doc2vec":
            questionMatcher = Doc2Vec(questions)
        else:
            raise ValueError(f"targetModel ({target_model}) is not valid")

        if target_interface == "cli":
            summariser = T5()
            self.__interface = BasicCLI(
                questionMatcher, summariser, questions, target_model)
        elif target_interface == "web":
            self.__interface = WebInterface(questionMatcher)
        else:
            raise ValueError(
                f"target_interface ({target_interface}) is not valid")

    def start(self) -> None:
        self.__interface.start()
