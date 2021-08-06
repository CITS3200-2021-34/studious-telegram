from app.interface import BasicCLI
from app.domain import KeywordMatcher

class App():
    def __init__(self) -> None:
        questionMatcher = KeywordMatcher()

        self.__cli = BasicCLI(questionMatcher)

    def start(self) -> None:
        self.__cli.start()