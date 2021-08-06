from ..domain import AbstractQuestionMatcher
from .userinterface import AbstractUserInterface

class BasicCLI(AbstractUserInterface):
    __matcher: AbstractQuestionMatcher = None

    def __init__(self, matcher: AbstractQuestionMatcher) -> None:
        super().__init__()
        self.setQuestionMatcher(matcher)

    def setQuestionMatcher(self, matcher: AbstractQuestionMatcher):
        self.__matcher = matcher

    def start(self):
        if self.__matcher == None:
            raise RuntimeError("Matcher has not been set.")
        
        while True:
            question = input("Please enter your question >> ")
            suggestions = self.__matcher.getSuggestions(question)

            print("Suggestions:")
            for suggestion in suggestions:
                print(suggestion)
