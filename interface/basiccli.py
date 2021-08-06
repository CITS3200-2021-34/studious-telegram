import domain
from interface import AbstractUserInterface

class BasicCLI(AbstractUserInterface):
    def setQuestionMatcher(self, matcher: domain.AbstractQuestionMatcher):
        self.matcher = matcher

    def start(self):
        if self.matcher == None:
            raise RuntimeError("Matcher has not been set.")
        while True:
            question = input("Please enter your question >> ")
            suggestions = self.matcher.getSuggestions(question)

            print("Suggestions:")
            for suggestion in suggestions:
                print(suggestion)
