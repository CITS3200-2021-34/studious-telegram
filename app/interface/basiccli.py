from ..domain import AbstractQuestionMatcher
from .userinterface import AbstractUserInterface

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
    

    def setQuestionMatcher(self, matcher: AbstractQuestionMatcher):
        self.__matcher = matcher

    def start(self):
        '''
        Prints the top 10 question suggestions, based on the user's command 
        line entry.

        :param self: Instance of the BasicCLI object
        '''
        if self.__matcher == None:
            raise RuntimeError("Matcher has not been set.")

        while True:
            question = input("Please enter your question >> ")
            print("Loading....")
            suggestions = self.__matcher.getSuggestions(question)

            print(f'QUESTIONS: {question}')
            
            for i in range(len(suggestions)):
                if i >= 10:
                    break
                print(f"{i + 1}: {suggestions[i]}")
