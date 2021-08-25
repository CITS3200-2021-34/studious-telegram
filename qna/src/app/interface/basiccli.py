from ..domain import AbstractQuestionMatcher
from .userinterface import AbstractUserInterface


class BasicCLI(AbstractUserInterface):
    '''
    Class for constructing a basic command line interface for the user 
    to interact with when asking questions and viewing suggestions.
    '''
    __matcher: AbstractQuestionMatcher = None

    def __init__(self, matcher: AbstractQuestionMatcher, questions):
        '''
        Constructor for the BasicCLI class.

        :param self: Instance of the BasicCLI object
        '''
        super().__init__()
        self.setQuestionMatcher(matcher)
        self.__questions = questions

    def setQuestionMatcher(self, matcher: AbstractQuestionMatcher):
        self.__matcher = matcher

    def print_question(self, chosen_questions):
        '''
        Prints the provided information of a chosen question.

        :param self: Instance of the BasicCLI object
        :param chosen_questions: A chosen question from the question dictionary
        '''
        print()
        print(f'Date: {self.__questions[chosen_questions]["Date"]}')
        print(f'To: {self.__questions[chosen_questions]["To"]}')
        print(
            f'Received: {self.__questions[chosen_questions]["Received"]}')
        print(f'Subject: { chosen_questions}')
        print(f'From: {self.__questions[ chosen_questions]["From"]}')
        print(f'X-smile: {self.__questions[ chosen_questions]["X-smile"]}')
        print(f'X-img: {self.__questions[chosen_questions]["X-img"]}')
        print()
        print(self.__questions[chosen_questions]["Text"])
        print()
        print("---------------------------------------------")
        print()

    def print_answers(self, chosen_questions):
        '''
        For every answer to a chosen question, prints the provided information.

        :param self: Instance of the BasicCLI object
        :param chosen_questions: A chosen question from the question dictionary
        '''
        for answers in self.__questions[chosen_questions]['Answers']:
            print(f'Date: {answers["Date"]}')
            print(f'To: {answers["To"]}')
            print(
                f'Received: {answers["Received"]}')
            print(f'Subject: {chosen_questions}')
            print(f'From: {answers["From"]}')
            print(
                f'X-smile: {answers["X-smile"]}')
            print(
                f'X-img: {answers["X-img"]}')
            print()
            print(answers['Text'])
            print()
            print("---------------------------------------------")
            print()

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
            for i in range(0, 10):
                print(f"{i + 1}: {suggestions[i]}")

            print()
            selected = input("Please enter the suggested question number >> ")
            print()
            question_selected = suggestions[int(selected)-1]

            self.print_question(question_selected)
            self.print_answers(question_selected)
