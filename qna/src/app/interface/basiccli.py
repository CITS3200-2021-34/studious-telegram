from ..domain import AbstractQuestionMatcher
from .userinterface import AbstractUserInterface
import questionary


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

            year = questionary.select(
                "What year do you want to search?",
                choices=["2017", "2018", "2019"],
            ).ask()
            week = questionary.text("What semester week is this (1-12)?").ask()
            question = questionary.text("What is your Question?").ask()
            
            '''
            If the user does not ask a question i.e. presses enter with no questions,
            the program is exited. 
            '''
            if not question:
                print("\nThank you for using our program :)\n")
                break

            print("\nLoading Suggestions....\n")
            suggestions = self.__matcher.getSuggestions(question)

            print(f'QUESTIONS: {question}\n')
            
            for i in range(len(suggestions)):
                if i >= 10:
                    break
                print(f"{i + 1}: {suggestions[i]}")   
            print("")

            if questionary.confirm("Would you like to view these suggestions?").ask():
                questionary.checkbox(
                    'Select questions',
                    choices=
                        suggestions[:10]
                    ).ask()

                '''
                show the user the contents 
                '''     
            if not questionary.confirm("Would you like to ask another question?").ask():
                print("\nThank you for using our program :)\n")
                break
