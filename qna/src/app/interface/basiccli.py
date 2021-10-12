from ..parser import write_to_json
from ..domain import AbstractQuestionMatcher
from ..domain import AbstractSummarisation
from .userinterface import AbstractUserInterface
import questionary


class BasicCLI(AbstractUserInterface):
    '''
    Class for constructing a basic command line interface for the user
    to interact with when asking questions and viewing suggestions.
    '''
    __matcher: AbstractQuestionMatcher = None

    def __init__(
            self,
            matcher: AbstractQuestionMatcher,
            summariser: AbstractSummarisation,
            questions,
            target_model: str):
        '''
        Constructor for the BasicCLI class.

        :param self: Instance of the BasicCLI object
        '''
        super().__init__()
        self.setQuestionMatcher(matcher)
        self.setSummarisation(summariser)
        self.__questions = questions
        self.__model = target_model

    def setQuestionMatcher(self, matcher: AbstractQuestionMatcher):
        self.__matcher = matcher

    def setSummarisation(self, summariser: AbstractSummarisation):
        self.__summariser = summariser

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
        if self.__matcher is None:
            raise RuntimeError("Matcher has not been set.")

        while True:
            '''
            year = questionary.select(
                "What year do you want to search?",
                choices=["2017", "2018", "2019"],
            ).ask()
            week = questionary.text("What semester week is this (1-12)?").ask()
            '''
            question = questionary.text(
                "what is the title of your Question").ask()

            '''
            If the user does not ask a question i.e. presses enter with no questions,
            the program is exited.
            '''
            if not question:
                print("\nThank you for using our program :)\n")
                break

            print("\nLoading Suggestions....\n")
            suggestions, title_vec = self.__matcher.getSuggestions(
                question, False)

            suggestions = [suggestion[0] for suggestion in suggestions]

            print(f'QUESTIONS: {question}\n')

            for i in range(len(suggestions)):
                if i >= 10:
                    break
                author = "Student"
                suggested_question = suggestions[i]
                if(self.__questions[suggested_question]['From'] == "chris.mcdonald@uwa.edu.au"):
                    author = "Lecturer"
                if(self.__questions[suggested_question]['From'] == "poster013@student.uwa.edu.au"):
                    author = "Tutor"
                print(f"{i + 1}: {suggestions[i]} ({author})")
            print("")
            flag = False
            if questionary.confirm(
                    "Would you like to view these suggestions?").ask():
                num = questionary.checkbox(
                    'Select questions',
                    choices=suggestions[:10]
                ).ask()
                if len(num) != 0:
                    flag = True
                    self.print_question(num[0])
                    self.print_answers(num[0])
            if not flag:
                body_text = questionary.text(
                    "What is your question?").ask()

                if not body_text:
                    print("\nThank you for using our nprogram :)\n")

                else:
                    summarisation = self.__summariser.getSummarisations(
                        str(body_text))
                    summarisation = question + summarisation
                    suggestions, text_vec = self.__matcher.getSuggestions(
                        summarisation)

                    suggestions = [suggestion[0] for suggestion in suggestions]

                    print(f'QUESTIONS: {question}\n')

                    for i in range(len(suggestions)):
                        if i >= 10:
                            break
                        author = "Student"
                        suggested_question = suggestions[i]
                        if(self.__questions[suggested_question]['From'] == "chris.mcdonald@uwa.edu.au"):
                            author = "Lecturer"
                        if(self.__questions[suggested_question]['From'] == "poster013@student.uwa.edu.au"):
                            author = "Tutor"
                        print(f"{i + 1}: {suggestions[i]} ({author})")
                    print("")

                    flag = False
                    if questionary.confirm(
                            "Would you like to view these suggestions?").ask():
                        num = questionary.checkbox(
                            'Select questions',
                            choices=suggestions[:10]
                        ).ask()
                        if len(num) != 0:
                            flag = True
                            self.print_question(num[0])
                            self.print_answers(num[0])
                    if not flag:
                        title_vec = title_vec.numpy().tolist()
                        text_vec = text_vec.numpy().tolist()
                        write_to_json(question, body_text,
                                      title_vec, text_vec, self.__model)

            if not questionary.confirm(
                    "Would you like to ask another question?").ask():
                print("\nThank you for using our program :)\n")
                break
