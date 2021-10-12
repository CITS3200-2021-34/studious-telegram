from ..parser import write_to_json
from ..parser.parser import preprocess
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

    def __init__(self, matcher: AbstractQuestionMatcher, summariser: AbstractSummarisation, questions, target_model: str):
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

    def get_first_suggestions(self, question: str):
        '''
        This method finds the suggestion based of the
        subject title only

        @param  question - the subject title
        @return text_vec - the embedding of the subject
                top_suggestions - the top suggestions above threshold
        '''
        suggestions, title_vec = self.__matcher.getSuggestions(
            question, "")
        print(f'QUESTIONS: {question}\n')
        top_questions = []
        for i, suggestion in enumerate(suggestions):
            if i >= 10:
                break
            if(suggestion[1] < 0.6):
                pass
            else:
                author = "Student"
                suggested_question = suggestion[0]
                top_questions.append(suggested_question)
                if(self.__questions[suggested_question]['From'] == "chris.mcdonald@uwa.edu.au"):
                    author = "Lecturer"
                if(self.__questions[suggested_question]['From'] == "poster013@student.uwa.edu.au"):
                    author = "Tutor"
                print(f"{i + 1}: {suggestions[i]} ({author})")
        if len(top_questions) == 0:
            print('No quality matches were found.')
        print("")
        return title_vec, top_questions

    def get_second_suggestions(self, question: str, body_text: str):
        '''
        This method find the suggested questions based of both the
        question and the subject line

        @param  question - the question 
                body_text - the question body
        @return text_vec - the embedding of the question + subject
                top_suggestions - the top suggestions above threshold
        '''

        # Summarise the response, and find the embedding
        summarisation = self.__summariser.getSummarisations(
            body_text)
        suggestions, text_vec = self.__matcher.getSuggestions(question,
                                                              summarisation)

        # show the top 10 suggestions above threshold
        top_questions = []
        print(f'QUESTIONS: {question}\n')
        for i, suggestion in enumerate(suggestions):
            if i >= 10:
                break
            if(suggestion[1] < 0.3):
                pass
            else:
                author = "Student"
                suggested_question = suggestion[0]
                top_questions.append(suggested_question)
                if(self.__questions[suggested_question]['From'] == "chris.mcdonald@uwa.edu.au"):
                    author = "Lecturer"
                if(self.__questions[suggested_question]['From'] == "poster013@student.uwa.edu.au"):
                    author = "Tutor"
                print(f"{i + 1}: {suggestions[i]} ({author})")
        if len(top_questions) == 0:
            print('No quality matches were found.')
        print("")
        return text_vec, top_questions

    def start(self):
        '''
        Prints the top 10 question suggestions, based on the user's command 
        line entry.

        :param self: Instance of the BasicCLI object
        '''
        if self.__matcher == None:
            raise RuntimeError("Matcher has not been set.")

        empty = False  # if the string by the user is empty set flag
        while True:

            question = questionary.text(
                "what is the title of your Question").ask()

            if question == "":
                empty = True
                print("ERROR : String can't be empty.")
                pass

            else:
                print("\nLoading Suggestions....\n")
                title_vec, top_questions = self. get_first_suggestions(
                    question)

                confirm = False
                if len(top_questions) != 0 and questionary.confirm("Would you like to view these suggestions? (Select 1 suggestion only)").ask():
                    num = questionary.checkbox(
                        'Select questions',
                        choices=top_questions
                    ).ask()
                    if len(num) != 0:
                        confirm = True
                        question = num[0]
                        self.print_question(question)
                        self.print_answers(question)

                # If the user declines the suggestions, prompt user to write a body to the question.
                if not confirm:
                    while(True):
                        body_text = questionary.text(
                            "What is your question?").ask()
                        if(body_text == 'q'):
                            break
                        if(body_text == ""):
                            print("ERROR : String can't be empty.")

                        if(body_text != ""):
                            break

                    if body_text == 'q':  # allow user to quit the prompt
                        print("\nThank you for using our program :)\n")
                        break

                    else:
                        # get suggestions based off subject + text body
                        text_vec, top_questions = self.get_second_suggestions(
                            question, body_text)

                    # Ask the user whether they wish to view the suggestions
                    confirm = False
                    if len(top_questions) != 0 and questionary.confirm("Would you like to view these suggestions? (Select 1 suggestion only)").ask():
                        num = questionary.checkbox(
                            'Select questions',
                            choices=top_questions
                        ).ask()
                        if len(num) != 0:
                            confirm = True
                            question = num[0]
                            self.print_question(question)
                            self.print_answers(question)
                    if not confirm:
                        # Write the response to the json file if they wish to submit
                        title_vec = title_vec.numpy().tolist()
                        text_vec = text_vec.numpy().tolist()
                        write_to_json(question, body_text,
                                      title_vec, text_vec, self.__model)

            if not empty:
                if not questionary.confirm("Would you like to ask another question?").ask():
                    print("\nThank you for using our program :)\n")
                    break
