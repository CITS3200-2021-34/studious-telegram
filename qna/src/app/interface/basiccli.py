from typing import Dict, List
from ..domain.question import Question
from ..parser import write_to_json
from ..domain import AbstractQuestionMatcher
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
            questions: List[Question]):
        '''
        Constructor for the BasicCLI class.

        :param self: Instance of the BasicCLI object
        '''
        super().__init__()
        self.__matcher = matcher
        self.__questions_map: Dict[str, Question] = {}

        for question in questions:
            self.__questions_map[question.subject] = question

    def print_question(self, question_subject_line):
        '''
        Prints the provided information of a chosen question.

        :param self: Instance of the BasicCLI object
        :param chosen_questions: A chosen question from the question dictionary
        '''

        question = self.__questions_map[question_subject_line]

        self.print_post(
            question.question_date,
            question.subject,
            question.question_author,
            question.body)

        for i, answer_body in enumerate(question.answers):
            self.print_post(
                question.answer_dates[i],
                question.subject,
                question.answer_authors[i],
                answer_body)

    def print_post(self, date: str, subject: str, author: str, body: str):
        print()
        print(f'Date: {date}')
        print(f'Subject: {subject}')
        print(f'From: {author}')
        print()
        print(body)
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
        suggestions, _ = self.__matcher.getSuggestions(question, "")

        top_questions = [suggestion[0]
                         for suggestion in suggestions[:10] if suggestion[1] > 0.6]

        return top_questions

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
        suggestions, _ = self.__matcher.getSuggestions(question, body_text)

        # show the top 10 suggestions above threshold
        top_suggestions = [suggestion[0]
                           for suggestion in suggestions[:10] if suggestion[1] > 0.3]

        return top_suggestions

    def get_nonempty_user_input(self, prompt: str):
        question = questionary.text(prompt).ask()

        while question == "":
            print("ERROR : Input cannot be empty")

            question = questionary.text(prompt).ask()

        return question

    def print_suggestion_summary(self, suggestions: List[str]):
        print("Suggestions")
        print()
        print("---------------------------------------------")
        print()

        for i, subject_line in enumerate(suggestions):
            suggestion = self.__questions_map[subject_line]

            author = "Student"
            if(suggestion.question_author == "chris.mcdonald@uwa.edu.au"):
                author = "Lecturer"
            if(suggestion.question_author == "poster013@student.uwa.edu.au"):
                author = "Tutor"

            print(f"{i + 1}: {suggestion.subject} ({author})")

        if len(suggestions) == 0:
            print('No quality matches were found.')

    def display_suggestion(self, suggestions):
        selected_suggestion = questionary.select(
            'Select a suggestions',
            choices=suggestions
        ).ask()

        self.print_question(selected_suggestion)

    def start(self):
        '''
        Prints the top 10 question suggestions, based on the user's command
        line entry.

        :param self: Instance of the BasicCLI object
        '''
        if self.__matcher is None:
            raise RuntimeError("Matcher has not been set.")

        while True:
            # First Suggestions
            question = self.get_nonempty_user_input(
                "What is the title of your question?")

            print("\nLoading Suggestions....\n")

            first_suggestions = self.get_first_suggestions(question)
            self.print_suggestion_summary(first_suggestions)

            if len(first_suggestions) != 0 and questionary.confirm(
                    "Would you like to view the answers to a suggestion (y) or write your question body (n)?").ask():
                self.display_suggestion(first_suggestions)

                continue

            # Second Suggestions
            # If the user declines the suggestions, prompt user to write a body to the question.
            body_text = self.get_nonempty_user_input(
                "What is the body of your question?")

            print("\nLoading Suggestions....\n")

            # get suggestions based off subject + text body
            second_suggestions = self.get_second_suggestions(
                question, body_text)
            self.print_suggestion_summary(second_suggestions)

            # Ask the user whether they wish to view the suggestions
            if len(second_suggestions) != 0 and questionary.confirm(
                    "Would you like to view the answers to a suggestions?").ask():
                self.display_suggestion(second_suggestions)

            if not questionary.confirm("Would you like to ask another question?").ask():
                break

        print("\nThank you for using our program :)\n")
