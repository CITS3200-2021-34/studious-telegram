from typing import List


class Question(object):
    def __init__(
            self,
            subject: str,
            body: str,
            question_author: str,
            question_date: str,
            answers: List[str],
            answer_authors: List[str],
            answer_dates: List[str]) -> None:
        self.subject = subject
        self.body = body
        self.question_author = question_author
        self.question_date = question_date
        self.answers = answers
        self.answer_authors = answer_authors
        self.answer_dates = answer_dates
