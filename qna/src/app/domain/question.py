from typing import List


class Question(object):
    def __init__(self, subject: str, body: str, question_author: str, answers: List[str], answer_authors: List[str]) -> None:
        self.subject = subject
        self.body = body
        self.question_author = question_author
        self.answers = answers
        self.answer_authors = answer_authors
