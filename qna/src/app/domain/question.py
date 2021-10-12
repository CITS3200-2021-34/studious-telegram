from typing import List


class Question(object):
    def __init__(self, subject: str, body: str, answers: List[str]) -> None:
        self.subject = subject
        self.body = body
        self.answers = answers
