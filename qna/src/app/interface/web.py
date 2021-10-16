from typing import Awaitable, List, Optional
import urllib.parse
import json

import tornado.ioloop
import tornado.web

from .userinterface import AbstractUserInterface
from ..domain.questionmatcher import AbstractQuestionMatcher
from ..domain.question import Question


class BaseHandler(tornado.web.RequestHandler):
    def prepare(self) -> Optional[Awaitable[None]]:
        if "Content-Type" not in self.request.headers:
            self.json_args = None
            return

        if self.request.headers["Content-Type"] != "application/json":
            self.json_args = None
            return

        self.json_args = json.loads(self.request.body)


class SuggestionHandler(BaseHandler):
    def initialize(self, questionMatcher: AbstractQuestionMatcher) -> None:
        self.questionMatcher = questionMatcher

    def get(self, rawQuestion, rawBody) -> None:
        question = urllib.parse.unquote(rawQuestion)
        body = urllib.parse.unquote(rawBody)

        suggestions, _ = self.questionMatcher.getSuggestions(question, body)

        response = {
            "matches": [],
            "question": question
        }

        for suggestion in suggestions:
            response["matches"].append({
                "question": suggestion[0],
                "similarity": suggestion[1],
                "flags": suggestion[2]
            })

        self.write(response)


class NewQuestionHandler(BaseHandler):
    def initialize(
            self,
            questionMatcher: AbstractQuestionMatcher,
            questions: List[Question]) -> None:
        self.questionMatcher = questionMatcher
        self.questions = questions

    async def post(self) -> None:
        if self.json_args is None:
            self.set_status(400)
            return

        # TODO add option to send other relevant data (date, answers, etc)
        newQuestion = Question(
            self.json_args['subject'],
            self.json_args['body'],
            "",
            [],
            [],
            [])

        self.questionMatcher.addQuestions([newQuestion])

        self.questions.append(newQuestion)

        self.set_status(200)


class GetQuestionHandler(BaseHandler):
    def initialize(self, questions: List[Question]) -> None:
        self.questions = questions

    async def get(self, rawQuestion) -> None:
        question = urllib.parse.unquote(rawQuestion)

        for pastQuestion in self.questions:
            if pastQuestion.subject == question:
                self.write({
                    "subject": pastQuestion.subject,
                    "body": pastQuestion.body,
                    "answers": pastQuestion.answers
                })
                return

        self.set_status(404)


class TornadoWebInterface(AbstractUserInterface):
    def __init__(
            self,
            port,
            questionMatcher: AbstractQuestionMatcher,
            questions: List[Question]) -> None:
        super().__init__()

        self.__port = port
        self.__questionMatcher = questionMatcher
        self.__questions = questions

    def start(self) -> None:
        app = tornado.web.Application([(r"/api/suggestion/([^/]+)/([^/]*)",
                                      SuggestionHandler,
                                      {"questionMatcher": self.__questionMatcher}),
            (r"/api/question/new",
             NewQuestionHandler,
             {"questionMatcher": self.__questionMatcher,
              "questions": self.__questions}),
            (r"/api/question/get/([^/]+)",
             GetQuestionHandler,
             {"questions": self.__questions}),
        ])

        app.listen(self.__port)

        print(f"Server started: http://localhost:{self.__port}")
        tornado.ioloop.IOLoop.current().start()
