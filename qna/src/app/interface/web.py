from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from ..domain.questionmatcher import AbstractQuestionMatcher
from urllib.parse import unquote
import json

from .userinterface import AbstractUserInterface

class MyServer(BaseHTTPRequestHandler):
    matcher: AbstractQuestionMatcher = None

    def do_GET(self):
        request_path = self.path.split('/')

        print(f"Received request: {request_path}")

        if len(request_path) != 4:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Page not found.", "utf-8"))
            return

        if request_path[1] != "api" or request_path[2] != "question" or request_path[3] == "":
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Page not found.", "utf-8"))
            return

        question = unquote(request_path[-1])

        print(f"Getting suggestions for {question}")
        suggestions = self.matcher.getSuggestions(question)
        print(f"Got {len(suggestions)} suggestions")

        response = {
            "matches": [],
            "question": question
        }

        for suggestion in suggestions:
            response["matches"].append({
                "question": suggestion[0], 
                "similarity": suggestion[1]
            })

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(json.dumps(response), "utf-8"))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """This class allows to handle requests in separated threads.
        No further content needed, don't touch this. """
class WebInterface(AbstractUserInterface):

    def __init__(self, matcher) -> None:
        super().__init__()
        MyServer.matcher = matcher

    def setQuestionMatcher(self, matcher: AbstractQuestionMatcher):
        pass

    def start(self):
        hostname = "0.0.0.0"
        server_port = 8080

        webServer = ThreadedHTTPServer((hostname, server_port), MyServer)
        print(f"Server started http://{hostname}:{server_port}")

        try:
            webServer.serve_forever()
        except KeyboardInterrupt:
            webServer.socket.close()

        webServer.server_close()
        print("Server stopped.")