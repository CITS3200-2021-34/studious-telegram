from typing import List
from domain import AbstractQuestionMatcher

class KeywordMatcher(AbstractQuestionMatcher):
    def __init__(self):
        super().__init__()

        print("Keyword matcher loaded")

    def getSuggestions(self, question: str) -> List[str]:
        pastQuestions = {"pointer": "How do I use pointers in C?", "code": "How do I code well?"}
        suggestions = []

        for word in question.split(" "):
            question = pastQuestions.get(word)

            if (question != None):
                suggestions.append(question)
        
        return suggestions