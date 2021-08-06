from .questionmatcher import AbstractQuestionMatcher

class KeywordMatcher(AbstractQuestionMatcher):
    def __init__(self):
        super().__init__()

        print("Keyword matcher loaded")

    def getSuggestions(self, question: str) -> list[str]:
        pastQuestions = {"pointer": "How do I use pointers in C?", "code": "How do I code well?"}
        suggestions = []

        for word in question.split(" "):
            question = pastQuestions.get(word)

            if (question != None):
                suggestions.append(question)
        
        return suggestions