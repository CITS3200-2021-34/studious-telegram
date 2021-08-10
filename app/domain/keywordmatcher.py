from .questionmatcher import AbstractQuestionMatcher

class KeywordMatcher(AbstractQuestionMatcher):
    '''
    Class matching suggestions to a question, based off specific key words.
    '''
    def __init__(self):
        '''
        Constructor for the KeywordMatcher class.

        :param self: Instance of the KeywordMatcher object
        '''
        super().__init__()

        print("Keyword matcher loaded")

    def getSuggestions(self, question: str) -> list[str]:
        '''
        Given a question, determines which question subjects contain the same 
        keywords and should be suggested.

        :param self: Instance of the KeywordMatcher object
        :param question: A question string
        :return suggestions: List of suggested question subjects
        '''
        # create a dictionary mapping key words to previous question subjects
        pastQuestions = {"pointer": "How do I use pointers in C?", "code": "How do I code well?"}
        suggestions = []

        for word in question.split(" "):
            question = pastQuestions.get(word)

            if (question != None):
                suggestions.append(question)
        
        return suggestions