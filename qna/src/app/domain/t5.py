
from .summarisation import AbstractSummarisation
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5(AbstractSummarisation):
    '''
    This is a class for text summarisation of question using Google T5
    Text Generator.
    '''

    def __init__(self, max_length, min_length, model_name):
        '''
        Load in the models on initialisation.
        '''
        self.max_length = max_length  # max amount of words in summarised text
        self.min_length = min_length  # min amount of words in summarised text
        self.__tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.__model = T5ForConditionalGeneration.from_pretrained(model_name)

    def getSummarisations(self, question: str):
        '''
        Creates a new summarised text from the question body.
        Currently, max length of 50 words and min length of 5 
        words

        @param - question - the question body to summarise
        @return - summarised text from question
        '''

        data = "summarize: " + question
        inputs = self.__tokenizer.encode(data,
                                         return_tensors='pt',
                                         max_length=512,
                                         truncation=True)

        outputs = self.__model.generate(
            inputs,
            max_length=self.max_length,
            min_length=self.min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True)

        return self.__tokenizer.decode(outputs[0], skip_special_tokens=True)
