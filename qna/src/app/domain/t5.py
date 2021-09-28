from .summarisation import AbstractSummarisation
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5(AbstractSummarisation):

    def __init__(self):
        self.__tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.__model = T5ForConditionalGeneration.from_pretrained(
            "t5-base")

    def getSummarisations(self, question: str):

        data = "summarize: "+question
        inputs = self.__tokenizer.encode(data,
                                         return_tensors='pt',
                                         max_length=512,
                                         truncation=True)

        outputs = self.__model.generate(
            inputs,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True)

        return self.__tokenizer.decode(outputs[0], skip_special_tokens=True)
