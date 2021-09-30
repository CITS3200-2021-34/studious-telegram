from typing import Dict, List, Tuple
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import string
import email

from ..domain.question import Question


def parseQuestionsAnswersFromFile(filePath: str) -> List[Question]:
    '''
    Returns a dictionary containing each question in filePath, and a
    dictionary containing each answer in filePath.

    :param filePath: File to be parsed
    :return questions: Dictionary of question threads
    :return answers: Dictionary of answer threads
    '''
    threads = parseThreadsFromFile(filePath)
    return getPostsFromThreads(threads)


def parseThreadsFromFile(filePath: str):
    '''
    Arranges the contents of a file into separate threads (ie. an individual
    question or answer) that are stored as list items.

    :param filePath: File to be parsed
    :return threads: List of question/answer strings
    '''
    with open(filePath, 'r') as file:
        line = file.readline()
        line = file.readline()  # Not sure best way to do this, needed to skip the first line
        str = ''
        threads = []

        while line:
            # a date line indicates a new question/answer
            if(line[:4] == 'Date' and len(str) > 0):
                str = str.rstrip('\n')
                threads.append(str)
                str = ''
            str += line
            line = file.readline()
        threads.append(str)
    return threads


def getPostsFromThreads(threads) -> List[Question]:
    '''
    For each item in a given threads list, a list containing all contents
    is stored in a json formatted dictionary to be stored in a JSON file

    :param threads: List of question/answer strings
    :return None:
    '''

    # Stores each question and answer into a json format and writes it to file
    # based on the target_model.
    added_questions = set()
    questions: Dict[str, Question] = {}

    for i in threads:

        msg = email.message_from_string(i)

        subject = msg['Subject']
        body = msg.get_payload()

        if subject not in added_questions:
            added_questions.add(subject)

            questions[subject] = Question(subject, body, [])
        else:
            questions[subject].answers += [body]

    return list(questions.values())


def get_summarisation(data, tokenizer, model):
    input = tokenizer.encode(data, return_tensors="pt",
                             max_length=512, truncation=True)
    # generate the summarization output
    outputs = model.generate(
        input,
        max_length=50,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True)
    # just for debugging
    return tokenizer.decode(outputs[0])


def preprocess(data):

    # Tokenize question and remove punctuation and lower strings
    data = word_tokenize(data)
    data = [i for i in data if i not in string.punctuation]
    data = [i.lower() for i in data]

    # Convert words to stem form
    # e.g. 'playing' is converted to 'play'
    lemmatizer = WordNetLemmatizer()
    data = [lemmatizer.lemmatize(i) for i in data]

    # Remove stopwords as they don't add value to the sentence meaning
    # and select only the top 10 stop words.
    # e.g. 'the' is not a valuable word
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords = stopwords[0:10]
    data = [i for i in data if i not in stopwords]

    return " ".join(data)
