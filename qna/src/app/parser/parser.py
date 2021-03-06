from typing import Dict, List, Tuple
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
    :param target_model: The target model
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
        author = msg['From']
        date = msg['Date']

        if subject not in added_questions:
            added_questions.add(subject)

            questions[subject] = Question(subject, body, author, date, [], [], [])
        else:
            questions[subject].answers += [body]
            questions[subject].answer_authors += [author]
            questions[subject].answer_dates += [date]

    return list(questions.values())
