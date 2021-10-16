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

        if subject not in added_questions:
            added_questions.add(subject)

            questions[subject] = Question(subject, body, author, [], [])
        else:
            questions[subject].answers += [body]
            questions[subject].answer_authors += [author]

    return list(questions.values())


def helper_preprocess(threads):

    # Load in the models
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    sum_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Get the unique questions from the subject line.
    # Summarise the body of the question as well.
    s = set()
    preprocessed_subjects = []
    preprocessed_texts = []
    begin = time.time()
    for i in threads:
        subject = email.message_from_string(i)['Subject']
        if subject not in s:
            p = preprocess(subject)
            text = email.message_from_string(i)._payload
            preprocessed_subjects.append(p)
            sum = get_summarisation(text, sum_model, tokenizer)
            sum_preprocess = preprocess(sum)
            final_sum = p + sum_preprocess
            preprocessed_texts.append(final_sum)
        s.add(subject)
    end = time.time()
    print(f'Time to compelete preprocess {end-begin}')

    # Get the embeddings for each the subject and the text
    begin = time.time()
    embeddings_subjects = model(preprocessed_subjects)  # for subject line
    embeddings_texts = model(preprocessed_texts)  # for text body
    end = time.time()
    print(f'Time to compelete embedding {end-begin}')

    return embeddings_subjects, embeddings_texts


def get_summarisation(data, model, tokenizer):
    '''
    This method summarises the text body below 50 tokens (words)

    @param: data  - the text body from a question/subject line.
            model - the t5 text summarisation model to inference from
    @retrun output_text - The below 50 summarised phrase/text.
    '''

    input = tokenizer.encode(data, return_tensors="pt",
                             max_length=512, truncation=True)
    # generate the summarization output
    outputs = model.generate(
        input,
        max_length=30,
        min_length=5,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True)
    # just for debugging
    return tokenizer.decode(outputs[0])
