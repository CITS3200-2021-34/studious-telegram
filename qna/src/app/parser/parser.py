import nltk  # Libary used in preprocessing
import tensorflow_hub as hub  # Load Universal Encoder model
import json  # Construct and use json methods
import time  # Test time of code
import numpy as np  # Help construct serialised array for json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize  # Splits data into list of tokens
import string  # Used in preprocessing
import email  # Used to construct and extract from email formated text
import logging
import tensorflow as tf                                                 #
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Remove TensorFlow Warnings

from.json_loader import JsonLoader

tf.get_logger().setLevel(logging.ERROR)


def parseQuestionsAnswersFromFile(filePath: str, target_model: str):
    '''
    Returns a dictionary containing each question in filePath, and a
    dictionary containing each answer in filePath.

    :param filePath: File to be parsed
    :return questions: Dictionary of question threads
    :return answers: Dictionary of answer threads
    '''
    threads = parseThreadsFromFile(filePath)
    file = getPostsFromThreads(threads, target_model)
    json = JsonLoader(file)
    return json.read_data()


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


def getPostsFromThreads(threads, target_model):
    '''
    For each item in a given threads list, a list containing all contents
    is stored in a json formatted dictionary to be stored in a JSON file

    :param threads: List of question/answer strings
    :return None:
    '''

    embeddings_subjects, embeddings_texts = helper_preprocess(threads)

    # Stores each question and answer into a json format and writes it to file
    # based on the target_model.
    questions = {}
    j = 0
    for i in threads:

        msg = email.message_from_string(i)

        # If the subject line is not already inside the dictionary get embeddings
        if msg['Subject'] not in questions.keys():

            # convert to a json serialized format
            text_vec = embeddings_texts[j].numpy().tolist()
            # convert to a json serialized format
            vec = embeddings_subjects[j].numpy().tolist()

            # Stores the question into correspoding key-value pair
            questions[msg['Subject']] = {'Date': msg['Date'],
                                         'To': msg['To'],
                                         'Received': msg['Received'],
                                         'Subject_vec': vec,
                                         'From': msg['From'],
                                         'X-smile': msg['X-smile'],
                                         'X-img': msg['X-img'],
                                         'Text': msg._payload,
                                         'Text_vec': text_vec,
                                         'Answers': [],
                                         }
            j += 1
        else:
            # Stores the answers to questions into its corresponding question key-value pair.
            questions[msg['Subject']]['Answers'].append({'Date': msg['Date'],
                                                         'To': msg['To'],
                                                         'Received': msg['Received'],
                                                         'Subject': msg['Subject'],
                                                         'From': msg['From'],
                                                         'X-smile': msg['X-smile'],
                                                         'X-img': msg['X-img'],
                                                         'Text': msg._payload,
                                                         })
    # Create a new file and store it inside storage folder. Name of json file is target model selected.
    file_path = f'app/storage/questions2017_{target_model}.json'
    with open(file_path, 'w') as outfile:
        json.dump(questions, outfile)
    print("Finished loading Json...")
    return file_path


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


def preprocess(data):
    '''
    Preprocessers the data by removing unnecessary words, 
    punctuation and grammar. The data is condensed into 
    only meaningful words/tokens used to input into models

    @param data - the text/subject that will be preprocessed
    @return preprocessed_text - The preprocessed text.
    '''

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
    # e.g. 'but' is not a valuable word
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords = stopwords[0:10]
    data = [i for i in data if i not in stopwords]

    return " ".join(data)
