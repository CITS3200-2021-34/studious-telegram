import email
import json
from .json_loader import JsonLoader
import datetime


def write_to_json(title, question, title_vec, text_vec):

    date = datetime.datetime.now()
    date = date.strftime("%c")

    '''
    this will need to be changed so that we dont keep loading in the json file.
    '''
    file = JsonLoader('app/storage/CITS2002_2021.json')
    questions = file.read_data()
    questions[title] = {'Date': date,
                        'To': "help2002@csse.uwa.edu.au",
                        'Received': "from 106.68.103.155",
                        'Subject_vec': title_vec,
                        'From': "poster013@student.uwa.edu.au",
                        'X-smile': "None",
                        'X-img': "None",
                        'Text': question,
                        'Text_vec': text_vec,
                        'Answers': [],
                        }

    with open('app/storage/CITS2002_2021.json', 'w') as outfile:
        json.dump(questions, outfile, indent=4)
    print("Finished writing to Json...")
