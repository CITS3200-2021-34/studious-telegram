import email

def parseQuestionsAnswersFromFile(filePath: str):
    '''
    Returns a dictionary containing each question in filePath, and a 
    dictionary containing each answer in filePath.

    :param filePath: File to be parsed
    :return questions: Dictionary of question threads
    :return answers: Dictionary of answer threads
    '''
    threads = parseThreadsFromFile(filePath)
    posts = getPostsFromThreads(threads)
    questions, answers = parseQuestionsAnswersFromPosts(posts)

    return questions, answers

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

def getPostsFromThreads(threads):
    '''
    For each item in a given threads list, a list containing the item's date, 
    subject, and body is created and stored in the posts list to be returned.

    :param threads: List of question/answer strings
    :return posts: List containing a list of the date, subject, and body for 
        each thread
    '''
    posts = []
    for i in range(0, len(threads)):
        msg = email.message_from_string(threads[i])
        p = []
        p.append(msg['Date'])
        p.append(msg['Subject'])
        # p.append(msg._payload)
        p.append(msg)
        posts.append(p)
    return posts

def parseQuestionsAnswersFromPosts(posts):
    '''
    Creates a dictionary containing each question and another dictionary 
    containing each answer, using the subject line as keys for each such 
    that the corresponding Q and As have the same key.

    :param posts: List containing a list of the date, subject, and body for 
        each thread
    :return dict_q: Dictionary of question threads
    :return dict_a: Dictionary of answer threads
    '''
    dict_q = {}
    dict_a = {}
    for i in range(len(posts)):
        li = []
        # add any post with subject not already in question dictionary
        if posts[i][1] not in dict_q:
            li.append(posts[i][0])
            li.append(posts[i][2])
            # key = subject, value = list containing date and thread body
            dict_q[posts[i][1]] = li
            dict_a[posts[i][1]] = []
        else:
            # all posts with subject already in question dictionary to be 
            # added to answer dictionary
            li.append(posts[i][0])
            li.append(posts[i][2])
            val = dict_a.get(posts[i][1])
            val.append(li)
            # key = subject, value = list containing subject, date, and thread body
            dict_a[posts[i][1]] = val
    return dict_q, dict_a