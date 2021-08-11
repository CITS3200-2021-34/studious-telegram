import email

def parseQuestionsAnswersFromFile(filePath: str):
    threads = parseThreadsFromFile(filePath)
    posts = getPostsFromThreads(threads)
    questions, answers = parseQuestionsAnswersFromPosts(posts)

    return questions, answers

def parseThreadsFromFile(filePath: str):
    with open(filePath, 'r') as file:
        line = file.readline()
        line = file.readline()  # Not sure best way to do this, needed to skip the first line
        str = ''
        threads = []

        while line:
            if(line[:4] == 'Date' and len(str) > 0):
                str = str.rstrip('\n')
                threads.append(str)
                str = ''
            str += line
            line = file.readline()
        threads.append(str)
    return threads

def getPostsFromThreads(threads):
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
    dict_q = {}
    dict_a = {}
    for i in range(len(posts)):
        li = []
        if posts[i][1] not in dict_q:
            li.append(posts[i][0])
            li.append(posts[i][2])
            dict_q[posts[i][1]] = li
            dict_a[posts[i][1]] = []
        else:
            li.append(posts[i][0])
            li.append(posts[i][2])
            val = dict_a.get(posts[i][1])
            val.append(li)
            dict_a[posts[i][1]] = val
    return dict_q, dict_a