import email


class PreProcessor():
    '''
    Class for converting threads into separate posts which are classified as 
    either a question or an answer in separate dictionaries, and can be accessed 
    using their subject line as the key.
    '''
    
    def __init__(self):
        '''
        Constructor for the PreProcessor class.

        :param self: Instance of the PreProcessor object
        '''
        print("Loaded Preprocessor")

    def get_posts(self, threads):
        '''
        Takes a list of the threads and, for each thread, creates a list containing 
        the date, subject, and body of that thread, and returns a list of these lists.

        :param self: Instance of the PreProcessor object
        "param threads: List of strings, where each string is an individual thread 
            (ie. question or answer)
        :return posts: List containing a list of the date, subject, and body for each 
            thread
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

    def create_data_structures(self, posts):
        '''
        Creates a dictionary containing each question and another dictionary 
        containing each answer, using the subject line as keys for each such 
        that the corresponding Q and As have the same key.

        :param self: Instance of the PreProcessor object
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
