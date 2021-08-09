import email


class PreProcessor():
    def __init__(self):
        print("Loaded Preprocessor")

    def get_posts(self, threads):
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
