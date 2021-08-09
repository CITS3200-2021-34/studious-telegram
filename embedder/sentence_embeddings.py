from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from .embeddingmodels import AbstractEmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as np
import operator


class SentEmbeddings(AbstractEmbeddingModel):
    def __init__(self, questions: dict[str, list[str]], answers: dict[str, list[list[str]]]):
        self.questions = questions
        self.answers = answers

    def doc2vec(self):
        question_list = [k for k in self.questions.keys()]

        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[
                                      str(i)]) for i, _d in enumerate(question_list)]
        tagged_dict = {}
        for i, k in enumerate(self.questions):
            i = str(i)
            tagged_dict[i] = k

        max_epochs = 100
        vec_size = 300
        alpha = 0.025

        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha,
                        min_alpha=0.00025,
                        min_count=1,
                        dm=1)

        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=50)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        return model, tagged_dict
        # model.save("d2v.model")
        # print("Model Saved")

    def load_execute_model(self, model, input):
        token_input = word_tokenize(input.lower())
        vect = model.infer_vector(token_input)
        similar_doc = model.dv.most_similar(positive=[vect], topn=3)
        return similar_doc

    def vectorised_data(self, model):
        vect_question = {}
        for k in self.questions.keys():
            token_input = word_tokenize(k.lower())
            vect = model.infer_vector(token_input)
            vect_question[k] = vect
        return vect_question

    def get_similarity(self, model, input, vect_question):
        sim_dict = {}
        token_input = word_tokenize(input.lower())
        vect = model.infer_vector(token_input)

        for k, v in vect_question.items():
            vec = v.reshape(1, -1)
            vect = vect.reshape(1, -1)
            sim_dict[k] = cosine_similarity(vect, vec)[0][0]

        sim_dict = sorted(sim_dict.items(),
                          key=operator.itemgetter(1), reverse=True)
        return sim_dict

    def get_top(self, sim_dict, n: int):
        top_suggestions = []
        for i, v in enumerate(sim_dict):
            if i == n:
                break
            top_suggestions.append(v[0])
            print(f'{i+1} {v[0]} : {v[1]}')
        return top_suggestions

    def get_question_answers(self, suggestion: str):
        print(self.questions.get(suggestion)[1])
        print()
        print()
        ans = self.answers.get(suggestion)
        for i in ans:
            print(i[1])
            print()
            print()
