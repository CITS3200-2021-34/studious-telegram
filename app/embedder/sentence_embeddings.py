from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from .embeddingmodels import AbstractEmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial.distance import cosine
import numpy as np
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

    def universal_encoder(self, input):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)
        question_list = [k for k in self.questions.keys()]
        sentence_embeddings = model(question_list)
        query_vec = model([input])[0]
        query_vec = tf.reshape(query_vec, (-1, 1))
        sim_dict = {}

        temp = self.questions.items()
        li = list(temp)

        for i, sent in enumerate(sentence_embeddings):
            sent = tf.reshape(sent, (-1, 1))
            sim_dict[li[i][0]] = 1 - cosine(sent, query_vec)

        sim_dict = sorted(sim_dict.items(),
                          key=operator.itemgetter(1), reverse=True)
        return sim_dict

    def sent_BERT(self, input) -> list[list[float]]:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        question_list = [k for k in self.questions.keys()]
        sentence_embeddings = model.encode(question_list)

        query = model.encode([input])[0]
        query = query.reshape(-1, 1)

        temp = self.questions.items()
        li = list(temp)

        sim_dict = {}
        for i, sent in enumerate(sentence_embeddings):
            sent = tf.reshape(sent, (-1, 1))
            sim_dict[li[i][0]] = 1 - cosine(sent, query)
        sim_dict = sorted(sim_dict.items(),
                          key=operator.itemgetter(1), reverse=True)
        return sim_dict

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
