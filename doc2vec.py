import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('question_set/train.csv')

questions = df['Title'].to_numpy()

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[
    str(i)]) for i, _d in enumerate(questions)]

max_epochs = 100
vec_size = 300
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=2,
                negative=5,
                hs=0,
                dm=1)

model.build_vocab(tagged_data)


for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=5)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
'''
model.train(tagged_data,
            total_examples=model.corpus_count,
            epochs=max_epochs)
'''

model.save("d2v.model")
