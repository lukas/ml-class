
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

#import gensim
# let X be a list of tokenized texts (i.e. list of lists of tokens)
#model = gensim.models.Word2Vec(X, size=100)
#w2v = dict(zip(model.wv.index2word, model.wv.syn0))

class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # pull out a single value from the dictionary and check its dimension
        # this is a little ugly to make it work in python 2 and 3
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        mean_vects = []
        for words in X:
            word_vects = []
            for w in words:
                if w in self.word2vec:
                    word_vects.append(self.word2vec[w])

            mean_vect = np.mean(word_vects, axis=0)
            mean_vects.append(np.array(mean_vect))

        mean_vects = np.array(mean_vects)

        return mean_vects
