from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

wnl = WordNetLemmatizer()

def tokenizer(doc):
    return [t for t in word_tokenize(doc)]

class Tokenizer(object):

    def fit(self, X, y):
        return self

    def transform(self, X):
        return_data = [tokenizer(x) for x in X]
        return return_data
