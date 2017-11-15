import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from embedding import MeanEmbeddingVectorizer
from tokenizer import Tokenizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier


from wandblog import log
import wandb
run = wandb.init(job_type='eval')
config = run.config

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

w2v = {}
with open("glove/glove.6B.50d.txt", "r") as lines:
    for line in lines:
        word, numbers = line.split(" ", 1)
        number_array = np.array(numbers.split()).astype(np.float)
        w2v[word] = number_array


text_clf = Pipeline([('token', Tokenizer()),
                     ('vect', MeanEmbeddingVectorizer(w2v)),
                     ("extra trees", ExtraTreesClassifier(n_estimators=200)),])

text_clf.fit(fixed_text, fixed_target)

scores = cross_val_score(text_clf, fixed_text, fixed_target)
print(scores)
print(scores.mean())

predictions = cross_val_predict(text_clf, fixed_text, fixed_target)
log(run, fixed_text, fixed_target, predictions)
