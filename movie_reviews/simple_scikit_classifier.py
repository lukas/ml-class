# download the data from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# and put in this directory
# to uncompress the file:
# tar xvfz aclImdb_v1.tar.gz

import pandas as pd
import numpy as np
import wandb
from sklearn.feature_extraction.text import CountVectorizer


run = wandb.init()
config = run.config   # for tracking model inputs

df = pd.read_csv('train.csv')
target = df['sentiment']
text = df['text']

df = pd.read_csv('test.csv')
test_target = df['sentiment']
test_text = df['text']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

p = Pipeline(steps=[('counts', CountVectorizer()),
                ('multinomialnb', MultinomialNB())])

p.fit(text, target)
predictions = p.predict(test_text)
correct_predictions = sum(predictions == test_target)
accuracy = (100.0 * correct_predictions / len(predictions))
print("Accuracy: ", accuracy)
run.summary['accuracy']=accuracy
