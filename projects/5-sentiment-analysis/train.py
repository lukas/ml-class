# This is a sentiment classifier on a fairly small dataset.
# This code adds the incredibly useful scikit learn package
# to your toolkit, which is especially useful for processing text data.
#
# This uses a "naive bayes" classifier instead of a neural net.
# You can add keras to do the classification as an additional challenge
# but the goal here is to improve the 66% validation accuracy to above
# 68%.  One approach is to use TfidfVectorizer instead of CountVectorizer
# and then SGDClassifier instead of naive bayes ("MultinomialNB", but there 
# are many other ways.
#
# Check out examples/scikit for inspiration.

import pandas as pd
import numpy as np
import wandb

wandb.init()

# Get a pandas DataFrame object of all the data in the csv file:
df = pd.read_csv('tweets.csv')

# Get pandas Series object of the "tweet text" column:
text = df['tweet_text']

# Get pandas Series object of the "emotion" column:
target = df['is_there_an_emotion_directed_at_a_brand_or_product']

# Remove the blank rows from the series:
target = target[pd.notnull(text)]
text = text[pd.notnull(text)]

# Perform feature extraction
# Try changing this!
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
count_vect = CountVectorizer()
count_vect.fit(text)
counts = count_vect.transform(text)

# Train with this data with a Naive Bayes classifier:
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
clf = MultinomialNB()

# (Tweets 0 to 5999 are used for training data)
clf.fit(counts[0:6000], target[0:6000])

# See what the classifier predicts for some new tweets:
# (Tweets 6000 to 9091 are used for testing)
predictions = clf.predict(counts[6000:9092])
correct_predictions = sum(predictions == target[6000:9092])
incorrect_predictions = (9092 - 6000) - correct_predictions

train_predictions = clf.predict(counts[0:6000])
train_correct_predictions = sum(train_predictions == target[0:6000])
train_incorrect_predictions = 6000 - train_correct_predictions

train_accuracy = train_correct_predictions/(train_correct_predictions+train_incorrect_predictions) 
val_accuracy = correct_predictions/(correct_predictions+incorrect_predictions)

wandb.log({"val_accuracy": val_accuracy, "train_accuracy": train_accuracy})

