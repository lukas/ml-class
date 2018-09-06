import pandas as pd
import numpy as np


# Get a pandas DataFrame object of all the data in the csv file:
df = pd.read_csv('tweets.csv')

# Get pandas Series object of the "tweet text" column:
text = df['tweet_text']

# Get pandas Series object of the "emotion" column:
target = df['is_there_an_emotion_directed_at_a_brand_or_product']

# Remove the blank rows from the series:
target = target[pd.notnull(text)]
text = text[pd.notnull(text)]

# Perform feature extraction:
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(text)
counts = count_vect.transform(text)

# Train with this data with a Naive Bayes classifier:
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(counts, target)

# See what the classifier predicts for some new tweets:
predictions = nb.predict(counts)
correct_predictions = sum(predictions == target)
print('Percent correct: ', 100.0 * correct_predictions / len(predictions))
