import pandas as pd
import numpy as np


# Get a pandas DataFrame object of all the data in the csv file:
df = pd.read_csv('tweets.csv')

# Get pandas Series object of the "tweet text" column:
text = df['tweet_text']

# Get pandas Series object of the "emotion" column:
target = df['is_there_an_emotion_directed_at_a_brand_or_product']

# The rows of  the "emotion" column have one of three strings:
# 'Positive emotion'
# 'Negative emotion'
# 'No emotion toward brand or product'

# Remove the blank rows from the series:
target = target[pd.notnull(text)]
text = text[pd.notnull(text)]

# Perform feature extraction:
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(text)
counts = count_vect.transform(text)
print(counts.shape)

# Train with this data with a Naive Bayes classifier:
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# (Tweets 0 to 5999 are used for training data)
nb.fit(counts[0:6000], target[0:6000])

# See what the classifier predicts for some new tweets:
# (Tweets 6000 to 9091 are used for testing)
predictions = nb.predict(counts[6000:9092])
print(len(predictions))
correct_predictions = sum(predictions == target[6000:9092])
print('Percent correct: ', 100.0 * correct_predictions / 3092)
