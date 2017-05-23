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

# Train with this data with a Naive Bayes classifier:
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(counts, target)

# See what the classifier predicts for some new tweets:
predictions = nb.predict(counts)
<<<<<<< HEAD
=======

>>>>>>> 031e1d93f095e4b3a3ecf1a5509b40b3b5b2fe28
correct_predictions = sum(predictions == target)
incorrect_predictions = 9092 - correct_predictions  # (there are 9,092 tweets in the csv)
print('# of correct predictions: ' + str(correct_predictions))
print('# of incorrect predictions: ' + str(incorrect_predictions))
print('Percent correct: ' + str(100.0 * correct_predictions / (correct_predictions + incorrect_predictions)))
<<<<<<< HEAD

=======
>>>>>>> 031e1d93f095e4b3a3ecf1a5509b40b3b5b2fe28
