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

# (Tweets 0 to 5999 are used for training data)
nb.fit(counts[0:6000], target[0:6000])

# See what the classifier predicts for some new tweets:
# (Tweets 6000 to 9091 are used for testing)
predictions = nb.predict(counts[6000:9092])
correct_predictions = sum(predictions == target[6000:9092])
incorrect_predictions = (9092 - 6000) - correct_predictions
print('# of correct predictions: ' + str(correct_predictions))
print('# of incorrect predictions: ' + str(incorrect_predictions))
print('Percent correct: ' + str(100.0 * correct_predictions / (correct_predictions + incorrect_predictions)))

from sklearn.metrics import confusion_matrix
## We're ignoring "I can't tell" here for simplicity
label_list = ['Positive emotion', 'No emotion toward brand or product', 'Negative emotion']
cm = confusion_matrix(target[6000:9092], predictions, labels=label_list)
print("Labels in data:")
print(label_list)
print("Rows: actual labels, Columns: Predicted labels")
print(cm)
