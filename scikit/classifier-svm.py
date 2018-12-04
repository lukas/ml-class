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

# Train with this data with an svm:
from sklearn.svm import SVC


clf = SVC()
clf.fit(counts, target)

#Try the classifier
print(clf.predict(count_vect.transform(['i do not love my iphone'])))

# See what the classifier predicts for some new tweets:
#for tweet in ('I love my iphone!!!', 'iphone costs too much!!!', 'the iphone is not good', 'I like turtles'):
#  print('Tweet: ' + tweet)
#  print('Prediction: ' + str(clf.predict(count_vect.transform([tweet]))))
#  print('\n')
