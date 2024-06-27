import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

text = pd.read_csv('tweets.csv')
text.columns = ['Tweets', 'device', 'Emotion']
text = text.dropna(how='any')

def cleantext(text):
  text = str(text).lower()
  #text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('[^a-zA-Z]', ' ', text)
  text = re.sub('<.*?>+', ' ', text)
  text = re.sub(' +', ' ', text)
  text = re.sub('\n', ' ', text)
  text = text.split()
  text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
  text = ' '.join(text)
  return text

text['Tweets'] = text['Tweets'].apply(cleantext)

x = text['Tweets']
y = text['Emotion']


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
Model_1 =  Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', MultinomialNB())
])
Model_2 =  Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', LogisticRegression())
])
Model_3 =  Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', SVC())
])

Model_1.fit(x_train, y_train)
Model_2.fit(x_train, y_train)
Model_3.fit(x_train, y_train)

y_pred_1 = Model_1.predict(x_test)
y_pred_2 = Model_2.predict(x_test)
y_pred_3 = Model_3.predict(x_test)

print('Naive Bayes:', accuracy_score(y_test, y_pred_1))
print('Logistic Regression:', accuracy_score(y_test, y_pred_2))
print('SVC:',accuracy_score(y_test, y_pred_3))