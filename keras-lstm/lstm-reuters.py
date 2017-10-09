# if you get an error, download nltk stopwords
# in python run
# import nltk
# nltk.download()

import re
import xml.sax.saxutils as saxutils

from gensim.models.word2vec import Word2Vec

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM

from multiprocessing import cpu_count

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer

from pandas import DataFrame

from sklearn.cross_validation import train_test_split

# Word2Vec number of features
num_features = 500
# Limit each newsline to a fixed number of words
document_max_num_words = 100
# Selected categories
selected_categories = ['pl_usa']

# Load stop-words
stop_words = set(stopwords.words('english'))

# Initialize tokenizer
# It's also possible to try with a stemmer or to mix a stemmer and a lemmatizer
tokenizer = RegexpTokenizer('[\'a-zA-Z]+')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Tokenized document collection
newsline_documents = []

def tokenize(document):
 words = []

 for sentence in sent_tokenize(document):
     tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(sentence) if t.lower() not in stop_words]
     words += tokens

 return words

# Create new Gensim Word2Vec model
w2v_model = Word2Vec(newsline_documents, size=num_features, min_count=1, window=10, workers=cpu_count())
w2v_model.init_sims(replace=True)
w2v_model.save(data_folder + 'reuters.word2vec')



num_categories = len(selected_categories)
X = zeros(shape=(number_of_documents, document_max_num_words, num_features)).astype(float32)
Y = zeros(shape=(number_of_documents, num_categories)).astype(float32)

empty_word = zeros(num_features).astype(float32)

for idx, document in enumerate(newsline_documents):
    for jdx, word in enumerate(document):
        if jdx == document_max_num_words:
            break

        else:
            if word in w2v_model:
                X[idx, jdx, :] = w2v_model[word]
            else:
                X[idx, jdx, :] = empty_word

for idx, key in enumerate(document_Y.keys()):
    Y[idx, :] = document_Y[key]


model.add(LSTM(int(document_max_num_words*1.5), input_shape=(document_max_num_words, num_features)))
model.add(Dropout(0.3))
model.add(Dense(num_categories))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, batch_size=128, nb_epoch=5, validation_data=(X_test, Y_test))

# Evaluate model
score, acc = model.evaluate(X_test, Y_test, batch_size=128)

print('Score: %1.4f' % score)
print('Accuracy: %1.4f' % acc)
