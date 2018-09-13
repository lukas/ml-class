import imdb
import numpy as np
from keras.preprocessing import text
import wandb
from sklearn.linear_model import LogisticRegression

wandb.init()
config = wandb.config
config.vocab_size = 2000

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train, mode="tfidf")
X_test = tokenizer.texts_to_matrix(X_test, mode="tfidf")

bow_model = LogisticRegression()
bow_model.fit(X_train, y_train)

pred_train = bow_model.predict(X_train)
acc = np.sum(pred_train==y_train)/len(pred_train)

pred_test = bow_model.predict(X_test)
val_acc = np.sum(pred_test==y_test)/len(pred_test)
wandb.log({"val_acc": val_acc, "acc": acc})