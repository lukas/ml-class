import pandas as pd
import csv
import numpy as np

words = pd.read_table('glove/glove.6B.50d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

def vec(w):
  return words.loc[w].as_matrix()

words_matrix = words.as_matrix()

def find_n_closest_words(v, n):
  diff = words_matrix - v
  delta = np.sum(diff * diff, axis=1)
  #i = np.argmin(delta)
  idx = (delta).argsort()[:n]
  close_words = []
  for i in idx:
    close_words.append(words.iloc[i].name)
  return close_words

print(find_n_closest_words( vec('srtgsrtg'), 5 ))
