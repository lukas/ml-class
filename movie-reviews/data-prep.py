# -*- coding: utf-8 -*-
import os
from random import shuffle
import csv

"""
We're going to need to have the data set from "http://ai.stanford.edu/~amaas/data/sentiment/"
When we extract aclImdb_v1.tar.gz it produces a folder called "aclImdb"
    that folder needs to be in the same directory as imdb_example.py and this file prep_data.py
Run this, and we should be able to run the example.
"""

train_doc_data = []

f_gen = os.walk('aclImdb/train/pos')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            train_doc_data.append(["Positive", open(os.path.join('aclImdb/train/pos', fname)).read()])

f_gen = os.walk('aclImdb/train/neg')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            train_doc_data.append(["Negative", open(os.path.join('aclImdb/train/neg', fname)).read()])



shuffle(train_doc_data)

with open("train.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows([['sentiment', 'text']])
    writer.writerows(train_doc_data)




test_doc_data = []

f_gen = os.walk('aclImdb/test/pos')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            test_doc_data.append(["Positive", open(os.path.join('aclImdb/test/pos', fname)).read()])

f_gen = os.walk('aclImdb/test/neg')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            test_doc_data.append(["Negative", open(os.path.join('aclImdb/test/neg', fname)).read()])


shuffle(test_doc_data)


with open("test.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows([['sentiment', 'text']])
    writer.writerows(test_doc_data)
