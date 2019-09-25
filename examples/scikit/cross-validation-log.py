import pandas as pd
import numpy as np
import wandb

run = wandb.init(job_type='eval')
config = run.config
config.lowercase=True
config.ngram_min=1
config.ngram_max=1

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(lowercase=config.lowercase,
                             ngram_range=(config.ngram_min,
                                          config.ngram_max)
                                        )
count_vect.fit(fixed_text)

counts = count_vect.transform(fixed_text)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

from sklearn.model_selection import cross_val_score, cross_val_predict

scores = cross_val_score(nb, counts, fixed_target)
print(scores)
print(scores.mean())

predictions = cross_val_predict(nb, counts, fixed_target)
wandb.log({"Accuracy": scores.mean()})
