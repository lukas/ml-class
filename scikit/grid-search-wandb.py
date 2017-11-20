import pandas as pd
import numpy as np
import wandb

run = wandb.init()
config = run.config
summary = run.summary

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

p = Pipeline(steps=[('counts', CountVectorizer()),
                ('multinomialnb', MultinomialNB())])

from sklearn.model_selection import GridSearchCV

# these parameters define the different configurations we are going to try
# in our grid search

parameters = {
#    'counts__max_df': (0.5, 0.75,1.0),
#    'counts__min_df': (0,1,2),
#    'counts__token_pattern': ('(?u)\b\w\w+\b', '(?u)\b\w\w+\b'),
    'counts__lowercase' : (True, False),
    'counts__ngram_range': ((1,1), (1,2)),
#    'feature_selection__k': (1000, 10000, 100000)
    'multinomialnb__alpha': (0.5, 1)
    }

# setup the grid search - increasing n_jobs will make more threads
grid_search = GridSearchCV(p, parameters, n_jobs=1, verbose=2, cv=2)

# do the actual grid search
grid_search.fit(fixed_text, fixed_target)

# output the results and the parameters that obtained those results
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
    config[param_name] = best_parameters[param_name]

summary['accuracy'] = grid_search.best_score_
