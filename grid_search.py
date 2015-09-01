import pandas as pd
import numpy as np


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
                ('feature_selection', SelectKBest(chi2)),
                ('multinomialnb', MultinomialNB())])

from sklearn.grid_search import GridSearchCV

parameters = {
    'counts__max_df': (0.5, 0.75, 1.0),
    'counts__min_df': (1, 2, 3),
    'counts__ngram_range': ((1,1), (1,2)),
#    'feature_selection__k': (1000, 10000, 100000)
    }

grid_search = GridSearchCV(p, parameters, n_jobs=1, verbose=1, cv=10)

grid_search.fit(fixed_text, fixed_target)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
