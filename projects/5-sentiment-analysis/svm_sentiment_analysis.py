#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


# In[52]:


data = pd.read_csv('tweets.csv')


# In[53]:


data.head()


# In[54]:


data['tweet_text'].fillna('',inplace=True)


# In[55]:


data['tweet_text'].isna().sum()


# In[56]:


data['is_there_an_emotion_directed_at_a_brand_or_product'].isna().sum()


# In[57]:


data['is_there_an_emotion_directed_at_a_brand_or_product'].unique()


# In[58]:


texts = data['tweet_text'].values 
target = data['is_there_an_emotion_directed_at_a_brand_or_product'].values


# In[59]:


texts


# In[60]:


target


# In[61]:


target_map = {'Negative emotion': -1,'Positive emotion':1, 'No emotion toward brand or product': 0}


# In[62]:


y = np.array([target_map.get(label, 0) for label in target])


# In[63]:


vectorizer = TfidfVectorizer(max_features=5000)  
X = vectorizer.fit_transform(texts)


# In[66]:


X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)


# In[67]:


svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)


# In[68]:


predictions = svm_model.predict(X_test)


# In[69]:


print(metrics.classification_report(y_test, predictions))


# In[ ]:




