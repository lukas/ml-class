#!/usr/bin/env python
# coding: utf-8

# In[2]:


import unittest
from svm_sentiment_analysis import svm_model, vectorizer 


# In[ ]:


import unittest
from svm_sentiment_analysis import svm_model, vectorizer 

class TestSentimentAnalysis(unittest.TestCase):
    def test_prediction(self):
        sample_text = ["Your cat is so cute! Can I pet her?"]
        
        sample_vector = vectorizer.transform(sample_text)
        
        prediction = svm_model.predict(sample_vector)
        
        self.assertIn(prediction[0], [1, 0, -1])

if __name__ == '__main__':
    unittest.main()



# In[ ]:





# In[ ]:




