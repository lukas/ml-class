import os
import json
import numpy as np

def load_amazon():
    filename = 'reviews_Video_Games_5.json'
    train_summary = []
    train_review_text = []
    train_labels = []
    
    test_summary = []
    test_review_text = []
    test_labels = []
    
    with open(filename, 'r') as f:
        for (i, line) in enumerate(f):
            data = json.loads(line)
            
            if data['overall'] == 3:
                next
            elif data['overall'] == 4 or data['overall'] == 5:
                label = 1
            elif data['overall'] == 1 or data['overall'] == 2:
                label = 0
            else:
                raise Exception("Unexpected value " + str(data['overall']))
                
            summary = data['summary']
            review_text = data['reviewText']
            
            if (i % 10 == 0):
                test_summary.append(summary)
                test_review_text.append(review_text)
                test_labels.append(label)
            else:
                train_summary.append(summary)
                train_review_text.append(review_text)
                train_labels.append(label)
                
    return (train_summary, train_review_text, train_labels), (test_summary, test_review_text, test_labels)

def load_amazon_smaller():
    size = 20000
    (train_summary, train_review_text, train_labels), (test_summary, test_review_text, test_labels) = load_amazon()
    return (train_summary[:size], train_review_text[:size], np.array(train_labels[:size])), (test_summary[:size], test_review_text[:size], np.array(test_labels[:size]))

