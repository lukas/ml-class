import os
import json

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

load_amazon()