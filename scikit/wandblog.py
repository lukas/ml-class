


def log(run, text, target, predictions):
    correct_predictions = sum(predictions == target)

#    run.examples.set_columns((
#        ('tweet', str),
#        ('target', str),
#        ('prediction', str),
#        ('accuracy', float)))

    run.summary['accuracy']=(100.0 * correct_predictions / len(predictions))


#    for idx, (tweet, t, prediction)  in enumerate(zip(text[:100], target[:100], predictions[:100])):
#        run.examples.add({
#                'tweet': tweet,
#                'target': t,
#                'prediction': prediction,
#                'accuracy': (prediction == tweet),
#        })
