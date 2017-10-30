from keras.datasets import cifar10
from keras.models import load_model
from tests import test_pixel_removal
import numpy as np
import wandb

num_examples = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

labels=[
"Airplane",
"Automobile",
"Bird",
"Cat",
"Deer",
"Dog",
"Frog",
"Horse",
"Ship",
"Truck"

]

run = wandb.init(job_type='eval')
config = run.config
run.examples.set_columns((
    ('id', int),
    ('loss', float),
    ('label', str),
    ('prediction', str),
    ('accuracy', float),
    ('confidence', wandb.types.Percentage),
    ('predictions', wandb.types.Histogram),
    ('image', wandb.types.Image),
    ('importance-image', wandb.types.Image)
))

model = load_model(config.model)
pred = np.zeros(10)

accuracies = []
losses = []

for idx, (image, label) in enumerate(list(zip(x_test, y_test))[0:num_examples]):
    image = image.reshape(1, 32, 32, 3)
    baseline = model.predict([image])[0]
    prediction = np.argmax(baseline)

    label_confidence = baseline[label]


    print("Label", label)
    print("Predictions", baseline)

    logloss = max(-100, np.log(label_confidence))
    accuracy = 1. if baseline[label] > 0.5 else 0.

    accuracies.append(accuracy)
    losses.append(logloss)

    (image_path, importance_image_path) = test_pixel_removal(model, image, idx)
    image = open(image_path, 'rb').read()
    importance_image = open(importance_image_path, 'rb').read()

    for i in range(10):
        logpred = max(0, (np.log(baseline[i])+30.)/40.)
        pred[i] = logpred

    run.examples.add(
            {'id': idx,
             'loss': logloss,
             'label': labels[label[0]],
             'accuracy': accuracy,
             'prediction': labels[prediction],
             'confidence': label_confidence,
             'predictions': pred,
             'image': image,
             'importance-image': importance_image
             })


run.summary['acc'] = np.mean(accuracies)
run.summary['loss'] = np.mean(losses)
