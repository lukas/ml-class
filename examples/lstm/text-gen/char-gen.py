import tensorflow as tf
import numpy as np
import random
import sys
import io
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("text", type=str)

args = parser.parse_args()

run = wandb.init()
config = run.config
config.hidden_nodes = 128
config.batch_size = 256
config.file = args.text
config.maxlen = 200
config.step = 3

# Only load first 100k charcters because we're not using memory efficiently
text = io.open(config.file, encoding='utf-8').read()[:100000]
chars = sorted(list(set(text)))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# build a sequence for every <config.step>-th character in the text

sentences = []
next_chars = []
for i in range(0, len(text) - config.maxlen, config.step):
    sentences.append(text[i: i + config.maxlen])
    next_chars.append(text[i + config.maxlen])

# build up one-hot encoded input x and output y where x is a character
# in the text y is the next character in the text

x = np.zeros((len(sentences), config.maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(
    128, input_shape=(config.maxlen, len(chars))))
model.add(tf.keras.layers.Dense(len(chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer="rmsprop")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class SampleText(tf.keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        start_index = random.randint(0, len(text) - config.maxlen - 1)

        for diversity in [0.5, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + config.maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated + " |!| ")

            for i in range(200):
                x_pred = np.zeros((1, config.maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


model.fit(x, y, batch_size=config.batch_size,
          epochs=100, callbacks=[SampleText(), wandb.keras.WandbCallback()])
