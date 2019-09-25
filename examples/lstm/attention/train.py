# Modified from https://github.com/datalogue/keras-attention

import os
import argparse
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Dense, Embedding, Activation, Permute
from keras.layers import Input, Flatten, Dropout
from keras.layers import CuDNNLSTM, LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from attention_decoder import AttentionDecoder
from reader import Data, Vocabulary
import numpy as np
from keras import backend as K
import wandb
from wandb.keras import WandbCallback
from util import Visualizer


# Set hyper-parameters
run = wandb.init()
config = run.config
config.encoder_units = 32
config.decoder_units = 32
config.padding = 50
config.batch_size = 128


EXAMPLES = ['26th January 2016', '3 April 1989', '5 Dec 09', 'Sat 8 Jun 2017']


def run_example(model, input_vocabulary, output_vocabulary, text):
    """Predict a single example"""
    encoded = input_vocabulary.string_to_int(text)
    prediction = model.predict(np.array([encoded]))
    prediction = np.argmax(prediction[0], axis=-1)
    return output_vocabulary.int_to_string(prediction)


def decode(chars, sanitize=False):
    """Join a list of chars removing <unk> and invalid utf-8"""
    string = "".join([c for c in chars if c != "<unk>"])
    if sanitize:
        string = "".join(i for i in string if ord(i) < 2048)
    return bytes(string, 'utf-8').decode('utf-8', 'ignore')


class Examples(Callback):
    """Keras callback to log examples"""

    def __init__(self, viz):
        self.visualizer = viz

    def on_epoch_end(self, epoch, logs):
        indicies = np.random.choice(len(validation.inputs), 18, replace=False)
        data_in = validation.inputs[indicies]
        data_out = validation.targets[indicies]
        examples = []
        viz = []
        # Swap the weights
        weights = self.visualizer.pred_model.get_layer(
            "attention_decoder_1").get_weights()
        self.visualizer.proba_model.get_layer(
            "attention_decoder_prob").set_weights(weights)
        for i, o in zip(data_in, data_out):
            text = decode(input_vocab.int_to_string(i)).replace('<eot>', '')
            truth = decode(output_vocab.int_to_string(np.argmax(o, -1)), True)
            pred = run_example(self.model, input_vocab, output_vocab, text)
            out = decode(pred, True)
            print(f"{decode(text, True)} -> {out} ({truth})")
            examples.append([decode(text, True), out, truth])
            amap = self.visualizer.attention_map(text)
            if amap:
                viz.append(wandb.Image(amap,))
                amap.close()
        if len(viz) > 0:
            logs["attention_map"] = viz[:5]
        logs["examples"] = wandb.Table(data=examples)
        wandb.log(logs)


def all_acc(y_true, y_pred):
    """
        All Accuracy
        https://github.com/rasmusbergpalm/normalization/blob/master/train.py#L10
    """
    return K.mean(
        K.all(
            K.equal(
                K.max(y_true, axis=-1),
                K.cast(K.argmax(y_pred, axis=-1), K.floatx())
            ),
            axis=1)
    )


# Configuration
training_data = './training.csv'
validation_data = './validation.csv'

# Dataset functions
input_vocab = Vocabulary('./human_vocab.json', padding=config.padding)
output_vocab = Vocabulary('./machine_vocab.json', padding=config.padding)

print('Loading datasets...')

training = Data(training_data, input_vocab, output_vocab)
validation = Data(validation_data, input_vocab, output_vocab)
training.load()
validation.load()
training.transform()
validation.transform()

print('Datasets Loaded.')


def build_models(pad_length=config.padding, n_chars=input_vocab.size(), n_labels=output_vocab.size(),
                 embedding_learnable=False, encoder_units=32, decoder_units=32, trainable=True, return_probabilities=False):
    """Build the model"""
    input_ = Input(shape=(pad_length,), dtype='float32')
    input_embed = Embedding(n_chars, n_chars,
                            input_length=pad_length,
                            trainable=embedding_learnable,
                            weights=[np.eye(n_chars)],
                            name='OneHot')(input_)

    rnn_encoded = Bidirectional(CuDNNLSTM(encoder_units, return_sequences=True),
                                name='bidirectional_1',
                                merge_mode='concat',
                                trainable=trainable)(input_embed)

    y_prob = AttentionDecoder(decoder_units,
                              name='attention_decoder_prob',
                              output_dim=n_labels,
                              return_probabilities=True,
                              trainable=False)(rnn_encoded)

    y_pred = AttentionDecoder(decoder_units,
                              name='attention_decoder_1',
                              output_dim=n_labels,
                              return_probabilities=return_probabilities,
                              trainable=trainable)(rnn_encoded)

    model = Model(inputs=input_, outputs=y_pred)
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    prob_model = Model(inputs=input_, outputs=y_prob)
    return model, prob_model


model, prob_model = build_models(encoder_units=config.encoder_units,
                                 decoder_units=config.decoder_units)

# Configure the visualizer
viz = Visualizer(input_vocab, output_vocab)
viz.set_models(model, prob_model)

# Save the network to wandb
wandb.run.summary['graph'] = wandb.Graph.from_keras(model)

model.fit_generator(generator=training.generator(config.batch_size),
                    steps_per_epoch=100,
                    validation_data=validation.generator(config.batch_size),
                    validation_steps=10,
                    workers=1,
                    verbose=1,
                    callbacks=[Examples(viz)],
                    epochs=100)

print('Model training complete.')
