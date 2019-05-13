import matplotlib  # pylint: disable
matplotlib.use("Agg")  # pylint: disable
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reader import Vocabulary

def run_example(model, input_vocabulary, output_vocabulary, text):
    encoded = input_vocabulary.string_to_int(text)
    prediction = model.predict(np.array([encoded]))
    prediction = np.argmax(prediction[0], axis=-1)
    return output_vocabulary.int_to_string(prediction)

class Visualizer(object):

    def __init__(self, input_vocab, output_vocab):
        """
            Visualizes attention maps
            :param input_vocab: the input human vocabulary
            :param output_vocab: the machine vocabulary
        """
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

    def set_models(self, pred_model, proba_model):
        """
            Sets the models to use
            :param pred_model: the prediction model
            :param proba_model: the model that outputs the activation maps
        """
        self.pred_model = pred_model
        self.proba_model = proba_model

    def attention_map(self, text):
        """
            Text to visualze attention map for.
        """
        # encode the string
        d = self.input_vocab.string_to_int(text)

        # get the output sequence
        predicted_text = run_example(
            self.pred_model, self.input_vocab, self.output_vocab, text)

        text_ = list(text) + ['<eot>'] + ['<unk>'] * self.input_vocab.padding
        # get the lengths of the string
        input_length = len(text)+1
        if '<eot>' not in predicted_text:
            return None
        output_length = predicted_text.index('<eot>')+1
        # get the activation map
        activation_map = np.squeeze(self.proba_model.predict(np.array([d])))[
            0:output_length, 0:input_length]

        # import seaborn as sns
        plt.clf()
        f = plt.figure(figsize=(8, 8.5))
        ax = f.add_subplot(1, 1, 1)

        # add image
        i = ax.imshow(activation_map, interpolation='nearest', cmap='gray')

        # add colorbar
        cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
        cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
        cbar.ax.set_xlabel('Probability', labelpad=2)

        # add labels
        ax.set_yticks(range(output_length))
        ax.set_yticklabels(predicted_text[:output_length])

        ax.set_xticks(range(input_length))
        ax.set_xticklabels(text_[:input_length], rotation=45)

        ax.set_xlabel('Input Sequence')
        ax.set_ylabel('Output Sequence')

        # add grid and legend
        ax.grid()
        # ax.legend(loc='best')

        return plt


def main(examples, args):
    print('Total Number of Examples:', len(examples))
    weights_file = os.path.expanduser(args.weights)
    print('Weights loading from:', weights_file)
    viz = Visualizer(padding=args.padding,
                     input_vocab=args.human_vocab,
                     output_vocab=args.machine_vocab)
    print('Loading models')
    pred_model = simpleNMT(trainable=False,
                           pad_length=args.padding,
                           n_chars=viz.input_vocab.size(),
                           n_labels=viz.output_vocab.size())

    pred_model.load_weights(weights_file, by_name=True)
    pred_model.compile(optimizer='adam', loss='categorical_crossentropy')

    proba_model = simpleNMT(trainable=False,
                            pad_length=args.padding,
                            n_chars=viz.input_vocab.size(),
                            n_labels=viz.output_vocab.size(),
                            return_probabilities=True)

    proba_model.load_weights(weights_file, by_name=True)
    proba_model.compile(optimizer='adam', loss='categorical_crossentropy')

    viz.set_models(pred_model, proba_model)

    print('Models loaded')

    for example in examples:
        viz.attention_map(example)

    print('Completed visualizations')