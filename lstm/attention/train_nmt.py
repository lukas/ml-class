# Modified from https://github.com/datalogue/keras-attention

import os
import argparse

from keras.callbacks import ModelCheckpoint

from nmt import simpleNMT
from reader import Data, Vocabulary
import numpy as np
from keras import backend as K


EXAMPLES = ['26th January 2016', '3 April 1989', '5 Dec 09', 'Sat 8 Jun 2017']

def run_example(model, input_vocabulary, output_vocabulary, text):
    encoded = input_vocabulary.string_to_int(text)
    prediction = model.predict(np.array([encoded]))
    prediction = np.argmax(prediction[0], axis=-1)
    return output_vocabulary.int_to_string(prediction)

def run_examples(model, input_vocabulary, output_vocabulary, examples=EXAMPLES):
    predicted = []
    for example in examples:
        print('~~~~~')
        predicted.append(''.join(run_example(model, input_vocabulary, output_vocabulary, example)))
        print('input:',example)
        print('output:',predicted[-1])
    return predicted


# cp = ModelCheckpoint("./weights/NMT.{epoch:02d}-{val_loss:.2f}.hdf5",
#                      monitor='val_loss',
#                      verbose=0,
#                      save_best_only=True,
#                      save_weights_only=True,
#                      mode='auto')

# # create a directory if it doesn't already exist
# if not os.path.exists('./weights'):
#     os.makedirs('./weights/')


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



def main(args):
    # Dataset functions
    input_vocab = Vocabulary('./human_vocab.json', padding=args.padding)
    output_vocab = Vocabulary('./machine_vocab.json',
                              padding=args.padding)

    print('Loading datasets.')

    training = Data(args.training_data, input_vocab, output_vocab)
    validation = Data(args.validation_data, input_vocab, output_vocab)
    training.load()
    validation.load()
    training.transform()
    validation.transform()

    print('Datasets Loaded.')
    print('Compiling Model.')
    model = simpleNMT(pad_length=args.padding,
                      n_chars=input_vocab.size(),
                      n_labels=output_vocab.size(),
                      embedding_learnable=False,
                      encoder_units=256,
                      decoder_units=256,
                      trainable=True,
                      return_probabilities=False)

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', all_acc])
    print('Model Compiled.')
    print('Training. Ctrl+C to end early.')

    try:
        model.fit_generator(generator=training.generator(args.batch_size),
                            steps_per_epoch=100,
                            validation_data=validation.generator(args.batch_size),
                            validation_steps=100,
                            workers=1,
                            verbose=1,
                            epochs=args.epochs)

    except KeyboardInterrupt as e:
        print('Model training stopped early.')

    print('Model training complete.')

    run_examples(model, input_vocab, output_vocab)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-e', '--epochs', metavar='|',
                            help="""Number of Epochs to Run""",
                            required=False, default=50, type=int)

    named_args.add_argument('-g', '--gpu', metavar='|',
                            help="""GPU to use""",
                            required=False, default='0', type=str)

    named_args.add_argument('-p', '--padding', metavar='|',
                            help="""Amount of padding to use""",
                            required=False, default=50, type=int)

    named_args.add_argument('-t', '--training-data', metavar='|',
                            help="""Location of training data""",
                            required=False, default='./training.csv')

    named_args.add_argument('-v', '--validation-data', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./validation.csv')

    named_args.add_argument('-b', '--batch-size', metavar='|',
                            help="""Location of validation data""",
                            required=False, default=32, type=int)
    args = parser.parse_args()
    print(args)

    main(args)