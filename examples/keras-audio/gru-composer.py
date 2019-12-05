""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream
import os
import subprocess
import wandb
import base64
wandb.init()


def ensure_midi(dataset="mario"):
    if dataset == "custom":
        if not os.path.exists("midi_songs"):
            raise ValueError(
                "Couldn't find custom soundtrack, please run python create_soundtrack.py")
        else:
            return True
    if not os.path.exists("data/%s" % dataset):
        print("Downloading %s dataset..." % dataset)
        subprocess.check_output(
            "curl -SL https://storage.googleapis.com/wandb/%s.tar.gz | tar xz" % dataset, shell=True)  # finalfantasy, hiphop, mario
        open("data/%s" % dataset, "w").close()


def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    if os.path.exists("data/notes"):
        return pickle.load(open("data/notes", "rb"))

    for file in glob.glob("midi_songs/*.mid"):
        try:
            midi = converter.parse(file)
        except TypeError:
            print("Invalid file %s" % file)
            continue

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    os.makedirs("data", exist_ok=True)
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(
        network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = tf.keras.utils.to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.CuDNNGRU(
        128,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(tf.keras.layers.CuDNNGRU(64, return_sequences=True))
    model.add(tf.keras.layers.CuDNNGRU(32))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(n_vocab))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


class Midi(tf.keras.callbacks.Callback):
    """
    Callback for sampling a midi file
    """

    def sample(self, preds, temperature=1):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_notes(self, network_input, pitchnames, n_vocab):
        """ Generate notes from the neural network based on a sequence of notes """
        # pick a random sequence from the input as a starting point for the prediction
        model = self.model
        start = np.random.randint(0, len(network_input)-1)

        int_to_note = dict((number, note)
                           for number, note in enumerate(pitchnames))

        pattern = list(network_input[start])
        prediction_output = []

        # generate 200 notes
        for note_index in range(200):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)

            prediction = model.predict(prediction_input, verbose=0)

            index = self.sample(prediction[0], temperature=0.5)  # np.argmax
            result = int_to_note[index]
            prediction_output.append(result)

            pattern.append(index)
            pattern = pattern[1:len(pattern)]

        return prediction_output

    def create_midi(self, prediction_output):
        """ convert the output from the prediction to notes and create a midi file
            from the notes """
        offset = 0
        output_notes = []

        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)

        return midi_stream.write('midi')

    def on_epoch_end(self, *args):
        notes = get_notes()
        # Get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # Get all pitch names
        n_vocab = len(set(notes))
        network_input, normalized_input = prepare_sequences(
            notes, n_vocab)
        music = self.generate_notes(network_input, pitchnames, n_vocab)
        midi = self.create_midi(music)
        midi = open(midi, "rb")
        data = "data:audio/midi;base64,%s" % base64.b64encode(
            midi.read()).decode("utf8")
        wandb.log({
            "midi": wandb.Html("""
                <script type="text/javascript" src="//www.midijs.net/lib/midi.js"></script>
                <button onClick="MIDIjs.play('%s')">Play midi</button>
                <button onClick="MIDIjs.stop()">Stop Playback</button>
            """ % data)
        }, commit=False)


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "mozart.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [Midi(), wandb.keras.WandbCallback(), checkpoint]

    model.fit(network_input, network_output, epochs=200,
              batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':
    ensure_midi("mario")
    train_network()
