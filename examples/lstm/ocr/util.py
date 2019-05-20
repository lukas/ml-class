"""Define utility functions."""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import ctc_ops  # pylint: disable=no-name-in-module
import numpy as np
import wandb


def slide_window(image, window_width, window_stride):
    """
    Takes (image_height, image_width, 1) input,
    Returns (num_windows, image_height, window_width, 1) output, where
    num_windows is floor((image_width - window_width) / window_stride) + 1
    """
    kernel = [1, 1, window_width, 1]
    strides = [1, 1, window_stride, 1]
    patches = tf.extract_image_patches(
        image, kernel, strides, [1, 1, 1, 1], 'VALID')
    patches = tf.transpose(patches, (0, 2, 1, 3))
    patches = tf.expand_dims(patches, -1)
    return patches


def ctc_decode(y_pred, input_length, max_output_length):
    """
    Cut down from https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py#L4170
    Decodes the output of a softmax.
    Uses greedy (best path) search.
    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        max_output_length: int giving the max output sequence length
    # Returns
        List: list of one element that contains the decoded sequence.
    """
    y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + K.epsilon())
    input_length = tf.to_int32((tf.squeeze(input_length, axis=-1)))

    (decoded, _) = ctc_ops.ctc_beam_search_decoder(  # ctc_greedy_decoder(
        inputs=y_pred, sequence_length=input_length, beam_width=1000)

    sparse = decoded[0]
    decoded_dense = tf.sparse_to_dense(
        sparse.indices, sparse.dense_shape, sparse.values, default_value=-1)

    # Unfortunately, decoded_dense will be of different number of columns, depending on the decodings.
    # We need to get it all in one standard shape, so let's pad if necessary.
    max_length = max_output_length + 2  # giving 2 extra characters for CTC leeway
    cols = tf.shape(decoded_dense)[-1]

    def pad():
        return tf.pad(decoded_dense, [[0, 0], [0, max_length - cols]], constant_values=-1)

    def noop():
        return decoded_dense

    return tf.cond(tf.less(cols, max_length), pad, noop)


def format_batch_ctc(batch_x, batch_y):
    """
    Because CTC loss needs to be computed inside of the network, we include information about outputs in the inputs.
    """
    batch_size = batch_y.shape[0]
    y_true = np.argmax(batch_y, axis=-1)

    label_lengths = []
    for ind in range(batch_size):
        # Find all of the indices in the label that are blank
        empty_at = np.where(batch_y[ind, :, -1] == 1)[0]
        # Length of the label is the pos of the first blank, or the max length
        if empty_at.shape[0] > 0:
            label_lengths.append(empty_at[0])
        else:
            label_lengths.append(batch_y.shape[1])

    batch_inputs = {
        'image_input': batch_x,
        'y_true': y_true,
        # dummy, will be set to num_windows in network (CURRENTLY HARDCODED)
        'input_length': np.ones((batch_size, 1)),
        'label_length': np.array(label_lengths)
    }
    batch_outputs = {
        'ctc_loss': np.zeros(batch_size),  # dummy
        'ctc_decoded': y_true
    }
    return batch_inputs, batch_outputs


def lev(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


class ExampleLogger(tf.keras.callbacks.Callback):
    def __init__(self, dataset, decode=None):
        self.dataset = dataset
        self.decode = decode

    def on_train_begin(self, logs):
        wandb.run.summary['graph'] = wandb.Graph.from_keras(self.model)

    def on_epoch_end(self, epoch, logs):
        indicies = np.random.choice(len(self.dataset.x_test), 9, replace=False)
        images = self.dataset.x_test[indicies]
        train_images = self.dataset.x_train[indicies]
        labels = self.dataset.y_test[indicies]
        train_labels = self.dataset.y_train[indicies]
        truth = np.argmax(labels, -1)
        truth_train = np.argmax(train_labels, -1)
        if self.decode:
            result_train = self.decode(train_images, train_labels)
            result = self.decode(images, labels)
        else:
            result_train = np.argmax(self.model.predict(train_images), -1)
            result = np.argmax(self.model.predict(images), -1)
        pred = [''.join(self.dataset.mapping.get(label, '')
                        for label in pred).strip(' |_') for pred in result]
        truth = [''.join(self.dataset.mapping.get(label, '')
                         for label in true).strip(' |_') for true in truth]
        pred_train = [''.join(self.dataset.mapping.get(label, '')
                              for label in pred).strip(' |_') for pred in result_train]
        truth_train = [''.join(self.dataset.mapping.get(label, '')
                               for label in true).strip(' |_') for true in truth_train]
        dists = [lev(list(a), list(b)) for a, b in zip(pred, truth)]
        print("Val Levenstein: ", np.mean(dists))
        print("Val Examples:\n" +
              "\n".join([f"{t}\n{p}\n---" for p, t in zip(pred, truth)]))
        print("Train :\n" +
              "\n".join([f"{t}\n{p}\n---" for p, t in zip(pred_train, truth_train)]))
        wandb.log({"examples": [
            wandb.Image(img, caption=f"Pred: \"{pred[i]}\" -- Truth: \"{truth[i]}\"") for i, img in enumerate(images)],
            "train_examples": [
                wandb.Image(img, caption=f"Pred: \"{pred_train[i]}\" -- Truth: \"{truth_train[i]}\"") for i, img in enumerate(train_images)
        ],
            **logs})
