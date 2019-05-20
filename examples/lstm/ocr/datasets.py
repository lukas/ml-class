"""
Simple and mnist datasets generated from https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project
IAM dataset from http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
"""

import h5py
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from urllib.request import urlretrieve
import argparse
import numpy as np
from tensorflow.keras.utils import Sequence


def _shuffle(x, y):
    """Shuffle x and y maintaining their association."""
    shuffled_indices = np.random.permutation(x.shape[0])
    return x[shuffled_indices], y[shuffled_indices]


DS = 'simple_lines.h5'  # emnist, iam
PROCESSED_DATA_DIRNAME = Path(__file__).parents[0].resolve() / 'data'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / DS
PROCESSED_DATA_URL = f'https://storage.googleapis.com/wandb/{DS}'
CHAR_CODE_MAPPING = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c',
                     39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z', 62: ' ', 63: '!', 64: '"', 65: '#', 66: '&', 67: "'", 68: '(', 69: ')', 70: '*', 71: '+', 72: ',', 73: '-', 74: '.', 75: '/', 76: ':', 77: ';', 78: '?', 79: '_'}
SIMPLE_CHAR_CODE_MAPPING = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm',
                            13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: ' ', 27: '_'}

parser = argparse.ArgumentParser()
parser.add_argument("--subsample_fraction",
                    type=float,
                    default=None,
                    help="If given, is used as the fraction of data to expose.")


class Generator(Sequence):
    """A simple generator to enable transforming batches"""

    def __init__(self, x, y, batch_size=32, augment_fn=None, format_fn=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augment_fn = augment_fn
        self.format_fn = format_fn

    def __len__(self):
        """Return length of the dataset."""
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        """Return a single batch."""
        # idx = 0  # If you want to intentionally overfit to just one batch
        begin = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        # batch_x = np.take(self.x, range(begin, end), axis=0, mode='clip')
        # batch_y = np.take(self.y, range(begin, end), axis=0, mode='clip')

        batch_x = self.x[begin:end]
        batch_y = self.y[begin:end]

        if batch_x.dtype == np.uint8:
            batch_x = (batch_x / 255).astype(np.float32)

        if self.augment_fn:
            batch_x, batch_y = self.augment_fn(batch_x, batch_y)

        if self.format_fn:
            batch_x, batch_y = self.format_fn(batch_x, batch_y)

        return batch_x, batch_y

    def on_epoch_end(self) -> None:
        """Shuffle data."""
        self.x, self.y = _shuffle(self.x, self.y)


class LinesDataset(object):
    """Dataset loader / transformer"""

    def __init__(self, batch_size: int = 32, subsample_fraction: float = None):
        self.mapping = SIMPLE_CHAR_CODE_MAPPING if "simple" in DS else CHAR_CODE_MAPPING
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.num_classes = len(self.mapping)

        if "iam" in DS:
            self.input_shape = (28, 952)
            self.output_shape = (97, self.num_classes)
        elif "simple" in DS:
            self.input_shape = (28, 280)
            self.output_shape = (10, self.num_classes)
        else:
            self.input_shape = (28, 952)
            self.output_shape = (34, self.num_classes)

        self.subsample_fraction = subsample_fraction
        self.batch_size = batch_size
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_or_generate_data(self):
        """Load or generate dataset data."""
        if not PROCESSED_DATA_FILENAME.exists():
            PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
            print(f'Downloading {DS} dataset...')
            urlretrieve(PROCESSED_DATA_URL, PROCESSED_DATA_FILENAME)
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['x_train'][:]
            if 'iam' in DS:
                self.y_train = to_categorical(
                    f['y_train'][:], self.num_classes)
            elif 'simple' in DS:
                # Translate the one-hot to simple chars
                tmp = [np.argmax(y, -1) - 36 for y in f['y_train']]
                tmp = np.clip(tmp, 0, 27)
                self.y_train = to_categorical(tmp)
            else:
                self.y_train = f['y_train'][:]
            self.x_test = f['x_test'][:]
            if 'iam' in DS:
                self.y_test = to_categorical(f['y_test'][:], self.num_classes)
            elif 'simple' in DS:
                # Translate the one-hot to simple chars
                tmp = [np.argmax(y, -1) - 36 for y in f['y_test']]
                tmp = np.clip(tmp, 0, 27)
                self.y_test = to_categorical(tmp)
            else:
                self.y_test = f['y_test'][:]
        self._subsample()

    def _subsample(self):
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return
        num_train = int(self.x_train.shape[0] * self.subsample_fraction)
        num_test = int(self.x_test.shape[0] * self.subsample_fraction)
        self.x_train = self.x_train[:num_train]
        self.y_train = self.y_train[:num_train]
        self.x_test = self.x_test[:num_test]
        self.y_test = self.y_test[:num_test]

    def __repr__(self):
        """Print info about the dataset."""
        return (
            'IAM Lines Dataset\n'  # pylint: disable=no-member
            f'Num classes: {self.num_classes}\n'
            f'Mapping: {self.mapping}\n'
            f'Train: {self.x_train.shape} {self.y_train.shape}\n'
            f'Test: {self.x_test.shape} {self.y_test.shape}\n'
        )


def main():
    """Load dataset and print info."""
    args = parser.parse_args()
    dataset = LinesDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == '__main__':
    main()
