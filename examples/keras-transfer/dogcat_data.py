import os
import glob
import subprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if not os.path.exists("dogcat-data"):
    print("Downloading dog/cat dataset...")
    subprocess.check_output("curl https://storage.googleapis.com/wandb-production.appspot.com/qualcomm/dogcat-data.tgz | tar xvz", shell=True)
    subprocess.check_output("rm dogcat-data/train/dog/._dog* dogcat-data/train/cat/._cat* dogcat-data/validation/cat/._cat* dogcat-data/validation/dog/._dog*", shell=True)

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

# data prep
def generators(preprocessing_function, img_width, img_height, batch_size=32, binary=False, shuffle=True,
               train_dir="dogcat-data/train", val_dir="dogcat-data/validation"):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary" if binary else "categorical",
        shuffle=shuffle
    )
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary" if binary else "categorical",
        shuffle=shuffle
    )
    return train_generator, validation_generator