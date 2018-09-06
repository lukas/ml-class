from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.callbacks import Callback
import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os

from PIL import Image
import numpy as np

run = wandb.init()
config = run.config

config.num_epochs = 100
config.batch_size = 4
config.img_dir = "images"
config.height = 256
config.width = 256

val_dir = 'dogcat-data/validation/cat'
train_dir = 'dogcat-data/train/cat'
if not os.path.exists("dogcat-data"):
      if os.path.exists("../keras-transfer/dogcat-data"):
            subprocess.check_output("cp -r ../keras-transfer/dogcat-data .", shell=True)
      else:
            print("Downloading dog/cat dataset...")
            subprocess.check_output("curl https://storage.googleapis.com/wandb-production.appspot.com/qualcomm/dogcat-data.tgz | tar xvz", shell=True)
            subprocess.check_output("rm dogcat-data/train/dog/._dog* dogcat-data/train/cat/._cat* dogcat-data/validation/cat/._cat* dogcat-data/validation/dog/._dog*", shell=True)


def my_generator(batch_size, img_dir):
      image_filenames = glob.glob(img_dir + "/*")
      counter = 0
      while True:
            bw_images = np.zeros((batch_size, config.width, config.height))
            color_images = np.zeros((batch_size, config.width, config.height, 3))
            random.shuffle(image_filenames) 
            if ((counter+1)*batch_size>=len(image_filenames)):
                  counter = 0
            for i in range(batch_size):
                  img = Image.open(image_filenames[counter + i]).resize((config.width, config.height))
                  color_images[i] = np.array(img)
                  bw_images[i] = np.array(img.convert('L'))
            yield (bw_images, color_images)
            counter += batch_size 

model = Sequential()
model.add(Reshape((config.height,config.width,1), input_shape=(config.height,config.width)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))


model.compile(optimizer='adam', loss='mse')

model.summary()

(val_bw_images, val_color_images) = next(my_generator(20, val_dir))

model.fit_generator( my_generator(config.batch_size, train_dir),
                     samples_per_epoch=20,
                     nb_epoch=config.num_epochs, callbacks=[WandbCallback(data_type='image', save_model=False)],
                     validation_data=(val_bw_images, val_color_images))
